"""End-to-end tests for ``--quant-source preserve`` on synthetic GGUFs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader
from safetensors import safe_open

from gguf2mlx import fidelity, gguf2mlx
from gguf2mlx.quantize import gguf_codes
from tests.quant_fixtures import DEFAULT_DIMS, write_llama_gguf, write_qwen35_gguf


@pytest.fixture(scope="module")
def source_gguf(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("preserve-src")
    return write_llama_gguf(root / "tiny.gguf", seed=3)


def _convert(source: Path, destination: Path, **kwargs) -> bool:
    return gguf2mlx.convert(str(source), str(destination), **kwargs)


def _report(destination: Path) -> dict:
    return json.loads((destination / "conversion_report.json").read_text())


def _read_tensor(destination: Path, name: str) -> np.ndarray:
    with safe_open(str(next(destination.glob("*.safetensors"))), framework="np") as handle:
        return handle.get_tensor(name)


def test_preserve_reuses_the_source_codes(source_gguf, tmp_path):
    destination = tmp_path / "out"

    assert _convert(source_gguf, destination, quant_source="preserve") is True

    report = _report(destination)
    assert report["quant_source"] == "preserve"
    assert report["gates"] == {"weight": "pass", "preserve": "pass"}
    assert report["preserve"]["group_size"] == 32

    # Q4_K and Q8_0 keep their codes. Q6_K cannot be expressed on a 32-wide grid
    # when its 16-value scales disagree, so it is re-fitted at 6 bits and
    # reported under its own tier -- not expanded to float16, which would have
    # cost 2.4x the bytes of the tensor.
    assert report["preserve"]["preserved_by_qtype"] == {"Q4_K": 7, "Q6_K": 1, "Q8_0": 1}
    assert report["preserve"]["preserved_by_bits"] == {"4": 7, "6": 1, "8": 1}
    assert report["preserve"]["fallback"] == "6bit"
    assert report["preserve"]["kept_float16"]["counts"] == {
        "F32_has_no_affine_reexpression": 3,
    }
    assert report["preserve"]["bit_exact_tensors"] >= 1

    tiers = report["preserve"]["tiers"]
    assert set(tiers) == {"exact", "inexact"}
    assert tiers["exact"]["status"] == "pass"
    assert tiers["exact"]["tolerance"] == gguf_codes.PRESERVE_TOLERANCE
    assert tiers["exact"]["aggregate_relative_l2"] < gguf_codes.PRESERVE_TOLERANCE
    assert tiers["inexact"]["status"] == "pass"
    assert tiers["inexact"]["tolerance"] == gguf_codes.PRESERVE_INEXACT_TOLERANCE["6bit"]
    assert tiers["inexact"]["tensors_compared"] == 1

    # The tier split is load-bearing: the re-fitted tensor is far outside the
    # exact ceiling and still inside its own.
    re_fitted = next(r for r in report["preserve"]["tensors"] if r["tier"] == "inexact")
    assert re_fitted["relative_l2"] > gguf_codes.PRESERVE_TOLERANCE
    assert re_fitted["relative_l2"] < gguf_codes.PRESERVE_INEXACT_TOLERANCE["6bit"]

    # Artifact size is the whole point of the mode.
    size = report["preserve"]["size"]
    assert size["source_bytes"] > 0
    assert 1.0 <= size["ratio"] < 1.15

    config = json.loads((destination / "config.json").read_text())
    quantization = config["quantization"]
    assert quantization["group_size"] == 32
    assert quantization["mode"] == "affine"
    assert quantization["scheme"] == "gguf_preserve_v1"
    # Only the tensors that differ from the 4-bit default carry an override.
    assert quantization["model.layers.0.mlp.up_proj"] == {
        "bits": 8, "group_size": 32, "mode": "affine",
    }
    assert quantization["model.layers.0.mlp.gate_proj"] == {
        "bits": 6, "group_size": 32, "mode": "affine",
    }

    # Spot-check the audited numbers against an independent decode of the GGUF.
    reader = GGUFReader(str(source_gguf))
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    reference = gguf_codes.reference_decode(
        by_name["blk.0.attn_v.weight"].data,
        by_name["blk.0.attn_v.weight"].tensor_type,
        (128, 256),
    )
    stored = _read_tensor(destination, "model.layers.0.self_attn.v_proj.weight")
    assert stored.dtype == np.uint32
    decoded = gguf_codes.affine_decode(
        stored,
        _read_tensor(destination, "model.layers.0.self_attn.v_proj.scales"),
        _read_tensor(destination, "model.layers.0.self_attn.v_proj.biases"),
        4,
    )
    assert np.linalg.norm(decoded - reference) / np.linalg.norm(reference) < 1e-3


def test_llama_rotary_rows_are_permuted_in_the_preserved_output(source_gguf, tmp_path):
    destination = tmp_path / "out"
    assert _convert(source_gguf, destination, quant_source="preserve") is True

    plan = _report(destination)["preserve"]["plan"]
    assert plan["model.layers.0.self_attn.q_proj.weight"]["rows"] == "rotary"
    assert plan["model.layers.0.self_attn.q_proj.weight"]["heads"] == DEFAULT_DIMS["heads"]
    assert plan["model.layers.0.self_attn.k_proj.weight"]["heads"] == DEFAULT_DIMS["kv_heads"]
    assert plan["model.layers.0.self_attn.o_proj.weight"]["rows"] == "identity"

    # The audit compares against the permuted reference; a missing permutation
    # would show up as a large relative L2 here.
    entry = next(
        record for record in _report(destination)["preserve"]["tensors"]
        if record["name"] == "model.layers.0.self_attn.q_proj.weight"
    )
    assert entry["relative_l2"] < gguf_codes.PRESERVE_TOLERANCE


def test_audit_fails_when_the_plan_points_at_the_wrong_source(source_gguf, tmp_path):
    destination = tmp_path / "out"
    assert _convert(source_gguf, destination, quant_source="preserve") is True
    report = _report(destination)

    plan = json.loads(json.dumps(report["preserve"]["plan"]))
    # Two same-shaped tensors: swapping them is exactly the mistake the audit
    # exists to catch.
    plan["model.layers.0.self_attn.q_proj.weight"]["source"] = "blk.0.attn_output.weight"
    plan["model.layers.0.self_attn.o_proj.weight"]["source"] = "blk.0.attn_q.weight"

    reader = GGUFReader(str(source_gguf))
    audit = fidelity.audit_preserved(destination, reader, plan)

    assert audit["gate"]["status"] == "fail"
    assert audit["gate"]["failure_count"] >= 1
    assert set(audit["gate"]["failures"]) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }


@pytest.mark.parametrize("arch", ["gemma", "phi3", "deepseek2"])
def test_preserve_fails_closed_for_unsupported_architectures(tmp_path, capsys, arch):
    source = write_llama_gguf(tmp_path / f"{arch}.gguf", arch=arch, seed=5)
    destination = tmp_path / "out"

    assert _convert(source, destination, quant_source="preserve") is False

    captured = capsys.readouterr().out
    assert arch in captured
    assert "--quant-source requant" in captured
    assert not destination.exists()


def test_preserve_never_silently_falls_back_to_requant(source_gguf, tmp_path, capsys):
    destination = tmp_path / "out"
    assert _convert(source_gguf, destination, quant_source="preserve") is True
    report = _report(destination)
    # A requantized artifact would carry no preserve block at all.
    assert "preserve" in report
    assert report["preserve"]["gate"] == "pass"
    assert "requantiz" not in capsys.readouterr().out.lower()


def test_fallback_float16_is_the_old_lossless_path(source_gguf, tmp_path):
    """The escape hatch still exists, and still costs the bytes this mode avoids."""
    destination = tmp_path / "out"
    assert _convert(
        source_gguf, destination, quant_source="preserve", preserve_fallback="float16",
    ) is True

    report = _report(destination)
    assert report["preserve"]["fallback"] == "float16"
    assert report["preserve"]["preserved_by_bits"] == {"4": 7, "8": 1}
    assert report["preserve"]["kept_float16"]["counts"] == {
        "F32_has_no_affine_reexpression": 3,
        "sub_block_scales_disagree": 1,
    }
    assert report["preserve"]["tiers"]["inexact"]["status"] == "not_run"
    # This is the regression: Q6_K expanded to float16 dominates the output.
    assert report["preserve"]["size"]["ratio"] > 1.3


def test_8bit_fallback_buys_accuracy_with_bytes(source_gguf, tmp_path):
    six_bit = tmp_path / "six"
    eight_bit = tmp_path / "eight"
    assert _convert(source_gguf, six_bit, quant_source="preserve") is True
    assert _convert(
        source_gguf, eight_bit, quant_source="preserve", preserve_fallback="8bit",
    ) is True

    six_report = _report(six_bit)["preserve"]
    eight_report = _report(eight_bit)["preserve"]

    assert eight_report["preserved_by_bits"] == {"4": 7, "8": 2}
    assert eight_report["tiers"]["inexact"]["tolerance"] == (
        gguf_codes.PRESERVE_INEXACT_TOLERANCE["8bit"]
    )
    six_error = six_report["tiers"]["inexact"]["max_relative_l2"]
    eight_error = eight_report["tiers"]["inexact"]["max_relative_l2"]
    assert eight_error < six_error / 2
    assert eight_report["size"]["ratio"] > six_report["size"]["ratio"]


def test_unknown_fallback_is_rejected(source_gguf, tmp_path, capsys):
    assert _convert(
        source_gguf, tmp_path / "out", quant_source="preserve", preserve_fallback="4bit",
    ) is False
    assert "--preserve-fallback" in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3", "mistral"])
def test_preserve_is_supported_for_the_documented_architectures(tmp_path, arch):
    source = write_llama_gguf(tmp_path / f"{arch}.gguf", arch=arch, seed=11)
    destination = tmp_path / "out"

    assert _convert(source, destination, quant_source="preserve") is True
    assert _report(destination)["preserve"]["gate"] == "pass"


def test_preserve_reuses_qwen35_projection_codes_and_keeps_ssm_vectors_float(tmp_path):
    source = write_qwen35_gguf(tmp_path / "qwen35.gguf")
    destination = tmp_path / "out"

    assert _convert(source, destination, quant_source="preserve") is True
    report = _report(destination)
    preserve = report["preserve"]
    assert preserve["gate"] == "pass"
    # The float fallbacks must still audit clean: keep-in-float is not a licence
    # to shift values, so A_log is the only tensor the plan expects to change.
    assert report["gates"]["weight"] == "pass"
    assert preserve["preserved_by_qtype"]["Q4_K"] > 0
    assert preserve["preserved_by_qtype"]["Q8_0"] == 1
    # SSM scalars and the depthwise convolution have no affine matrix form.
    assert preserve["kept_float16"]["counts"] == {
        "F32_has_no_affine_reexpression": 7,
    }

    plan = preserve["plan"]
    assert plan["model.layers.0.linear_attn.in_proj_qkv.weight"]["rows"] == "identity"
    assert plan["model.layers.0.linear_attn.in_proj_b.weight"]["bits"] == 4
    assert plan["model.layers.0.linear_attn.in_proj_a.weight"]["bits"] == 4
    assert "model.layers.0.linear_attn.A_log" not in plan
    assert "model.layers.0.linear_attn.conv1d.weight" not in plan

    layout = report["tensor_layout"]
    assert layout["model.layers.0.linear_attn.A_log"]["layout"] == "transformed"
    assert layout["model.layers.0.linear_attn.conv1d.weight"]["layout"] == "reshaped"
    assert layout["model.layers.0.linear_attn.norm.weight"]["layout"] == "identity"

    # A_log is log(-ssm_a) and the squeezed kernel gains MLX's trailing axis;
    # the norms are *not* offset (llama.cpp skips its +1 centering here).
    source_tensors = {t.name: fidelity.reference_tensor(t) for t in GGUFReader(str(source)).tensors}
    with safe_open(str(next(destination.glob("*.safetensors"))), framework="np") as handle:
        a_log = handle.get_tensor("model.layers.0.linear_attn.A_log")
        conv1d = handle.get_tensor("model.layers.0.linear_attn.conv1d.weight")
        norm = handle.get_tensor("model.layers.0.linear_attn.norm.weight")

    ssm_a = source_tensors["blk.0.ssm_a"]
    assert np.all(ssm_a < 0)  # GGUF stores -exp(A_log)
    assert np.allclose(a_log, np.log(-ssm_a), atol=1e-3)
    assert conv1d.shape == (768, 4, 1)
    assert np.allclose(conv1d[..., 0], source_tensors["blk.0.ssm_conv1d.weight"], atol=1e-3)
    assert np.allclose(norm, source_tensors["blk.0.ssm_norm.weight"], atol=1e-3)

    quantization = json.loads((destination / "config.json").read_text())["quantization"]
    assert quantization["bits"] == 4 and quantization["group_size"] == 32
    assert quantization["model.layers.0.mlp.up_proj"]["bits"] == 8
    # Neither the gated norm nor any SSM state vector may be quantized.
    assert not [
        key
        for key in quantization
        if "conv1d" in key or "A_log" in key or "dt_bias" in key or "norm" in key
    ]



@pytest.mark.parametrize("arch", ["qwen36", "qwen38"])
def test_preserve_accepts_qwen35_alias_architectures(tmp_path, arch):
    source = write_qwen35_gguf(tmp_path / f"{arch}.gguf", arch=arch)
    destination = tmp_path / "out"

    assert _convert(source, destination, quant_source="preserve") is True
    assert _report(destination)["preserve"]["gate"] == "pass"


def test_preserve_rejects_incompatible_output_options(source_gguf, tmp_path, capsys):
    assert _convert(
        source_gguf, tmp_path / "fp32", quant_source="preserve", dtype="float32"
    ) is False
    assert "float16" in capsys.readouterr().out

    assert _convert(
        source_gguf, tmp_path / "bf16", quant_source="preserve", dtype="bfloat16"
    ) is False
    assert "bfloat16" in capsys.readouterr().out

    assert _convert(
        source_gguf, tmp_path / "mxfp4", quant_source="preserve",
        q_mode="mxfp4", quantize=True,
    ) is False
    assert "affine" in capsys.readouterr().out

    assert _convert(
        source_gguf, tmp_path / "mixed", quant_source="preserve",
        mixed_precision=True, quantize=True, direct_quant=True,
    ) is False
    assert "mixed-precision" in capsys.readouterr().out


def test_deprecated_alias_still_means_preserve(source_gguf, tmp_path, capsys):
    destination = tmp_path / "alias"
    assert _convert(source_gguf, destination, preserve_source_quant=True) is True
    assert "deprecated" in capsys.readouterr().err
    assert _report(destination)["quant_source"] == "preserve"

    conflict = tmp_path / "conflict"
    assert _convert(
        source_gguf, conflict, preserve_source_quant=True, quant_source="requant"
    ) is False
    assert "conflicts" in capsys.readouterr().out
    assert not conflict.exists()


def test_requant_remains_the_default(source_gguf, tmp_path):
    destination = tmp_path / "default"

    assert _convert(source_gguf, destination) is True

    report = _report(destination)
    assert report.get("quant_source") is None
    assert "preserve" not in report
    config = json.loads((destination / "config.json").read_text())
    assert "quantization" not in config
    names = set()
    with safe_open(str(next(destination.glob("*.safetensors"))), framework="np") as handle:
        names = set(handle.keys())
    assert not any(name.endswith(".scales") for name in names)


def _run_cli(argv: list[str]) -> int:
    try:
        gguf2mlx.main(argv)
    except SystemExit as exit_info:  # non-zero paths call sys.exit
        return int(exit_info.code or 0)
    return 0


def test_cli_accepts_quant_source_and_the_alias(source_gguf, tmp_path, capsys):
    explicit = tmp_path / "cli-explicit"
    assert _run_cli([
        "convert", "--input", str(source_gguf), "--output", str(explicit),
        "--quant-source", "preserve",
    ]) == 0
    assert _report(explicit)["quant_source"] == "preserve"

    alias = tmp_path / "cli-alias"
    assert _run_cli([
        "convert", "--input", str(source_gguf), "--output", str(alias),
        "--preserve-source-quant",
    ]) == 0
    assert "deprecated" in capsys.readouterr().err
    assert _report(alias)["quant_source"] == "preserve"


def test_mlx_lm_loads_the_mixed_width_artifact(source_gguf, tmp_path):
    # Loading the artifact is a numerical check: it needs MLX, which only ships
    # Apple Silicon wheels. The rest of the preserve suite is numpy-only.
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm import load

    destination = tmp_path / "out"
    assert _convert(source_gguf, destination, quant_source="preserve") is True

    model, _tokenizer = load(str(destination))
    modules = dict(model.named_modules())
    assert modules["model.layers.0.self_attn.q_proj"].bits == 4
    assert modules["model.layers.0.self_attn.q_proj"].group_size == 32
    assert modules["model.layers.0.mlp.up_proj"].bits == 8

    result = fidelity.load_and_run(destination)
    assert result["status"] == "pass", result


def test_cli_accepts_preserve_fallback(source_gguf, tmp_path):
    destination = tmp_path / "cli-8bit"
    assert _run_cli([
        "convert", "--input", str(source_gguf), "--output", str(destination),
        "--quant-source", "preserve", "--preserve-fallback", "8bit",
    ]) == 0
    assert _report(destination)["preserve"]["fallback"] == "8bit"


def test_preserve_report_matches_the_documented_schema(source_gguf, tmp_path):
    destination = tmp_path / "out"
    assert _convert(source_gguf, destination, quant_source="preserve") is True
    report = _report(destination)

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "conversion-report.schema.json")
        .read_text()
    )
    preserve_schema = schema["properties"]["preserve"]
    for key in preserve_schema["required"]:
        assert key in report["preserve"]

    entry = report["preserve"]["tensors"][0]
    for key in ("name", "source", "qtype", "bits", "group_size", "max_abs_error",
                "relative_l2", "bit_exact"):
        assert key in entry
