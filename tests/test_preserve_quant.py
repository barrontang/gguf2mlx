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
from tests.quant_fixtures import DEFAULT_DIMS, write_llama_gguf


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
    assert report["preserve"]["aggregate_relative_l2"] < gguf_codes.PRESERVE_TOLERANCE

    # Q4_K and Q8_0 keep their codes; Q6_K cannot be expressed on a 32-wide grid
    # when its 16-value scales disagree, so it is decoded instead of requantized.
    assert report["preserve"]["preserved_by_qtype"] == {"Q4_K": 7, "Q8_0": 1}
    assert report["preserve"]["preserved_by_bits"] == {"4": 7, "8": 1}
    assert report["preserve"]["kept_float16"]["counts"] == {
        "F32_has_no_affine_reexpression": 3,
        "sub_block_scales_disagree": 1,
    }
    assert report["preserve"]["bit_exact_tensors"] >= 1

    config = json.loads((destination / "config.json").read_text())
    quantization = config["quantization"]
    assert quantization["group_size"] == 32
    assert quantization["mode"] == "affine"
    assert quantization["scheme"] == "gguf_preserve_v1"
    # The 8-bit tensor is the only module that needs an explicit override.
    assert quantization["model.layers.0.mlp.up_proj"] == {
        "bits": 8, "group_size": 32, "mode": "affine",
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


@pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3", "mistral"])
def test_preserve_is_supported_for_the_documented_architectures(tmp_path, arch):
    source = write_llama_gguf(tmp_path / f"{arch}.gguf", arch=arch, seed=11)
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
    import mlx.core as mx  # noqa: F401  (skip when MLX is unavailable)
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
