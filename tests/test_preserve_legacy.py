"""``preserve`` on the legacy 32-value block types, and what it refuses.

Two claims are tested here, and they are different claims:

1. **Exactness.** ``Q4_0``/``Q4_1``/``Q5_0``/``Q5_1`` cover exactly 32 values --
   MLX's minimum group size -- so their codes can be copied unchanged. The
   emitted tensor must decode *bit for bit* to what ``gguf.quants.dequantize``
   says the source block means. Anything less would mean the re-expression lost
   precision it did not need to lose.

2. **No dequantization on the conversion path.** ``preserve`` exists to avoid
   the float16 round trip; a call to ``dequantize`` inside the writer would
   reintroduce it silently, and would only show up as a slightly larger and
   slightly different artifact. The test replaces the symbol in both modules
   that could reach it with a function that raises, so the conversion has to
   succeed without it.

Claim 2 is deliberately scoped: ``audit_preserved`` *does* decode, because it is
the independent oracle the gate compares against. The tests therefore call the
preserve writer with ``audit=False`` and assert on the bytes, not on the audit.

The remaining tests cover the reverse direction: formats that cannot be
re-expressed are rejected for a *named structural* reason rather than falling
through to a generic label or, worse, being reported as preserved.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader
from safetensors import safe_open

from gguf2mlx import gguf2mlx
from gguf2mlx.quantize import gguf_codes
from tests.quant_fixtures import llama_tensor_plan, write_llama_gguf

LEGACY = ("Q4_0", "Q4_1", "Q5_0", "Q5_1")

#: Formats ``preserve`` rejects, and the reason the report must print for each.
REJECTED = {
    "Q2_K": "sub_block_scale_is_per_16_values",
    "Q3_K": "sub_block_scale_is_per_16_values",
    "IQ1_S": "codebook_has_no_affine_form",
    "IQ2_XXS": "codebook_has_no_affine_form",
    "IQ2_XS": "codebook_has_no_affine_form",
    "IQ2_S": "codebook_has_no_affine_form",
    "IQ3_XXS": "codebook_has_no_affine_form",
    "IQ3_S": "codebook_has_no_affine_form",
    "IQ4_NL": "codebook_has_no_affine_form",
    "IQ4_XS": "codebook_has_no_affine_form",
    "IQ1_M": "codebook_has_no_affine_form",
    "TQ1_0": "ternary_not_an_integer_grid",
    "TQ2_0": "ternary_not_an_integer_grid",
    "MXFP4": "block_float_has_no_affine_form",
}

#: Names of the F32 tensors in the fixture plan.
F32_TENSORS = 3


def _plan_for(kind: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The stock one-layer model with every matrix tensor in ``kind``."""
    return {
        name: (kind if qtype != "F32" else "F32", shape)
        for name, (qtype, shape) in llama_tensor_plan().items()
    }


@pytest.fixture(scope="module", params=LEGACY)
def legacy_gguf(request, tmp_path_factory) -> tuple[str, object]:
    root = tmp_path_factory.mktemp(f"legacy-{request.param}")
    path = write_llama_gguf(root / "tiny.gguf", plan=_plan_for(request.param), seed=13)
    return request.param, path


def _preserve(source, destination, **kwargs):
    """Run only the preserve writer, with the rotary permutation switched off.

    Head counts are ``None`` on purpose: the row interleave llama.cpp applies to
    ``attn_q``/``attn_k`` is correct and separately tested, but it would make a
    byte comparison against the source block orders meaningless here.
    """
    reader = GGUFReader(str(source))
    return gguf2mlx.extract_and_convert_weights_preserve(
        reader,
        "llama",
        destination,
        "float16",
        num_heads=None,
        kv_heads=None,
        audit=kwargs.pop("audit", False),
        **kwargs,
    )


def _stored(destination, name: str) -> np.ndarray:
    for shard in sorted(destination.glob("*.safetensors")):
        with safe_open(str(shard), framework="np") as handle:
            # ``safe_open`` is not iterable, so the key list has to be built
            # explicitly rather than using ``name in handle``.
            available = set(handle.keys())
            if name in available:
                return handle.get_tensor(name)
    raise AssertionError(f"{name} is not in any shard under {destination}")


def _plan_qtypes(report: dict) -> dict[str, int]:
    """Block types in the preserve plan, which is filled even without an audit."""
    counts: dict[str, int] = {}
    for record in report["preserve"]["plan"].values():
        counts[record["qtype"]] = counts.get(record["qtype"], 0) + 1
    return counts


def _matrix_count() -> int:
    return sum(1 for qtype, _shape in llama_tensor_plan().values() if qtype != "F32")


def test_legacy_block_type_is_preserved_bit_exactly(legacy_gguf, tmp_path):
    kind, source = legacy_gguf
    destination = tmp_path / f"out-{kind}"
    destination.mkdir()

    report = _preserve(source, destination, audit=True)
    matrices = _matrix_count()

    assert report["preserve"]["preserved_by_qtype"] == {kind: matrices}, (
        f"every matrix tensor should be preserved as {kind}, "
        f"got {report['preserve']['preserved_by_qtype']}"
    )
    assert report["preserve"]["gate"] == "pass"
    assert report["preserve"]["tiers"]["exact"]["status"] == "pass"
    assert report["preserve"]["tiers"]["inexact"]["tensors_compared"] == 0, (
        f"{kind} aligns on 32 values, so nothing may land in the inexact tier"
    )
    assert report["preserve"]["bit_exact_tensors"] == matrices

    bits = gguf_codes.mlx_bits(kind)
    assert bits is not None
    checked = 0
    for tensor in GGUFReader(str(source)).tensors:
        if str(tensor.tensor_type.name) != kind:
            continue
        logical = tuple(int(dim) for dim in reversed(tuple(tensor.shape)))
        expected = gguf_codes.reference_decode(tensor.data, tensor.tensor_type, logical)
        name = gguf2mlx._map_tensor_name(tensor.name, "llama").removesuffix(".weight")
        got = gguf_codes.affine_decode(
            _stored(destination, f"{name}.weight"),
            _stored(destination, f"{name}.scales"),
            _stored(destination, f"{name}.biases"),
            bits,
        )
        assert got.shape == expected.shape
        assert np.array_equal(got, expected), (
            f"{tensor.name}: {kind} round trip is not bit-exact "
            f"(max |diff| {np.abs(got - expected).max():.3e})"
        )
        checked += 1
    assert checked == matrices


def test_preserve_never_dequantizes(monkeypatch, legacy_gguf, tmp_path):
    """The writer must not reach ``dequantize`` -- that is the whole point."""
    kind, source = legacy_gguf
    destination = tmp_path / f"nodequant-{kind}"
    destination.mkdir()

    def _forbidden(*_args, **_kwargs):
        raise AssertionError(
            f"dequantize() was called while preserving {kind}; the codes are "
            f"meant to be copied, not decoded and re-encoded"
        )

    # Two independent routes to the oracle: the imported symbol in the
    # converter, and the module attribute that ``gguf_codes.reference_decode``
    # imports lazily. Patch both so neither can be taken. The audit is off,
    # because decoding *is* its job -- the claim being tested is about the
    # writer.
    monkeypatch.setattr(gguf2mlx, "dequantize", _forbidden)
    import gguf.quants

    monkeypatch.setattr(gguf.quants, "dequantize", _forbidden)

    report = _preserve(source, destination)
    assert _plan_qtypes(report) == {kind: _matrix_count()}


def test_f32_tensors_are_written_straight_through(legacy_gguf, tmp_path):
    """Norms bypass the quantized path entirely, and are reported as such."""
    kind, source = legacy_gguf
    destination = tmp_path / f"norms-{kind}"
    destination.mkdir()

    report = _preserve(source, destination)

    assert report["preserve"]["kept_float16"]["counts"] == {
        "F32_has_no_affine_reexpression": F32_TENSORS,
    }


@pytest.mark.parametrize(("kind", "reason"), sorted(REJECTED.items()))
def test_unmappable_types_are_rejected_for_a_named_reason(kind, reason):
    """A rejected type must say *why*, and must not claim a mapping."""
    qtype = GGMLQuantizationType[kind]

    assert gguf_codes.qtype_name(qtype) is None
    assert gguf_codes.non_affine_reason(qtype) == reason
    with pytest.raises(gguf_codes.UnsupportedQuantType):
        gguf_codes.map_tensor(b"\x00" * 64, qtype, (1, 32))

    skip = gguf2mlx._preserve_skip_reason((32, 32), qtype)
    assert skip == reason, (
        f"{kind} should be reported as {reason!r}, not {skip!r}; a generic label "
        f"hides whether the cause is sub-block scales or a codebook"
    )


def test_imatrix_exports_land_on_the_unmappable_types():
    """iMatrix-guided quantization becomes an IQ type, so this is the answer users hit.

    ``--quant-source preserve`` still converts such a model; it just cannot
    shrink it the way a ``Q4_K`` source does. Pinning the reason here keeps the
    prose in ``docs/quality-gates.md`` honest.
    """
    for name, reason in REJECTED.items():
        qtype = GGMLQuantizationType[name]
        assert gguf_codes.non_affine_reason(qtype) == reason, name
        assert gguf_codes.qtype_name(qtype) is None, name


def test_unknown_types_are_distinguished_from_known_rejections():
    """An unlabelled type must not borrow a known type's explanation."""
    class Bogus:
        name = "Q9_FAKE"

    assert gguf_codes.qtype_name(Bogus()) is None
    assert gguf_codes.non_affine_reason(Bogus()) is None
    assert gguf2mlx._preserve_skip_reason((32, 32), Bogus()) == (
        "Q9_FAKE_has_no_affine_reexpression"
    )


def test_a_source_with_nothing_preservable_is_refused(tmp_path):
    """An all-Q2_K source must fail before writing, and say what to do instead.

    Converting it would work -- every tensor would simply decode to float -- and
    the artifact would be *larger* than the GGUF, which is the opposite of what
    the mode promises. Silence there is worse than a refusal.
    """
    source = write_llama_gguf(tmp_path / "q2k.gguf", plan=_plan_for("Q2_K"), seed=17)
    destination = tmp_path / "refused"
    destination.mkdir()

    with pytest.raises(RuntimeError, match="quantized bytes") as info:
        _preserve(source, destination)

    message = str(info.value)
    assert "Q2_K (100%)" in message, "the refusal must name the block type and its share"
    assert "--direct-quant" in message, "the refusal must offer the alternative"
    assert not list(destination.glob("*.safetensors")), (
        "the refusal must happen before any shard is written"
    )


def test_the_majority_rule_is_measured_in_bytes_not_tensor_count(tmp_path):
    """A file with one preservable tensor can still be refused.

    The real ``Qwen1.5-MoE-A2.7B`` ``Q2_K`` file is ~96% ``Q2_K``/``IQ4_NL``/
    ``Q3_K`` by bytes but ships a single ``Q6_K`` tensor, so a count-based "is
    anything preservable?" test would answer yes and hand back a ~2.4x artifact.
    This fixture reproduces the shape of that problem in miniature: a small
    preservable tensor beside a larger un-preservable one.
    """
    plan = {
        "token_embd.weight": ("Q2_K", (64, 256)),
        "blk.0.attn_q.weight": ("Q4_K", (32, 256)),
        "output_norm.weight": ("F32", (256,)),
    }
    source = write_llama_gguf(tmp_path / "bytes.gguf", plan=plan, seed=23)
    destination = tmp_path / "refused"
    destination.mkdir()

    with pytest.raises(RuntimeError, match="quantized bytes") as info:
        _preserve(source, destination)

    message = str(info.value)
    assert "Q2_K" in message
    # 32 Q4_K blocks * 144 B = 4608 of 9984 quantized bytes, i.e. 46.2%. The
    # share has to be reported: "you asked for the wrong mode" is only credible
    # next to the number that says so.
    assert re.search(r"keep only 4\d\.\d%", message), message


def test_an_all_preservable_source_is_never_refused(tmp_path):
    """The rule must not touch the sources preserve was built for."""
    plan = {
        "token_embd.weight": ("Q4_K", (64, 256)),
        "output_norm.weight": ("F32", (256,)),
    }
    source = write_llama_gguf(tmp_path / "fine.gguf", plan=plan, seed=29)
    destination = tmp_path / "ok"
    destination.mkdir()

    report = _preserve(source, destination)
    assert _plan_qtypes(report) == {"Q4_K": 1}


def test_a_partly_preservable_source_still_converts(tmp_path):
    """Refusing on *some* unmappable tensors would be worse than warning."""
    plan = _plan_for("Q4_K")
    plan["blk.0.ffn_up.weight"] = ("Q6_K", plan["blk.0.ffn_up.weight"][1])
    source = write_llama_gguf(tmp_path / "mixed.gguf", plan=plan, seed=19)
    destination = tmp_path / "mixed"
    destination.mkdir()

    report = _preserve(source, destination)
    assert _plan_qtypes(report) == {"Q4_K": _matrix_count() - 1, "Q6_K": 1}


def test_the_32_value_rule_is_stated_in_the_module_docstring():
    """The rule is the reason Q6_K is special and Q4_0 is not; it must be written."""
    from pathlib import Path

    text = (
        Path(__file__).resolve().parent.parent
        / "src" / "gguf2mlx" / "quantize" / "gguf_codes.py"
    ).read_text(encoding="utf-8")
    assert "iMatrix" in text, "the docstring must name iMatrix, the case users hit"
    assert "divides 32" in text, "the exactness rule must be stated, not implied"
    for kind in ("Q2_K", "Q3_K", "Q6_K"):
        assert kind in text


def test_full_conversion_declares_the_legacy_width_to_mlx_lm(legacy_gguf, tmp_path):
    """The public path must emit a ``quantization`` block mlx-lm can act on.

    Widths of 4 and 5 bits are not exotic for mlx-lm -- ``Q5_K`` already ships
    5-bit -- but this is the assertion that a legacy source reaches the config,
    not just the safetensors shards.
    """
    import json

    kind, source = legacy_gguf
    destination = tmp_path / f"full-{kind}"

    assert gguf2mlx.convert(str(source), str(destination), quant_source="preserve") is True

    report = json.loads((destination / "conversion_report.json").read_text())
    assert report["quant_source"] == "preserve"
    assert report["gates"] == {"weight": "pass", "preserve": "pass"}

    config = json.loads((destination / "config.json").read_text())
    quantization = config["quantization"]
    assert quantization["bits"] == gguf_codes.mlx_bits(kind)
    assert quantization["group_size"] == gguf_codes.AFFINE_GROUP_SIZE
    assert quantization["mode"] == "affine"
    # One uniform width means no per-module overrides are needed.
    overrides = {k: v for k, v in quantization.items() if isinstance(v, dict)}
    assert overrides == {}
