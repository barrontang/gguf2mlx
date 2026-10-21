"""Unit tests for the GGUF block -> MLX affine re-expression."""

from __future__ import annotations

import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from gguf2mlx.quantize import gguf_codes
from tests.quant_fixtures import (
    encode_q4_k,
    encode_q6_k,
    encode_q8_0,
    q4_k_blocks,
    q6_k_blocks,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    )


# ---------------------------------------------------------------------------
# Known codes -> known scale/bias
# ---------------------------------------------------------------------------


def test_q4_k_maps_known_codes_to_known_scale_and_bias():
    """d=0.5, dmin=0.25, sc=2, min=3 => scale 1.0 and bias -0.75, codes copied."""
    pattern = np.arange(32, dtype=np.uint8) % 16
    codes = np.tile(pattern, (1, 8, 1))
    raw = q4_k_blocks(
        d=np.array([0.5], dtype=np.float32),
        dmin=np.array([0.25], dtype=np.float32),
        scales=np.full((1, 8), 2, dtype=np.int32),
        mins=np.full((1, 8), 3, dtype=np.int32),
        codes=codes,
    )

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q4_K, (1, 256))

    assert mapped.bits == 4
    assert mapped.group_size == 32
    assert mapped.exact is True
    assert mapped.scales.dtype == np.float16
    np.testing.assert_array_equal(mapped.scales, np.full((1, 8), 1.0, dtype=np.float16))
    np.testing.assert_array_equal(mapped.biases, np.full((1, 8), -0.75, dtype=np.float16))

    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )
    expected = codes.astype(np.float32) * 1.0 - 0.75
    np.testing.assert_allclose(decoded, expected.reshape(1, 256), rtol=0, atol=1e-6)

    # The oracle agrees: this is a re-expression, not a fit.
    oracle = dequantize(raw, GGMLQuantizationType.Q4_K).reshape(1, 256)
    np.testing.assert_allclose(decoded, oracle, rtol=0, atol=1e-6)


def test_q8_0_is_a_bit_exact_shift_of_the_source_codes():
    generator = np.random.default_rng(3)
    d = generator.uniform(5e-4, 4e-3, 16).astype(np.float16)
    codes = generator.integers(-127, 128, (16, 32)).astype(np.int8)
    raw = np.zeros((16, 34), dtype=np.uint8)
    raw[:, 0:2] = d.view(np.uint8).reshape(16, 2)
    raw[:, 2:] = codes.view(np.uint8)

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q8_0, (4, 128))

    assert mapped.bits == 8
    assert mapped.exact is True
    np.testing.assert_array_equal(mapped.scales, d.astype(np.float32).reshape(4, 4).astype(np.float16))
    np.testing.assert_array_equal(
        mapped.biases, (-128.0 * d.astype(np.float32)).reshape(4, 4).astype(np.float16)
    )
    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )
    oracle = dequantize(raw, GGMLQuantizationType.Q8_0).reshape(4, 128)
    # Every stored value is exactly d*q in both representations.
    np.testing.assert_allclose(decoded, oracle, rtol=0, atol=1e-7)


def test_q6_k_is_exact_when_pair_scales_match():
    generator = np.random.default_rng(11)
    values = generator.normal(0, 0.05, (4, 256)).astype(np.float32)
    raw = encode_q6_k(values, equal_pair_scales=True)

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (4, 256))

    assert mapped.bits == 6
    assert mapped.exact is True
    assert mapped.inexact_groups == 0
    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )
    oracle = dequantize(raw, GGMLQuantizationType.Q6_K).reshape(4, 256)
    assert _relative_l2(oracle, decoded) < gguf_codes.PRESERVE_TOLERANCE


def test_q6_k_flags_the_error_mlx_group_32_cannot_express():
    """Mismatched 16-value scales cost ~1e-2, an order above the shipped gate."""
    generator = np.random.default_rng(12)
    values = generator.normal(0, 0.05, (4, 256)).astype(np.float32)
    raw = encode_q6_k(values, equal_pair_scales=False)

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (4, 256))

    assert mapped.exact is False
    assert mapped.inexact_groups > 0
    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )
    oracle = dequantize(raw, GGMLQuantizationType.Q6_K).reshape(4, 256)
    error = _relative_l2(oracle, decoded)
    assert error > gguf_codes.PRESERVE_TOLERANCE
    # ... but inside the ceiling the pipeline gates this tier on, so the tensor
    # can stay at 6 bits instead of being expanded to float16.
    assert error < gguf_codes.PRESERVE_INEXACT_TOLERANCE["6bit"]
    assert np.isfinite(decoded).all()


def test_q6_k_output_does_not_depend_on_the_chunk_size(monkeypatch):
    """Chunking exists for embedding tables; it must not change a single byte."""
    generator = np.random.default_rng(21)
    values = generator.normal(0, 0.05, (16, 512)).astype(np.float32)
    raw = encode_q6_k(values)

    whole = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (16, 512))
    monkeypatch.setattr(gguf_codes, "_CHUNK_VALUES", 512)
    chunked = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (16, 512))

    assert np.array_equal(whole.packed, chunked.packed)
    assert np.array_equal(whole.scales, chunked.scales)
    assert np.array_equal(whole.biases, chunked.biases)
    assert whole.inexact_groups == chunked.inexact_groups


def test_refit_to_affine_8bit_is_more_accurate_than_6bit():
    generator = np.random.default_rng(31)
    values = generator.normal(0, 0.05, (8, 256)).astype(np.float32)

    six = gguf_codes.refit_tensor_to_affine(values, 6)
    eight = gguf_codes.refit_tensor_to_affine(values, 8)

    assert six.exact is False and eight.exact is False
    six_error = _relative_l2(
        values, gguf_codes.affine_decode(six.packed, six.scales, six.biases, 6)
    )
    eight_error = _relative_l2(
        values, gguf_codes.affine_decode(eight.packed, eight.scales, eight.biases, 8)
    )
    assert eight_error < six_error / 2
    assert six_error < gguf_codes.PRESERVE_INEXACT_TOLERANCE["6bit"]
    assert eight_error < gguf_codes.PRESERVE_INEXACT_TOLERANCE["8bit"]


def test_refit_to_affine_rejects_misaligned_shapes():
    with pytest.raises(ValueError):
        gguf_codes.refit_tensor_to_affine(np.zeros((4, 48), dtype=np.float32), 6)
    with pytest.raises(ValueError):
        gguf_codes.refit_tensor_to_affine(np.zeros((4, 32, 2), dtype=np.float32), 6)


def test_q6_k_rounds_ties_away_from_zero_like_ggml():
    """A 2.5-code tie must become 3, matching ``std::round`` (not banker's)."""
    # Sub-block scales 4 and 2 inside one 32-value group: ratio 0.5. m = 5 gives
    # exactly 2.5, the tie that distinguishes the two rounding modes.
    codes = np.full((1, 16, 16), 32, dtype=np.uint8)
    codes[0, 1, :] = 32 + 5      # sub-block 1: scale 2, m = +5 => 2.5
    codes[0, 3, :] = 32 - 5      # sub-block 3: scale 2, m = -5 => -2.5
    raw = q6_k_blocks(
        d=np.array([0.001], dtype=np.float32),
        sub_scales=np.array([[4, 2, 4, 2] + [4] * 12], dtype=np.int8),
        codes=codes,
    )

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (1, 256))
    emitted = gguf_codes.unpack_codes(mapped.packed, 6, 256).reshape(1, 8, 2, 16)

    assert emitted[0, 0, 1, 0] == 35      # round( 2.5) == 3, away from zero
    assert emitted[0, 1, 1, 0] == 29      # round(-2.5) == -3, away from zero


def test_q6_k_zero_scale_group_does_not_produce_nan():
    codes = np.full((1, 16, 16), 60, dtype=np.uint8)
    raw = q6_k_blocks(
        d=np.array([0.002], dtype=np.float32),
        sub_scales=np.array([[0, 0] + [7] * 14], dtype=np.int8),
        codes=codes,
    )

    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q6_K, (1, 256))
    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )

    assert np.isfinite(decoded).all()
    assert mapped.scales[0, 0] == 0
    assert np.all(decoded.reshape(1, 8, 32)[:, 0, :] == 0)


# ---------------------------------------------------------------------------
# Round trips against the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("encoder", "qtype"),
    [
        (encode_q4_k, GGMLQuantizationType.Q4_K),
        (encode_q8_0, GGMLQuantizationType.Q8_0),
        (lambda values: encode_q6_k(values, equal_pair_scales=True),
         GGMLQuantizationType.Q6_K),
    ],
)
def test_round_trip_stays_inside_the_audit_tolerance(encoder, qtype):
    generator = np.random.default_rng(21)
    values = generator.normal(0, 0.04, (8, 256)).astype(np.float32)
    raw = encoder(values)
    shape = (8, 256)

    mapped = gguf_codes.map_tensor(raw, qtype, shape)
    decoded = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, mapped.bits
    )
    oracle = gguf_codes.reference_decode(raw, qtype, shape)

    assert decoded.shape == oracle.shape
    assert np.isfinite(decoded).all()
    assert _relative_l2(oracle, decoded) < gguf_codes.PRESERVE_TOLERANCE


def test_pack_and_unpack_round_trip_every_width():
    generator = np.random.default_rng(5)
    for bits in (4, 6, 8):
        codes = generator.integers(0, 1 << bits, (3, 128)).astype(np.uint8)
        packed = gguf_codes.pack_codes(codes, bits)
        assert packed.dtype == np.uint32
        assert packed.shape == (3, 128 * bits // 32)
        np.testing.assert_array_equal(gguf_codes.unpack_codes(packed, bits, 128), codes)


def test_packing_matches_mlx_core_dequantize():
    """The stored word layout must be exactly what MLX reads back."""
    mx = pytest.importorskip("mlx.core")
    generator = np.random.default_rng(6)
    ones = None
    for bits in (4, 6, 8):
        codes = generator.integers(0, 1 << bits, (3, 64)).astype(np.uint8)
        packed = gguf_codes.pack_codes(codes, bits)
        ones = mx.ones((3, 2), dtype=mx.float16)
        zeros = mx.zeros((3, 2), dtype=mx.float16)
        decoded = np.array(
            mx.dequantize(mx.array(packed), ones, zeros, group_size=32, bits=bits)
        )
        np.testing.assert_allclose(decoded, codes.astype(np.float32), rtol=0, atol=0)


def test_affine_decode_matches_mlx_for_scaled_groups():
    mx = pytest.importorskip("mlx.core")
    generator = np.random.default_rng(8)
    codes = generator.integers(0, 16, (2, 64)).astype(np.uint8)
    packed = gguf_codes.pack_codes(codes, 4)
    scales = generator.normal(0, 0.01, (2, 2)).astype(np.float16)
    biases = generator.normal(0, 0.01, (2, 2)).astype(np.float16)

    mine = gguf_codes.affine_decode(packed, scales, biases, 4)
    theirs = np.array(
        mx.dequantize(
            mx.array(packed), mx.array(scales), mx.array(biases), group_size=32, bits=4
        )
    )
    # MLX evaluates ``scale * code + bias`` in float16; numpy mirrors it in
    # float32, so the two agree to float16 rounding rather than bit-for-bit.
    np.testing.assert_allclose(mine, theirs, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Contract / failure modes
# ---------------------------------------------------------------------------


def test_map_tensor_rejects_a_block_type_without_a_mapping():
    with pytest.raises(gguf_codes.UnsupportedQuantType, match="Q5_K"):
        gguf_codes.map_tensor(
            np.zeros(176, dtype=np.uint8), GGMLQuantizationType.Q5_K, (1, 256)
        )


def test_map_tensor_rejects_a_payload_that_does_not_match_the_shape():
    with pytest.raises(ValueError, match="payload"):
        gguf_codes.map_tensor(
            np.zeros(100, dtype=np.uint8), GGMLQuantizationType.Q4_K, (1, 256)
        )


@pytest.mark.parametrize("shape", [(256,), (0, 256), (2, 3, 256)])
def test_map_tensor_rejects_non_matrix_shapes(shape):
    with pytest.raises(ValueError, match="2-D shape"):
        gguf_codes.map_tensor(
            np.zeros(1, dtype=np.uint8), GGMLQuantizationType.Q4_K, shape
        )


def test_unaligned_row_length_is_rejected():
    with pytest.raises(ValueError, match="multiple of 256"):
        gguf_codes.map_tensor(
            np.zeros(144, dtype=np.uint8), GGMLQuantizationType.Q4_K, (1, 32)
        )


@pytest.mark.parametrize("bits", [2, 3, 5, 16])
def test_unsupported_affine_widths_are_rejected(bits):
    with pytest.raises(ValueError, match="bit-width"):
        gguf_codes.pack_codes(np.zeros((1, 32), dtype=np.uint8), bits)


def test_affine_decode_validates_group_shapes():
    with pytest.raises(ValueError, match="equal in shape"):
        gguf_codes.affine_decode(
            np.zeros((1, 4), dtype=np.uint32),
            np.zeros((1, 2), dtype=np.float16),
            np.zeros((1, 3), dtype=np.float16),
            4,
        )


def test_qtype_helpers_agree_on_the_supported_set():
    assert gguf_codes.qtype_name(GGMLQuantizationType.Q4_K) == "Q4_K"
    assert gguf_codes.qtype_name(12) == "Q4_K"
    assert gguf_codes.qtype_name(GGMLQuantizationType.Q5_K) is None
    assert gguf_codes.mlx_bits("Q8_0") == 8
    assert gguf_codes.block_elements("Q6_K") == 256
    assert gguf_codes.block_elements("nope") is None


def test_rotary_row_permutation_commutes_with_decoding():
    """Permuting stored rows must equal permuting the decoded values."""
    generator = np.random.default_rng(13)
    values = generator.normal(0, 0.05, (16, 256)).astype(np.float32)
    raw = encode_q4_k(values)
    mapped = gguf_codes.map_tensor(raw, GGMLQuantizationType.Q4_K, (16, 256))

    permuted = gguf_codes.llama_rotary_rows(mapped.packed, heads=4)
    permuted_scales = gguf_codes.llama_rotary_rows(mapped.scales, heads=4)
    permuted_biases = gguf_codes.llama_rotary_rows(mapped.biases, heads=4)
    after = gguf_codes.affine_decode(permuted, permuted_scales, permuted_biases, 4)

    before = gguf_codes.affine_decode(
        mapped.packed, mapped.scales, mapped.biases, 4
    )
    np.testing.assert_array_equal(after, gguf_codes.llama_rotary_rows(before, heads=4))
    assert not np.array_equal(after, before)


def test_rotary_row_permutation_rejects_undivisible_rows():
    with pytest.raises(ValueError, match="head_dim"):
        gguf_codes.llama_rotary_rows(np.zeros((10, 4), dtype=np.float32), heads=4)
