"""Executable MVP layout research, intentionally outside the converter."""

from __future__ import annotations

import os
import platform

import numpy as np
import pytest
from gguf import GGMLQuantizationType
from gguf.quants import dequantize


def _fixture(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bits = 4 if kind == "Q4_0" else 8
    indices = np.arange(256)
    codes = (
        (indices + indices // 16) % 16 if bits == 4 else indices
    ).astype(np.uint8).reshape(2, 128)
    scales = np.array(
        [[0.5, -0.25, 0.125, -1.0], [0.25, 0.5, -0.5, 0.0625]], dtype="<f2"
    )
    blocks = codes.reshape(-1, 32)
    if bits == 4:
        payload = blocks[:, :16] | (blocks[:, 16:] << 4)
    else:
        payload = (blocks.astype(np.int16) - 128).astype(np.int8).view(np.uint8)
    raw = np.concatenate((scales.reshape(-1, 1).view(np.uint8), payload), axis=1)
    return raw.reshape(2, -1), codes, scales, -scales * (1 << (bits - 1))


def _research_pack(raw: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_bytes = 18 if bits == 4 else 34
    blocks = raw.reshape(-1, block_bytes)
    scales = blocks[:, :2].copy().view("<f2").reshape(raw.shape[0], -1)
    payload = blocks[:, 2:]
    if bits == 4:
        codes = np.concatenate((payload & 15, payload >> 4), axis=1)
    else:
        codes = payload.copy().view(np.int8).astype(np.int16) + 128
    per_word = 32 // bits
    codes = codes.astype(np.uint32).reshape(raw.shape[0], -1, per_word)
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    packed = np.bitwise_or.reduce(codes << shifts, axis=-1)
    biases = -scales * (1 << (bits - 1))
    return packed, scales, biases


@pytest.mark.parametrize("kind,bits,block_bytes", [("Q4_0", 4, 18), ("Q8_0", 8, 34)])
def test_layout_against_gguf_oracle(kind: str, bits: int, block_bytes: int):
    raw, codes, expected_scales, expected_biases = _fixture(kind)
    packed, scales, biases = _research_pack(raw, bits)
    assert raw.nbytes == 8 * block_bytes
    assert packed.dtype == np.uint32
    assert packed.shape == (2, 128 // (32 // bits))
    assert scales.shape == biases.shape == (2, 4)
    assert scales.tobytes() == expected_scales.tobytes()
    np.testing.assert_array_equal(biases, expected_biases)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    unpacked = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(2, 128)
    np.testing.assert_array_equal(unpacked, codes)
    decoded = (
        unpacked.astype(np.float32) * np.repeat(scales.astype(np.float32), 32, axis=1)
        + np.repeat(biases.astype(np.float32), 32, axis=1)
    )
    oracle = dequantize(raw, GGMLQuantizationType[kind])
    np.testing.assert_array_equal(decoded, oracle)
    differing_bits = decoded.view(np.uint32) != oracle.view(np.uint32)
    assert not differing_bits[oracle != 0].any()
    if bits == 4:
        # d*(q-8) retains negative zero; affine cancellation produces positive zero.
        assert differing_bits.any()
        assert np.signbit(oracle[differing_bits]).all()
        assert not np.signbit(decoded[differing_bits]).any()
    if bits == 4:
        assert packed[0, 0] == 0x76543210
        assert packed[0, 1] == 0xFEDCBA98
    else:
        assert packed[0, 0] == 0x03020100
        assert unpacked.min() == 0 and unpacked.max() == 255


@pytest.mark.parametrize("bits", [4, 8])
def test_fp16_bias_overflow_is_not_lossless(bits: int):
    scale = np.array([65504], dtype=np.float16)
    with np.errstate(over="ignore"):
        bias16 = -scale * (1 << (bits - 1))
    assert not np.isfinite(bias16).all()
    bias32 = -scale.astype(np.float32) * (1 << (bits - 1))
    assert np.isfinite(bias32).all()


@pytest.mark.skipif(
    os.getenv("GGUF2MLX_RUN_E2E") != "1"
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Set GGUF2MLX_RUN_E2E=1 on Apple Silicon for the MLX packed-array contract",
)
@pytest.mark.parametrize("kind,bits", [("Q4_0", 4), ("Q8_0", 8)])
def test_mlx_accepts_external_packed_arrays(kind: str, bits: int):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    raw, _, _, _ = _fixture(kind)
    packed, scales, biases = _research_pack(raw, bits)
    weight, scale, bias = mx.array(packed), mx.array(scales), mx.array(biases)
    decoded = mx.dequantize(weight, scale, bias, group_size=32, bits=bits)
    oracle = dequantize(raw, GGMLQuantizationType[kind]).astype(np.float16)
    mx.eval(decoded)
    np.testing.assert_array_equal(np.asarray(decoded), oracle)
    layer = nn.QuantizedLinear(128, 2, bias=False, group_size=32, bits=bits)
    layer.weight, layer.scales, layer.biases = weight, scale, bias
    inputs = mx.array(np.eye(128, dtype=np.float16))
    result = layer(inputs)
    expected = mx.quantized_matmul(inputs, weight, scale, bias, transpose=True,
                                   group_size=32, bits=bits)
    mx.eval(result, expected)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(result), oracle.T)
