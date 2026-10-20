"""Fidelity checks for the affine 4-bit quantizer used by ``--direct-quant``.

These tests pin the two properties that used to be implicit:

* rounding is half-away-from-zero (``np.round`` is half-to-even, which produced
  one-LSB differences against MLX's C++ quantizer), and
* non-finite inputs are rejected instead of being silently turned into garbage.
"""

from __future__ import annotations

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core


def _unpack(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    shifts = np.arange(8, dtype=np.uint32) * 4
    return ((packed[..., None] >> shifts) & 15).reshape(shape)


def _dequantize(packed, scales, biases, shape: tuple[int, ...]) -> np.ndarray:
    codes = _unpack(packed, shape).astype(np.float32)
    repeat = shape[-1] // scales.shape[-1]
    values = codes * np.repeat(scales.astype(np.float32), repeat, axis=-1)
    values += np.repeat(biases.astype(np.float32), repeat, axis=-1)
    return values


def test_affine_quantize_ties_round_away_from_zero():
    # min=0, max=15 makes the step exactly 1.0, so 2.5 lands on a tie.
    arr = np.array([[0.0, 2.5, 15.0] + [2.5] * 61], dtype=np.float32)
    packed, scales, biases = core._quantize_affine_4bit(arr, 64)

    assert float(scales[0, 0]) == pytest.approx(1.0)
    assert float(biases[0, 0]) == pytest.approx(0.0)
    assert _unpack(packed, arr.shape)[0, 1] == 3  # not 2 (banker's rounding)
    assert _dequantize(packed, scales, biases, arr.shape)[0, 1] == pytest.approx(3.0)


def test_affine_quantize_rejects_non_finite_values():
    arr = np.zeros((2, 64), dtype=np.float32)
    arr[1, 7] = np.nan
    with pytest.raises(ValueError, match=r"Tensor blk\.0\.attn_q\.weight contains non-finite"):
        core._quantize_affine_4bit(arr, 64, tensor_name="blk.0.attn_q.weight")

    arr[1, 7] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        core._quantize_affine_4bit(arr, 64, tensor_name="blk.0.attn_q.weight")


def test_affine_quantize_constant_group_stays_finite():
    # A constant group has zero scale; it must not emit NaN codes.
    arr = np.full((1, 64), 0.25, dtype=np.float32)
    packed, scales, biases = core._quantize_affine_4bit(arr, 64)
    assert np.isfinite(scales).all()
    np.testing.assert_array_equal(_unpack(packed, arr.shape), 0)
    assert float(biases[0, 0]) == pytest.approx(0.25)


def test_affine_quantize_tie_semantics_match_mlx():
    """Cross-check the tie rule against MLX's own quantizer.

    MLX stores affine groups with a *negative* scale and ``bias = max``, i.e.
    ``w = -code * step + max``, mirroring the code axis relative to the
    ``w = code * step + min`` convention used here. Bit-identical packed words
    are therefore not expected -- but both quantizers must round a .5 tie away
    from zero in their own space (banker's rounding would land on the other
    side), which is what this test pins.
    """
    mx = pytest.importorskip("mlx.core")

    # min=0, max=15 -> step is exactly 1.0, so 2.5 is a true tie.
    weights = np.array([[0.0, 2.5, 15.0] + [2.5] * 61], dtype=np.float16)

    packed, scales, biases = core._quantize_affine_4bit(weights, 64)
    ours = _dequantize(packed, scales, biases, weights.shape)[0, 1]

    q, mlx_scales, mlx_biases = mx.quantize(mx.array(weights), group_size=64, bits=4)
    reference = np.array(mx.dequantize(q, mlx_scales, mlx_biases, group_size=64, bits=4))[0, 1]

    # Ours: 2.5 -> code 3 -> 3.0. MLX mirrors: 15 - 13 = 2.
    assert float(ours) == pytest.approx(3.0)
    assert float(reference) == pytest.approx(2.0)
    # Both moved exactly half a step off the tie, in opposite directions.
    assert abs(float(ours) - 2.5) == pytest.approx(0.5)
    assert abs(float(reference) - 2.5) == pytest.approx(0.5)
