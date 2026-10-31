"""Re-express llama.cpp quantized blocks on MLX's affine grid.

The central claim of this module
--------------------------------
A GGUF quantization is a *storage format*, not just a precision. ``Q4_K`` stores
32-value groups as 4-bit codes against a 6-bit scale and a 6-bit min, both scaled
by one ``fp16`` per 256 values. MLX's affine quantization stores a group of
``group_size`` values as ``bits``-wide codes against an ``fp16`` scale and an
``fp16`` bias: ``w = scale * code + bias``.

Those two schemes describe the same number. For ``Q4_K`` a 32-value group is
exactly ``d * sc * q - dmin * m`` and an MLX affine group of 32 is exactly
``scale * c + bias``, so with

    scale = d * sc        bias = -dmin * m        c = q

the two representations are the same values. This is a *re-expression*: the
integer codes are copied, not recomputed, and the only arithmetic that happens
is the ``fp16`` rounding of the scale and the bias. It is not a requantization
of decoded floats, which is what ``mlx_lm.convert`` must do when it starts from
HF weights and what this module exists to avoid.

Per type:

``Q4_K`` (block 256, 144 bytes)
    Per 32 values: 6-bit unsigned scale, 6-bit unsigned min, one ``fp16`` ``d``
    and one ``fp16`` ``dmin`` per 256 values. ``w = d*sc*q - dmin*m`` for
    ``q in [0, 15]``, which *is* affine 4-bit with ``group_size=32``. Exact up
    to the ``fp16`` rounding of ``scale``/``bias``. Measured relative L2 on
    synthetic blocks: ~2e-4, entirely from that rounding.

``Q8_0`` (block 32, 34 bytes)
    Per 32 values: one ``fp16`` ``d`` and 32 ``int8`` codes. ``w = d*q`` for
    ``q in [-127, 127]``. Affine 8-bit is ``w = scale*c + bias`` with
    ``c in [0, 255]``, so ``scale = d``, ``bias = -128*d``, ``c = q + 128``.
    ``d`` is already ``fp16`` and ``128*d`` is an exact power-of-two scaling, so
    this mapping is bit-exact: measured relative L2 is 0.

Legacy non-K types (block 32)
    ``Q4_0`` (18 bytes), ``Q4_1`` (20), ``Q5_0`` (22) and ``Q5_1`` (24) all cover
    exactly 32 values -- MLX's group size -- so they are the *easiest* exact
    mappings, not the hardest:

    ====== ================== ===================== ======== ==========
    type   stored scales      decode                scale    bias
    ====== ================== ===================== ======== ==========
    Q4_0   ``d``              ``d*(q-8)``           ``d``    ``-8*d``
    Q4_1   ``d``, ``m``       ``d*q+m``             ``d``    ``m``
    Q5_0   ``d``              ``d*(q-16)``          ``d``    ``-16*d``
    Q5_1   ``d``, ``m``       ``d*q+m``             ``d``    ``m``
    ====== ================== ===================== ======== ==========

    Every code is copied unchanged. For the ``_0`` variants the bias is a
    power-of-two multiple of ``d``, which is exact in ``fp16``, so those are
    bit-exact; for the ``_1`` variants ``m`` is already ``fp16`` and is copied
    through. Relative L2 is 0 for all four (verified against
    ``gguf.quants.dequantize`` in ``tests/test_gguf_codes.py``).

    "Bit-exact" holds while the derived constant stays representable: ``-8*d``
    leaves the ``fp16`` range for ``d > 8192`` and ``-16*d`` for ``d > 4096``.
    A real model never reaches that (it implies weights around 1e6), but a
    non-finite bias is not silent -- ``audit_preserved`` compares against the
    reference decode and reports the tensor as a mismatch rather than passing.

    The 5th bit of ``Q5_0``/``Q5_1`` is the one place ggml's layout is easy to
    get wrong: ``qh`` is four bytes that must be read as a *single little-endian
    ``uint32``*, value ``i`` taking bit ``i``. It is not four independent bytes
    of eight bits each, which is how it is sometimes described.

``Q6_K`` (block 256, 210 bytes)
    Per 16 values: one signed 8-bit sub-block scale, 6-bit codes, one ``fp16``
    ``d`` per 256 values. ``w = d*sc*(q - 32)``.

    Here the two formats disagree structurally: ``Q6_K``'s scale is per 16
    values, MLX's smallest supported group is 32 (``mlx.core.quantize`` rejects
    ``group_size=16``), so one affine group has to cover two ``Q6_K`` sub-blocks
    with potentially different scales. Exactness requires
    ``sc[2i] == sc[2i+1]``, which holds for only ~0.7% of groups in a real
    model, so this is the one format where ``preserve`` cannot copy codes: the
    values are re-fitted onto the 6-bit grid and the error is *reported*, not
    hidden.

    Two candidates are computed per group and the lower squared error wins:

    * the source-lattice fit -- scale ``d*max|sc|``, codes divided through by
      the scale ratio, which is exact for the coarser sub-block;
    * a least-squares refit of ``scale``/``bias`` onto the decoded values,
      which recovers most of what the coarse fit loses on the finer sub-block.

    Measured on real Llama-3.2-1B ``Q6_K`` tensors: relative L2 **1.7e-2**,
    against 4.3e-3 if the same tensor is stored at 8 bits. Groups whose pair
    scales do agree keep their codes bit for bit, so ``MappingResult.exact``
    is still an honest per-tensor statement.

    The conversion pipeline never falls back to ``float16`` for these tensors:
    that would turn a 0.82 byte/value tensor into a 2 byte/value one (2.4x) and
    defeat the point of the mode. It emits them at 6 bits under a separate
    gate with the tolerance in :data:`PRESERVE_INEXACT_TOLERANCE`; only block
    types with no mapping at all still decode to float.

``Q2_K`` (84 bytes) and ``Q3_K`` (110 bytes)
    Not mapped, and not for lack of effort: both also carry their scale **and
    min** per 16 values, so they hit the same 16-versus-32 wall as ``Q6_K``, but
    with only 2 or 3 code bits a least-squares refit has so few levels that the
    error stops being "bounded and reported" and starts being a real loss of
    precision. Fitting them would produce a tensor that is *worse* than the
    source while still occupying the source's byte budget, so they are decoded
    to float and recorded, not silently re-fitted.

    The general rule this module encodes: a block format maps exactly onto MLX's
    affine grid only when its scale granularity divides 32 -- that is, when its
    scale is per 32 or per 256 values. ``Q4_0``/``Q4_1``/``Q5_0``/``Q5_1``/
    ``Q8_0`` (32) and ``Q4_K``/``Q5_K`` (256, with 32-value sub-blocks) satisfy
    it. ``Q2_K``/``Q3_K``/``Q6_K`` store 16-value sub-blocks and cannot.

``IQ*`` and other codebook formats
    ``IQ2_XXS``/``IQ2_XS``/``IQ3_XXS``/``IQ3_S``/``IQ2_S``/``IQ1_S``/``IQ1_M``/
    ``IQ4_NL``/``IQ4_XS`` decode as ``d * kvalues[code]`` where ``kvalues`` is a
    fixed non-uniform table (and, for the ``_XXS``/``_XS``/``_S`` variants, a
    multi-level sign/hierarchy scheme). There is no ``scale*code + bias`` form
    at all, so these fail closed: :func:`map_tensor` raises, the tensor is
    decoded to float under the ``<TYPE>_has_no_affine_reexpression`` reason, and
    the report names the type. That is the layout an iMatrix-guided
    quantization produces, so an iMatrix source converts but does not preserve.
    ``TQ1_0``/``TQ2_0``/``MXFP4`` are excluded for the same reason.

Packing
-------
MLX stores codes as little-endian ``uint32`` words, filled LSB-first and
continuing across word boundaries (4-bit: 8 codes per word, 8-bit: 4, 6-bit: a
contiguous bit stream, 6 words per 32-value group). :func:`pack_codes` and
:func:`unpack_codes` implement exactly that layout; ``tests/test_gguf_codes.py``
verifies it against ``mlx.core.dequantize`` when MLX is installed.

The module is numpy-only on purpose: auditing a conversion must work on Linux
CI, where MLX is not installable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

AFFINE_GROUP_SIZE = 32

#: Architectures whose tensor layout this pipeline can re-express without
#: transforming a quantized payload. Everything else fails closed in the CLI.
PRESERVE_ARCHES = frozenset({
    "llama",
    "qwen2",
    "qwen3",
    "mistral",
    # Dense Qwen3.5 uses the same linear tensors plus SSM parameters. The SSM
    # vectors and convolution are not matrices and therefore stay in float;
    # only the quantized projections are re-expressed on MLX's affine grid.
    "qwen35", "qwen3_5", "qwen3.5",
    "qwen36", "qwen3_6", "qwen3.6",
    "qwen38", "qwen3_8", "qwen3.8",
})

#: Relative-L2 ceiling for a preserved tensor. Only ``fp16`` scale/bias rounding
#: is allowed inside it, so a mis-mapped block cannot hide underneath.
PRESERVE_TOLERANCE = 1.0e-3

#: Ceilings for tensors that cannot keep their codes and are re-fitted at the
#: source bit width instead. Keyed by the fallback the caller asked for.
#: Measured on Llama-3.2-1B: 1.7e-2 at 6 bits, 4.4e-3 at 8 bits.
PRESERVE_INEXACT_TOLERANCE = {
    "6bit": 2.5e-2,
    "8bit": 8.0e-3,
}

#: Block formats whose affine re-expression cannot be exact for every group.
#: Inexactness here is structural and bounded, not a mapping bug -- so the
#: pipeline re-fits instead of dropping the tensor to ``float16``.
INEVITABLY_INEXACT = frozenset({"Q6_K"})

#: How a tensor whose codes cannot be reused verbatim is stored.
PRESERVE_FALLBACK_CHOICES = ("6bit", "8bit", "float16")

#: Values processed per chunk. Keeps the transient float32 arrays bounded for
#: embedding tables, where one ``Q6_K`` tensor holds a quarter of a billion
#: values.
_CHUNK_VALUES = 4 * 1024 * 1024

#: MLX's own limit: ``group_size`` must be 32, 64, or 128.
MIN_MLX_GROUP_SIZE = 32

_BLOCK_ELEMENTS = {
    "Q4_0": 32,
    "Q4_1": 32,
    "Q5_0": 32,
    "Q5_1": 32,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "Q8_0": 32,
}
_TYPE_BYTES = {
    "Q4_0": 18,
    "Q4_1": 20,
    "Q5_0": 22,
    "Q5_1": 24,
    "Q4_K": 144,
    "Q5_K": 176,
    "Q6_K": 210,
    "Q8_0": 34,
}
_AFFINE_BITS = {
    "Q4_0": 4,
    "Q4_1": 4,
    "Q5_0": 5,
    "Q5_1": 5,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8,
}

#: ``GGMLQuantizationType`` values this module understands.
QTYPE_IDS = {
    "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
    "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
}
_NAME_BY_ID = {value: key for key, value in QTYPE_IDS.items()}

#: Block types that exist in GGUF and are rejected by ``preserve``, with the
#: reason to show a reader. Anything here *converts* -- it is decoded to float
#: and recorded -- but its codes are not reused, so the artifact does not shrink
#: the way a preserved one does.
NON_AFFINE_QTYPES = {
    # 16-value sub-block scale and min; MLX's smallest group is 32.
    "Q2_K": "sub_block_scale_is_per_16_values",
    "Q3_K": "sub_block_scale_is_per_16_values",
    # Non-uniform codebook: w = d * kvalues[code], which has no affine form.
    "IQ1_S": "codebook_has_no_affine_form",
    "IQ1_M": "codebook_has_no_affine_form",
    "IQ2_XXS": "codebook_has_no_affine_form",
    "IQ2_XS": "codebook_has_no_affine_form",
    "IQ2_S": "codebook_has_no_affine_form",
    "IQ3_XXS": "codebook_has_no_affine_form",
    "IQ3_S": "codebook_has_no_affine_form",
    "IQ4_NL": "codebook_has_no_affine_form",
    "IQ4_XS": "codebook_has_no_affine_form",
    # Ternary / block-float, neither of which is an affine integer grid.
    "TQ1_0": "ternary_not_an_integer_grid",
    "TQ2_0": "ternary_not_an_integer_grid",
    "MXFP4": "block_float_has_no_affine_form",
    # 8-bit with per-block sums, used only by ggml's CUDA kernels.
    "Q8_1": "intermediate_type_never_stored_in_a_file",
    "Q8_K": "intermediate_type_never_stored_in_a_file",
}


class UnsupportedQuantType(ValueError):
    """Raised when a GGUF block type has no defined affine re-expression."""


@dataclass(frozen=True)
class MappingResult:
    """One GGUF tensor re-expressed as MLX affine groups of 32."""

    packed: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    bits: int
    group_size: int
    qtype: str
    exact: bool
    #: Number of 32-value groups whose two ``Q6_K`` sub-block scales disagreed.
    inexact_groups: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.scales.shape[0]), int(self.scales.shape[1]) * self.group_size)


def qtype_name(qtype: Any) -> str | None:
    """Return the block-format name for a ``GGMLQuantizationType`` or int."""
    if isinstance(qtype, str):
        return qtype if qtype in _AFFINE_BITS else None
    try:
        value = int(qtype)
    except (TypeError, ValueError):
        return None
    name = getattr(qtype, "name", None)
    if isinstance(name, str) and name in _AFFINE_BITS:
        return name
    return _NAME_BY_ID.get(value)


def non_affine_reason(qtype: Any) -> str | None:
    """Why a known GGUF block type cannot be re-expressed, or ``None``.

    Distinguishes "this type is understood and rejected for a stated reason"
    from "this type is unknown to the module", which is what the conversion
    report needs to tell a user apart.
    """
    name = qtype if isinstance(qtype, str) else getattr(qtype, "name", None)
    if not isinstance(name, str):
        return None
    return NON_AFFINE_QTYPES.get(name)


def mlx_bits(qtype: Any) -> int | None:
    """Bit-width an MLX affine group needs to hold ``qtype``'s codes."""
    name = qtype_name(qtype)
    return None if name is None else _AFFINE_BITS[name]


def block_elements(qtype: Any) -> int | None:
    """Values covered by one GGUF block of ``qtype`` (``None`` if unknown)."""
    name = qtype_name(qtype)
    return None if name is None else _BLOCK_ELEMENTS[name]


# ---------------------------------------------------------------------------
# MLX packing layout
# ---------------------------------------------------------------------------


def _pack_bitstream(codes: np.ndarray, bits: int, group: int) -> np.ndarray:
    """Pack ``group`` codes of ``bits`` each into ``uint32`` words, LSB-first."""
    rows = codes.shape[0]
    groups = codes.shape[1] // group
    words = group * bits // 32
    out = np.empty((rows, groups * words), dtype=np.uint32)
    # Bound the temporary bit matrix: bits=6 needs 6 bytes per code.
    step = max(1, (8 * 1024 * 1024) // max(1, group * bits))
    for start in range(0, rows, step):
        block = codes[start:start + step].reshape(-1, groups, group, 1)
        shifts = np.arange(bits, dtype=np.uint8).reshape(1, 1, 1, bits)
        bit_matrix = ((block >> shifts) & 1).astype(np.uint8).reshape(-1, groups, group * bits)
        packed = np.packbits(bit_matrix, axis=-1, bitorder="little")
        out[start:start + step] = np.ascontiguousarray(packed).view(np.uint32).reshape(
            -1, groups * words
        )
    return out


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``(rows, cols)`` integer codes the way MLX stores them.

    ``cols`` must be a multiple of the 32-value group used by the mappings so
    that every group starts on a word boundary.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}")
    rows, cols = codes.shape
    if bits not in (4, 5, 6, 8):
        raise ValueError(f"unsupported MLX affine bit-width: {bits}")
    if cols % AFFINE_GROUP_SIZE:
        raise ValueError(
            f"column count {cols} is not a multiple of the affine group size "
            f"{AFFINE_GROUP_SIZE}"
        )
    if bits == 4:
        chunks = codes.reshape(rows, cols // 8, 8).astype(np.uint32)
        shifts = np.arange(8, dtype=np.uint32) * 4
        return cast(np.ndarray, np.bitwise_or.reduce(chunks << shifts, axis=-1))
    if bits == 8:
        chunks = codes.reshape(rows, cols // 4, 4).astype(np.uint32)
        shifts = np.arange(4, dtype=np.uint32) * 8
        return cast(np.ndarray, np.bitwise_or.reduce(chunks << shifts, axis=-1))
    return _pack_bitstream(codes, bits, AFFINE_GROUP_SIZE)


def unpack_codes(packed: np.ndarray, bits: int, cols: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`; the basis of the conversion audit."""
    packed = np.ascontiguousarray(packed)
    if bits not in (4, 5, 6, 8):
        raise ValueError(f"unsupported MLX affine bit-width: {bits}")
    if cols % AFFINE_GROUP_SIZE:
        raise ValueError(f"column count {cols} is not a multiple of {AFFINE_GROUP_SIZE}")
    rows = packed.shape[0]
    if bits == 4:
        words = packed.astype(np.uint32)[:, :, None]
        shifts = np.arange(8, dtype=np.uint32) * 4
        codes = (words >> shifts) & 0x0F
        return codes.astype(np.uint8).reshape(rows, cols)
    if bits == 8:
        words = packed.astype(np.uint32)[:, :, None]
        shifts = np.arange(4, dtype=np.uint32) * 8
        codes = (words >> shifts) & 0xFF
        return codes.astype(np.uint8).reshape(rows, cols)
    nbytes = int(packed.size // rows) * 4
    raw = np.ascontiguousarray(packed).view(np.uint8).reshape(rows, nbytes)
    unpacked = np.unpackbits(raw, axis=1, bitorder="little")
    unpacked = unpacked.reshape(rows, cols, bits).astype(np.uint16)
    weights = (np.uint16(1) << np.arange(bits, dtype=np.uint16)).reshape(1, 1, bits)
    return cast(np.ndarray, np.sum(unpacked * weights, axis=-1).astype(np.uint8))


def affine_decode(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int,
    group_size: int = AFFINE_GROUP_SIZE,
) -> np.ndarray:
    """Decode stored affine groups, mirroring ``mlx.core.dequantize``."""
    packed = np.ascontiguousarray(packed)
    scales = np.asarray(scales)
    biases = np.asarray(biases)
    if scales.shape != biases.shape or scales.ndim != 2:
        raise ValueError(
            f"scales{scales.shape} and biases{biases.shape} must be 2-D and equal in shape"
        )
    rows = scales.shape[0]
    cols = int(scales.shape[1]) * group_size
    if packed.shape[0] != rows:
        raise ValueError(f"packed has {packed.shape[0]} rows but scales have {rows}")
    codes = unpack_codes(packed, bits, cols).astype(np.float32)
    scale = scales.astype(np.float32).repeat(group_size, axis=1)
    bias = biases.astype(np.float32).repeat(group_size, axis=1)
    return codes * scale + bias


# ---------------------------------------------------------------------------
# GGUF block decoding
# ---------------------------------------------------------------------------


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """C++ ``std::round`` semantics, which MLX and ggml both use."""
    return cast(np.ndarray, np.trunc(values + np.copysign(0.5, values)))


def _blocks(raw: np.ndarray, rows: int, cols: int, qtype: str) -> np.ndarray:
    block = _BLOCK_ELEMENTS[qtype]
    if cols % block:
        raise ValueError(
            f"{qtype} needs a row length that is a multiple of {block}, got {cols}"
        )
    expected = rows * cols // block * _TYPE_BYTES[qtype]
    flat = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    if flat.size != expected:
        raise ValueError(
            f"{qtype} payload is {flat.size} bytes, expected {expected} for shape "
            f"({rows}, {cols})"
        )
    return flat.reshape(rows * (cols // block), _TYPE_BYTES[qtype])


def _check_shape(shape: Any, qtype: str) -> tuple[int, int]:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 2 or min(dims) <= 0:
        raise ValueError(f"{qtype} mapping needs a positive 2-D shape, got {dims}")
    return dims


def _split_nibbles(qs: np.ndarray) -> np.ndarray:
    """Unpack ``(blocks, 16)`` packed bytes into ``(blocks, 32)`` 4-bit codes.

    ggml stores value ``i`` in the low nibble of byte ``i`` and value ``i + 16``
    in the high nibble, so the two halves come out as separate runs rather than
    interleaved pairs. Every 32-value format here (``Q4_0``, ``Q4_1``, ``Q4_K``)
    uses that order.
    """
    packed = qs.reshape(-1, 1, 1, 16)
    halves = (
        packed >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ) & np.uint8(0x0F)
    return cast(np.ndarray, halves.reshape(qs.shape[0], 32))


def _fifth_bits(qh: np.ndarray) -> np.ndarray:
    """The 5th bit of 32 codes from the 4-byte ``qh`` plane.

    ``qh`` must be read as **one little-endian uint32**, with value ``i`` taking
    bit ``i``. Reading it as four independent bytes of eight bits each produces
    codes that decode to plausible-looking but wrong weights, which is the kind
    of error a shape check cannot see.
    """
    words = np.ascontiguousarray(qh).copy().view("<u4").reshape(qh.shape[0], 1)
    return ((words >> np.arange(32, dtype=np.uint32)) & np.uint32(1)).astype(np.uint8)


def _legacy_blocks(
    raw: np.ndarray, shape: Any, qtype: str
) -> tuple[int, int, np.ndarray, int]:
    """Validate a legacy 32-value block tensor and return its block view."""
    rows, cols = _check_shape(shape, qtype)
    blocks = _blocks(raw, rows, cols, qtype)
    return rows, cols, blocks, cols // AFFINE_GROUP_SIZE


def q4_0_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express ``Q4_0`` as affine 4-bit groups of 32.

    ``w = d*(q-8)`` with ``q`` in ``[0, 15]``, so ``scale = d``, ``bias = -8*d``
    and the source codes are reused unchanged. ``-8*d`` is a power-of-two
    scaling of an ``fp16``, so the result is bit-exact.
    """
    rows, cols, blocks, groups = _legacy_blocks(raw, shape, "Q4_0")
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    codes = _split_nibbles(blocks[:, 2:]).reshape(rows, cols)
    return MappingResult(
        packed=pack_codes(codes, 4),
        scales=d.reshape(rows, groups).astype(floating_dtype),
        biases=(-8.0 * d).reshape(rows, groups).astype(floating_dtype),
        bits=4,
        group_size=AFFINE_GROUP_SIZE,
        qtype="Q4_0",
        exact=True,
    )


def q4_1_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express ``Q4_1`` as affine 4-bit groups of 32.

    ``w = d*q + m`` with ``q`` in ``[0, 15]`` and both ``d`` and ``m`` stored as
    ``fp16``, so the mapping is an identity on the codes and on both constants.
    """
    rows, cols, blocks, groups = _legacy_blocks(raw, shape, "Q4_1")
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    m = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    codes = _split_nibbles(blocks[:, 4:]).reshape(rows, cols)
    return MappingResult(
        packed=pack_codes(codes, 4),
        scales=d.reshape(rows, groups).astype(floating_dtype),
        biases=m.reshape(rows, groups).astype(floating_dtype),
        bits=4,
        group_size=AFFINE_GROUP_SIZE,
        qtype="Q4_1",
        exact=True,
    )


def q5_0_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express ``Q5_0`` as affine 5-bit groups of 32.

    ``w = d*(q-16)`` with ``q`` in ``[0, 31]``; the low nibble comes from ``qs``
    and the fifth bit from ``qh``.
    """
    rows, cols, blocks, groups = _legacy_blocks(raw, shape, "Q5_0")
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    high = _fifth_bits(blocks[:, 2:6])
    low = _split_nibbles(blocks[:, 6:])
    codes = (low | (high << np.uint8(4))).reshape(rows, cols)
    return MappingResult(
        packed=pack_codes(codes, 5),
        scales=d.reshape(rows, groups).astype(floating_dtype),
        biases=(-16.0 * d).reshape(rows, groups).astype(floating_dtype),
        bits=5,
        group_size=AFFINE_GROUP_SIZE,
        qtype="Q5_0",
        exact=True,
    )


def q5_1_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express ``Q5_1`` as affine 5-bit groups of 32.

    ``w = d*q + m`` with ``q`` in ``[0, 31]``; both constants are stored as
    ``fp16``, so this is exact like ``Q4_1``.
    """
    rows, cols, blocks, groups = _legacy_blocks(raw, shape, "Q5_1")
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    m = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    high = _fifth_bits(blocks[:, 4:8])
    low = _split_nibbles(blocks[:, 8:])
    codes = (low | (high << np.uint8(4))).reshape(rows, cols)
    return MappingResult(
        packed=pack_codes(codes, 5),
        scales=d.reshape(rows, groups).astype(floating_dtype),
        biases=m.reshape(rows, groups).astype(floating_dtype),
        bits=5,
        group_size=AFFINE_GROUP_SIZE,
        qtype="Q5_1",
        exact=True,
    )


def q4_k_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express a ``Q4_K`` tensor as affine 4-bit groups of 32."""
    rows, cols = _check_shape(shape, "Q4_K")
    blocks = _blocks(raw, rows, cols, "Q4_K")

    from gguf.quants import Q4_K

    scales, mins = Q4_K.get_scale_min(blocks[:, 4:16].copy())
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    scale = (d * scales.astype(np.float32)).reshape(rows, cols // 32)
    bias = (-dmin * mins.astype(np.float32)).reshape(rows, cols // 32)

    codes = (blocks[:, 16:].reshape(-1, 4, 1, 32) >> np.array(
        [0, 4], dtype=np.uint8
    ).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
    codes = codes.reshape(rows, cols // 32, 32)

    return MappingResult(
        packed=pack_codes(codes.reshape(rows, cols), 4),
        scales=scale.astype(floating_dtype),
        biases=bias.astype(floating_dtype),
        bits=4,
        group_size=32,
        qtype="Q4_K",
        exact=True,
    )


def q5_k_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express a ``Q5_K`` tensor as affine 5-bit groups of 32.

    ``Q5_K`` keeps its 6-bit scale and min per 32 values -- exactly MLX's group
    -- and its 5-bit codes are the low nibble of ``qs`` plus one high bit taken
    from ``qh``. So the codes are reusable verbatim, like ``Q4_K``, and only the
    fp16 rounding of ``d*sc`` and ``-dmin*m`` is new.
    """
    rows, cols = _check_shape(shape, "Q5_K")
    blocks = _blocks(raw, rows, cols, "Q5_K")

    from gguf.quants import Q4_K

    scales, mins = Q4_K.get_scale_min(blocks[:, 4:16].copy())
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    scale = (d * scales.astype(np.float32)).reshape(rows, cols // 32)
    bias = (-dmin * mins.astype(np.float32)).reshape(rows, cols // 32)

    # Layout per 176-byte block: d, dmin, scales[12], qh[32], qs[128]. ``qs``
    # supplies the low nibble of 4 pairs of 32-value groups; ``qh`` supplies the
    # fifth bit for all 8 groups, one bit per group per byte.
    qh = blocks[:, 16:48].reshape(-1, 1, 32)
    qs = blocks[:, 48:].reshape(-1, 4, 1, 32)
    low = (qs >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
    high = (qh >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & np.uint8(0x01)
    codes = low.reshape(-1, cols) | (high.reshape(-1, cols) << np.uint8(4))
    codes = codes.reshape(rows, cols // 32, 32)

    return MappingResult(
        packed=pack_codes(codes.reshape(rows, cols), 5),
        scales=scale.astype(floating_dtype),
        biases=bias.astype(floating_dtype),
        bits=5,
        group_size=32,
        qtype="Q5_K",
        exact=True,
    )


def q8_0_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express a ``Q8_0`` tensor as affine 8-bit groups of 32."""
    rows, cols = _check_shape(shape, "Q8_0")
    blocks = _blocks(raw, rows, cols, "Q8_0")
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    codes = blocks[:, 2:].copy().view(np.int8).astype(np.int16)

    scale = d.reshape(rows, cols // 32)
    bias = (-128.0 * d).reshape(rows, cols // 32)
    shifted = (codes + 128).astype(np.uint8).reshape(rows, cols)
    return MappingResult(
        packed=pack_codes(shifted, 8),
        scales=scale.astype(floating_dtype),
        biases=bias.astype(floating_dtype),
        bits=8,
        group_size=32,
        qtype="Q8_0",
        exact=True,
    )


def _group_sse(
    values: np.ndarray, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Squared error of one reconstructed group, summed over its values."""
    residual = codes * scale + bias - values
    return cast(np.ndarray, np.sum(residual * residual, axis=-1))


def _codes_for(
    values: np.ndarray, scale: np.ndarray, bias: np.ndarray, levels: int
) -> np.ndarray:
    """Nearest code in ``[0, levels]`` for every value, given a group grid."""
    safe = np.where(scale > 0, scale, 1.0)
    return cast(
        np.ndarray,
        np.clip(_round_half_away_from_zero((values - bias) / safe), 0, levels),
    )


def _refit_groups(
    values: np.ndarray,
    codes: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    levels: int,
    *,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares refit of ``scale``/``bias`` for groups of 32 values.

    Starts from the caller's mapping and keeps it whenever a pass does not lower
    the squared error, so a refit can never make a group worse. ``values`` is
    ``(..., 32)``, ``scale``/``bias`` are ``(..., 1)``.
    """
    best_codes = codes.astype(np.float32)
    best_scale = scale.astype(np.float32)
    best_bias = bias.astype(np.float32)
    best_error = _group_sse(values, best_codes, best_scale, best_bias)
    for _ in range(iterations):
        code_mean = best_codes.mean(-1, keepdims=True)
        value_mean = values.mean(-1, keepdims=True)
        centred = best_codes - code_mean
        variance = np.sum(centred * centred, axis=-1, keepdims=True)
        covariance = np.sum(centred * (values - value_mean), axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate_scale = np.where(variance > 1e-12, covariance / variance, best_scale)
        candidate_scale = np.where(candidate_scale > 0, candidate_scale, best_scale)
        candidate_bias = value_mean - candidate_scale * code_mean
        candidate_codes = _codes_for(values, candidate_scale, candidate_bias, levels)
        candidate_error = _group_sse(
            values, candidate_codes, candidate_scale, candidate_bias
        )
        better = (candidate_error < best_error)[..., None]
        best_codes = np.where(better, candidate_codes, best_codes)
        best_scale = np.where(better, candidate_scale, best_scale)
        best_bias = np.where(better, candidate_bias, best_bias)
        best_error = np.where(better[..., 0], candidate_error, best_error)
    return best_codes, best_scale, best_bias


def _q6k_chunk(
    blocks: np.ndarray, floating_dtype: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map one run of ``Q6_K`` blocks; returns packed, scales, biases, mismatch mask."""
    count = int(blocks.shape[0])
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    pair = blocks[:, 192:208].copy().view(np.int8).astype(np.float32).reshape(count, 8, 2)
    d = blocks[:, 208:210].copy().view("<f2").astype(np.float32).reshape(count, 1, 1)

    low = ((ql.reshape(-1, 2, 1, 64) >> np.array(
        [0, 4], dtype=np.uint8
    ).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(count, 8, 32)
    high = ((qh.reshape(-1, 2, 1, 32) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8
    ).reshape(1, 1, 4, 1)) & np.uint8(0x03)).reshape(count, 8, 32)
    codes = (low | (high << np.uint8(4))).astype(np.float32)

    # Exact source values: per 16-value sub-block, w = d * sc * (q - 32).
    sub = (codes.reshape(count, 8, 2, 16) - 32.0) * d[..., None] * pair[..., None]
    values = sub.reshape(count, 8, 32)

    # Candidate 1: the coarser sub-block scale, codes divided through by the
    # scale ratio. Exact for the coarser of the two sub-blocks.
    magnitudes = np.abs(pair)
    coarse = magnitudes.max(axis=-1)
    scale = d * coarse[..., None]
    zero = coarse == 0
    ratio = pair / np.where(zero, 1.0, coarse)[..., None]
    source_codes = np.clip(_round_half_away_from_zero(
        (codes.reshape(count, 8, 2, 16) - 32.0) * ratio[..., None]
    ) + 32.0, 0, 63).reshape(count, 8, 32)
    source_codes[
        zero.repeat(2, axis=-1).repeat(16, axis=-1).reshape(values.shape)
    ] = 32.0
    source_bias = -32.0 * scale

    # Candidates 2 and 3: least squares from the source lattice, and from the
    # group's own span. Neither is uniformly better -- the lattice start wins
    # where codes land on the grid, the span start where the sub-block scales
    # differ a lot -- so the lower error wins per group. Applied only where the
    # pair scales disagree: matched groups must keep their exact codes.
    mismatched = pair[..., 0] != pair[..., 1]
    codes_out = source_codes
    scale_out = scale
    bias_out = source_bias
    if mismatched.any():
        lattice = _refit_groups(values, source_codes, scale, source_bias, 63)
        group_low = values.min(axis=-1, keepdims=True)
        group_high = values.max(axis=-1, keepdims=True)
        span_scale = np.maximum(group_high - group_low, 1e-12) / 63.0
        span_bias = group_low
        span_codes = _codes_for(values, span_scale, span_bias, 63)
        span = _refit_groups(values, span_codes, span_scale, span_bias, 63)

        take_span = _group_sse(values, *span) < _group_sse(values, *lattice)
        fit_codes = np.where(take_span[..., None], span[0], lattice[0])
        fit_scale = np.where(take_span[..., None], span[1], lattice[1])
        fit_bias = np.where(take_span[..., None], span[2], lattice[2])

        keep = mismatched[..., None]
        codes_out = np.where(keep, fit_codes, source_codes)
        scale_out = np.where(keep, fit_scale, scale)
        bias_out = np.where(keep, fit_bias, source_bias)

    packed = pack_codes(codes_out.astype(np.uint8).reshape(count, 256), 6)
    return (
        packed,
        scale_out.astype(floating_dtype),
        bias_out.astype(floating_dtype),
        mismatched,
    )


def q6_k_to_affine(
    raw: np.ndarray, shape: Any, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Re-express a ``Q6_K`` tensor as affine 6-bit groups of 32.

    Exact only where the two 16-value sub-block scales inside a 32-value group
    agree; elsewhere the group is least-squares re-fitted and the error is
    reported through :attr:`MappingResult.inexact_groups`.
    """
    rows, cols = _check_shape(shape, "Q6_K")
    blocks = _blocks(raw, rows, cols, "Q6_K")
    per_row = cols // _BLOCK_ELEMENTS["Q6_K"]
    step = max(1, _CHUNK_VALUES // max(cols, 1))

    packed_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    bias_parts: list[np.ndarray] = []
    inexact_groups = 0
    packed_cols = cols * 6 // 32
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        packed, scale, bias, mismatched = _q6k_chunk(
            blocks[start * per_row: stop * per_row], floating_dtype
        )
        rows_in_chunk = stop - start
        packed_parts.append(packed.reshape(rows_in_chunk, packed_cols))
        scale_parts.append(scale.reshape(rows_in_chunk, cols // 32))
        bias_parts.append(bias.reshape(rows_in_chunk, cols // 32))
        inexact_groups += int(mismatched.sum())

    packed = packed_parts[0] if len(packed_parts) == 1 else np.concatenate(packed_parts)
    scales = scale_parts[0] if len(scale_parts) == 1 else np.concatenate(scale_parts)
    biases = bias_parts[0] if len(bias_parts) == 1 else np.concatenate(bias_parts)
    return MappingResult(
        packed=packed,
        scales=scales,
        biases=biases,
        bits=6,
        group_size=32,
        qtype="Q6_K",
        exact=inexact_groups == 0,
        inexact_groups=inexact_groups,
    )


def refit_tensor_to_affine(
    values: np.ndarray, bits: int, *, floating_dtype: str = "float16"
) -> MappingResult:
    """Fit arbitrary float values onto an MLX affine grid at ``bits``.

    Used when a block format cannot keep its codes at the source width and the
    caller asked for a different width. ``values`` is the decoded tensor, so
    the result is a requantization -- callers must record it as inexact.
    """
    if bits not in (4, 6, 8):
        raise ValueError(f"unsupported MLX affine bit-width: {bits}")
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] % AFFINE_GROUP_SIZE:
        raise ValueError(
            f"affine refit needs a 2-D tensor with a row length that is a "
            f"multiple of {AFFINE_GROUP_SIZE}, got {array.shape}"
        )
    rows, cols = array.shape
    levels = float((1 << bits) - 1)
    step = max(1, _CHUNK_VALUES // max(cols, 1))
    packed_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    bias_parts: list[np.ndarray] = []
    for start in range(0, rows, step):
        groups = np.ascontiguousarray(array[start:start + step]).reshape(
            -1, cols // AFFINE_GROUP_SIZE, AFFINE_GROUP_SIZE
        )
        low = groups.min(axis=-1, keepdims=True)
        high = groups.max(axis=-1, keepdims=True)
        scale = np.maximum(high - low, 1e-12) / levels
        bias = low
        codes = _codes_for(groups, scale, bias, int(levels))
        codes, scale, bias = _refit_groups(groups, codes, scale, bias, int(levels))
        packed_parts.append(
            pack_codes(codes.astype(np.uint8).reshape(-1, cols), bits)
        )
        scale_parts.append(scale.reshape(-1, cols // AFFINE_GROUP_SIZE))
        bias_parts.append(bias.reshape(-1, cols // AFFINE_GROUP_SIZE))

    packed = packed_parts[0] if len(packed_parts) == 1 else np.concatenate(packed_parts)
    scales = scale_parts[0] if len(scale_parts) == 1 else np.concatenate(scale_parts)
    biases = bias_parts[0] if len(bias_parts) == 1 else np.concatenate(bias_parts)
    return MappingResult(
        packed=packed,
        scales=scales.astype(floating_dtype),
        biases=biases.astype(floating_dtype),
        bits=bits,
        group_size=AFFINE_GROUP_SIZE,
        qtype=f"AFFINE{bits}",
        exact=False,
        inexact_groups=int(packed.shape[0] * (cols // AFFINE_GROUP_SIZE)),
    )



_MAPPERS = {
    "Q4_0": q4_0_to_affine,
    "Q4_1": q4_1_to_affine,
    "Q5_0": q5_0_to_affine,
    "Q5_1": q5_1_to_affine,
    "Q4_K": q4_k_to_affine,
    "Q5_K": q5_k_to_affine,
    "Q6_K": q6_k_to_affine,
    "Q8_0": q8_0_to_affine,
}


def map_tensor(
    raw: np.ndarray,
    qtype: Any,
    shape: Any,
    *,
    floating_dtype: str = "float16",
) -> MappingResult:
    """Re-express one GGUF tensor, or raise :class:`UnsupportedQuantType`."""
    name = qtype_name(qtype)
    if name is None:
        raise UnsupportedQuantType(
            f"no affine re-expression is defined for GGUF block type {qtype!r}"
        )
    return _MAPPERS[name](raw, shape, floating_dtype=floating_dtype)


def reference_decode(raw: np.ndarray, qtype: Any, shape: Any) -> np.ndarray:
    """Decode a GGUF tensor to float32 with ``gguf.quants`` (the audit oracle)."""
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    name = qtype_name(qtype)
    if name is None:
        raise UnsupportedQuantType(f"cannot decode GGUF block type {qtype!r}")
    decoded = dequantize(
        np.ascontiguousarray(raw), GGMLQuantizationType(QTYPE_IDS[name])
    )
    return np.asarray(decoded, dtype=np.float32).reshape(tuple(int(d) for d in shape))


def llama_rotary_rows(array: np.ndarray, heads: int) -> np.ndarray:
    """Row permutation llama.cpp applies to ``attn_q``/``attn_k``.

    Only the row (output-feature) axis moves, so the same call permutes packed
    codes, scales, biases, and float tensors identically.
    """
    rows = int(array.shape[0])
    if heads <= 0 or rows % (2 * heads):
        raise ValueError(
            f"cannot split {rows} rows into {heads} heads with an even head_dim"
        )
    return array.reshape(heads, rows // (2 * heads), 2, *array.shape[1:]).swapaxes(
        1, 2
    ).reshape(rows, *array.shape[1:])
