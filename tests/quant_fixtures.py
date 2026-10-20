"""Synthetic GGUF blocks and files for the quantized-passthrough tests.

Everything here is generated locally: no test in this suite downloads a model.
The encoders below mirror the layouts documented in ``ggml`` and are validated
against ``gguf.quants.dequantize`` in ``tests/test_gguf_codes.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter

Q4_K_BYTES = 144
Q6_K_BYTES = 210
Q8_0_BYTES = 34


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Inverse of ``gguf.quants.Q4_K.get_scale_min`` for 8 sub-blocks."""
    count = scales.shape[0]
    blocked = np.zeros((count, 12), dtype=np.uint8)
    for index in range(4):
        blocked[:, index] = (scales[:, index] & 0x3F) | (
            ((scales[:, index + 4] >> 4) & 0x03) << 6
        )
        blocked[:, index + 4] = (mins[:, index] & 0x3F) | (
            ((mins[:, index + 4] >> 4) & 0x03) << 6
        )
        blocked[:, index + 8] = (scales[:, index + 4] & 0x0F) | (
            (mins[:, index + 4] & 0x0F) << 4
        )
    return blocked


def q4_k_blocks(
    d: np.ndarray,
    dmin: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    """Assemble ``Q4_K`` blocks from explicit fields (``codes`` is (n, 8, 32))."""
    count = codes.shape[0]
    raw = np.zeros((count, Q4_K_BYTES), dtype=np.uint8)
    raw[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(count, 2)
    raw[:, 2:4] = dmin.astype(np.float16).view(np.uint8).reshape(count, 2)
    raw[:, 4:16] = _pack_q4_k_scales(scales.astype(np.uint16), mins.astype(np.uint16))
    grouped = codes.reshape(count, 4, 2, 32)
    for chunk in range(4):
        start = 16 + chunk * 32
        raw[:, start:start + 32] = grouped[:, chunk, 0, :] | (grouped[:, chunk, 1, :] << 4)
    return raw


def encode_q4_k(array: np.ndarray) -> np.ndarray:
    """Reference-style ``Q4_K`` encoder used to build realistic fixtures.

    The sub-block scale covers a 4-bit code range, so ``sc`` is derived from
    ``sub_max / (15 * d)``; using ``sub_max / d`` instead would produce blocks
    whose codes never leave 0/1 and would not exercise the mapping.
    """
    values = np.asarray(array, dtype=np.float32).reshape(-1, 256)
    count = values.shape[0]
    sub = values.reshape(count, 8, 32)
    upper = np.clip(sub.max(-1), 0, None)
    lower = np.clip(-sub.min(-1), 0, None)
    # A 4-bit code spans 15 steps of d*sc and sc itself is 6-bit, so d carries
    # the remaining 63x: d = max/945 keeps sc in [1, 63] and codes in [0, 15].
    d = np.maximum(upper.max(1), 1e-8) / 945.0
    dmin = np.maximum(lower.max(1), 1e-8) / 945.0
    scales = np.clip(np.round(upper / (15.0 * d[:, None])), 0, 63)
    mins = np.clip(np.round(lower / (15.0 * dmin[:, None])), 0, 63)
    codes = np.clip(
        np.round((sub + dmin[:, None, None] * mins[:, :, None])
                 / (d[:, None, None] * scales[:, :, None] + 1e-30)),
        0,
        15,
    ).astype(np.uint8)
    raw = q4_k_blocks(d, dmin, scales, mins, codes)
    return raw.reshape(array.shape[0], array.shape[1] // 256 * Q4_K_BYTES)


def q6_k_blocks(
    d: np.ndarray, sub_scales: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Assemble ``Q6_K`` blocks; ``codes`` is (n, 16, 16) with values in 0..63."""
    count = codes.shape[0]
    flat = codes.reshape(count, 256)
    low = np.zeros((count, 128), dtype=np.uint8)
    high = np.zeros((count, 64), dtype=np.uint8)
    for element in range(256):
        half, rest = element // 128, element % 128
        chunk, column = rest // 32, rest % 32
        value = flat[:, element]
        if element < 64:
            low[:, element] |= value & 0x0F
        elif element < 128:
            low[:, element - 64] |= (value & 0x0F) << 4
        elif element < 192:
            low[:, element - 64] |= value & 0x0F
        else:
            low[:, element - 128] |= (value & 0x0F) << 4
        high[:, half * 32 + column] |= ((value >> 4) & 0x03) << (2 * chunk)
    raw = np.zeros((count, Q6_K_BYTES), dtype=np.uint8)
    raw[:, 0:128] = low
    raw[:, 128:192] = high
    raw[:, 192:208] = sub_scales.astype(np.int8).view(np.uint8)
    raw[:, 208:210] = d.astype(np.float16).view(np.uint8).reshape(count, 2)
    return raw


def encode_q6_k(array: np.ndarray, *, equal_pair_scales: bool = False) -> np.ndarray:
    """Reference-style ``Q6_K`` encoder; optional pair-scale flattening.

    ``Q6_K`` codes are 6-bit centred on 32, so a sub-block scale covers 32 code
    steps: ``sc = sub_max / (32 * d)``, matching ggml's ``-128 * sub_max/d``
    derivation up to sign convention.
    """
    values = np.asarray(array, dtype=np.float32).reshape(-1, 256)
    count = values.shape[0]
    sub = values.reshape(count, 16, 16)
    magnitudes = np.clip(np.abs(sub).max(-1), 1e-8, None)
    # 6-bit codes span 32 steps around the midpoint and sc is int8, so
    # d = max/(32*127) keeps sc in [1, 127] and codes in [0, 63].
    d = (magnitudes.max(1) / 4064.0).astype(np.float16)
    scales = np.clip(
        np.round(magnitudes / (32.0 * d.reshape(-1, 1))), 1, 127
    ).astype(np.int8)
    if equal_pair_scales:
        scales = np.repeat(scales[:, 0::2], 2, axis=1)
    codes = np.clip(
        np.round(sub / (d.reshape(-1, 1, 1) * scales.reshape(count, 16, 1) + 1e-30)) + 32,
        0,
        63,
    ).astype(np.uint8)
    raw = q6_k_blocks(d, scales, codes)
    return raw.reshape(array.shape[0], array.shape[1] // 256 * Q6_K_BYTES)


def encode_q8_0(array: np.ndarray) -> np.ndarray:
    """Reference-style ``Q8_0`` encoder."""
    values = np.asarray(array, dtype=np.float32).reshape(-1, 32)
    count = values.shape[0]
    d = (np.abs(values).max(1) / 127.0).astype(np.float16)
    codes = np.clip(np.round(values / d.reshape(-1, 1)), -127, 127).astype(np.int8)
    raw = np.zeros((count, Q8_0_BYTES), dtype=np.uint8)
    raw[:, 0:2] = d.view(np.uint8).reshape(count, 2)
    raw[:, 2:] = codes.view(np.uint8)
    return raw.reshape(array.shape[0], array.shape[1] // 32 * Q8_0_BYTES)


ENCODERS = {
    "Q4_K": (encode_q4_k, GGMLQuantizationType.Q4_K),
    "Q6_K": (encode_q6_k, GGMLQuantizationType.Q6_K),
    "Q8_0": (encode_q8_0, GGMLQuantizationType.Q8_0),
}


DEFAULT_DIMS = {
    "hidden": 256,
    "ffn": 512,
    "heads": 4,
    "kv_heads": 2,
    "vocab": 64,
    "layers": 1,
}


def llama_tensor_plan(dims: dict[str, int] | None = None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """A one-layer dense model: name -> (block type or "F32", logical shape)."""
    dims = {**DEFAULT_DIMS, **(dims or {})}
    hidden, ffn, kv_width = dims["hidden"], dims["ffn"], (
        dims["kv_heads"] * dims["hidden"] // dims["heads"]
    )
    return {
        "token_embd.weight": ("Q4_K", (dims["vocab"], hidden)),
        "output.weight": ("Q4_K", (dims["vocab"], hidden)),
        "output_norm.weight": ("F32", (hidden,)),
        "blk.0.attn_norm.weight": ("F32", (hidden,)),
        "blk.0.ffn_norm.weight": ("F32", (hidden,)),
        "blk.0.attn_q.weight": ("Q4_K", (hidden, hidden)),
        "blk.0.attn_k.weight": ("Q4_K", (kv_width, hidden)),
        "blk.0.attn_v.weight": ("Q4_K", (kv_width, hidden)),
        "blk.0.attn_output.weight": ("Q4_K", (hidden, hidden)),
        "blk.0.ffn_gate.weight": ("Q6_K", (ffn, hidden)),
        "blk.0.ffn_up.weight": ("Q8_0", (ffn, hidden)),
        "blk.0.ffn_down.weight": ("Q4_K", (hidden, ffn)),
    }


def write_llama_gguf(
    path: Path,
    *,
    arch: str = "llama",
    dims: dict[str, int] | None = None,
    plan: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    equal_q6_scales: bool = False,
    seed: int = 7,
) -> Path:
    """Write a synthetic GGUF that exercises every preserved block type."""
    dims = {**DEFAULT_DIMS, **(dims or {})}
    plan = plan or llama_tensor_plan(dims)
    writer = GGUFWriter(str(path), arch)
    writer.add_context_length(64)
    writer.add_embedding_length(dims["hidden"])
    writer.add_block_count(dims["layers"])
    writer.add_feed_forward_length(dims["ffn"])
    writer.add_head_count(dims["heads"])
    writer.add_head_count_kv(dims["kv_heads"])
    writer.add_rope_dimension_count(dims["hidden"] // dims["heads"])
    writer.add_token_list([f"token{i}" for i in range(dims["vocab"])])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    generator = np.random.default_rng(seed)
    for name, (kind, shape) in reversed(list(plan.items())):
        values = generator.normal(0, 0.05, shape).astype(np.float32)
        if "norm" in name:
            values = (values * 0.1 + 1).astype(np.float32)
        if kind == "F32":
            writer.add_tensor(name, values)
            continue
        encoder, qtype = ENCODERS[kind]
        if kind == "Q6_K":
            packed = encoder(values, equal_pair_scales=equal_q6_scales)
        else:
            packed = encoder(values)
        writer.add_tensor(name, packed, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
