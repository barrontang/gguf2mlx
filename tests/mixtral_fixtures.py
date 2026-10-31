"""Synthetic Mixtral GGUFs in both on-disk layouts.

Mixtral is the one MoE family with two encodings in the wild, and the difference
is not cosmetic:

* **legacy** (`TheBloke/*-Mixtral-8x7B-*-GGUF`, 2023): one tensor per expert,
  ``blk.0.ffn_gate.3.weight``. This is what llama.cpp wrote before its MoE
  refactor.
* **stacked** (anything converted since, e.g. ``dolphin-2.5-mixtral-8x7b``):
  ``blk.0.ffn_gate_exps.weight`` with the expert axis already present.

Both declare ``general.architecture = llama`` plus ``llama.expert_count``; there
is no ``mixtral`` architecture in GGUF. The fixtures keep each expert's weights
constant and distinct (expert ``e`` is filled with ``e + 1``) so a test can tell
expert stacking order from expert *reversal*, which a shape check cannot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gguf import GGUFWriter

DEFAULT_DIMS = {
    "vocab": 16,
    "hidden": 8,
    "ffn": 12,
    "heads": 2,
    "kv_heads": 1,
    "layers": 1,
    "experts": 2,
    "used": 2,
}

#: Architecture the fixture declares in the KV namespace.
SOURCE_ARCH = "llama"


def mixtral_tensor_plan(
    dims: dict[str, int] | None = None, *, layout: str = "legacy",
) -> dict[str, tuple[int, ...]]:
    """name -> logical shape. ``layout`` is ``"legacy"`` or ``"stacked"``."""
    dims = {**DEFAULT_DIMS, **(dims or {})}
    hidden, ffn, experts = dims["hidden"], dims["ffn"], dims["experts"]
    kv_width = dims["kv_heads"] * hidden // dims["heads"]

    plan: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (dims["vocab"], hidden),
        "output.weight": (dims["vocab"], hidden),
        "output_norm.weight": (hidden,),
        "blk.0.attn_norm.weight": (hidden,),
        "blk.0.ffn_norm.weight": (hidden,),
        "blk.0.attn_q.weight": (hidden, hidden),
        "blk.0.attn_k.weight": (kv_width, hidden),
        "blk.0.attn_v.weight": (kv_width, hidden),
        "blk.0.attn_output.weight": (hidden, hidden),
        "blk.0.ffn_gate_inp.weight": (experts, hidden),
    }
    projections = {"gate": (ffn, hidden), "up": (ffn, hidden), "down": (hidden, ffn)}
    if layout == "stacked":
        for projection, shape in projections.items():
            plan[f"blk.0.ffn_{projection}_exps.weight"] = (experts, *shape)
    elif layout == "legacy":
        for expert in range(experts):
            for projection, shape in projections.items():
                plan[f"blk.0.ffn_{projection}.{expert}.weight"] = shape
    else:
        raise ValueError(f"Unknown Mixtral layout: {layout!r}")
    return plan


def write_mixtral_gguf(
    path: Path,
    *,
    layout: str = "legacy",
    dims: dict[str, int] | None = None,
    drop_expert: int | None = None,
    expert_count: int | None = None,
    plan: dict[str, tuple[int, ...]] | None = None,
) -> Path:
    """Write a one-layer Mixtral GGUF.

    ``drop_expert`` omits one expert's three projections, to exercise the
    incomplete-group refusal. ``expert_count`` overrides the metadata without
    changing the tensors, to exercise the missing/contradictory-metadata cases.
    """
    dims = {**DEFAULT_DIMS, **(dims or {})}
    plan = plan or mixtral_tensor_plan(dims, layout=layout)
    if drop_expert is not None:
        plan = {
            name: shape for name, shape in plan.items()
            if not name.endswith(f".{drop_expert}.weight") or "_exps" in name
        }

    writer = GGUFWriter(str(path), SOURCE_ARCH)
    writer.add_context_length(64)
    writer.add_embedding_length(dims["hidden"])
    writer.add_block_count(dims["layers"])
    writer.add_feed_forward_length(dims["ffn"])
    writer.add_head_count(dims["heads"])
    writer.add_head_count_kv(dims["kv_heads"])
    writer.add_rope_dimension_count(dims["hidden"] // dims["heads"])
    writer.add_expert_count(dims["experts"] if expert_count is None else expert_count)
    writer.add_expert_used_count(dims["used"])
    writer.add_token_list([f"token{index}" for index in range(dims["vocab"])])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    for name, shape in reversed(list(plan.items())):
        values = _values_for(name, shape, dims)
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _values_for(name: str, shape: tuple[int, ...], dims: dict[str, int]) -> np.ndarray:
    """Weights that identify their own position, so order errors are visible."""
    if "norm" in name:
        return np.full(shape, 1.0, dtype=np.float32)
    expert = _expert_index(name)
    if expert is not None:
        # Expert ``e`` is filled with ``e + 1`` everywhere: stacking the wrong
        # expert, or reversing the order, changes the value, not just the shape.
        return np.full(shape, float(expert + 1), dtype=np.float32)
    if name.endswith("_exps.weight"):
        experts = shape[0]
        return np.stack([
            np.full(shape[1:], float(index + 1), dtype=np.float32)
            for index in range(experts)
        ])
    if name == "blk.0.ffn_gate_inp.weight":
        # A router whose logits are ordered by expert index: routing weights are
        # the one tensor where a silent transpose still "works".
        return np.tile(np.arange(shape[1], dtype=np.float32), (shape[0], 1)) / shape[1]
    return np.full(shape, 0.5, dtype=np.float32)


def _expert_index(name: str) -> int | None:
    """The expert index in ``blk.<n>.ffn_<projection>.<e>.weight``, if any."""
    parts = name.split(".")
    if len(parts) == 5 and parts[2] in ("ffn_gate", "ffn_up", "ffn_down"):
        return int(parts[3])
    return None
