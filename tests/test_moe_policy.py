from __future__ import annotations

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core


class _Tensor:
    def __init__(self, name: str):
        self.name = name


class _Reader:
    def __init__(self, names: list[str]):
        self.tensors = [_Tensor(name) for name in names]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("blk.0.ffn_gate_inp.weight", "router"),
        ("model.layers.0.mlp.gate.weight", "router"),
        ("blk.0.ffn_up_exps.weight", "expert_mlp"),
        ("model.layers.0.mlp.switch_mlp.down_proj.weight", "expert_mlp"),
        ("blk.0.attn_q.weight", "attention"),
        ("blk.0.ffn_norm.weight", "unknown"),
        ("token_embd.weight", "other"),
    ],
)
def test_tensor_policy_classification(name: str, expected: str):
    assert core.classify_tensor(name) == expected


def test_architecture_kind_detects_dense_moe_and_hybrid(monkeypatch):
    monkeypatch.setattr(core, "rust_classify_architecture", lambda *_args: None)
    assert core.classify_architecture(_Reader([]), "llama") == "dense"
    assert core.classify_architecture(
        _Reader(["blk.0.ffn_gate_inp.weight", "blk.0.ffn_up_exps.weight"]),
        "qwen3moe",
    ) == "moe"
    assert core.classify_architecture(
        _Reader(
            [
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_up_exps.weight",
                "blk.1.ffn_up.weight",
            ]
        ),
        "qwen3moe",
    ) == "hybrid"


def test_moe_layout_without_router_fails_closed():
    reader = _Reader(["blk.0.ffn_up_exps.weight"])
    with pytest.raises(RuntimeError, match="no recognized router tensor"):
        core.validate_moe_layout(reader, "qwen3moe", "moe")


def test_affine_4bit_supports_stacked_expert_weights():
    weights = np.arange(2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)
    packed, scales, biases = core._quantize_affine_4bit(weights, 64)
    assert packed.shape == (2, 3, 32)
    assert scales.shape == (2, 3, 1)
    assert biases.shape == (2, 3, 1)


def test_mixed_precision_requires_direct_quant(tmp_path, capsys):
    assert core.convert("missing.gguf", str(tmp_path / "out"), mixed_precision=True) is False
    assert "requires --quantize and --direct-quant" in capsys.readouterr().out
