"""Opt-in integration tests for the quantized MLX output path."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import pytest

from gguf2mlx import gguf2mlx as core

RUN_E2E = os.getenv("GGUF2MLX_RUN_E2E") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_E2E,
        reason="Set GGUF2MLX_RUN_E2E=1 to run integration tests.",
    ),
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="MLX integration tests require Apple Silicon.",
    ),
]


def _write_tiny_tokenizer(model_dir: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "hello": 3,
        "world": 4,
        "mlx": 5,
        "gguf": 6,
        "test": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<unk>",
    )
    fast_tokenizer.model_max_length = 32
    fast_tokenizer.save_pretrained(model_dir)


def _write_tiny_mlx_llama_model(model_dir: Path) -> None:
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import save_model

    model_dir.mkdir(parents=True, exist_ok=True)

    args = ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    model = Model(args)
    mx.eval(model.parameters())

    save_model(model_dir, model)
    (model_dir / "config.json").write_text(
        json.dumps({**asdict(args), "torch_dtype": "float32"}, indent=2)
    )
    _write_tiny_tokenizer(model_dir)


def test_quantized_output_loads_with_mlx_lm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")

    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm import load

    output_dir = tmp_path / "quantized-model"

    def fake_convert(_input: str, staging_dir: str, _dtype: str) -> bool:
        _write_tiny_mlx_llama_model(Path(staging_dir))
        return True

    monkeypatch.setattr(core, "_convert", fake_convert)

    assert (
        core.convert(
            "dummy.gguf",
            str(output_dir),
            quantize=True,
            q_bits=4,
            q_group_size=32,
            q_mode="affine",
        )
        is True
    )

    model, tokenizer, config = load(str(output_dir), return_config=True)
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["group_size"] == 32
    assert tokenizer.encode("hello world", add_special_tokens=False) == [3, 4]

    quantized_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda module: isinstance(module, nn.Module)
    )
    assert any(isinstance(module, nn.QuantizedLinear) for _, module in quantized_modules)

    logits = model(mx.array([[3, 4]], dtype=mx.int32))
    assert logits.shape == (1, 2, 32)
