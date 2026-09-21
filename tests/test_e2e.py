"""Opt-in integration tests for the quantized MLX output path."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import pytest

import numpy as np

from gguf2mlx import gguf2mlx as core

RUN_E2E = os.getenv("GGUF2MLX_RUN_E2E") == "1"
ARCH_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "architectures"

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


def _write_tiny_mlx_llama_model_quantized(model_dir: Path) -> None:
    """Write a tiny direct-quant-style MLX llama model (packed int4 weights)."""
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import save_model
    from mlx import nn

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
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())

    save_model(model_dir, model)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                **{
                    "model_type": "llama",
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 32,
                    "max_position_embeddings": 32,
                    "tie_word_embeddings": True,
                    "torch_dtype": "float16",
                },
                "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
            },
            indent=2,
        )
    )
    _write_tiny_tokenizer(model_dir)


def test_direct_quant_output_loads_with_mlx_lm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """mlx_lm.load() must succeed on direct-quant output and produce finite logits."""
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")

    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm import load

    output_dir = tmp_path / "direct-quant-model"

    def fake_direct_convert(
        gguf_path: str,
        output_path: str,
        dtype: str,
        q_bits: int,
        q_group_size: int,
        q_mode: str,
    ) -> bool:
        _write_tiny_mlx_llama_model_quantized(Path(output_path))
        return True

    monkeypatch.setattr(core, "_convert_direct_quantized", fake_direct_convert)

    assert (
        core.convert(
            "dummy.gguf",
            str(output_dir),
            quantize=True,
            q_bits=4,
            q_group_size=32,
            q_mode="affine",
            direct_quant=True,
        )
        is True
    )

    model, tokenizer, config = load(str(output_dir), return_config=True)
    assert config["quantization"]["bits"] == 4
    assert tokenizer.encode("hello world", add_special_tokens=False) == [3, 4]

    quantized_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda module: isinstance(module, nn.Module)
    )
    assert any(isinstance(module, nn.QuantizedLinear) for _, module in quantized_modules)

    logits = model(mx.array([[3, 4]], dtype=mx.int32))
    assert logits.shape == (1, 2, 32)
    assert mx.isfinite(logits).all().item()


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


def _build_tiny_gguf(path: Path, arch: str) -> None:
    """Synthesize a minimal but structurally valid GGUF for a fixture-backed arch.

    Weights are small random values; the goal is a file that survives the full
    convert() pipeline and loads under mlx_lm with finite logits, not numerical
    fidelity.
    """
    import gguf

    vocab, hidden, heads, ffn = 32, 16, 4, 32
    head_dim = hidden // heads

    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_name(f"tiny-{arch}")
    writer.add_block_count(1)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(heads)
    writer.add_file_type(1)

    tokens = [f"<t{i}>" for i in range(vocab)]
    tokens[0], tokens[1], tokens[2] = "<unk>", "<s>", "</s>"
    token_types = [1] * vocab
    token_types[0], token_types[1], token_types[2] = 2, 3, 3
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    rng = np.random.default_rng(abs(hash(arch)) % (2**32))

    def rand(*shape: int) -> "np.ndarray":
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    if arch == "gemma":
        kv_heads = 2
        writer.add_head_count_kv(kv_heads)
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
        writer.add_context_length(128)
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_tensor("token_embd.weight", rand(vocab, hidden))
        writer.add_tensor("output_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.attn_q.weight", rand(heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_k.weight", rand(kv_heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_v.weight", rand(kv_heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_output.weight", rand(hidden, heads * head_dim))
        writer.add_tensor("blk.0.attn_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.ffn_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.ffn_gate.weight", rand(ffn, hidden))
        writer.add_tensor("blk.0.ffn_up.weight", rand(ffn, hidden))
        writer.add_tensor("blk.0.ffn_down.weight", rand(hidden, ffn))
    elif arch == "phi3":
        kv_heads = heads  # keep q/k/v head counts equal for the fused qkv path
        writer.add_head_count_kv(kv_heads)
        writer.add_context_length(4096)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_dimension_count(head_dim)
        writer.add_tensor("token_embd.weight", rand(vocab, hidden))
        writer.add_tensor("output.weight", rand(vocab, hidden))  # untied lm_head
        writer.add_tensor("output_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.attn_q.weight", rand(heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_k.weight", rand(kv_heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_v.weight", rand(kv_heads * head_dim, hidden))
        writer.add_tensor("blk.0.attn_output.weight", rand(hidden, heads * head_dim))
        writer.add_tensor("blk.0.attn_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.ffn_norm.weight", rand(hidden))
        writer.add_tensor("blk.0.ffn_up.weight", rand(2 * ffn, hidden))  # gate_up_proj
        writer.add_tensor("blk.0.ffn_down.weight", rand(hidden, ffn))
    else:  # pragma: no cover - guard against silent misuse
        raise ValueError(f"No tiny-GGUF builder for arch {arch!r}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("arch", ["gemma", "phi3"])
def test_real_gguf_converts_and_loads_with_finite_logits(arch: str, tmp_path: Path):
    """Convert a synthesized real GGUF end-to-end and validate mlx_lm forward pass.

    This exercises the full pipeline (no monkeypatching): GGUFReader, tokenizer
    extraction, tensor dequant/remap, safetensors shard finalization, then
    mlx_lm.load() and a forward pass that must produce finite logits.
    """
    pytest.importorskip("gguf")
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")

    import mlx.core as mx
    from mlx_lm import load

    gguf_path = tmp_path / f"{arch}.gguf"
    _build_tiny_gguf(gguf_path, arch)

    output_dir = tmp_path / f"{arch}-mlx"
    assert core.convert(str(gguf_path), str(output_dir), dtype="float16") is True

    index_path = output_dir / "model.safetensors.index.json"
    assert index_path.exists(), "conversion did not finalize a safetensors index"

    model, tokenizer, config = load(str(output_dir), return_config=True)
    assert config["model_type"] == arch

    logits = model(mx.array([[1, 3, 4, 5]], dtype=mx.int32))
    assert logits.shape == (1, 4, config["vocab_size"])
    assert mx.isfinite(logits).all().item()


@pytest.mark.parametrize("fixture_name", ["gemma", "phi3"])
def test_adapter_fixture_matches_mlx_lm_parameter_contract(fixture_name: str):
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")

    from mlx.utils import tree_flatten

    fixture = json.loads(
        (ARCH_FIXTURE_DIR / f"{fixture_name}.json").read_text(encoding="utf-8")
    )
    if fixture_name == "gemma":
        from mlx_lm.models.gemma import Model, ModelArgs

        model = Model(
            ModelArgs(
                model_type="gemma",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=4,
                head_dim=4,
                rms_norm_eps=1e-6,
                vocab_size=32,
                num_key_value_heads=2,
            )
        )
    else:
        from mlx_lm.models.phi3 import Model, ModelArgs

        model = Model(
            ModelArgs(
                model_type="phi3",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=32,
                num_key_value_heads=2,
                max_position_embeddings=4096,
                original_max_position_embeddings=4096,
            )
        )

    model_keys = {name for name, _ in tree_flatten(model.parameters())}
    expected_keys = set(fixture["tensor_map"].values())
    assert expected_keys <= model_keys
