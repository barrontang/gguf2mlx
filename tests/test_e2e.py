"""Opt-in integration tests for the quantized MLX output path."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core
from gguf2mlx.mixed_validation import validate_mixed_artifacts

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
    from mlx import nn
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
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())

    save_model(model_dir, model)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
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
        moe_router_protect: bool = True,
        mixed_precision: bool = False,
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


def _build_tiny_gguf(path: Path, arch: str, hidden: int = 16, quantized: bool = False) -> None:
    """Synthesize a minimal but structurally valid GGUF for a fixture-backed arch.

    Weights are small random values; the goal is a file that survives the full
    convert() pipeline and loads under mlx_lm with finite logits, not numerical
    fidelity.
    """
    import gguf

    vocab, heads, ffn = 32, 4, max(32, hidden)
    if arch == "qwen2moe":
        ffn = 2 * hidden
    head_dim = hidden // heads

    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_name(f"tiny-{arch}")
    writer.add_block_count(1)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(heads)
    writer.add_file_type(2 if quantized else 1)

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

    def rand(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def add_tensor(name: str, array: np.ndarray) -> None:
        if quantized and array.ndim >= 2:
            from gguf.quants import quantize

            qtype = gguf.GGMLQuantizationType.Q4_0
            writer.add_tensor(name, quantize(array, qtype), raw_dtype=qtype)
        else:
            writer.add_tensor(name, array)

    if arch == "gemma":
        kv_heads = 2
        writer.add_head_count_kv(kv_heads)
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
        writer.add_context_length(128)
        writer.add_layer_norm_rms_eps(1e-6)
        add_tensor("token_embd.weight", rand(vocab, hidden))
        add_tensor("output_norm.weight", rand(hidden))
        add_tensor("blk.0.attn_q.weight", rand(heads * head_dim, hidden))
        add_tensor("blk.0.attn_k.weight", rand(kv_heads * head_dim, hidden))
        add_tensor("blk.0.attn_v.weight", rand(kv_heads * head_dim, hidden))
        add_tensor("blk.0.attn_output.weight", rand(hidden, heads * head_dim))
        add_tensor("blk.0.attn_norm.weight", rand(hidden))
        add_tensor("blk.0.ffn_norm.weight", rand(hidden))
        add_tensor("blk.0.ffn_gate.weight", rand(ffn, hidden))
        add_tensor("blk.0.ffn_up.weight", rand(ffn, hidden))
        add_tensor("blk.0.ffn_down.weight", rand(hidden, ffn))
    elif arch == "qwen2moe":
        writer.add_head_count_kv(heads)
        writer.add_context_length(128)
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_expert_count(4)
        writer.add_expert_used_count(2)
        add_tensor("token_embd.weight", rand(vocab, hidden))
        add_tensor("output.weight", rand(vocab, hidden))
        add_tensor("output_norm.weight", rand(hidden))
        for projection in ("q", "k", "v", "output"):
            add_tensor(f"blk.0.attn_{projection}.weight", rand(hidden, hidden))
            if projection != "output":
                add_tensor(f"blk.0.attn_{projection}.bias", rand(hidden))
        add_tensor("blk.0.attn_norm.weight", rand(hidden))
        add_tensor("blk.0.ffn_norm.weight", rand(hidden))
        add_tensor("blk.0.ffn_gate_inp.weight", rand(4, hidden))
        add_tensor("blk.0.ffn_gate_inp_shexp.weight", rand(hidden))
        for projection in ("gate", "up", "down"):
            expert_shape = (4, hidden, hidden)
            shared_shape = (hidden, 2 * hidden) if projection == "down" else (2 * hidden, hidden)
            add_tensor(f"blk.0.ffn_{projection}_exps.weight", rand(*expert_shape))
            add_tensor(f"blk.0.ffn_{projection}_shexp.weight", rand(*shared_shape))
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

    model, _tokenizer, config = load(str(output_dir), return_config=True)
    assert config["model_type"] == arch
    if arch == "gemma":
        from mlx import nn

        assert config["model_file"] == "gemma_model.py"
        values = mx.array([-2.0, -0.5, 0.5, 2.0])
        np.testing.assert_array_equal(
            np.asarray(model.layers[0].mlp.activation(values)),
            np.asarray(nn.gelu_approx(values)),
        )

    logits = model(mx.array([[1, 3, 4, 5]], dtype=mx.int32))
    assert logits.shape == (1, 4, config["vocab_size"])
    assert mx.isfinite(logits).all().item()


@pytest.mark.parametrize("direct_quant", [False, True])
def test_gemma_quantized_conversion_preserves_activation_adapter(tmp_path, direct_quant):
    import mlx.core as mx
    from mlx import nn
    from mlx_lm import load

    source, output = tmp_path / "gemma.gguf", tmp_path / "quantized"
    _build_tiny_gguf(source, "gemma", hidden=64, quantized=True)
    assert core.convert(
        str(source), str(output), quantize=True, direct_quant=direct_quant, q_group_size=64,
    )
    config = json.loads((output / "config.json").read_text())
    assert config["model_file"] == "gemma_model.py"
    assert (output / "gemma_model.py").read_bytes() == (
        Path(core.__file__).parent / "data" / "gemma_model.py"
    ).read_bytes()
    assert config["hidden_activation"] == "gelu_pytorch_tanh"
    model, _ = load(str(output))
    values = mx.array([-2.0, -0.5, 0.5, 2.0])
    np.testing.assert_array_equal(
        np.asarray(model.layers[0].mlp.activation(values)), np.asarray(nn.gelu_approx(values)),
    )
    assert mx.isfinite(model(mx.array([[3, 4]], dtype=mx.int32))).all().item()


@pytest.mark.parametrize("direct_quant", [False, True])
def test_tiny_qwen2_moe_shared_experts_load_and_run(tmp_path, direct_quant):
    import mlx.core as mx
    from mlx_lm import load

    source, output = tmp_path / "qwen.gguf", tmp_path / "qwen-mlx"
    _build_tiny_gguf(source, "qwen2moe", hidden=64, quantized=True)
    assert core.convert(
        str(source), str(output), quantize=True, direct_quant=direct_quant,
    )
    model, _, config = load(str(output), return_config=True)
    assert config["moe_intermediate_size"] == 64
    assert config["shared_expert_intermediate_size"] == 128
    assert model.layers[0].mlp.shared_expert_gate(mx.zeros((1, 64), dtype=mx.float16)).shape == (1, 1)
    logits = model(mx.array([[3, 4]], dtype=mx.int32))
    assert logits.shape == (1, 2, 32)
    assert mx.isfinite(logits).all().item()
    router_logits = _extract_router_logits(model, [3, 4])
    assert router_logits.shape == (1, 2, 4)
    assert np.isfinite(router_logits).all()


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


def _normalized_router_entropy(router_logits: np.ndarray) -> float:
    shifted = router_logits - router_logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=-1)
    return float(np.mean(entropy / np.log(router_logits.shape[-1])))


def _extract_router_logits(model, token_ids: list[int]) -> np.ndarray:
    import mlx.core as mx

    model_body = getattr(model, "model", model)
    gate = None
    for layer in model_body.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if callable(gate):
            break
    assert callable(gate), "Converted MoE model exposes no callable router gate"
    captured = []
    original_call = type(gate).__call__

    def capture(module, *args, **kwargs):
        result = original_call(module, *args, **kwargs)
        if module is gate:
            captured.append(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(gate), "__call__", capture)
        mx.eval(model(mx.array([token_ids], dtype=mx.int32)))
    assert len(captured) == 1, "Expected exactly one first-layer router evaluation"
    return np.asarray(captured[0])


@pytest.mark.parametrize("arch", ["qwen2moe", "qwen3moe", "deepseek2"])
def test_real_moe_model_has_finite_logits_routing_entropy_and_coherent_output(
    arch: str, tmp_path: Path
):
    """Opt-in real-model contract used by the scheduled Apple Silicon workflow."""
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")

    model_paths = json.loads(os.getenv("GGUF2MLX_MOE_MODELS", "{}"))
    source = model_paths.get(arch)
    if not source:
        pytest.skip(f"GGUF2MLX_MOE_MODELS does not provide {arch}")
    output_root = Path(os.getenv("GGUF2MLX_MOE_OUTPUT_ROOT", str(tmp_path)))
    output_dir = output_root / f"{arch}-mixed"
    assert core.convert(
        source,
        str(output_dir),
        quantize=True,
        direct_quant=True,
        mixed_precision=True,
    )
    mixed_report = validate_mixed_artifacts(output_dir)

    import mlx.core as mx
    from mlx_lm import generate, load

    uniform_dir = output_root / f"{arch}-uniform"
    assert core.convert(source, str(uniform_dir), quantize=True, direct_quant=True)
    model, tokenizer = load(str(uniform_dir))
    token_ids = tokenizer.encode("Explain why the sky is blue.", add_special_tokens=False)
    logits = model(mx.array([token_ids], dtype=mx.int32))
    assert mx.isfinite(logits).all().item()

    router_logits = _extract_router_logits(model, token_ids)
    entropy = _normalized_router_entropy(router_logits)
    assert 0.05 < entropy < 0.95

    prompt = "Explain why the sky is blue."
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
    output = generate(model, tokenizer, prompt=prompt, max_tokens=32)
    generated_tokens = tokenizer.encode(output, add_special_tokens=False)
    assert output.isprintable()
    assert len(set(generated_tokens)) >= max(2, len(generated_tokens) // 8)
    if report_dir := os.getenv("GGUF2MLX_MOE_REPORT_DIR"):
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as handle:
            source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        result = {
            "success": True,
            "architecture": arch,
            "source": source,
            "source_sha256": source_hash,
            "source_bytes": Path(source).stat().st_size,
            "mixed_model_directory": str(output_dir),
            "uniform_model_directory": str(uniform_dir),
            "mixed_artifacts_valid": True,
            "mixed_protected_routers": len(mixed_report["protected_router_tensors"]),
            "mixed_compressed_experts": len(mixed_report["compressed_expert_tensors"]),
            "uniform_model_loaded": True,
            "logits_shape": list(logits.shape),
            "logits_finite": bool(mx.isfinite(logits).all().item()),
            "first_layer_router_logits_shape": list(router_logits.shape),
            "normalized_router_entropy": entropy,
            "router_entropy_bounds": [0.05, 0.95],
            "generation_prompt": prompt,
            "generated_text": output,
            "generated_token_count": len(generated_tokens),
            "generated_unique_tokens": len(set(generated_tokens)),
            "generation_checks": "printable and non-degenerate; not a semantic-quality proof",
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("gguf", "mlx", "mlx-lm", "numpy")
            },
        }
        (directory / f"{arch}-real-validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
