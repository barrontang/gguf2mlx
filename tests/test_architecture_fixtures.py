from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "architectures"


class _FakeField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _FakeTensor:
    def __init__(self, name: str, data=()):
        self.name = name
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = tuple(reversed(self.data.shape))


class _FakeReader:
    def __init__(self, mapping, tensor_names=()):
        self._mapping = mapping
        self.tensors = [_FakeTensor(name) for name in tensor_names]

    def get_field(self, key):
        value = self._mapping.get(key)
        return None if value is None else _FakeField(value)


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("separate_output", [False, True])
def test_llama_config_detects_tied_embeddings_from_tensor_layout(separate_output):
    metadata = {
        "llama.vocab_size": 32,
        "llama.embedding_length": 64,
        "llama.block_count": 1,
        "llama.attention.head_count": 4,
        "llama.context_length": 8192,
        "llama.rope.freq_base": 100000.0,
    }
    tensors = ["token_embd.weight"]
    if separate_output:
        tensors.append("output.weight")
    config = core.build_config(_FakeReader(metadata, tensors), "llama")
    assert config["tie_word_embeddings"] is not separate_output
    assert config["rope_theta"] == 100000.0


@pytest.mark.parametrize(("projection", "heads"), [("q", 4), ("k", 2)])
def test_llama_rotary_projection_restores_hf_order(projection, heads):
    original = np.arange(heads * 8 * 16, dtype=np.float16).reshape(heads * 8, 16)
    gguf_weight = original.reshape(heads, 2, 4, 16).swapaxes(1, 2).reshape(original.shape)
    emitted = core._plan_tensor_emit(
        f"blk.0.attn_{projection}.weight", gguf_weight, "llama",
        {"num_heads": 4, "num_key_value_heads": 2}, {}, {},
    )
    np.testing.assert_array_equal(emitted[0][1], original)


def _reader_with_rope_table(metadata: dict, table, tensors=("token_embd.weight",)):
    reader = _FakeReader(metadata, tensors)
    if table is not None:
        reader.tensors.append(_FakeTensor("rope_freqs.weight", table))
    return reader


def test_llama3_rope_table_becomes_a_scaling_block():
    """llama.cpp ships llama3 scaling as rop_freqs only; recover the metadata.

    Llama-3.2-1B's real GGUF has no ``llama.rope.scaling.*`` fields at all, so
    dropping the tensor without this inference would leave plain RoPE in the
    config and rotate every position slightly wrong.
    """
    dim, base, factor, original = 64, 500000.0, 32.0, 8192
    table = core._rope_freq_multiplier(dim, base, factor, original)
    metadata = {
        "llama.vocab_size": 32,
        "llama.embedding_length": 2048,
        "llama.block_count": 1,
        "llama.attention.head_count": 32,
        "llama.context_length": 131072,
        "llama.rope.freq_base": base,
        "llama.rope.dimension_count": dim,
    }
    config = core.build_config(_reader_with_rope_table(metadata, table), "llama")

    assert config["rope_scaling"] == {
        "type": "llama3",
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    # The inference is only accepted because it regenerates the table.
    np.testing.assert_allclose(
        core._rope_freq_multiplier(dim, base, 32.0, 8192), table, rtol=0, atol=1e-12
    )


def test_rope_table_without_a_matching_context_is_rejected():
    table = core._rope_freq_multiplier(64, 500000.0, 32.0, 8192)
    table[3] *= 7.0  # not any llama3 table we can express
    with pytest.raises(ValueError, match="does not match llama3 rope scaling"):
        core._rope_scaling_from_table(table, 64, 500000.0, "llama")


def test_rope_table_is_rejected_for_architectures_without_rope_scaling():
    table = core._rope_freq_multiplier(64, 500000.0, 32.0, 8192)
    with pytest.raises(ValueError, match="unsupported for gemma"):
        core._rope_scaling_from_table(table, 64, 500000.0, "gemma")


def test_qwen3_ties_embeddings_when_output_weight_is_absent():
    # Qwen3-0.6B ships token_embd only; forcing tie_word_embeddings=False made
    # mlx_lm expect an lm_head the file never had ("Missing 1 parameters").
    metadata = {
        "qwen3.embedding_length": 1024,
        "qwen3.block_count": 28,
        "qwen3.attention.head_count": 16,
        "qwen3.context_length": 40960,
        "qwen3.rope.freq_base": 1000000.0,
    }
    tied = core.build_config(_FakeReader(metadata, ["token_embd.weight"]), "qwen3")
    untied = core.build_config(
        _FakeReader(metadata, ["token_embd.weight", "output.weight"]), "qwen3"
    )
    assert tied["tie_word_embeddings"] is True
    assert untied["tie_word_embeddings"] is False


@pytest.mark.parametrize(
    ("gguf_name", "expected"),
    [
        ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
        ("blk.0.post_attention_norm.weight",
         "model.layers.0.post_attention_layernorm.weight"),
        ("blk.0.ffn_norm.weight", "model.layers.0.pre_feedforward_layernorm.weight"),
        ("blk.0.post_ffw_norm.weight",
         "model.layers.0.post_feedforward_layernorm.weight"),
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
    ],
)
def test_gemma2_maps_all_four_block_norms(gguf_name, expected):
    assert core._map_tensor_name(gguf_name, "gemma2") == expected


def test_gemma1_keeps_the_two_norm_layout():
    # Gemma 1 really does use ffn_norm as the post-attention norm.
    assert core._map_tensor_name("blk.0.ffn_norm.weight", "gemma") == (
        "model.layers.0.post_attention_layernorm.weight"
    )


def test_gemma_post_norms_are_de_shifted_like_the_other_norms():
    # llama.cpp stores Gemma RMSNorm weights as 1 + w; mlx_lm adds the 1 back.
    for name in ("blk.0.post_attention_norm.weight", "blk.0.post_ffw_norm.weight"):
        arr = np.array([1.5, 2.0], dtype=np.float32)
        restored = core._restore_architecture_tensor(name, arr, "gemma2")
        np.testing.assert_allclose(restored, [0.5, 1.0])


def test_llama_rotary_projection_leaves_degenerate_tensor_untouched():
    # Streaming/smoke fixtures use tiny tensors that cannot represent the head
    # count declared in metadata; they must pass through instead of raising.
    arr = np.arange(12, dtype=np.float16).reshape(3, 4)
    emitted = core._plan_tensor_emit(
        "blk.0.attn_q.weight", arr, "llama",
        {"num_heads": 64, "num_key_value_heads": 64}, {}, {},
    )
    np.testing.assert_array_equal(emitted[0][1], arr)


def test_llama_rotary_projection_rejects_uneven_head_split():
    arr = np.zeros((12, 8), dtype=np.float16)  # 12 rows / 4 heads -> odd head_dim 3
    with pytest.raises(ValueError, match="cannot be split into 4 heads"):
        core._plan_tensor_emit(
            "blk.0.attn_q.weight", arr, "llama",
            {"num_heads": 4, "num_key_value_heads": 4}, {}, {},
        )


def test_build_config_strict_rejects_defaulted_dimensions():
    reader = _FakeReader({"llama.vocab_size": 32}, tensor_names=["token_embd.weight"])
    with pytest.raises(ValueError, match="embedding_length"):
        core.build_config(reader, "llama", strict=True)

    config = core.build_config(reader, "llama")  # inspect-style call stays lenient
    assert config["hidden_size"] == 4096
    assert config["num_hidden_layers"] == 32
    assert config["num_attention_heads"] == 32


def test_build_config_strict_accepts_vocab_from_tokenizer():
    metadata = {
        "llama.embedding_length": 64,
        "llama.block_count": 1,
        "llama.attention.head_count": 4,
        "tokenizer.ggml.tokens": ["<s>", "</s>", "a"],
    }
    config = core.build_config(_FakeReader(metadata), "llama", strict=True)
    assert config["vocab_size"] == 3


@pytest.fixture
def deepseek_v2_metadata():
    return {
        "deepseek2.vocab_size": 102400,
        "deepseek2.embedding_length": 2048,
        "deepseek2.block_count": 27,
        "deepseek2.context_length": 163840,
        "deepseek2.attention.head_count": 16,
        "deepseek2.attention.key_length": 192,
        "deepseek2.attention.value_length": 128,
        "deepseek2.attention.kv_lora_rank": 512,
        "deepseek2.feed_forward_length": 10944,
        "deepseek2.expert_feed_forward_length": 1408,
        "deepseek2.expert_count": 64,
        "deepseek2.expert_shared_count": 2,
        "deepseek2.expert_used_count": 6,
        "deepseek2.leading_dense_block_count": 1,
        "deepseek2.rope.dimension_count": 64,
        "deepseek2.rope.scaling.type": "yarn",
        "deepseek2.rope.scaling.factor": 40.0,
        "deepseek2.rope.scaling.original_context_length": 4096,
        "deepseek2.rope.scaling.yarn_log_multiplier": 0.0707,
    }


def test_deepseek_v2_config_preserves_mla_dense_experts_and_yarn(deepseek_v2_metadata):
    config = core.build_config(_FakeReader(deepseek_v2_metadata), "deepseek2")
    expected = {
        "intermediate_size": 10944,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "first_k_dense_replace": 1,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "routed_scaling_factor": 1.0,
        "topk_method": "greedy",
    }
    for key, value in expected.items():
        assert config[key] == value
    assert config["rope_scaling"]["type"] == "yarn"
    assert config["rope_scaling"]["factor"] == 40.0
    assert config["rope_scaling"]["original_max_position_embeddings"] == 4096
    assert config["rope_scaling"]["mscale"] == pytest.approx(0.707)
    assert config["rope_scaling"]["mscale_all_dim"] == pytest.approx(0.707)


def test_deepseek_v2_config_preserves_query_lora_and_unscaled_rope(deepseek_v2_metadata):
    deepseek_v2_metadata["deepseek2.attention.q_lora_rank"] = 1536
    del deepseek_v2_metadata["deepseek2.rope.scaling.type"]
    config = core.build_config(_FakeReader(deepseek_v2_metadata), "deepseek2")
    assert config["q_lora_rank"] == 1536
    assert config["rope_scaling"] == {"type": "yarn", "factor": 1.0}


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("attention.kv_lora_rank", None, "requires positive attention.kv_lora_rank"),
        ("expert_feed_forward_length", 0, "requires positive expert_feed_forward_length"),
        ("rope.dimension_count", 192, "rotary dimension"),
        ("leading_dense_block_count", -1, "dense block count"),
        ("leading_dense_block_count", 28, "dense block count"),
        ("expert_shared_count", -1, "shared expert count"),
        ("attention.q_lora_rank", 0, "query LoRA rank"),
        ("expert_weights_scale", float("nan"), "expert weights scale"),
        ("rope.scaling.type", "linear", "Unsupported DeepSeek-V2 rope scaling"),
        ("rope.scaling.factor", None, "YaRN factor"),
        ("rope.scaling.factor", float("inf"), "YaRN factor"),
        ("rope.scaling.yarn_log_multiplier", -0.1, "YaRN log multiplier"),
        ("rope.scaling.original_context_length", None, "requires positive"),
    ],
)
def test_deepseek_v2_config_rejects_invalid_metadata(
    deepseek_v2_metadata, key, value, message,
):
    deepseek_v2_metadata[f"deepseek2.{key}"] = value
    with pytest.raises(ValueError, match=message):
        core.build_config(_FakeReader(deepseek_v2_metadata), "deepseek2")


@pytest.mark.parametrize("fixture_name", ["gemma", "phi3", "phi4"])
def test_architecture_fixture_config_and_tensor_map(fixture_name: str):
    fixture = _load_fixture(fixture_name)
    arch = fixture["architecture"]
    config = core.build_config(_FakeReader(fixture["metadata"]), arch)

    for key, expected in fixture["expected_config"].items():
        assert config[key] == expected
    for gguf_name, expected in fixture["tensor_map"].items():
        assert core._map_tensor_name(gguf_name, arch) == expected


def test_gemma_fixture_restores_shifted_norm_weights():
    shifted = np.array([1.25, 0.5], dtype=np.float16)

    restored = core._restore_architecture_tensor(
        "blk.0.attn_norm.weight", shifted, "gemma"
    )

    np.testing.assert_array_equal(restored, np.array([0.25, -0.5], dtype=np.float16))


@pytest.mark.parametrize("arch", ["phi3", "phi4"])
def test_phi_fixture_fuses_split_qkv_weights(arch):
    pending_qkv = {}
    mla_dims = {"num_heads": 4, "qk_nope_head_dim": 4, "v_head_dim": 4}
    pending_kv_b = {}

    assert core._plan_tensor_emit(
        "blk.0.attn_q.weight",
        np.full((4, 2), 1, dtype=np.float16),
        arch,
        mla_dims,
        pending_kv_b,
        pending_qkv,
    ) == []
    assert core._plan_tensor_emit(
        "blk.0.attn_k.weight",
        np.full((2, 2), 2, dtype=np.float16),
        arch,
        mla_dims,
        pending_kv_b,
        pending_qkv,
    ) == []
    emitted = core._plan_tensor_emit(
        "blk.0.attn_v.weight",
        np.full((2, 2), 3, dtype=np.float16),
        arch,
        mla_dims,
        pending_kv_b,
        pending_qkv,
    )

    assert emitted[0][0] == "model.layers.0.self_attn.qkv_proj.weight"
    np.testing.assert_array_equal(
        emitted[0][1][:, 0],
        np.array([1, 1, 1, 1, 2, 2, 3, 3], dtype=np.float16),
    )
    assert pending_qkv == {}


def test_new_adapters_reject_unknown_tensor_names():
    with pytest.raises(ValueError, match="Unsupported gemma tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "gemma")
    with pytest.raises(ValueError, match="Unsupported gemma2 tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "gemma2")
    with pytest.raises(ValueError, match="Unsupported qwen3 tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "qwen3")
    with pytest.raises(ValueError, match="Unsupported phi3 tensor"):
        core._map_tensor_name("blk.0.unknown.weight", "phi3")


def test_phi4_adapter_uses_phi3_metadata_and_tensor_layout():
    fixture = _load_fixture("phi4")
    reader = _FakeReader(fixture["metadata"])

    assert core.detect_architecture(reader) == "phi4"
    assert "phi4" in core.CONVERTIBLE_ARCHES
    assert "phi4" in core.STRICT_ADAPTER_ARCHES


def test_phi3_longrope_factors_are_written_to_config_not_weights():
    fixture = _load_fixture("phi3")
    reader = _FakeReader(fixture["metadata"])
    reader.tensors = [
        _FakeTensor("rope_factors_long.weight", [1.25, 2.5]),
        _FakeTensor("rope_factors_short.weight", [1.0, 1.5]),
    ]
    config = core.build_config(reader, "phi3")

    assert config["rope_scaling"] == {
        "type": "longrope",
        "long_factor": [1.25, 2.5],
        "short_factor": [1.0, 1.5],
    }
    assert list(core._iter_weight_arrays(reader, "phi3", "float16")) == []
    assert core.validate_architecture_variant(reader, "phi3") is None


def test_phi3_longrope_requires_matching_factor_tensors():
    reader = _FakeReader({})
    reader.tensors = [_FakeTensor("rope_factors_long.weight", [1.0, 1.0])]

    error = core.validate_architecture_variant(reader, "phi3")

    assert error is not None
    assert "requires both" in error


def test_phi3_longrope_rejects_invalid_factor_dimensions():
    fixture = _load_fixture("phi3")
    reader = _FakeReader(fixture["metadata"])
    reader.tensors = [
        _FakeTensor("rope_factors_long.weight", [1.0]),
        _FakeTensor("rope_factors_short.weight", [1.0, 1.0]),
    ]

    with pytest.raises(ValueError, match="factor lengths"):
        core.build_config(reader, "phi3")


def test_python_name_detection_prefers_specific_gemma_variants():
    assert core.detect_architecture(_FakeReader({"general.name": "Gemma-2-9B"})) == "gemma"
    assert core.detect_architecture(_FakeReader({"general.name": "gemma2-9b"})) == "gemma2"
    assert core.detect_architecture(_FakeReader({"general.name": "gemma3-4b"})) == "gemma3"
    assert core.detect_architecture(_FakeReader({"general.name": "Gemma4-E4B"})) == "gemma4"
    assert core.detect_architecture(_FakeReader({"general.name": "Qwen3-8B"})) == "qwen3"
    assert core.detect_architecture(_FakeReader({"general.name": "Qwen3.5-9B"})) == "qwen35"
    assert core.detect_architecture(_FakeReader({"general.name": "Qwen3.6-9B"})) == "qwen36"
    assert core.detect_architecture(_FakeReader({"general.name": "Qwen3.8-9B"})) == "qwen38"
    assert core.detect_architecture(_FakeReader({"general.name": "phi2"})) == "phi2"
    assert core.detect_architecture(_FakeReader({"general.name": "Phi-4-mini-instruct"})) == "phi4"


@pytest.mark.parametrize(
    "arch,model_type",
    [("gemma2", "gemma2"), ("gemma3", "gemma3")],
)
def test_gemma_2_and_3_configuration_and_norm_restore(arch, model_type):
    metadata = {
        f"{arch}.vocab_size": 32,
        f"{arch}.embedding_length": 16,
        f"{arch}.block_count": 1,
        f"{arch}.attention.head_count": 4,
        f"{arch}.attention.head_count_kv": 2,
        f"{arch}.attention.key_length": 4,
        f"{arch}.feed_forward_length": 32,
        f"{arch}.context_length": 128,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.attention.sliding_window": 64,
    }
    reader = _FakeReader(metadata)

    config = core.build_config(reader, arch)

    assert config["model_type"] == model_type
    assert config["head_dim"] == 4
    assert config["hidden_activation"] == "gelu_pytorch_tanh"
    assert config["sliding_window"] == 64
    assert config["attention_bias"] is False
    assert config["tie_word_embeddings"] is True
    np.testing.assert_array_equal(
        core._restore_architecture_tensor(
            "blk.0.attn_norm.weight", np.array([1.5]), arch
        ),
        np.array([0.5]),
    )


def test_gemma4_text_configuration_and_tensor_mapping():
    reader = _FakeReader({
        "gemma4.vocab_size": 32,
        "gemma4.embedding_length": 16,
        "gemma4.block_count": 2,
        "gemma4.attention.head_count": 4,
        "gemma4.attention.head_count_kv": 2,
        "gemma4.attention.key_length": 8,
        "gemma4.attention.key_length_swa": 4,
        "gemma4.feed_forward_length": 32,
        "gemma4.context_length": 128,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.attention.sliding_window_pattern": [1, 0],
    })

    config = core.build_config(reader, "gemma4")

    assert config["model_type"] == "gemma4_text"
    assert config["head_dim"] == 4
    assert config["global_head_dim"] == 8
    assert config["layer_types"] == ["sliding_attention", "full_attention"]
    assert core._map_tensor_name("blk.0.attn_post_norm.weight", "gemma4") == (
        "model.layers.0.post_attention_layernorm.weight"
    )
    assert core._map_tensor_name("per_layer_token_embd.weight", "gemma4") == (
        "model.embed_tokens_per_layer.weight"
    )
    assert core._map_tensor_name("blk.0.ffn_gate_inp.weight", "gemma4") == (
        "model.layers.0.router.proj.weight"
    )


def test_qwen3_dense_and_qwen35_family_mappings_and_configuration():
    assert core._map_tensor_name("blk.0.attn_q_norm.weight", "qwen3") == (
        "model.layers.0.self_attn.q_norm.weight"
    )
    assert core._map_tensor_name("blk.0.ssm_beta.weight", "qwen35") == (
        "model.layers.0.linear_attn.in_proj_b.weight"
    )
    assert core._map_tensor_name("blk.0.attn_q.weight", "qwen3.6") == (
        "model.layers.0.self_attn.q_proj.weight"
    )

    config = core.build_config(_FakeReader({
        "qwen35.vocab_size": 32,
        "qwen35.embedding_length": 16,
        "qwen35.block_count": 4,
        "qwen35.attention.head_count": 4,
        "qwen35.attention.head_count_kv": 2,
        "qwen35.attention.key_length": 4,
        "qwen35.feed_forward_length": 32,
        "qwen35.context_length": 128,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.ssm.group_count": 2,
        "qwen35.ssm.time_step_rank": 4,
        "qwen35.ssm.state_size": 8,
        "qwen35.ssm.inner_size": 32,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.full_attention_interval": 4,
        "qwen35.rope.dimension_sections": [1, 1, 1, 0],
    }), "qwen3.5")

    assert config["model_type"] == "qwen3_5"
    assert config["linear_num_key_heads"] == 2
    assert config["linear_num_value_heads"] == 4
    assert config["linear_key_head_dim"] == 8
    assert config["linear_value_head_dim"] == 8
    assert config["full_attention_interval"] == 4


def test_qwen35_restores_exported_gated_delta_parameters():
    source = np.array([-np.e, -np.e**2], dtype=np.float32)

    np.testing.assert_allclose(
        core._restore_architecture_tensor("ssm_a", source, "qwen35"),
        np.array([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        core._restore_architecture_tensor(
            "blk.0.ssm_norm.weight", np.array([1.25], dtype=np.float16), "qwen35"
        ),
        np.array([0.25], dtype=np.float16),
    )


def test_qwen3_dense_adapters_reject_expert_tensors():
    reader = _FakeReader({}, tensor_names=["blk.0.ffn_gate_exps.weight"])

    assert "dense layouts only" in core.validate_architecture_variant(reader, "qwen3")
    assert "dense layouts only" in core.validate_architecture_variant(reader, "qwen35")
