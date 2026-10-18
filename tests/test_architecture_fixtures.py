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


def test_python_name_detection_prefers_specific_gemma_variants(monkeypatch):
    monkeypatch.setattr(core, "rust_detect_architecture", lambda arch, name: None)

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
