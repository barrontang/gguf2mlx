from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

from gguf2mlx import vlm
from gguf2mlx.vision_minicpm import MiniCPMV47Adapter


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class Reader:
    def __init__(self, metadata, tensors=()):
        self.metadata = metadata
        self.fields = metadata
        self.tensors = list(tensors)

    def get_field(self, name):
        return Field(self.metadata[name]) if name in self.metadata else None


def tensor(name, array):
    array = np.asarray(array, dtype=np.float32)
    return SimpleNamespace(
        name=name, data=array, shape=np.array(array.shape[::-1]), tensor_type=0,
    )


def model_config():
    return {
        "model_type": "minicpmv4_7",
        "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
        "mrope_mode": "canvas",
        "downsample_mode": "16x",
        "insert_layer_id": 0,
        "image_token_id": 4,
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 4,
            "intermediate_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1, "vocab_size": 5,
            "head_dim": 4, "tie_word_embeddings": False, "attention_bias": False,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "rope_parameters": {"mrope_section": [1, 1, 1]},
        },
        "vision_config": {
            "model_type": "siglip", "hidden_size": 4, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 1,
            "image_size": 4, "patch_size": 2, "num_channels": 3,
            "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
        },
    }


def make_readers(config):
    text_metadata = {
        "general.architecture": "qwen35",
        "qwen35.vocab_size": 5,
        "qwen35.embedding_length": 4,
        "qwen35.feed_forward_length": 8,
        "qwen35.block_count": 1,
        "qwen35.attention.head_count": 1,
        "qwen35.attention.head_count_kv": 1,
        "qwen35.attention.key_length": 4,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.rope.freq_base": 1000000.0,
        "qwen35.ssm.group_count": 2,
        "qwen35.ssm.time_step_rank": 4,
        "qwen35.ssm.state_size": 8,
        "qwen35.ssm.inner_size": 32,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.full_attention_interval": 4,
        "qwen35.rope.dimension_sections": [1, 1, 1, 0],
        "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "word", "<image>"],
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }
    text_names = {
        "token_embd.weight": (5, 4),
        "output_norm.weight": (4,),
        "output.weight": (5, 4),
        "blk.0.attn_norm.weight": (4,),
        "blk.0.attn_q.weight": (4, 4),
        "blk.0.attn_k.weight": (4, 4),
        "blk.0.attn_v.weight": (4, 4),
        "blk.0.attn_output.weight": (4, 4),
        "blk.0.ffn_norm.weight": (4,),
        "blk.0.ffn_gate.weight": (8, 4),
        "blk.0.ffn_up.weight": (8, 4),
        "blk.0.ffn_down.weight": (4, 8),
    }
    text = Reader(text_metadata, [
        tensor(name, np.ones(shape, dtype=np.float32) * (index + 1))
        for index, (name, shape) in enumerate(text_names.items())
    ])
    adapter = MiniCPMV47Adapter(config)
    vision_metadata = {
        "general.architecture": "clip",
        "clip.projector_type": "minicpmv4_7",
        "clip.vision.embedding_length": 4,
        "clip.vision.feed_forward_length": 8,
        "clip.vision.block_count": 1,
        "clip.vision.attention.head_count": 1,
        "clip.vision.image_size": 448,
        "clip.vision.patch_size": 2,
        "clip.vision.projector.scale_factor": (
            2 if config.get("downsample_mode") == "4x" else 4
        ),
        "clip.vision.wa_layer_indexes": [0],
        "clip.use_gelu": True,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
    }
    vision = Reader(vision_metadata, [
        tensor(name, np.ones(shape, dtype=np.float32) * (index + 20))
        for index, (name, shape) in enumerate(adapter.source_shapes.items())
    ] + [tensor("v.tok_embd_sep", np.full((4, 4), 99, dtype=np.float32))])
    return text, vision


@pytest.fixture
def model(tmp_path, monkeypatch):
    config = model_config()
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokens = ["<unk>", "<s>", "</s>", "word", "<image>"]
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {token: i for i, token in enumerate(tokens)}},
        "added_tokens": [],
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "MiniCPMV4_7ImageProcessorPil",
        "scale_resolution": 448,
    }), encoding="utf-8")
    text, vision = make_readers(config)
    readers = {"main.gguf": text, "mmproj.gguf": vision}
    monkeypatch.setattr(vlm, "GGUFReader", lambda name: readers[str(name)])
    return SimpleNamespace(root=root, readers=readers, output=tmp_path / "out")


def convert(model, **kwargs):
    vlm.convert_vlm(
        "main.gguf", model.output, "float32", mmproj="mmproj.gguf",
        hf_model=str(model.root), offline=True, **kwargs,
    )


def test_minicpm_v47_dense_conversion_maps_both_mergers_and_validates_separator(model):
    convert(model)

    weights = load_file(model.output / "model.safetensors")
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in weights
    assert "vision_tower.embeddings.patch_embedding.bias" in weights
    assert "vit_merger.self_attn.q_proj.weight" in weights
    assert "merger.mlp.0.pre_norm.weight" in weights
    assert "merger.mlp.0.linear_2.weight" in weights
    assert "v.tok_embd_sep" not in weights
    config = json.loads((model.output / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "minicpmv4_7"
    assert config["text_config"]["model_type"] == "qwen3_5_text"
    report = json.loads((model.output / "vlm_conversion_report.json").read_text())
    assert report["tensor_counts"]["separator_embedding_ignored"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "minicpmv4_6", "minicpmv4_7"),
        ("text_model_type", "qwen3_5_moe_text", "dense qwen3_5_text"),
    ],
)
def test_minicpm_v47_rejects_unverified_architectures(field, value, message):
    config = model_config()
    if field == "text_model_type":
        config["text_config"]["model_type"] = value
    else:
        config[field] = value
    with pytest.raises(ValueError, match=message):
        MiniCPMV47Adapter(config)


def test_minicpm_v47_rejects_moe_text_tensors(model):
    reader = model.readers["main.gguf"]
    reader.tensors.append(tensor("blk.0.ffn_gate_exps.weight", np.ones((2, 4, 4))))
    with pytest.raises(ValueError, match="dense layouts only"):
        convert(model)


def test_minicpm_v47_4x_omits_window_merger():
    config = model_config()
    config["downsample_mode"] = "4x"
    adapter = MiniCPMV47Adapter(config)

    assert not any(name.startswith("vit_merger.") for name in adapter.expected)
    assert adapter.map_projector("mm.up.weight") == "merger.mlp.0.linear_1.weight"
    with pytest.raises(ValueError, match="Unsupported MiniCPM-V 4.7 vision tensor"):
        adapter.map_vision("v.vit_merger.ln1.weight")


def test_plain_minicpm3_is_not_detected_as_vlm():
    assert not vlm.is_vlm(Reader({"general.architecture": "minicpm3"}))
