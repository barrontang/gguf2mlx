from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors import safe_open
from tokenizers import Tokenizer

from gguf2mlx import gguf2mlx as core
from gguf2mlx.qwen_vl_text import QWEN_VL_TEXT_ARCHES, Qwen3VLTextEncoderConverter
from gguf2mlx.streaming import StreamingShardWriter

TOKENS = ["<unk>", "a", "<|image_pad|>", "<|video_pad|>",
          "<|vision_start|>", "<|vision_end|>"]
TENSOR_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    **{f"blk.0.{source}.weight": f"model.layers.0.{target}.weight" for source, target in {
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
        "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
    }.items()},
}


def metadata(arch="qwen3vl"):
    return {
        "general.architecture": arch,
        "general.name": "Qwen3-VL-8B text conditioning",
        f"{arch}.embedding_length": 4096,
        f"{arch}.block_count": 32,
        f"{arch}.attention.head_count": 32,
        f"{arch}.attention.head_count_kv": 8,
        f"{arch}.attention.key_length": 128,
        f"{arch}.feed_forward_length": 12288,
        f"{arch}.context_length": 262144,
        f"{arch}.rope.freq_base": 5000000.0,
        f"{arch}.rope.dimension_sections": [24, 20, 20, 0],
        "tokenizer.ggml.tokens": TOKENS,
    }


class Reader:
    def __init__(self, fields, tensors=()):
        self.fields = fields
        self.tensors = [SimpleNamespace(name=name) for name in tensors]

    def get_field(self, key):
        if key not in self.fields:
            return None
        return SimpleNamespace(contents=lambda: self.fields[key])


@pytest.mark.parametrize("arch", sorted(QWEN_VL_TEXT_ARCHES))
def test_qwen3_topology_and_mapping(arch, monkeypatch):
    reader = Reader(metadata(arch), TENSOR_MAP)
    monkeypatch.setattr(core, "rust_detect_architecture", lambda *args: "qwen2")
    assert core.detect_architecture(reader) == arch
    config = core.build_config(reader, arch, "bfloat16")
    assert config["model_type"] == "qwen3_vl_text"
    assert config["architectures"] == ["Qwen3VLTextModel"]
    assert config["max_position_embeddings"] == 262144
    assert config["head_dim"] == 128
    assert config["num_key_value_heads"] == 8
    assert config["rope_theta"] == 5000000.0
    assert config["rope_scaling"] == {
        "rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True,
    }
    assert config["torch_dtype"] == "bfloat16"
    assert config["attention_bias"] is False
    assert config["tie_word_embeddings"] is False
    for key, value in {"image": 2, "video": 3, "vision_start": 4, "vision_end": 5}.items():
        assert config[f"{key}_token_id"] == value
    for source, target in TENSOR_MAP.items():
        assert core._map_tensor_name(source, arch) == target
    with pytest.raises(ValueError, match="Unsupported Qwen-VL text tensor"):
        core._map_tensor_name("blk.0.unknown.weight", arch)


def test_qwen2_is_not_relabelled_qwen3():
    fields = metadata("qwen2vl")
    fields["general.name"] = "Qwen2-VL"
    fields["qwen2vl.rope.dimension_sections"] = [16, 16, 32]
    config = core.build_config(Reader(fields), "qwen2vl", "bfloat16")
    assert config["model_type"] == "qwen2_vl_text"
    assert config["attention_bias"] is True
    assert config["tie_word_embeddings"] is True
    assert "mrope_interleaved" not in config["rope_scaling"]


@pytest.mark.parametrize("sections", [None, [], [16, 16], [16, 16, 31], [0, 32, 32],
                                     [24.5, 20, 19.5], [True, 31, 32], [24, 20, 20, 1]])
def test_invalid_mrope_fails_closed(sections):
    fields = metadata()
    if sections is None:
        del fields["qwen3vl.rope.dimension_sections"]
    else:
        fields["qwen3vl.rope.dimension_sections"] = sections
    with pytest.raises(ValueError, match="rope.dimension_sections"):
        core.build_config(Reader(fields), "qwen3vl", "bfloat16")


def test_metadata_namespace_alias_and_explicit_token_ids():
    fields = metadata("qwen2")
    fields["general.architecture"] = "qwen3_vl"
    fields["tokenizer.ggml.tokens"] = ["a"] * 6
    for key, value in {"image": 2, "video": 3, "vision_start": 4, "vision_end": 5}.items():
        fields[f"qwen2.{key}_token_id"] = value
    assert core.build_config(Reader(fields), "qwen3_vl", "float32")["image_token_id"] == 2
    del fields["qwen2.image_token_id"]
    with pytest.raises(ValueError, match="image_token_id"):
        core.build_config(Reader(fields), "qwen3_vl", "float32")


@pytest.mark.parametrize("name", ["visual.patch_embed.weight", "vision_tower.blk.0.weight",
                                 "model.visual.merger.weight", "merger.weight",
                                 "v.blk.0.attn_q.weight", "mm.0.weight"])
def test_visual_nodes_are_recognized(name):
    assert Qwen3VLTextEncoderConverter.is_visual_tensor(name)
    assert not Qwen3VLTextEncoderConverter.is_visual_tensor("blk.0.attn_v.weight")


def write_gguf(path, arch="qwen3vl", *, unknown=False, bf16=False):
    writer = GGUFWriter(path, arch)
    fields = metadata(arch)
    fields["GGUF.version"] = 3
    for key, value in fields.items():
        if key in {"general.architecture", "GGUF.version"}:
            continue
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_uint32(key, value)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "qwen2")
    array = np.full((2, 64), 99840.0, dtype=np.float32)
    if bf16:
        writer.add_tensor("blk.0.attn_q.weight", quantize(array, GGMLQuantizationType.BF16),
                          raw_dtype=GGMLQuantizationType.BF16)
    else:
        writer.add_tensor("blk.0.attn_q.weight", array)
    writer.add_tensor("visual.patch_embed.weight", np.ones((2, 2), dtype=np.float32))
    writer.add_tensor("merger.weight", np.ones((2, 2), dtype=np.float32))
    if unknown:
        writer.add_tensor("blk.0.unknown.weight", np.ones(2, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def read_payload(path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        payload = handle.read()
    return header, payload


@pytest.mark.parametrize("source_bf16", [False, True])
def test_portable_conversion_preserves_bf16_range_and_skips_vision(tmp_path, capsys, source_bf16):
    source, output = tmp_path / "source.gguf", tmp_path / "output"
    write_gguf(source, bf16=source_bf16)
    assert core.convert(str(source), str(output), model_type="text-encoder")
    config = json.loads((output / "config.json").read_text())
    assert config["torch_dtype"] == "bfloat16"
    assert config["rope_scaling"]["mrope_section"] == [24, 20, 20]
    assert config["image_token_id"] == 2
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert list(index["weight_map"]) == ["model.layers.0.self_attn.q_proj.weight"]
    assert index["metadata"]["total_size"] == 256
    shard = output / next(iter(index["weight_map"].values()))
    with safe_open(shard, framework="np") as handle:
        assert handle.get_slice("model.layers.0.self_attn.q_proj.weight").get_dtype() == "BF16"
    header, payload = read_payload(shard)
    assert header["model.layers.0.self_attn.q_proj.weight"]["shape"] == [2, 64]
    decoded = dequantize(np.frombuffer(payload, dtype=np.uint8), GGMLQuantizationType.BF16)
    np.testing.assert_array_equal(decoded, np.full(128, 99840, dtype=np.float32))
    assert "Skipping vision/projector tensor: visual.patch_embed.weight" in capsys.readouterr().out
    tokenizer = json.loads((output / "tokenizer.json").read_text())
    assert tokenizer["model"]["vocab"]["<|image_pad|>"] == 2
    assert tokenizer["normalizer"] is None
    loaded = Tokenizer.from_file(str(output / "tokenizer.json"))
    assert loaded.encode("<|vision_start|><|image_pad|><|vision_end|>").ids == [4, 2, 5]


@pytest.mark.parametrize("options", [
    {"dtype": "float16"}, {"quantize": True}, {"direct_quant": True},
    {"mmproj": "vision.gguf"}, {"mixed_precision": True},
])
def test_precision_and_source_options_fail_closed(tmp_path, options):
    source, output = tmp_path / "source.gguf", tmp_path / "output"
    write_gguf(source)
    assert not core.convert(str(source), str(output), model_type="text-encoder", **options)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))


def test_unknown_language_tensor_does_not_publish_partial_output(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "output"
    write_gguf(source, unknown=True)
    assert not core.convert(str(source), str(output), model_type="text-encoder")
    assert not output.exists()


def test_text_encoder_rejects_non_vl_architecture(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "output"
    write_gguf(source, arch="qwen2")
    assert not core.convert(str(source), str(output), model_type="text-encoder")
    assert not output.exists()


def test_explicit_direct_quantization_and_float32_output(tmp_path):
    source = tmp_path / "source.gguf"
    write_gguf(source)
    assert core.convert(str(source), str(tmp_path / "fp32"), "float32", model_type="text-encoder")
    assert core.convert(str(source), str(tmp_path / "quant"), model_type="text-encoder",
                        quantize=True, direct_quant=True)
    config = json.loads((tmp_path / "quant" / "config.json").read_text())
    assert config["quantization"]["bits"] == 4
    assert config["torch_dtype"] == "bfloat16"
    assert config["rope_scaling"]["mrope_interleaved"] is True
    index = json.loads((tmp_path / "quant" / "model.safetensors.index.json").read_text())
    shard = tmp_path / "quant" / next(iter(index["weight_map"].values()))
    header, payload = read_payload(shard)
    bias = header["model.layers.0.self_attn.q_proj.biases"]
    assert bias["dtype"] == "BF16"
    start, end = bias["data_offsets"]
    decoded = dequantize(np.frombuffer(payload[start:end], dtype=np.uint8),
                         GGMLQuantizationType.BF16)
    np.testing.assert_array_equal(decoded, [99840, 99840])


def test_bf16_streaming_rounding_and_shard_sizes(tmp_path):
    values = np.array([1.00390625, 1.01171875, -0.0, 99840, np.inf, np.nan], dtype=np.float32)
    with StreamingShardWriter(tmp_path, 12) as writer:
        writer.add("a", values, dtype="bfloat16")
        writer.add("b", values, dtype="bfloat16")
        writer.finish()
    assert writer.total == 24
    assert len(writer.files) == 2
    header, payload = read_payload(writer.files[0])
    assert header["a"]["dtype"] == "BF16"
    decoded = dequantize(np.frombuffer(payload, dtype=np.uint8), GGMLQuantizationType.BF16)
    np.testing.assert_array_equal(decoded, [1.0, 1.015625, -0.0, 99840, np.inf, np.nan])


def test_cli_accepts_text_encoder_and_bf16(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(core, "convert", lambda *args, **kwargs: calls.append((args, kwargs)) or True)
    core.main(["convert", "--input", str(tmp_path / "model.gguf"),
               "--type", "text-encoder", "--dtype", "bfloat16"])
    assert calls[0][0][2] == "bfloat16"
    assert calls[0][1]["model_type"] == "text-encoder"
