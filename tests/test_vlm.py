from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors.numpy import load_file, save_file

from gguf2mlx import vlm


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class Reader:
    def __init__(self, metadata, tensors=()):
        self.metadata = metadata
        self.tensors = list(tensors)

    def get_field(self, name):
        return Field(self.metadata[name]) if name in self.metadata else None


def tensor(name, array):
    return SimpleNamespace(
        name=name, data=np.asarray(array, dtype=np.float32),
        shape=np.array(array.shape[::-1]), tensor_type=0,
    )


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def tiny_config():
    return {
        "model_type": "llava",
        "image_token_index": 4,
        "text_config": {
            "model_type": "llama", "hidden_size": 4, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
            "vocab_size": 5, "bos_token_id": 1, "eos_token_id": 2,
        },
        "vision_config": {
            "model_type": "clip_vision_model", "hidden_size": 4, "intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 1,
            "image_size": 4, "patch_size": 2,
        },
    }


def gguf_name(name):
    roots = {
        "language_model.model.embed_tokens.weight": "token_embd.weight",
        "language_model.lm_head.weight": "output.weight",
        "language_model.model.norm.weight": "output_norm.weight",
    }
    if name in roots:
        return roots[name]
    if name.startswith("language_model.model.layers.0."):
        parts = {
            "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
            "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
            "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
        }
        fragment = name.removeprefix("language_model.model.layers.0.").removesuffix(".weight")
        return f"blk.0.{parts[fragment]}.weight"
    if name.startswith("multi_modal_projector."):
        return name.replace("multi_modal_projector.linear_1", "mm.0").replace(
            "multi_modal_projector.linear_2", "mm.2"
        )
    suffix = name.removeprefix(vlm.VISION_PREFIX)
    for src, dst in vlm._VISION_ROOT_MAP.items():
        if suffix == dst:
            return src
    parts = suffix.split(".")
    index = parts[2]
    fragment = ".".join(parts[3:-1])
    inverse = {value: key for key, value in vlm._VISION_BLOCK_MAP.items()}
    return f"v.blk.{index}.{inverse[fragment]}.{parts[-1]}"


@pytest.fixture
def model(tmp_path, monkeypatch):
    root = tmp_path / "hf"
    root.mkdir()
    config = tiny_config()
    write_json(root / "config.json", config)
    tokens = ["<unk>", "<s>", "</s>", "word", "<image>"]
    write_json(root / "tokenizer.json", {
        "model": {"type": "BPE", "vocab": {token: i for i, token in enumerate(tokens)}},
        "added_tokens": [{"id": 4, "content": "<image>", "special": True}],
    })
    write_json(root / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    write_json(root / "preprocessor_config.json", {"image_processor_type": "CLIPImageProcessor"})
    metadata = {
        "general.architecture": "llava", "llama.embedding_length": 4,
        "llama.feed_forward_length": 8, "llama.block_count": 1,
        "llama.attention.head_count": 1, "llama.attention.head_count_kv": 1,
        "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }
    arrays = {}
    tensors = []
    for i, (name, shape) in enumerate(
        vlm._expected_shapes(config["text_config"], config["vision_config"]).items()
    ):
        array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i * 100
        arrays[name] = array
        src = gguf_name(name)
        stored = array
        if src in ("blk.0.attn_q.weight", "blk.0.attn_k.weight"):
            stored = array.reshape(1, 2, 2, 4).swapaxes(1, 2).reshape(4, 4)
        tensors.append(tensor(src, stored))
    reader = Reader(metadata, tensors)
    readers = {"main.gguf": reader}
    monkeypatch.setattr(vlm, "GGUFReader", lambda name: readers[str(name)])
    return SimpleNamespace(
        root=root, config=config, arrays=arrays, reader=reader, readers=readers,
        output=tmp_path / "out",
    )


def convert(model, **kwargs):
    vlm.convert_vlm(
        "main.gguf", model.output, "float32",
        hf_model=str(model.root), offline=True, **kwargs,
    )


@pytest.mark.parametrize("architecture", sorted(vlm.VLM_ARCHITECTURES))
def test_detection_raw_architecture(architecture):
    assert vlm.is_vlm(Reader({"general.architecture": architecture}))


@pytest.mark.parametrize("name", ["v.patch_embd.weight", "mm.0.weight", "vision_tower.foo"])
def test_detection_vision_tensors(name):
    assert vlm.is_vlm(Reader({"general.architecture": "llama"}, [SimpleNamespace(name=name)]))


@pytest.mark.parametrize("key", ["clip.has_vision_encoder", "clip.has_llava_projector"])
def test_detection_presence_even_false(key):
    assert vlm.is_vlm(Reader({key: False}))
    assert not vlm.is_vlm(Reader({"general.architecture": "llama"}))


def test_direct_conversion_numeric_layout_and_assets(model):
    (model.root / "model.py").write_text("untrusted", encoding="utf-8")
    (model.root / "pytorch_model.bin").write_bytes(b"untrusted")
    convert(model)
    output = load_file(str(model.output / "model.safetensors"))
    assert set(output) == set(model.arrays)
    for name, expected in model.arrays.items():
        np.testing.assert_array_equal(output[name], expected)
    assert output[vlm.VISION_PREFIX + "embeddings.patch_embedding.weight"].shape == (4, 3, 2, 2)
    assert not (model.output / "model.py").exists()
    assert not (model.output / "pytorch_model.bin").exists()
    assert (model.output / "tokenizer.json").read_bytes() == (model.root / "tokenizer.json").read_bytes()
    report = json.loads((model.output / "vlm_conversion_report.json").read_text())
    assert report["mode"] == "direct"
    assert report["tensor_count"] == len(model.arrays)
    assert report["tensor_counts"]["vision_hf"] == 0
    assert str(model.root) not in json.dumps(report)
    index = json.loads((model.output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(output)
    assert set(index["weight_map"].values()) == {"model.safetensors"}


def test_split_llama_mmproj(model):
    original = model.reader.tensors
    model.reader.metadata["general.architecture"] = "llama"
    model.reader.tensors = [t for t in original if not t.name.startswith(("v.", "mm."))]
    model.readers["mmproj.gguf"] = Reader(
        {"general.architecture": "clip", "clip.projector_type": "mlp"},
        [t for t in original if t.name.startswith(("v.", "mm."))],
    )
    convert(model, mmproj="mmproj.gguf")
    assert len(load_file(str(model.output / "model.safetensors"))) == len(model.arrays)


def test_hybrid_replaces_only_vision_and_implies_hf_model(model):
    vision = {name: array + 7 for name, array in model.arrays.items()
              if name.startswith(vlm.VISION_PREFIX)}
    vision["language_model.model.embed_tokens.weight"] = np.zeros((2, 2), dtype=np.float32)
    save_file(vision, str(model.root / "model.safetensors"))
    vlm.convert_vlm(
        "main.gguf", model.output, "float16", hf_fallback_vision=str(model.root), offline=True
    )
    output = load_file(str(model.output / "model.safetensors"))
    for name, array in model.arrays.items():
        expected = array + 7 if name.startswith(vlm.VISION_PREFIX) else array
        np.testing.assert_array_equal(output[name], expected.astype(np.float16))
    report = json.loads((model.output / "vlm_conversion_report.json").read_text())
    assert report["mode"] == "hybrid"
    assert report["tensor_counts"]["vision_gguf_ignored"] == len(vision) - 1


def test_hybrid_accepts_truncated_gguf_vision(model):
    model.reader.tensors = [t for t in model.reader.tensors if not t.name.startswith("v.")]
    save_file(
        {name: array for name, array in model.arrays.items() if name.startswith(vlm.VISION_PREFIX)},
        str(model.root / "model.safetensors"),
    )
    convert(model, hf_fallback_vision=str(model.root))
    assert (model.output / "model.safetensors").is_file()


def test_bounded_shards_and_index(model, monkeypatch):
    monkeypatch.setattr(vlm, "SHARD_BYTES", 100)
    convert(model)
    index = json.loads((model.output / "model.safetensors.index.json").read_text())
    output = {}
    for file in set(index["weight_map"].values()):
        output.update(load_file(str(model.output / file)))
    assert set(output) == set(model.arrays)
    assert index["metadata"]["total_size"] == sum(a.nbytes for a in model.arrays.values())


def test_actual_gguf_reader_preserves_row_major_and_patch(tmp_path):
    path = tmp_path / "layout.gguf"
    arrays = {
        "matrix": np.arange(12, dtype=np.float32).reshape(3, 4),
        "patch": np.arange(48, dtype=np.float32).reshape(4, 3, 2, 2),
    }
    writer = GGUFWriter(str(path), "clip")
    for name, array in arrays.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for current in GGUFReader(str(path)).tensors:
        np.testing.assert_array_equal(vlm._decode(current, "float32"), arrays[current.name])


@pytest.mark.parametrize("source_dtype", [np.float16, np.float32])
def test_actual_combined_gguf_end_to_end_numeric_fixture(model, monkeypatch, source_dtype):
    path = model.root.parent / "combined.gguf"
    writer = GGUFWriter(str(path), "llava")
    for name, value in model.reader.metadata.items():
        if name == "general.architecture":
            continue
        if isinstance(value, list):
            writer.add_array(name, value)
        else:
            writer.add_uint32(name, value)
    for current in model.reader.tensors:
        writer.add_tensor(current.name, current.data.astype(source_dtype))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    monkeypatch.setattr(vlm, "GGUFReader", GGUFReader)
    vlm.convert_vlm(
        str(path), model.output, "float32", hf_model=str(model.root), offline=True,
    )
    output = load_file(str(model.output / "model.safetensors"))
    for name, expected in model.arrays.items():
        np.testing.assert_array_equal(output[name], expected.astype(source_dtype).astype(np.float32))


@pytest.mark.parametrize("heads,kv", [(2, 1), (2, 2), (4, 2)])
def test_inverse_rope_permutation(heads, kv):
    text = {"num_attention_heads": heads, "num_key_value_heads": kv}
    for fragment, count in [("q", heads), ("k", kv)]:
        original = np.arange(count * 4 * 8, dtype=np.float32).reshape(count * 4, 8)
        exported = original.reshape(count, 2, 2, 8).swapaxes(1, 2).reshape(count * 4, 8)
        np.testing.assert_array_equal(
            vlm._restore_llama(f"blk.0.attn_{fragment}.weight", exported, text), original
        )


def test_quantized_decode_retains_canonical_layout():
    array = np.arange(64, dtype=np.float32).reshape(2, 32) / 10
    packed = quantize(array, GGMLQuantizationType.Q4_0)
    current = SimpleNamespace(
        name="matrix", shape=np.array([32, 2]),
        data=packed, tensor_type=GGMLQuantizationType.Q4_0,
    )
    np.testing.assert_array_equal(
        vlm._decode(current, "float32"), dequantize(packed, GGMLQuantizationType.Q4_0)
    )


@pytest.mark.parametrize("name", ["mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"])
def test_projector_mapping(name):
    assert vlm._map_projector(name).startswith("multi_modal_projector.linear_")


@pytest.mark.parametrize("source", ["qwen2vl", "llava_next", "qwen2"])
def test_hf_config_cannot_mask_unsupported_gguf_architecture(model, source):
    model.reader.metadata["general.architecture"] = source
    with pytest.raises(ValueError, match="Unsupported GGUF architecture"):
        convert(model)


@pytest.mark.parametrize("field,value,message", [
    ("model_type", "llava_next", "Only HF"),
    ("vision_config", {"model_type": "siglip_vision_model"}, "CLIP"),
    ("text_config", {"model_type": "qwen2"}, "llama"),
    ("mm_projector_type", "linear", "two-linear"),
    ("vision_feature_select_strategy", "full", "patch feature"),
    ("architectures", ["LlavaNextForConditionalGeneration"], "Unsupported HF LLaVA"),
    ("image_grid_pinpoints", [[224, 224]], "tiled"),
    ("vision_aspect_ratio", "anyres", "tiled"),
])
def test_unsupported_hf_configuration(model, field, value, message):
    config = copy.deepcopy(model.config)
    config[field] = value
    write_json(model.root / "config.json", config)
    with pytest.raises(ValueError, match=message):
        convert(model)


def test_requires_same_hf_source(model):
    with pytest.raises(ValueError, match="same HF source"):
        convert(model, hf_fallback_vision="different/repo")


def test_requires_original_assets(model):
    with pytest.raises(ValueError, match="requires --hf-model"):
        vlm.convert_vlm("main.gguf", model.output, "float32")
    (model.root / "preprocessor_config.json").unlink()
    with pytest.raises(ValueError, match="preprocessor_config.json"):
        convert(model)


def test_requires_chat_template(model):
    write_json(model.root / "tokenizer_config.json", {})
    with pytest.raises(ValueError, match="chat template"):
        convert(model)


def test_separate_chat_template_asset(model):
    write_json(model.root / "tokenizer_config.json", {})
    (model.root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    convert(model)
    assert (model.output / "chat_template.jinja").is_file()


@pytest.mark.parametrize("kind", ["tokens", "bos", "image"])
def test_tokenizer_mismatch(model, kind):
    if kind == "tokens":
        model.reader.metadata["tokenizer.ggml.tokens"][3] = "different"
    elif kind == "bos":
        model.reader.metadata["tokenizer.ggml.bos_token_id"] = 3
    else:
        model.config["image_token_index"] = 3
        write_json(model.root / "config.json", model.config)
    with pytest.raises(ValueError, match="mismatch|IDs differ|<image>"):
        convert(model)


@pytest.mark.parametrize("key", [
    "llama.embedding_length", "llama.block_count", "llama.attention.head_count",
    "llama.attention.head_count_kv", "llama.feed_forward_length",
])
def test_text_metadata_mismatch(model, key):
    model.reader.metadata[key] += 1
    with pytest.raises(ValueError, match="incompatible"):
        convert(model)


def test_llava_adapted_text_metadata(model):
    model.reader.metadata = {
        name.replace("llama.", "llava."): value for name, value in model.reader.metadata.items()
    }
    convert(model)
    assert (model.output / "model.safetensors").is_file()


@pytest.mark.parametrize("key,value,message", [
    ("llama.rope.freq_base", 500000.0, "rope_theta"),
    ("llama.rope.dimension_count", 2, "Partial rotary"),
    ("llama.rope.scaling.type", "linear", "incompatible"),
    ("llama.attention.layer_norm_rms_epsilon", 1e-5, "rms_norm_eps"),
    ("clip.vision.patch_size", 14, "patch_size"),
    ("clip.vision.embedding_length", 1024, "hidden_size"),
    ("clip.vision.block_count", 24, "num_hidden_layers"),
    ("clip.use_gelu", True, "hidden_act"),
    ("clip.use_silu", True, "SiLU"),
    ("clip.has_siglip", True, "SigLIP"),
    ("clip.projector_type", "ldp", "projector type"),
])
def test_configuration_semantics_mismatch(model, key, value, message):
    model.reader.metadata[key] = value
    with pytest.raises(ValueError, match=message):
        convert(model)


def test_hf_extended_rope_rejected(model):
    model.config["text_config"]["rope_scaling"] = {"type": "yarn", "factor": 2}
    write_json(model.root / "config.json", model.config)
    with pytest.raises(ValueError, match="Only linear"):
        convert(model)


def test_linear_rope_matches_gguf_and_normalizes_hf_key(model):
    model.config["text_config"]["rope_scaling"] = {"rope_type": "linear", "factor": 2}
    write_json(model.root / "config.json", model.config)
    model.reader.metadata["llama.rope.scaling.type"] = "linear"
    model.reader.metadata["llama.rope.scaling.factor"] = 2.0
    convert(model)
    config = json.loads((model.output / "config.json").read_text())
    assert config["text_config"]["rope_scaling"] == {"type": "linear", "factor": 2.0}


@pytest.mark.parametrize("factor", [None, 0, False, float("inf")])
def test_invalid_linear_rope_factor(model, factor):
    model.config["text_config"]["rope_scaling"] = {"type": "linear", "factor": factor}
    write_json(model.root / "config.json", model.config)
    with pytest.raises(ValueError, match="finite factor"):
        convert(model)


def test_linear_rope_factor_must_match_gguf(model):
    model.config["text_config"]["rope_scaling"] = {"type": "linear", "factor": 2}
    write_json(model.root / "config.json", model.config)
    model.reader.metadata["llama.rope.scaling.type"] = "linear"
    model.reader.metadata["llama.rope.scaling.factor"] = 4.0
    with pytest.raises(ValueError, match="scaling factor"):
        convert(model)


def test_modern_hf_rounded_vocab_retains_gguf_embedding_rows(model):
    model.config["text_config"]["vocab_size"] = 8
    write_json(model.root / "config.json", model.config)
    model.reader.metadata["tokenizer.ggml.tokens"].extend(
        ["[PAD5]", "[PAD6]", "[PAD7]"]
    )
    for name in ("token_embd.weight", "output.weight"):
        current = next(t for t in model.reader.tensors if t.name == name)
        current.data = np.arange(32, dtype=np.float32).reshape(8, 4)
        current.shape = np.array([4, 8])
    convert(model)
    output = load_file(str(model.output / "model.safetensors"))
    assert output["language_model.model.embed_tokens.weight"].shape == (8, 4)
    np.testing.assert_array_equal(
        output["language_model.model.embed_tokens.weight"],
        np.arange(32, dtype=np.float32).reshape(8, 4),
    )
    assert json.loads((model.output / "config.json").read_text())["text_config"]["vocab_size"] == 8


def test_hf_vocab_never_silently_pads_actual_gguf_embeddings(model):
    model.config["text_config"]["vocab_size"] = 8
    write_json(model.root / "config.json", model.config)
    model.reader.metadata["tokenizer.ggml.tokens"].extend(
        ["[PAD5]", "[PAD6]", "[PAD7]"]
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        convert(model)


def test_unrecognized_unused_tokenizer_rows_rejected(model):
    model.config["text_config"]["vocab_size"] = 8
    write_json(model.root / "config.json", model.config)
    model.reader.metadata["tokenizer.ggml.tokens"].extend(
        ["some-other-token", "[PAD6]", "[PAD7]"]
    )
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        convert(model)


def test_hf_traditional_rope_rejected(model):
    model.config["text_config"]["rope_traditional"] = True
    write_json(model.root / "config.json", model.config)
    with pytest.raises(ValueError, match="nontraditional"):
        convert(model)


def test_image_token_outside_actual_embeddings_rejected(model):
    model.config["image_token_index"] = 5
    write_json(model.root / "config.json", model.config)
    tokenizer = json.loads((model.root / "tokenizer.json").read_text())
    del tokenizer["model"]["vocab"]["<image>"]
    tokenizer["model"]["vocab"]["<pad>"] = 4
    tokenizer["added_tokens"][0]["id"] = 5
    write_json(model.root / "tokenizer.json", tokenizer)
    model.reader.metadata["tokenizer.ggml.tokens"][4] = "<pad>"
    with pytest.raises(ValueError, match="compatibly resized text embeddings"):
        convert(model)


@pytest.mark.parametrize("token", ["<pad>", "<im_start>", "extra-word"])
def test_any_hf_tokenizer_id_outside_actual_embeddings_rejected(model, token):
    tokenizer = json.loads((model.root / "tokenizer.json").read_text())
    tokenizer["added_tokens"].append({
        "id": model.config["text_config"]["vocab_size"], "content": token, "special": True,
    })
    write_json(model.root / "tokenizer.json", tokenizer)
    with pytest.raises(ValueError, match="HF tokenizer IDs exceed"):
        convert(model)


def test_valid_hf_added_pad_token_inside_embeddings(model):
    tokenizer = json.loads((model.root / "tokenizer.json").read_text())
    del tokenizer["model"]["vocab"]["word"]
    tokenizer["added_tokens"].append({"id": 3, "content": "<pad>", "special": True})
    write_json(model.root / "tokenizer.json", tokenizer)
    model.reader.metadata["tokenizer.ggml.tokens"][3] = "<pad>"
    convert(model)
    output = load_file(str(model.output / "model.safetensors"))
    assert np.isfinite(output["language_model.model.embed_tokens.weight"]).all()


def test_missing_vision_fails_with_hybrid_suggestion(model):
    model.reader.tensors = [t for t in model.reader.tensors if t.name != "v.post_ln.weight"]
    with pytest.raises(ValueError, match="hf-fallback-vision"):
        convert(model)


def test_missing_language_layer_fails_closed(model):
    model.reader.tensors = [t for t in model.reader.tensors if t.name != "blk.0.ffn_gate.weight"]
    with pytest.raises(ValueError, match="Missing required"):
        convert(model)


def test_tied_language_head(model):
    model.config["text_config"]["tie_word_embeddings"] = True
    write_json(model.root / "config.json", model.config)
    model.reader.tensors = [t for t in model.reader.tensors if t.name != "output.weight"]
    convert(model)
    output = load_file(str(model.output / "model.safetensors"))
    assert "language_model.lm_head.weight" not in output
    assert "language_model.model.embed_tokens.weight" in output


def test_tied_language_head_source_is_validated_and_not_emitted(model):
    model.config["text_config"]["tie_word_embeddings"] = True
    write_json(model.root / "config.json", model.config)
    embedding = next(t for t in model.reader.tensors if t.name == "token_embd.weight")
    head = next(t for t in model.reader.tensors if t.name == "output.weight")
    head.data = embedding.data.copy()
    convert(model)
    output = load_file(str(model.output / "model.safetensors"))
    assert "language_model.lm_head.weight" not in output


def test_distinct_gguf_head_conflicts_with_hf_tied_config(model):
    model.config["text_config"]["tie_word_embeddings"] = True
    write_json(model.root / "config.json", model.config)
    with pytest.raises(ValueError, match="tie_word_embeddings=True conflicts"):
        convert(model)


def test_tied_source_heads_compared_before_fp16_rounding(model):
    model.config["text_config"]["tie_word_embeddings"] = True
    write_json(model.root / "config.json", model.config)
    embedding = next(t for t in model.reader.tensors if t.name == "token_embd.weight")
    head = next(t for t in model.reader.tensors if t.name == "output.weight")
    head.data = embedding.data.copy()
    head.data.flat[1] += 1e-5
    with pytest.raises(ValueError, match="tie_word_embeddings=True conflicts"):
        vlm.convert_vlm(
            "main.gguf", model.output, "float16", hf_model=str(model.root), offline=True,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gguf_weights_rejected(model, value):
    model.reader.tensors[0].data.flat[0] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        convert(model)


def test_fp16_overflow_rejected(model):
    model.reader.tensors[0].data.flat[0] = 1e10
    with pytest.raises(ValueError, match="overflows float16"):
        vlm.convert_vlm(
            "main.gguf", model.output, "float16", hf_model=str(model.root), offline=True,
        )


def test_nonfinite_hf_vision_rejected(model):
    vision = {name: array.copy() for name, array in model.arrays.items()
              if name.startswith(vlm.VISION_PREFIX)}
    vision[next(iter(vision))].flat[0] = float("nan")
    save_file(vision, str(model.root / "model.safetensors"))
    with pytest.raises(ValueError, match="Nonfinite"):
        convert(model, hf_fallback_vision=str(model.root))


@pytest.mark.parametrize("name", ["v.blk.0.attn_q.weight", "mm.0.weight", "token_embd.weight"])
def test_shape_mismatch(model, name):
    target = next(t for t in model.reader.tensors if t.name == name)
    target.shape = np.array([1, target.data.size])
    with pytest.raises(ValueError, match="shape mismatch"):
        convert(model)


@pytest.mark.parametrize("name", [
    "v.blk.0.attn_qkv.weight", "v.blk.99.attn_q.weight", "mm.1.weight",
    "vision_tower.unknown", "blk.0.unknown.weight",
])
def test_unknown_weights_fail_closed(model, name):
    model.reader.tensors.append(tensor(name, np.zeros((4, 4), dtype=np.float32)))
    with pytest.raises(ValueError, match="Unsupported|Unexpected"):
        convert(model)


def test_hybrid_unknown_vision_layout_still_fails_closed(model):
    model.reader.tensors.append(tensor("v.unknown.weight", np.zeros(4, dtype=np.float32)))
    with pytest.raises(ValueError, match="Unsupported CLIP"):
        convert(model, hf_fallback_vision=str(model.root))


def test_tensor_collision(model):
    model.reader.tensors.append(model.reader.tensors[0])
    with pytest.raises(ValueError, match="Duplicate"):
        convert(model)


def test_unsupported_decode_type(model):
    model.reader.tensors[0].tensor_type = 99999
    with pytest.raises(ValueError, match="Cannot decode"):
        convert(model)


@pytest.mark.parametrize("unsafe", ["../vision.safetensors", "/vision.safetensors",
                                  "nested/../../x.safetensors", "x.bin", r"..\x.safetensors",
                                  "*.safetensors", "vision?.safetensors", "a\x00.safetensors"])
def test_unsafe_index_path(model, unsafe):
    write_json(model.root / "model.safetensors.index.json", {
        "weight_map": {"language_model.any.weight": unsafe,
                       vlm.VISION_PREFIX + "embeddings.class_embedding": "vision.safetensors"},
    })
    with pytest.raises(ValueError, match="Unsafe"):
        convert(model, hf_fallback_vision=str(model.root))


def test_local_shard_symlink_escape(model):
    other = model.root.parent / "outside.safetensors"
    save_file({name: array for name, array in model.arrays.items()
               if name.startswith(vlm.VISION_PREFIX)}, str(other))
    (model.root / "model.safetensors").symlink_to(other)
    with pytest.raises(ValueError, match="Unsafe HF shard symlink"):
        convert(model, hf_fallback_vision=str(model.root))


def test_hybrid_index_missing_tensor(model):
    vision = {name: array for name, array in model.arrays.items()
              if name.startswith(vlm.VISION_PREFIX)}
    write_json(model.root / "model.safetensors.index.json", {
        "weight_map": {name: "vision.safetensors" for name in vision},
    })
    del vision[next(iter(vision))]
    save_file(vision, str(model.root / "vision.safetensors"))
    with pytest.raises(ValueError, match="references missing"):
        convert(model, hf_fallback_vision=str(model.root))


def test_remote_download_targets_only_indexed_vision_shards(model, monkeypatch):
    vision = {name: array for name, array in model.arrays.items()
              if name.startswith(vlm.VISION_PREFIX)}
    save_file(vision, str(model.root / "vision-00001.safetensors"))
    write_json(model.root / "model.safetensors.index.json", {
        "weight_map": dict.fromkeys(vision, "vision-00001.safetensors") | {
            "language_model.model.embed_tokens.weight": "language-00001.safetensors",
        },
    })
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(model.root)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=snapshot_download,
    ))
    vlm.convert_vlm(
        "main.gguf", model.output, "float32",
        hf_fallback_vision="owner/model", hf_revision="v1", offline=True,
    )
    assert calls[0]["allow_patterns"] == list(vlm.ASSETS) + ["model.safetensors.index.json"]
    assert calls[1]["allow_patterns"] == ["vision-00001.safetensors"]
    assert all(call["revision"] == "v1" and call["local_files_only"] for call in calls)
    assert not any("language-00001.safetensors" in call["allow_patterns"] for call in calls)


def test_local_sources_do_not_import_hub(model, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    convert(model)
    assert (model.output / "model.safetensors").is_file()


def test_hybrid_bfloat16_vision_weights(model):
    header = {}
    payload = bytearray()
    expected = {}
    for name, array in model.arrays.items():
        if name.startswith(vlm.VISION_PREFIX):
            bits = (array.view(np.uint32) >> 16).astype(np.uint16)
            start = len(payload)
            payload.extend(bits.tobytes())
            header[name] = {"dtype": "BF16", "shape": array.shape,
                            "data_offsets": [start, len(payload)]}
            expected[name] = (bits.astype(np.uint32) << 16).view(np.float32)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (model.root / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + payload
    )
    convert(model, hf_fallback_vision=str(model.root))
    output = load_file(str(model.output / "model.safetensors"))
    for name, array in expected.items():
        np.testing.assert_array_equal(output[name], array)


def test_hybrid_rejects_quantized_hf_vision_weights(model):
    vision = {name: array.astype(np.int16) for name, array in model.arrays.items()
              if name.startswith(vlm.VISION_PREFIX)}
    save_file(vision, str(model.root / "model.safetensors"))
    with pytest.raises(ValueError, match="unsupported dtype"):
        convert(model, hf_fallback_vision=str(model.root))
