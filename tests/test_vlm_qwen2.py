"""Real GGUF fixtures with independently exported split QKV/temporal kernels."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors.numpy import load_file, save_file

from gguf2mlx import gguf2mlx as core
from gguf2mlx import vlm
from gguf2mlx.vision_qwen2 import Qwen2VisionAdapter
from tests.test_vlm import Reader, gguf_name, tiny_config


def qwen_config():
    return {
        "model_type": "qwen2_vl", "architectures": ["Qwen2VLForConditionalGeneration"],
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 16,
        "bos_token_id": 1, "eos_token_id": 2, "image_token_id": 4,
        "video_token_id": 5, "vision_start_token_id": 6, "vision_end_token_id": 7,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        "vision_config": {
            "depth": 1, "embed_dim": 16, "hidden_size": 64, "num_heads": 2,
            "patch_size": 4, "spatial_patch_size": 4, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "mlp_ratio": 2, "in_chans": 3,
        },
    }


def export_qwen_vision(name, array):
    name = name.removeprefix("vision_tower.")
    if name == "patch_embed.proj.weight":
        return {"v.patch_embd.weight": array[:, :, 0], "v.patch_embd.weight.1": array[:, :, 1]}
    if name.startswith("merger.ln_q."):
        return {name.replace("merger.ln_q", "v.post_ln"): array}
    if name.startswith("merger.mlp."):
        return {name.replace("merger.mlp", "mm"): array}
    name = name.replace("blocks.0.", "v.blk.0.")
    if ".attn.qkv." in name:
        return {name.replace("attn.qkv", "attn_" + part): value
                for part, value in zip(("q", "k", "v"), np.split(array, 3, axis=0))}
    for hf, gguf in {
        "attn.proj": "attn_out", "norm1": "ln1", "norm2": "ln2",
        "mlp.fc1": "ffn_up", "mlp.fc2": "ffn_down",
    }.items():
        if hf in name:
            return {name.replace(hf, gguf): array}
    raise AssertionError(name)


def write_fixture(
    tmp_path, architecture="qwen2_vl", *, quantized_source=False, legacy=False,
    k_quantized_source=False,
):
    root = tmp_path / "hf"
    root.mkdir()
    if architecture == "qwen2_vl":
        config = qwen_config()
        if k_quantized_source:
            config.update(hidden_size=256, intermediate_size=512)
            config["rope_scaling"]["mrope_section"] = [8, 12, 12]
            config["vision_config"]["hidden_size"] = 256
        adapter = Qwen2VisionAdapter(config)
        text, shapes, gguf_arch = adapter.text, adapter.expected, "qwen2vl"
        tokens = ["<unk>", "<s>", "</s>", "word", "<|image_pad|>", "<|video_pad|>",
                  "<|vision_start|>", "<|vision_end|>"]
        processor = {
            "image_processor_type": "Qwen2VLImageProcessor",
            "processor_class": "Qwen2VLProcessor", "patch_size": 4,
            "temporal_patch_size": 2, "merge_size": 2, "min_pixels": 64, "max_pixels": 64,
        }
    else:
        config = tiny_config()
        config["text_config"].update(hidden_size=64, intermediate_size=128,
                                     num_attention_heads=4, num_key_value_heads=2, vocab_size=16)
        config["vision_config"].update(hidden_size=16, intermediate_size=32,
                                       image_size=8, patch_size=4, num_attention_heads=2)
        text, gguf_arch = config["text_config"], "llama"
        shapes = vlm._expected_shapes(text, config["vision_config"])
        tokens = ["<unk>", "<s>", "</s>", "word", "<image>", "USER", "ASSISTANT", ":"]
        processor = {
            "image_processor_type": "CLIPImageProcessor", "processor_class": "LlavaProcessor",
            "size": {"shortest_edge": 8}, "crop_size": {"height": 8, "width": 8},
        }
    tokens += [f"token{i}" for i in range(len(tokens), 16)]
    processor.update(image_mean=[0.5] * 3, image_std=[0.5] * 3)
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(processor))
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {token: i for i, token in enumerate(tokens)}},
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    rng = np.random.default_rng(42)
    arrays = {}
    source_arrays = {}
    source_targets = {}
    for name, shape in shapes.items():
        array = rng.normal(0, 0.08, shape).astype(np.float32)
        if ("norm" in name or "ln_q" in name) and name.endswith(".weight"):
            array += 1
        arrays[name] = array
        if architecture == "qwen2_vl" and name.startswith("vision_tower."):
            source_arrays.update(export_qwen_vision(name, array))
        else:
            suffix = ".bias" if name.endswith(".bias") else ".weight"
            src = gguf_name(name.removesuffix(suffix) + ".weight") if (
                name.startswith("language_model.") and suffix == ".bias"
            ) else gguf_name(name)
            if name.startswith("language_model.") and suffix == ".bias":
                src = src.removesuffix(".weight") + ".bias"
            if architecture == "llava" and src in ("blk.0.attn_q.weight", "blk.0.attn_k.weight"):
                heads = text["num_attention_heads"] if ".attn_q." in src else text["num_key_value_heads"]
                array = array.reshape(heads, 2, -1, array.shape[1]).swapaxes(1, 2).reshape(array.shape)
            source_arrays[src] = array
            source_targets[src] = name
    if legacy:
        assert architecture == "qwen2_vl"
        source_arrays = {
            (name.replace(".ffn_up.", ".ffn_down.") if ".ffn_up." in name
             else name.replace(".ffn_down.", ".ffn_up."))
            if name.startswith("v.") else name: array
            for name, array in source_arrays.items()
        }
        source_arrays["v.position_embd.weight"] = np.zeros((10, 10), dtype=np.float32)
    main, mmproj = tmp_path / "language.gguf", tmp_path / "mmproj.gguf"
    for path, arch in ((main, gguf_arch), (mmproj, "clip")):
        writer = GGUFWriter(str(path), arch)
        if path == main:
            for src, dst in {
                "embedding_length": "hidden_size", "feed_forward_length": "intermediate_size",
                "block_count": "num_hidden_layers", "attention.head_count": "num_attention_heads",
                "attention.head_count_kv": "num_key_value_heads",
            }.items():
                writer.add_uint32(f"{gguf_arch}.{src}", text[dst])
            writer.add_token_list(tokens)
            writer.add_bos_token_id(1)
            writer.add_eos_token_id(2)
            if architecture == "qwen2_vl":
                writer.add_array("qwen2vl.rope.dimension_sections",
                                 config["rope_scaling"]["mrope_section"] + [0])
        else:
            projector = "qwen2vl_merger" if legacy else (
                "qwen2vl" if architecture == "qwen2_vl" else "mlp"
            )
            writer.add_string("clip.projector_type", projector)
            if legacy:
                writer.add_uint32("clip.vision.feed_forward_length", 0)
                writer.add_uint32("clip.vision.projection_dim", text["hidden_size"])
            writer.add_array("clip.vision.image_mean", processor["image_mean"])
            writer.add_array("clip.vision.image_std", processor["image_std"])
        for name in reversed(list(source_arrays)):
            is_vision = name.startswith(("v.", "mm."))
            if is_vision == (path == mmproj):
                value = np.ascontiguousarray(source_arrays[name])
                if k_quantized_source and path == main and value.ndim == 2:
                    qtype = (GGMLQuantizationType.Q6_K if name in (
                        "blk.0.ffn_down.weight", "token_embd.weight",
                    ) else GGMLQuantizationType.Q4_K)
                    packed = k_quant_blocks(value.shape, qtype, rng)
                    writer.add_tensor(name, packed, raw_dtype=qtype)
                    arrays[source_targets[name]] = dequantize(packed, qtype)
                elif quantized_source and path == main and value.ndim == 2:
                    packed = quantize(value, GGMLQuantizationType.Q4_0)
                    writer.add_tensor(name, packed, raw_dtype=GGMLQuantizationType.Q4_0)
                    decoded = dequantize(packed, GGMLQuantizationType.Q4_0)
                    if architecture == "llava":
                        decoded = vlm._restore_llama(name, decoded, text)
                    arrays[source_targets[name]] = decoded
                else:
                    writer.add_tensor(name, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    save_file(arrays, str(root / "model.safetensors"))
    return SimpleNamespace(root=root, main=main, mmproj=mmproj, config=config,
                           arrays=arrays, tokens=tokens, output=tmp_path / "out")


def k_quant_blocks(shape, qtype, rng):
    block_bytes = 144 if qtype == GGMLQuantizationType.Q4_K else 210
    data = rng.integers(0, 256, (int(np.prod(shape)) // 256, block_bytes), dtype=np.uint8)
    if qtype == GGMLQuantizationType.Q4_K:
        data[:, :2] = np.array([0.0002], dtype="<f2").view(np.uint8)
        data[:, 2:4] = np.array([0.0003], dtype="<f2").view(np.uint8)
    else:
        data[:, -2:] = np.array([0.00002], dtype="<f2").view(np.uint8)
    return data.reshape(shape[0], shape[1] // 256 * block_bytes)


@pytest.fixture
def qwen(tmp_path):
    return write_fixture(tmp_path)


@pytest.mark.parametrize("hybrid", [False, True])
def test_real_split_qwen_roundtrip(qwen, hybrid):
    assert core.convert(str(qwen.main), str(qwen.output), "float32",
                        mmproj=str(qwen.mmproj), offline=True,
                        hf_model=None if hybrid else str(qwen.root),
                        hf_fallback_vision=str(qwen.root) if hybrid else None)
    actual = load_file(str(qwen.output / "model.safetensors"))
    assert set(actual) == set(qwen.arrays)
    for name, expected in qwen.arrays.items():
        np.testing.assert_array_equal(actual[name], expected)
    config = json.loads((qwen.output / "config.json").read_text())
    assert config["model_type"] == "qwen2_vl"
    assert config["hidden_size"] == 64
    assert config["vision_config"]["embed_dim"] == 16
    assert config["rope_scaling"] == {"type": "mrope", "mrope_section": [2, 3, 3]}


@pytest.mark.parametrize("architecture", ["llava", "qwen2_vl"])
@pytest.mark.parametrize("quantized_source", [False, True])
def test_language_only_direct_quantization(tmp_path, architecture, quantized_source):
    fixture = write_fixture(tmp_path, architecture, quantized_source=quantized_source)
    assert core.convert(
        str(fixture.main), str(fixture.output), "float32", quantize=True, direct_quant=True,
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )
    actual = load_file(str(fixture.output / "model.safetensors"))
    quantized = 0
    for name, expected in fixture.arrays.items():
        if name.startswith("language_model.") and name.endswith("_proj.weight"):
            packed = actual[name]
            stem = name.removesuffix(".weight")
            codes = ((packed[..., None] >> (np.arange(8) * 4)) & 15).reshape(expected.shape)
            scales, biases = actual[stem + ".scales"], actual[stem + ".biases"]
            restored = codes * scales.repeat(64, axis=-1) + biases.repeat(64, axis=-1)
            assert packed.dtype == np.uint32
            assert packed.shape == (expected.shape[0], expected.shape[1] // 8)
            error = np.abs(restored - expected)
            assert np.all(error <= scales.repeat(64, axis=-1) * 0.51 + 0.001)
            quantized += 1
        else:
            np.testing.assert_array_equal(actual[name], expected)
    assert quantized == 7
    config = json.loads((fixture.output / "config.json").read_text())
    assert config["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert config["vision_config"]["skip_vision"] is True
    report = json.loads((fixture.output / "vlm_conversion_report.json").read_text())
    assert len(report["quantization"]["language_modules"]) == quantized


@pytest.mark.parametrize("field,value", [
    ("model_type", "qwen3_5"), ("use_sliding_window", True), ("attention_bias", False),
    ("head_dim", 32), ("rope_scaling", {"type": "mrope", "mrope_section": [1, 1, 1]}),
])
def test_qwen_config_rejects_incompatible_text(field, value):
    config = qwen_config()
    config[field] = value
    with pytest.raises(ValueError):
        Qwen2VisionAdapter(config)


@pytest.mark.parametrize("field,value", [
    ("model_type", "qwen2_5_vl"), ("temporal_patch_size", 1), ("spatial_merge_size", 3),
    ("in_chans", 1), ("hidden_act", "silu"), ("hidden_size", 128),
    ("mlp_ratio", float("nan")), ("embed_dim", 15), ("layer_norm_eps", 1e-5),
])
def test_qwen_config_rejects_incompatible_vision(field, value):
    config = qwen_config()
    config["vision_config"][field] = value
    with pytest.raises(ValueError):
        Qwen2VisionAdapter(config)


def test_nested_qwen_config_normalized():
    flat = qwen_config()
    nested = {
        "model_type": "qwen2_vl", "vision_config": flat["vision_config"],
        "text_config": {k: v for k, v in flat.items() if k != "vision_config"},
    }
    nested["text_config"]["model_type"] = "qwen2_vl_text"
    assert Qwen2VisionAdapter(nested).expected == Qwen2VisionAdapter(flat).expected


def test_qwen_plan_has_independent_qkv_and_temporal_shapes():
    adapter = Qwen2VisionAdapter(qwen_config())
    assert adapter.source_shapes["v.patch_embd.weight.1"] == (16, 3, 4, 4)
    assert adapter.expected["vision_tower.patch_embed.proj.weight"] == (16, 3, 2, 4, 4)
    assert adapter.expected["vision_tower.blocks.0.attn.qkv.weight"] == (48, 16)
    assert adapter.expected["vision_tower.merger.mlp.0.weight"] == (64, 64)


@pytest.mark.parametrize("key,value", [
    ("image_mean", [0, 0, 0]), ("image_std", [0, 0, 0]), ("merge_size", 3),
    ("image_processor_type", "CLIPImageProcessor"), ("temporal_patch_size", 1),
])
def test_processor_mismatch_fails_atomically(qwen, key, value):
    path = qwen.root / "preprocessor_config.json"
    processor = json.loads(path.read_text())
    processor[key] = value
    path.write_text(json.dumps(processor))
    assert not core.convert(str(qwen.main), str(qwen.output), mmproj=str(qwen.mmproj),
                            hf_model=str(qwen.root), offline=True)
    assert not qwen.output.exists()
    assert not list(qwen.output.parent.glob(".out.*"))


def test_qwen_hybrid_never_replaces_merger(qwen):
    changed = copy.deepcopy(qwen.arrays)
    for name in changed:
        if ".merger." in name:
            changed[name] += 10
    save_file(changed, str(qwen.root / "model.safetensors"))
    vlm.convert_vlm(str(qwen.main), qwen.output, "float32", mmproj=str(qwen.mmproj),
                    hf_fallback_vision=str(qwen.root), offline=True)
    actual = load_file(str(qwen.output / "model.safetensors"))
    for name in qwen.arrays:
        if ".merger." in name:
            np.testing.assert_array_equal(actual[name], qwen.arrays[name])


def test_qwen_hybrid_index_selects_encoder_not_merger_or_language(qwen):
    vision = {
        name.replace("vision_tower.", "visual."): array
        for name, array in qwen.arrays.items()
        if name.startswith("vision_tower.") and ".merger." not in name
    }
    save_file(vision, str(qwen.root / "encoder.safetensors"))
    index = dict.fromkeys(vision, "encoder.safetensors")
    index["visual.merger.mlp.0.weight"] = "unused-projector.safetensors"
    index["model.embed_tokens.weight"] = "unused-language.safetensors"
    (qwen.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    vlm.convert_vlm(str(qwen.main), qwen.output, "float32", mmproj=str(qwen.mmproj),
                    hf_fallback_vision=str(qwen.root), offline=True)
    actual = load_file(str(qwen.output / "model.safetensors"))
    for name, expected in qwen.arrays.items():
        np.testing.assert_array_equal(actual[name], expected)


@pytest.mark.parametrize("name", [
    "v.patch_embd.weight", "v.patch_embd.weight.1", "v.blk.0.attn_q.weight",
    "v.blk.0.attn_k.bias", "v.blk.0.attn_v.weight", "v.post_ln.weight", "mm.2.bias",
])
def test_qwen_incomplete_components_fail_atomically(qwen, monkeypatch, name):
    main, companion = GGUFReader(str(qwen.main)), GGUFReader(str(qwen.mmproj))
    companion.tensors = [tensor for tensor in companion.tensors if tensor.name != name]
    monkeypatch.setattr(vlm, "GGUFReader", lambda path: main if str(path) == str(qwen.main) else companion)
    assert not core.convert(str(qwen.main), str(qwen.output), mmproj=str(qwen.mmproj),
                            hf_model=str(qwen.root), offline=True)
    assert not qwen.output.exists()
    assert not list(qwen.output.parent.glob(".out.*"))


@pytest.mark.parametrize("name", ["v.patch_embd.weight.1", "v.blk.0.attn_q.weight", "mm.2.weight"])
def test_qwen_source_shape_mismatch(qwen, monkeypatch, name):
    main, companion = GGUFReader(str(qwen.main)), GGUFReader(str(qwen.mmproj))
    companion.tensors = [
        SimpleNamespace(name=tensor.name, shape=np.array([1, tensor.data.size]),
                        data=tensor.data, tensor_type=tensor.tensor_type)
        if tensor.name == name else tensor for tensor in companion.tensors
    ]
    monkeypatch.setattr(vlm, "GGUFReader", lambda path: main if str(path) == str(qwen.main) else companion)
    with pytest.raises(ValueError, match="shape mismatch"):
        vlm.convert_vlm(str(qwen.main), qwen.output, "float32", mmproj=str(qwen.mmproj),
                        hf_model=str(qwen.root), offline=True)


@pytest.mark.parametrize("key,value", [
    ("general.architecture", "qwen2"), ("qwen2vl.embedding_length", 128),
    ("qwen2vl.rope.dimension_sections", [3, 3, 2, 0]),
    ("qwen2vl.rope.dimension_sections", []), ("qwen2vl.rope.scaling.type", "linear"),
    ("qwen2vl.rope.freq_base", 10000.0), ("qwen2vl.rope.dimension_count", 8),
])
def test_qwen_text_metadata_validation(qwen, key, value):
    original = GGUFReader(str(qwen.main))
    metadata = {name: field.contents() for name, field in original.fields.items()}
    metadata[key] = value
    with pytest.raises(ValueError):
        Qwen2VisionAdapter(qwen.config).validate_text_metadata(Reader(metadata))


@pytest.mark.parametrize("key,value", [
    ("clip.projector_type", "qwen2.5vl"), ("clip.vision.embedding_length", 1280),
    ("clip.vision.feed_forward_length", 32), ("clip.vision.patch_size", 14),
    ("clip.use_silu", True), ("clip.has_audio_encoder", True),
])
def test_qwen_vision_metadata_validation(key, value):
    with pytest.raises(ValueError):
        Qwen2VisionAdapter(qwen_config()).validate_vision_metadata(Reader({key: value}))


@pytest.mark.parametrize("hybrid", [False, True])
def test_legacy_merger_roundtrip(tmp_path, hybrid):
    fixture = write_fixture(tmp_path, legacy=True)
    assert core.convert(str(fixture.main), str(fixture.output), "float32",
                        mmproj=str(fixture.mmproj), offline=True,
                        hf_model=None if hybrid else str(fixture.root),
                        hf_fallback_vision=str(fixture.root) if hybrid else None)
    actual = load_file(str(fixture.output / "model.safetensors"))
    assert set(actual) == set(fixture.arrays)
    for name, expected in fixture.arrays.items():
        np.testing.assert_array_equal(actual[name], expected)
    report = json.loads((fixture.output / "vlm_conversion_report.json").read_text())
    assert report["gguf_vision_layout"] == "qwen2vl_merger"
    assert report["tensor_counts"]["legacy_position_placeholder_ignored"] == 1


@pytest.mark.parametrize("shape,value", [((10, 10), 1), ((1, 100), 0), ((10, 10), np.nan)])
def test_legacy_placeholder_must_be_known_zero_tensor(tmp_path, monkeypatch, shape, value):
    fixture = write_fixture(tmp_path, legacy=True)
    main, companion = GGUFReader(str(fixture.main)), GGUFReader(str(fixture.mmproj))
    companion.tensors = [
        SimpleNamespace(name=t.name, shape=np.array(shape[::-1]),
                        data=np.full(shape, value, dtype=np.float32),
                        tensor_type=GGMLQuantizationType.F32)
        if t.name == "v.position_embd.weight" else t for t in companion.tensors
    ]
    monkeypatch.setattr(vlm, "GGUFReader",
                        lambda path: main if str(path) == str(fixture.main) else companion)
    assert not core.convert(str(fixture.main), str(fixture.output),
                            mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True)
    assert not fixture.output.exists()


def test_conflicting_vision_layouts_rejected():
    adapter = Qwen2VisionAdapter(qwen_config())
    with pytest.raises(ValueError, match="Conflicting"):
        adapter.configure_layout([
            Reader({"clip.projector_type": name}) for name in ("qwen2vl", "qwen2vl_merger")
        ])


@pytest.mark.parametrize("key,value", [
    ("clip.vision.feed_forward_length", 64), ("clip.vision.projection_dim", 16),
])
def test_legacy_metadata_still_validated(key, value):
    adapter = Qwen2VisionAdapter(qwen_config())
    reader = Reader({"clip.projector_type": "qwen2vl_merger", key: value})
    adapter.configure_layout([reader])
    with pytest.raises(ValueError, match="incompatible"):
        adapter.validate_vision_metadata(reader)


def test_image_tokens_only_in_added_tokens_decoder(qwen):
    path = qwen.root / "tokenizer.json"
    tokenizer = json.loads(path.read_text())
    decoder = {}
    for token in ("<|image_pad|>", "<|video_pad|>"):
        index = tokenizer["model"]["vocab"].pop(token)
        decoder[str(index)] = {"content": token, "special": True}
    path.write_text(json.dumps(tokenizer))
    (qwen.root / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": decoder, "chat_template": "{{ messages }}",
    }))
    assert core.convert(str(qwen.main), str(qwen.output), mmproj=str(qwen.mmproj),
                        hf_model=str(qwen.root), offline=True)
    assert (qwen.output / "tokenizer_config.json").read_bytes() == (
        qwen.root / "tokenizer_config.json"
    ).read_bytes()


@pytest.mark.parametrize("decoder,error", [
    ([], "must be an object"),
    ({"bad": {"content": "new"}}, "Invalid HF added_tokens_decoder"),
    ({"8": "new"}, "Invalid HF added_tokens_decoder"),
    ({"8": {}}, "Invalid HF tokenizer added_tokens"),
    ({"8": {"content": "<|image_pad|>"}}, "conflicting added token IDs"),
    ({"4": {"content": "new"}}, "duplicate token IDs"),
    ({"16": {"content": "new"}}, "IDs exceed"),
])
def test_added_tokens_decoder_rejects_inconsistency(qwen, decoder, error):
    (qwen.root / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": decoder, "chat_template": "{{ messages }}",
    }))
    with pytest.raises(ValueError, match=error):
        vlm._validate_tokenizer(GGUFReader(str(qwen.main)), qwen.root, qwen.config, qwen.config)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_source_preserving_k_quantization(tmp_path, dtype):
    fixture = write_fixture(tmp_path, k_quantized_source=True)
    assert core.convert(
        str(fixture.main), str(fixture.output), dtype, mmproj=str(fixture.mmproj),
        hf_model=str(fixture.root), offline=True, quantize=True, direct_quant=True,
        preserve_source_quant=True,
    )
    actual = load_file(str(fixture.output / "model.safetensors"))
    config = json.loads((fixture.output / "config.json").read_text())
    report = json.loads((fixture.output / "vlm_conversion_report.json").read_text())
    assert config["quantization"]["group_size"] == 32
    policies = report["quantization"]["source_precision"]
    assert len(policies) == 7
    assert sum(p["method"] == "repack_q4_k" for p in policies.values()) == 6
    assert policies["language_model.model.layers.0.mlp.down_proj"]["source_type"] == "Q6_K"
    assert report["quantization"]["method"] == "preserve_source_q4_k"
    for name, expected in fixture.arrays.items():
        stem = name.removesuffix(".weight")
        if stem + ".scales" not in actual:
            np.testing.assert_array_equal(actual[name], expected.astype(dtype))
            continue
        codes = ((actual[name][..., None] >> (np.arange(8) * 4)) & 15).reshape(expected.shape)
        scales = actual[stem + ".scales"].astype(np.float32).repeat(32, axis=-1)
        biases = actual[stem + ".biases"].astype(np.float32).repeat(32, axis=-1)
        restored = codes.astype(np.float32) * scales + biases
        if dtype == "float32":
            np.testing.assert_array_equal(restored, expected)
        else:
            bound = np.finfo(np.float16).eps * (15 * np.abs(scales) + np.abs(biases))
            assert np.all(np.abs(restored - expected) <= bound + np.finfo(np.float16).smallest_subnormal)


def test_source_preserving_rejects_no_q4_k_projections(qwen):
    assert not core.convert(
        str(qwen.main), str(qwen.output), mmproj=str(qwen.mmproj), hf_model=str(qwen.root),
        offline=True, quantize=True, direct_quant=True, preserve_source_quant=True,
    )
    assert not qwen.output.exists()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_q4_k_repack_zero_scales_and_nonfinite(dtype):
    data = np.zeros((1, 144), dtype=np.uint8)
    tensor = SimpleNamespace(name="weight", shape=np.array([256, 1]), data=data,
                             tensor_type=GGMLQuantizationType.Q4_K)
    packed, scales, biases = vlm._repack_q4_k(tensor, dtype)
    assert not packed.any() and not scales.any() and not biases.any()
    data[0, :2] = np.array([np.inf], dtype="<f2").view(np.uint8)
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="Nonfinite"):
        vlm._repack_q4_k(tensor, dtype)


@pytest.mark.parametrize("shape,qtype", [
    ([128, 2], GGMLQuantizationType.Q4_K), ([256], GGMLQuantizationType.Q4_K),
    ([256, 2], GGMLQuantizationType.Q4_K), ([256, 1], GGMLQuantizationType.Q6_K),
])
def test_q4_k_repack_rejects_invalid_layout(shape, qtype):
    tensor = SimpleNamespace(name="weight", shape=np.array(shape),
                             data=np.zeros((1, 144), dtype=np.uint8), tensor_type=qtype)
    with pytest.raises(ValueError, match="Invalid Q4_K"):
        vlm._repack_q4_k(tensor, "float16")


def test_source_preserving_rejects_llava(tmp_path):
    fixture = write_fixture(tmp_path, "llava")
    assert not core.convert(
        str(fixture.main), str(fixture.output), mmproj=str(fixture.mmproj),
        hf_model=str(fixture.root), offline=True, quantize=True, direct_quant=True,
        preserve_source_quant=True,
    )
    assert not fixture.output.exists()
