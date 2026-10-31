"""Plan-driven fixtures for the Qwen vision adapters.

The fixture writes exactly the GGUF tensors an adapter declares, and the expected
output is built by the adapter's own plan, so a test verifies **assembly** (which
sources a target is built from, and along which axis) instead of restating the
name table a third time.

Plan-versus-reality coverage is separate: ``tests/test_vlm_real_mmproj.py``
checks the declared source set and shapes against downloaded ``mmproj`` files.
Neither test alone would be convincing, which is the point of having both.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from gguf import GGUFWriter

TOKENS = [
    "<unk>", "<s>", "</s>", "word",
    "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
] + [f"t{i}" for i in range(8, 16)]

_LAYER_NAMES = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
_ROOTS = {
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.lm_head.weight": "output.weight",
    "language_model.model.norm.weight": "output_norm.weight",
}


def language_gguf_name(name: str) -> str:
    """Invert the dense language mapping for the fixture's one-layer toy model."""
    if name in _ROOTS:
        return _ROOTS[name]
    prefix = "language_model.model.layers.0."
    if not name.startswith(prefix):
        raise AssertionError(f"fixture cannot express {name}")
    stem, _, kind = name.removeprefix(prefix).rpartition(".")
    if stem not in _LAYER_NAMES:
        raise AssertionError(f"fixture cannot express {name}")
    return f"blk.0.{_LAYER_NAMES[stem]}.{kind}"


def qwen25_config() -> dict:
    return {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 16,
        "bos_token_id": 1, "eos_token_id": 2,
        "image_token_id": 4, "video_token_id": 5,
        "vision_start_token_id": 6, "vision_end_token_id": 7,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6, "attention_bias": True,
        "hidden_act": "silu", "tie_word_embeddings": False,
        "rope_scaling": {"type": "mrope", "mrope_section": [4, 2, 2]},
        "vision_config": {
            "model_type": "qwen2_5_vl", "depth": 2, "hidden_size": 16,
            "intermediate_size": 32, "num_heads": 2, "patch_size": 4,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "in_chans": 3,
            "out_hidden_size": 64, "window_size": 8, "hidden_act": "silu",
            "fullatt_block_indexes": [1], "layer_norm_eps": 1e-6,
        },
    }


def qwen3_config() -> dict:
    return {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 4, "video_token_id": 5,
        "vision_start_token_id": 6, "vision_end_token_id": 7,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_vl_text", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "vocab_size": 16, "head_dim": 16, "attention_bias": False,
            "hidden_act": "silu", "tie_word_embeddings": False,
            "rms_norm_eps": 1e-6, "rope_theta": 5e6,
            "rope_scaling": {
                "mrope_interleaved": True, "mrope_section": [4, 2, 2],
                "rope_type": "default",
            },
        },
        "vision_config": {
            "model_type": "qwen3_vl", "depth": 2, "hidden_size": 16,
            "intermediate_size": 32, "num_heads": 2, "patch_size": 4,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "in_chans": 3,
            "out_hidden_size": 64, "num_position_embeddings": 16,
            "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6,
            "deepstack_visual_indexes": [1],
        },
    }


def _vision_metadata(adapter: Any, config: dict) -> dict[str, Any]:
    vision = adapter.vision
    blocks = vision["depth"]
    return {
        "clip.projector_type": (
            "qwen2.5vl_merger" if adapter.name == "Qwen2.5-VL" else "qwen3vl_merger"
        ),
        "clip.vision.embedding_length": vision["hidden_size"],
        "clip.vision.block_count": blocks,
        "clip.vision.attention.head_count": vision["num_heads"],
        "clip.vision.patch_size": vision["patch_size"],
        "clip.vision.spatial_merge_size": vision["spatial_merge_size"],
        "clip.vision.temporal_patch_size": vision["temporal_patch_size"],
        "clip.vision.feed_forward_length": vision["intermediate_size"],
        "clip.vision.projection_dim": vision["out_hidden_size"],
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
    }


def build_plan_arrays(adapter: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create the expected outputs, then split them back into their GGUF sources."""
    expected: dict[str, np.ndarray] = {}
    for name, shape in adapter.expected.items():
        array = rng.normal(0, 0.08, shape).astype(np.float32)
        if name.endswith(".weight") and ("norm" in name or "ln_q" in name):
            array += 1.0
        expected[name] = array
    sources: dict[str, np.ndarray] = {}
    for target, inputs, axis in adapter.vision_plan:
        output = expected[target]
        if len(inputs) == 1:
            sources[inputs[0]] = output.reshape(adapter.source_shapes[inputs[0]])
            continue
        chunks = np.split(output, len(inputs), axis=axis)
        for source, chunk in zip(inputs, chunks):
            sources[source] = chunk.reshape(adapter.source_shapes[source])
    for name, array in expected.items():
        if name.startswith("language_model."):
            sources[language_gguf_name(name)] = array
    return {"expected": expected, "sources": sources}


def write_plan_fixture(
    tmp_path: Path, adapter: Any, config: dict, *, gguf_arch: str,
    vision_overrides: dict[str, Any] | None = None,
    drop_vision_kv: tuple[str, ...] = (),
    drop_sources: tuple[str, ...] = (),
) -> SimpleNamespace:
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "Qwen2VLImageProcessor",
        "processor_class": "Qwen2VLProcessor",
        "patch_size": adapter.vision["patch_size"],
        "merge_size": adapter.vision["spatial_merge_size"],
        "temporal_patch_size": adapter.vision["temporal_patch_size"],
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }))
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {token: i for i, token in enumerate(TOKENS)}},
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ msgs }}"}))

    rng = np.random.default_rng(7)
    built = build_plan_arrays(adapter, rng)
    expected, sources = built["expected"], built["sources"]
    text = adapter.text
    main, mmproj = tmp_path / "language.gguf", tmp_path / "mmproj.gguf"

    writer = GGUFWriter(str(main), gguf_arch)
    for key, value in {
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"],
        "block_count": text["num_hidden_layers"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "vocab_size": text["vocab_size"],
        "rope.dimension_count": text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
    }.items():
        writer.add_uint32(f"{gguf_arch}.{key}", value)
    writer.add_float32(f"{gguf_arch}.rope.freq_base", text["rope_theta"])
    writer.add_float32(
        f"{gguf_arch}.attention.layer_norm_rms_epsilon", text["rms_norm_eps"]
    )
    writer.add_array(
        f"{gguf_arch}.rope.dimension_sections",
        list(text["rope_scaling"]["mrope_section"]) + [0],
    )
    writer.add_token_list(TOKENS)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    for name, array in sources.items():
        if not name.startswith(("v.", "mm.")):
            writer.add_tensor(name, np.ascontiguousarray(array))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    metadata = _vision_metadata(adapter, config)
    metadata.update(vision_overrides or {})
    for key in drop_vision_kv:
        metadata.pop(key, None)
    if (
        "clip.vision.is_deepstack_layers" not in metadata
        and "clip.vision.is_deepstack_layers" not in drop_vision_kv
        and adapter.name == "Qwen3-VL"
    ):
        layers = adapter.vision["depth"]
        wanted = set(adapter.vision["deepstack_visual_indexes"])
        metadata["clip.vision.is_deepstack_layers"] = [i in wanted for i in range(layers)]
    if adapter.name == "Qwen2.5-VL":
        metadata.setdefault("clip.use_silu", True)
    else:
        metadata.setdefault("clip.use_gelu", True)

    writer = GGUFWriter(str(mmproj), "clip")
    for key, value in metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list) and value and isinstance(value[0], bool):
            writer.add_array(key, [bool(v) for v in value])
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_uint32(key, value)
    for name, array in sources.items():
        if name.startswith(("v.", "mm.")) and name not in drop_sources:
            writer.add_tensor(name, np.ascontiguousarray(array))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return SimpleNamespace(
        root=root, main=main, mmproj=mmproj, config=config,
        expected=expected, sources=sources, output=tmp_path / "out",
    )
