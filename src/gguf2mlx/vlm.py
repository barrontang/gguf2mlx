"""Strict VLM conversion; HF assets supply processor and tokenizer semantics.

Vision patch kernels are stored in HF OIHW layout. mlx-vlm's CLIP sanitizer
transposes those kernels to OHWI when loading the resulting model.
Original releases with image tokens outside their text embedding tables require
matching, compatibly resized HF/GGUF checkpoints; this adapter never pads weights.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import Q4_K, dequantize
from safetensors import safe_open

from . import gguf2mlx as core
from .streaming import StreamingShardWriter

# Detection is deliberately broader than the conversion registry.
VLM_ARCHITECTURES = {
    "llava", "llava_next", "llava-next", "llava_next_video", "llava_onevision",
    "qwen2vl", "qwen2_vl", "qwen2.5vl", "qwen2_5_vl", "qwen2.5_vl",
    "qwen3vl", "qwen3_vl", "mllama", "minicpmv", "minicpm-v",
    "internvl", "idefics2", "idefics3", "paligemma", "gemma3",
    "qwen2-vl", "qwen2.5-vl", "qwen2_5vl", "llava-next-video",
    "llava-onevision", "llava1.5", "llava_1_5",
    "clip", "pixtral", "mplug_owl",
    "qwen3.5-vl", "qwen3_5_vl",
}
ARCHITECTURE_REGISTRY = {
    "llava": "llama_clip_mlp",
    "qwen2_vl": "qwen2_vit_patch_merger",
    "minicpmv4_7": "qwen35_siglip_window_merger",
}
ASSETS = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "preprocessor_config.json", "added_tokens.json", "special_tokens_map.json",
    "processor_config.json", "generation_config.json", "chat_template.jinja",
)
REQUIRED_ASSETS = ASSETS[:4]
VISION_PREFIX = "vision_tower.vision_model."
SHARD_BYTES = 256 * 1024 * 1024


def is_vlm(reader: Any) -> bool:
    """Recognize VLM metadata without normalizing it into a language-only arch."""
    arch = (core.get_metadata_str(reader, "general.architecture") or "").lower()
    return (
        arch in VLM_ARCHITECTURES
        or any(t.name.startswith((
            "v.", "mm.", "vision_tower.", "vision_model.", "visual.",
            "mm_projector.", "multi_modal_projector.", "model.vision_tower.",
            "model.visual.",
        )) for t in reader.tensors)
        or any(key.startswith("clip.vision.") for key in getattr(reader, "fields", {}))
        or any(
            reader.get_field(key) is not None
            for key in ("clip.has_vision_encoder", "clip.has_llava_projector")
        )
    )


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read required HF JSON asset {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"HF asset {path.name} must contain a JSON object")  # noqa: TRY004
    return result


def _source_identity(source: str) -> str:
    path = Path(source).expanduser()
    return str(path.resolve()) if path.is_dir() else source


def _resolve_source(source: str, revision: str, offline: bool) -> tuple[Path, str | None]:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve(), None
    if path.exists() or source.startswith(("/", "./", "../", "~")):
        raise ValueError("--hf-model must be a full local HF directory or HF repository ID")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Remote HF assets require the VLM extra: pip install 'gguf2mlx[vlm]'; "
            "alternatively use --hf-model with a local directory."
        ) from exc
    try:
        root = Path(snapshot_download(
            repo_id=source, revision=revision, local_files_only=offline,
            allow_patterns=list(ASSETS) + ["model.safetensors.index.json"],
        ))
    except Exception as exc:
        raise RuntimeError(
            "Cannot obtain HF assets; check --hf-model, --hf-revision and "
            "--offline/cache availability."
        ) from exc
    resolved = root.name if root.parent.name == "snapshots" else None
    return root, resolved


def _positive(config: dict, key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"HF config requires a positive integer {key}")
    return value


def _validate_config(config: dict) -> tuple[dict, dict]:
    if config.get("model_type") != "llava":
        raise ValueError("Only HF model_type='llava' or 'qwen2_vl' is supported.")
    if config.get("architectures") not in (None, ["LlavaForConditionalGeneration"]):
        raise ValueError("Unsupported HF LLaVA architecture; only LlavaForConditionalGeneration works")
    if config.get("image_grid_pinpoints") is not None or config.get(
        "vision_aspect_ratio", "square"
    ) != "square":
        raise ValueError("Any-resolution/tiled LLaVA-next vision layouts are unsupported")
    text, vision = config.get("text_config", {}), config.get("vision_config", {})
    if not isinstance(text, dict) or text.get("model_type") != "llama":
        raise ValueError("Supported LLaVA requires text_config.model_type='llama'")
    if not isinstance(vision, dict) or vision.get("model_type") != "clip_vision_model":
        raise ValueError("Supported LLaVA requires CLIP vision; SigLIP is not supported")
    if text.get("architectures") not in (None, ["LlamaForCausalLM"]):
        raise ValueError("Unsupported HF text architecture; only LlamaForCausalLM works")
    if vision.get("architectures") not in (None, ["CLIPVisionModel"]):
        raise ValueError("Unsupported HF vision architecture; only CLIPVisionModel works")
    for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "vocab_size",
    ):
        _positive(text, key)
    text = dict(text)
    text.setdefault("num_key_value_heads", text["num_attention_heads"])
    _positive(text, "num_key_value_heads")
    heads, kv, hidden = (
        text["num_attention_heads"], text["num_key_value_heads"], text["hidden_size"]
    )
    if hidden % heads or heads % kv or (hidden // heads) % 2:
        raise ValueError("Llama head dimensions must support the GGUF RoPE permutation")
    if text.get("head_dim", hidden // heads) != hidden // heads:
        raise ValueError("Nonstandard Llama head_dim is unsupported")
    if text.get("attention_bias", False) or text.get("mlp_bias", False):
        raise ValueError("Biased Llama attention/MLP is unsupported")
    if not isinstance(text.get("tie_word_embeddings", False), bool):
        raise ValueError("HF tie_word_embeddings must be a boolean")  # noqa: TRY004
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("Only the Llama SiLU gated MLP is supported")
    if text.get("rope_traditional", False) is not False:
        raise ValueError("LLaVA requires HF nontraditional Llama RoPE (rope_traditional=False)")
    scaling = text.get("rope_scaling")
    if scaling is not None:
        if (
            not isinstance(scaling, dict)
            or set(scaling) - {"type", "rope_type", "factor"}
            or scaling.get("type", scaling.get("rope_type")) != "linear"
            or scaling.get("rope_type", "linear") != "linear"
        ):
            raise ValueError("Only linear Llama rope_scaling with factor/type is supported")
        factor = scaling.get("factor")
        if (
            isinstance(factor, bool) or not isinstance(factor, (int, float))
            or not np.isfinite(factor) or factor < 1
        ):
            raise ValueError("Linear Llama rope_scaling requires a finite factor >= 1")
        text["rope_scaling"] = {"type": "linear", "factor": float(factor)}
    if text.get("sliding_window") is not None:
        raise ValueError("Sliding-window Llama attention is unsupported")
    for key, default in (("rope_theta", 10000.0), ("rms_norm_eps", 1e-6)):
        value = text.get(key, default)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0
        ):
            raise ValueError(f"Unsupported HF Llama {key}")
    for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "image_size", "patch_size",
    ):
        _positive(vision, key)
    if (
        vision["image_size"] % vision["patch_size"]
        or vision["hidden_size"] % vision["num_attention_heads"]
        or vision.get("num_channels", 3) != 3
    ):
        raise ValueError("Unsupported CLIP image/patch/head/channel dimensions")
    if vision.get("hidden_act", "quick_gelu") != "quick_gelu":
        raise ValueError("mlx-vlm's LLaVA CLIP requires hidden_act='quick_gelu'")
    epsilon = vision.get("layer_norm_eps", 1e-5)
    if (
        isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
        or not np.isfinite(epsilon) or epsilon != 1e-5
    ):
        raise ValueError("Unsupported HF CLIP layer_norm_eps")
    if config.get("vision_feature_select_strategy", "default") != "default":
        raise ValueError("Only LLaVA default patch feature selection is supported")
    feature_layer = config.get("vision_feature_layer", -2)
    if isinstance(feature_layer, bool) or not isinstance(feature_layer, int) or not (
        -(vision["num_hidden_layers"] + 1) <= feature_layer <= vision["num_hidden_layers"]
    ):
        raise ValueError("Unsupported CLIP vision_feature_layer")
    if config.get("projector_hidden_act", "gelu") != "gelu":
        raise ValueError("Only the two-linear GELU LLaVA projector is supported")
    if config.get("mm_projector_type", "mlp2x_gelu") != "mlp2x_gelu":
        raise ValueError("Only a two-linear mlp2x_gelu projector is supported")
    if config.get("multimodal_projector_bias", True) is not True:
        raise ValueError("The supported two-linear projector requires bias tensors")
    return text, vision


def _language_shapes(text: dict, *, attention_bias: bool = False) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    h, f, vocab = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
    kv = h // text["num_attention_heads"] * text["num_key_value_heads"]
    shapes["language_model.model.embed_tokens.weight"] = (vocab, h)
    if not text.get("tie_word_embeddings", False):
        shapes["language_model.lm_head.weight"] = (vocab, h)
    shapes["language_model.model.norm.weight"] = (h,)
    for i in range(text["num_hidden_layers"]):
        prefix = f"language_model.model.layers.{i}."
        for name, shape in {
            "self_attn.q_proj.weight": (h, h), "self_attn.k_proj.weight": (kv, h),
            "self_attn.v_proj.weight": (kv, h), "self_attn.o_proj.weight": (h, h),
            "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
            "mlp.gate_proj.weight": (f, h), "mlp.up_proj.weight": (f, h),
            "mlp.down_proj.weight": (h, f),
        }.items():
            shapes[prefix + name] = shape
        if attention_bias:
            for part, size in (("q", h), ("k", kv), ("v", kv)):
                shapes[prefix + f"self_attn.{part}_proj.bias"] = (size,)
    return shapes


def _expected_shapes(text: dict, vision: dict) -> dict[str, tuple[int, ...]]:
    shapes = _language_shapes(text)
    h = text["hidden_size"]
    v, vf, patch = vision["hidden_size"], vision["intermediate_size"], vision["patch_size"]
    positions = (vision["image_size"] // patch) ** 2 + 1
    shapes[VISION_PREFIX + "embeddings.class_embedding"] = (v,)
    shapes[VISION_PREFIX + "embeddings.patch_embedding.weight"] = (v, 3, patch, patch)
    shapes[VISION_PREFIX + "embeddings.position_embedding.weight"] = (positions, v)
    for norm in ("pre_layrnorm", "post_layernorm"):
        for suffix in ("weight", "bias"):
            shapes[VISION_PREFIX + f"{norm}.{suffix}"] = (v,)
    for i in range(vision["num_hidden_layers"]):
        prefix = VISION_PREFIX + f"encoder.layers.{i}."
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[prefix + f"self_attn.{part}.weight"] = (v, v)
            shapes[prefix + f"self_attn.{part}.bias"] = (v,)
        for part in ("layer_norm1", "layer_norm2"):
            for suffix in ("weight", "bias"):
                shapes[prefix + f"{part}.{suffix}"] = (v,)
        for part, shape in (("fc1", (vf, v)), ("fc2", (v, vf))):
            shapes[prefix + f"mlp.{part}.weight"] = shape
            shapes[prefix + f"mlp.{part}.bias"] = (shape[0],)
    for part, shape in (("linear_1", (h, v)), ("linear_2", (h, h))):
        shapes[f"multi_modal_projector.{part}.weight"] = shape
        shapes[f"multi_modal_projector.{part}.bias"] = (h,)
    return shapes


def _validate_text_metadata(reader: Any, text: dict) -> None:
    architecture = core.get_metadata_str(reader, "general.architecture")
    if architecture not in ("llama", "llava"):
        raise ValueError(
            f"Unsupported GGUF architecture {architecture!r}; only llama/llava text "
            "with CLIP LLaVA is supported, regardless of supplied HF config."
        )
    dimensions = {
        "embedding_length": "hidden_size", "feed_forward_length": "intermediate_size",
        "block_count": "num_hidden_layers", "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
    }
    for gguf_key, hf_key in dimensions.items():
        values = [
            core.get_metadata_int(reader, f"{prefix}.{gguf_key}")
            for prefix in ("llama", "llava")
        ]
        present = [value for value in values if value is not None]
        if not present or any(value != text[hf_key] for value in present):
            raise ValueError(f"GGUF {gguf_key} missing or incompatible with HF {hf_key}")
    for prefix in ("llama", "llava"):
        vocab = core.get_metadata_int(reader, f"{prefix}.vocab_size")
        if vocab is not None and vocab != text["vocab_size"]:
            raise ValueError("GGUF vocab_size does not match HF text embeddings")
        rotary_dims = core.get_metadata_int(reader, f"{prefix}.rope.dimension_count")
        if rotary_dims is not None and rotary_dims != (
            text["hidden_size"] // text["num_attention_heads"]
        ):
            raise ValueError("Partial rotary Llama embeddings are unsupported")
        scaling = core.get_metadata_str(reader, f"{prefix}.rope.scaling.type")
        if scaling not in (None, "none", "linear"):
            raise ValueError("Unsupported GGUF Llama RoPE scaling type")
        hf_scaling = text.get("rope_scaling")
        expected_scaling = "linear" if hf_scaling else "none"
        if scaling is not None and scaling != expected_scaling:
            raise ValueError("GGUF RoPE scaling type is incompatible with HF rope_scaling")
        factor = core.get_metadata_float(reader, f"{prefix}.rope.scaling.factor")
        expected_factor = hf_scaling["factor"] if hf_scaling else 1.0
        if factor is not None and not np.isclose(factor, expected_factor, rtol=1e-5, atol=0):
            raise ValueError("GGUF RoPE scaling factor is incompatible with HF rope_scaling")
        for gguf_key, hf_key, default in (
            ("rope.freq_base", "rope_theta", 10000.0),
            ("attention.layer_norm_rms_epsilon", "rms_norm_eps", 1e-6),
        ):
            value = core.get_metadata_float(reader, f"{prefix}.{gguf_key}")
            if value is not None and not np.isclose(
                value, text.get(hf_key, default), rtol=1e-5, atol=0
            ):
                raise ValueError(f"GGUF {gguf_key} is incompatible with HF {hf_key}")


def _validate_vision_metadata(reader: Any, vision: dict) -> None:
    for gguf_key, hf_key in {
        "embedding_length": "hidden_size", "feed_forward_length": "intermediate_size",
        "block_count": "num_hidden_layers", "attention.head_count": "num_attention_heads",
        "image_size": "image_size", "patch_size": "patch_size",
    }.items():
        value = core.get_metadata_int(reader, f"clip.vision.{gguf_key}")
        if value is not None and value != vision[hf_key]:
            raise ValueError(f"GGUF CLIP {gguf_key} is incompatible with HF vision {hf_key}")
    epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
    if epsilon is not None and not np.isclose(
        epsilon, vision.get("layer_norm_eps", 1e-5), rtol=1e-5, atol=0
    ):
        raise ValueError("GGUF CLIP layer norm epsilon is incompatible with HF vision")
    if core.get_metadata_bool(reader, "clip.use_silu") is True:
        raise ValueError("SiLU CLIP vision encoders are unsupported")
    gelu = core.get_metadata_bool(reader, "clip.use_gelu")
    if gelu is not None and gelu != (vision.get("hidden_act", "quick_gelu") == "gelu"):
        raise ValueError("GGUF CLIP activation is incompatible with HF vision hidden_act")


def _validate_tokenizer(reader: Any, root: Path, text: dict, config: dict) -> None:
    document = _read_json(root / "tokenizer.json")
    tokenizer_config = _read_json(root / "tokenizer_config.json")
    model = document.get("model", {})
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        token_ids = vocab
    elif isinstance(vocab, list):
        if any(not isinstance(entry, list) or not entry or not isinstance(entry[0], str)
               for entry in vocab):
            raise ValueError("Unsupported HF tokenizer vocabulary layout")
        token_ids = {entry[0]: i for i, entry in enumerate(vocab)}
    else:
        raise ValueError("HF tokenizer.json must contain a verifiable vocabulary")  # noqa: TRY004
    token_ids = dict(token_ids)
    added_tokens = list(document.get("added_tokens", []))
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise ValueError("HF added_tokens_decoder must be an object")  # noqa: TRY004
    for index_text, entry in decoder.items():
        if (
            not isinstance(index_text, str) or not index_text.isdecimal()
            or not isinstance(entry, dict)
        ):
            raise ValueError("Invalid HF added_tokens_decoder entry")
        added_tokens.append({**entry, "id": int(index_text)})
    for added in added_tokens:
        if not isinstance(added, dict) or not isinstance(added.get("content"), str):
            raise ValueError("Invalid HF tokenizer added_tokens")  # noqa: TRY004
        token, index = added["content"], added.get("id")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("HF tokenizer contains invalid token IDs")
        if token in token_ids and token_ids[token] != index:
            raise ValueError("HF tokenizer has conflicting added token IDs")
        token_ids[token] = index
    by_id: dict[int, str] = {}
    for token, index in token_ids.items():
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("HF tokenizer contains invalid token IDs")
        if index in by_id and by_id[index] != token:
            raise ValueError("HF tokenizer has duplicate token IDs")
        by_id[index] = token
    if any(index >= text["vocab_size"] for index in by_id):
        raise ValueError(
            "HF tokenizer IDs exceed the actual GGUF/HF text vocab_size; "
            "supply matching checkpoints with compatibly resized text embeddings."
        )
    tokens = core.get_metadata_array_str(reader, "tokenizer.ggml.tokens")
    if len(tokens) != text["vocab_size"]:
        raise ValueError("GGUF tokenizer vocabulary must match HF text embedding vocab_size")
    last_hf_token = max(by_id, default=-1)
    for i, token in enumerate(tokens):
        if i not in by_id and i > last_hf_token and token == f"[PAD{i}]":
            # llama.cpp names unused, rounded-up embedding rows this way.
            # Keep their original rows; do not invent tokenizer IDs or pad weights.
            continue
        if by_id.get(i) != token:
            raise ValueError(
                f"HF/GGUF tokenizer mismatch at token ID {i}; use the original matching HF model."
            )
    for kind in ("bos", "eos", "pad"):
        gguf_id = core.get_metadata_int(reader, f"tokenizer.ggml.{kind}_token_id")
        hf_id = text.get(f"{kind}_token_id")
        if gguf_id is not None and hf_id is not None and gguf_id != hf_id:
            raise ValueError(f"HF/GGUF {kind} token IDs differ")
    special_tokens = (
        {"image_token_id": "<|image_pad|>", "video_token_id": "<|video_pad|>",
         "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>"}
        if config.get("model_type") == "qwen2_vl" else {"image_token_index": "<image>"}
    )
    for key, token in special_tokens.items():
        image_id = config.get(key, 32000 if key == "image_token_index" else None)
        if (
            isinstance(image_id, bool) or not isinstance(image_id, int)
            or not 0 <= image_id < text["vocab_size"]
        ):
            raise ValueError(
                f"HF {key} must be inside the actual GGUF/HF text vocab_size; "
                "supply matching checkpoints with compatibly resized text embeddings. "
                "mlx-vlm embeds image tokens before replacing their features."
            )
        if by_id.get(image_id) != token:
            raise ValueError(f"HF tokenizer must define {token} at config.{key}")
    _read_json(root / "preprocessor_config.json")
    if not (
        tokenizer_config.get("chat_template")
        or (root / "chat_template.jinja").is_file()
        or _read_optional_json(root / "processor_config.json").get("chat_template")
    ):
        raise ValueError("Original HF assets must include a chat template for image prompts")


def _read_optional_json(path: Path) -> dict:
    return _read_json(path) if path.is_file() else {}


def _validate_processor(readers: list[Any], root: Path, vision: dict, architecture: str) -> None:
    processor = _read_json(root / "preprocessor_config.json")
    expected_type = "Qwen2VLImageProcessor" if architecture == "qwen2_vl" else "CLIPImageProcessor"
    if processor.get("image_processor_type") not in (expected_type, expected_type + "Fast"):
        raise ValueError(f"HF image processor must be {expected_type}")
    for key in ("image_mean", "image_std"):
        value = processor.get(key)
        if value is not None and (
            not isinstance(value, list) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (float, int))
                   or not np.isfinite(x) or (key == "image_std" and x <= 0) for x in value)
        ):
            raise ValueError(f"Invalid HF processor {key}")
        for reader in readers:
            if reader.get_field("clip.vision." + key) is not None:
                gguf_value = core.get_metadata_array_float(reader, "clip.vision." + key)
                if value is None or len(gguf_value) != 3 or not np.allclose(
                    gguf_value, value, rtol=1e-5, atol=1e-7
                ):
                    raise ValueError(f"GGUF/HF image processor {key} mismatch")
    if architecture == "qwen2_vl":
        for key, vision_key in (
            ("patch_size", "patch_size"), ("temporal_patch_size", "temporal_patch_size"),
            ("merge_size", "spatial_merge_size"),
        ):
            if processor.get(key) != vision[vision_key]:
                raise ValueError(f"Qwen2-VL processor {key} missing or incompatible with vision")
    else:
        crop = processor.get("crop_size")
        if crop is not None and crop != {
            "height": vision["image_size"], "width": vision["image_size"],
        }:
            raise ValueError("CLIP processor crop_size is incompatible with vision image_size")


_VISION_ROOT_MAP = {
    "v.class_embd": "embeddings.class_embedding",
    "v.class_embd.weight": "embeddings.class_embedding",
    "v.patch_embd.weight": "embeddings.patch_embedding.weight",
    "v.position_embd.weight": "embeddings.position_embedding.weight",
    "v.pre_ln.weight": "pre_layrnorm.weight", "v.pre_ln.bias": "pre_layrnorm.bias",
    "v.post_ln.weight": "post_layernorm.weight", "v.post_ln.bias": "post_layernorm.bias",
}
_VISION_BLOCK_MAP = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_out": "self_attn.out_proj",
    "ln1": "layer_norm1", "ln2": "layer_norm2",
    "ffn_up": "mlp.fc1", "ffn_down": "mlp.fc2",
}


def _map_vision(name: str) -> str:
    if name in _VISION_ROOT_MAP:
        return VISION_PREFIX + _VISION_ROOT_MAP[name]
    match = re.fullmatch(r"v\.blk\.(\d+)\.([a-z0-9_]+)\.(weight|bias)", name)
    if match and match[2] in _VISION_BLOCK_MAP:
        return (
            VISION_PREFIX + f"encoder.layers.{match[1]}."
            + _VISION_BLOCK_MAP[match[2]] + f".{match[3]}"
        )
    raise ValueError(f"Unsupported CLIP vision tensor {name}; no generic VLM mapping exists")


def _map_projector(name: str) -> str:
    match = re.fullmatch(r"mm\.(0|2)\.(weight|bias)", name)
    if not match:
        raise ValueError(
            f"Unsupported projector tensor {name}; expected mm.0/mm.2 weight and bias "
            "for the two-linear LLaVA MLP."
        )
    linear = "linear_1" if match[1] == "0" else "linear_2"
    return f"multi_modal_projector.{linear}.{match[2]}"


def _decode(tensor: Any, dtype: str) -> np.ndarray:
    shape = tuple(int(dim) for dim in reversed(tensor.shape))
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid GGUF tensor shape: {tensor.name}")
    try:
        qtype = GGMLQuantizationType(int(tensor.tensor_type))
        if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            array = np.asarray(tensor.data)
        else:
            array = dequantize(tensor.data, qtype)
        array = array.reshape(shape)
    except (ValueError, TypeError, NotImplementedError, KeyError) as exc:
        raise ValueError(
            f"Cannot decode GGUF tensor {tensor.name} ({tensor.tensor_type}); "
            "export a supported GGUF quantization or F16/F32."
        ) from exc
    return _cast_finite(array, tensor.name, dtype)


def _cast_finite(array: np.ndarray, name: str, dtype: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite values in VLM tensor {name}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(
            array, dtype=np.float16 if dtype == "float16" else np.float32
        )
    if not np.isfinite(result).all():
        raise ValueError(
            f"VLM tensor {name} overflows {dtype}; use --dtype float32 "
            "or verify the original weights."
        )
    return result


def _restore_llama(name: str, array: np.ndarray, text: dict) -> np.ndarray:
    array = core._restore_architecture_tensor(name, array, "llama")
    if re.fullmatch(r"blk\.\d+\.attn_[qk]\.weight", name):
        heads = text["num_attention_heads"] if ".attn_q." in name else text["num_key_value_heads"]
        if array.ndim != 2 or array.shape[0] % (2 * heads):
            raise ValueError(f"Invalid Llama RoPE projection shape for {name}")
        # HF -> GGUF splits even/odd rotary rows. Restore HF's contiguous halves.
        array = array.reshape(heads, array.shape[0] // heads // 2, 2, array.shape[1])
        array = array.swapaxes(1, 2).reshape(-1, array.shape[-1])
    return np.ascontiguousarray(array)


def _hf_vision_name(
    name: str, architecture: str = "llava", minicpm: Any = None,
) -> str | None:
    if architecture == "minicpmv4_7":
        return minicpm.hf_vision_name(name) if minicpm else None
    if architecture == "qwen2_vl":
        for prefix in ("visual.", "model.visual.", "vision_tower."):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                # The merger (including its norm) always comes from GGUF.
                return None if suffix.startswith("merger.") else "vision_tower." + suffix
        return None
    for prefix in (VISION_PREFIX, "model." + VISION_PREFIX, "vision_model."):
        if name.startswith(prefix):
            return VISION_PREFIX + name[len(prefix):]
    return None


def _safe_shard_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ValueError("Unsafe HF safetensors index shard path")  # noqa: TRY004
    path = PurePosixPath(name)
    if (
        not name or "\\" in name or path.is_absolute()
        or any(character in name for character in "*?[]:")
        or any(ord(character) < 32 for character in name)
        or any(part in ("", ".", "..") for part in name.split("/"))
        or path.suffix != ".safetensors"
    ):
        raise ValueError(f"Unsafe HF safetensors index shard path: {name!r}")
    return name


def _vision_files(
    root: Path, source: str, revision: str, offline: bool, architecture: str = "llava",
    minicpm: Any = None,
) -> tuple[list[Path], dict[str, str] | None]:
    index_path = root / "model.safetensors.index.json"
    vision_index = None
    if index_path.is_file():
        index = _read_json(index_path).get("weight_map")
        if not isinstance(index, dict):
            raise ValueError("HF safetensors index is missing weight_map")
        # Validate every entry, not merely the selected subset.
        for name, shard in index.items():
            if not isinstance(name, str):
                raise ValueError("Invalid tensor name in HF safetensors index")  # noqa: TRY004
            _safe_shard_name(shard)
        vision_index = {
            name: shard for name, shard in index.items()
            if _hf_vision_name(name, architecture, minicpm) is not None
        }
        if not vision_index:
            raise ValueError("HF safetensors index contains no supported vision weights")
        names = sorted(set(vision_index.values()))
    else:
        names = ["model.safetensors"]
    if not Path(source).expanduser().is_dir():
        try:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(
                repo_id=source, revision=revision, local_files_only=offline,
                allow_patterns=names,
            ))
        except Exception as exc:
            raise RuntimeError(
                "Cannot obtain original HF vision safetensors; check cache/revision. "
                "Pickle/PyTorch checkpoints are intentionally unsupported."
            ) from exc
    paths = []
    for name in names:
        path = root / name
        # Hub cache symlinks are legitimate; local shard symlinks must stay in the source.
        if Path(source).expanduser().is_dir() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe HF shard symlink: {name}")
        if not path.is_file():
            raise ValueError(
                f"Missing HF vision safetensors shard {name}; provide the full original HF model."
            )
        paths.append(path)
    return paths, vision_index


def _read_hf_tensor(weights: Any, path: Path, name: str) -> np.ndarray:
    tensor = weights.get_slice(name)
    dtype = tensor.get_dtype()
    if dtype not in ("F16", "F32", "BF16"):
        raise ValueError(
            f"HF vision tensor {name} has unsupported dtype {dtype}; "
            "provide original floating-point vision weights, not quantized tensors."
        )
    if dtype != "BF16":
        return np.asarray(weights.get_tensor(name))
    # NumPy has no native BF16. safe_open has already validated the file;
    # read just this tensor's payload without loading its enclosing LM shard.
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        descriptor = json.loads(handle.read(header_length))[name]
        handle.seek(8 + header_length + descriptor["data_offsets"][0])
        values = np.fromfile(handle, dtype="<u2", count=int(np.prod(tensor.get_shape())))
    return (values.astype(np.uint32) << 16).view(np.float32).reshape(tensor.get_shape())


class _ShardWriter(StreamingShardWriter):
    def __init__(self, root: Path, max_shard_bytes: int | None = None):
        super().__init__(root, SHARD_BYTES if max_shard_bytes is None else max_shard_bytes)

    def finish(self) -> None:
        super().finish()
        if len(self.files) == 1:
            renamed = self.root / "model.safetensors"
            self.files[0].rename(renamed)
            self.files[0] = renamed
            self.weight_map = {name: renamed.name for name in self.weight_map}
        _write_json(self.root / "model.safetensors.index.json", {
            "metadata": {"total_size": self.total}, "weight_map": self.weight_map,
        })


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _repack_q4_k(tensor: Any, dtype: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep Q4_K codes and 32-value grids, changing only their storage layout."""
    shape = tuple(int(size) for size in reversed(tensor.shape))
    if (
        tensor.tensor_type != GGMLQuantizationType.Q4_K
        or len(shape) != 2 or any(size <= 0 for size in shape) or shape[-1] % 256
        or tensor.data.dtype != np.uint8
        or tensor.data.size != shape[0] * shape[1] // 256 * 144
    ):
        raise ValueError(f"Invalid Q4_K linear tensor for source-preserving conversion: {tensor.name}")
    blocks = np.ascontiguousarray(tensor.data).reshape(-1, 144)
    scales, minima = Q4_K.get_scale_min(blocks[:, 4:16].copy())
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    scales = (d * scales).reshape(shape[0], shape[1] // 32)
    biases = (-dmin * minima).reshape(scales.shape)
    # Each 64-value pair stores the first 32 codes in low nibbles.
    codes = (
        blocks[:, 16:].reshape(-1, 4, 1, 32)
        >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ) & 15
    codes = codes.reshape(shape[0], shape[1] // 8, 8).astype(np.uint32)
    packed = np.bitwise_or.reduce(codes << (np.arange(8, dtype=np.uint32) * 4), axis=-1)
    return (
        packed, _cast_finite(scales, tensor.name + ".scales", dtype),
        _cast_finite(biases, tensor.name + ".biases", dtype),
    )


def convert_vlm(
    gguf_path: str,
    output_path: Path,
    dtype: str,
    *,
    mmproj: str | None = None,
    hf_model: str | None = None,
    hf_fallback_vision: str | None = None,
    hf_revision: str = "main",
    offline: bool = False,
    direct_quant: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    preserve_source_quant: bool = False,
    max_shard_bytes: int | None = None,
) -> None:
    """Convert supported VLMs to mlx-vlm weights in the caller's staging dir."""
    if dtype not in ("float16", "float32"):
        raise ValueError("VLM conversion supports --dtype float16 or float32 only")
    if direct_quant and (q_bits != 4 or q_group_size != 64 or q_mode != "affine"):
        raise ValueError("VLM direct quantization supports only affine 4-bit, group size 64")
    if preserve_source_quant and not direct_quant:
        raise ValueError("--preserve-source-quant requires --quantize --direct-quant")
    if hf_model and hf_fallback_vision and (
        _source_identity(hf_model) != _source_identity(hf_fallback_vision)
    ):
        raise ValueError("--hf-model and --hf-fallback-vision must refer to the same HF source")
    source = hf_model or hf_fallback_vision
    if not source:
        raise ValueError(
            "VLM conversion requires --hf-model (original HF repo or local directory) "
            "for config, tokenizer, processor and chat-template semantics."
        )
    root, resolved = _resolve_source(source, hf_revision, offline)
    for asset in REQUIRED_ASSETS:
        if not (root / asset).is_file():
            raise ValueError(f"Missing original HF asset {asset}; supply a full --hf-model")
    config = _read_json(root / "config.json")
    if any(config.get(key) is not None for key in ("quantization", "quantization_config")):
        raise ValueError("Supply an original unquantized HF config, not a quantized checkpoint")
    architecture = config.get("model_type")
    if not isinstance(architecture, str) or architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            "Only HF model_type='llava', 'qwen2_vl', or 'minicpmv4_7' is supported."
        )
    qwen = minicpm = None
    if architecture == "qwen2_vl":
        from .vision_qwen2 import Qwen2VisionAdapter

        qwen = Qwen2VisionAdapter(config)
        text, vision = qwen.text, qwen.vision
    elif architecture == "minicpmv4_7":
        from .vision_minicpm import MiniCPMV47Adapter

        preprocessor = _read_json(root / "preprocessor_config.json")
        config.setdefault(
            "downsample_mode", preprocessor.get("downsample_mode", "16x")
        )
        minicpm = MiniCPMV47Adapter(config)
        text, vision = minicpm.text, minicpm.vision
    else:
        text, vision = _validate_config(config)
    if preserve_source_quant and not qwen:
        raise ValueError("--preserve-source-quant currently supports Qwen2-VL only")
    if minicpm and direct_quant:
        raise ValueError("MiniCPM-V 4.7 direct quantization is not supported")
    reader = GGUFReader(gguf_path)
    if qwen:
        qwen.validate_text_metadata(reader)
    elif minicpm:
        minicpm.validate_text_metadata(reader)
        minicpm_text_expected = minicpm.validate_text_tensors(reader)
    else:
        _validate_text_metadata(reader, text)
    if minicpm:
        minicpm.validate_tokenizer(reader, root)
    else:
        _validate_tokenizer(reader, root, text, config)
    readers = [reader]
    if mmproj:
        companion = GGUFReader(mmproj)
        companion_arch = core.get_metadata_str(companion, "general.architecture")
        if companion_arch not in ("clip", "llava"):
            raise ValueError("The --mmproj companion must be a CLIP/LLaVA projector GGUF")
        readers.append(companion)
    if minicpm and not mmproj:
        raise ValueError("MiniCPM-V 4.7 requires its separate vision GGUF via --mmproj")
    if qwen:
        qwen.configure_layout(readers)
    processor = minicpm.validate_processor(root) if minicpm else None
    for index, current in enumerate(readers):
        if qwen:
            qwen.validate_vision_metadata(current)
            continue
        if minicpm:
            if index == 0:
                continue
            if core.get_metadata_str(current, "general.architecture") != "clip":
                raise ValueError("MiniCPM-V 4.7 requires a separate CLIP projector GGUF")
            minicpm.validate_vision_metadata(current, processor)
            continue
        projector_type = core.get_metadata_str(current, "clip.projector_type")
        if projector_type not in (None, "mlp"):
            raise ValueError(f"Unsupported CLIP projector type {projector_type!r}")
        if core.get_metadata_bool(current, "clip.has_siglip") is True:
            raise ValueError("SigLIP projector companions are unsupported")
        _validate_vision_metadata(current, vision)
    if not minicpm:
        _validate_processor(readers, root, vision, architecture)
    expected = (
        {**minicpm.expected, **minicpm_text_expected}
        if minicpm else qwen.expected if qwen else _expected_shapes(text, vision)
    )
    seen: set[str] = set()
    raw_seen: set[str] = set()
    counts = {"language": 0, "projector": 0, "vision_gguf": 0, "vision_hf": 0,
              "vision_gguf_ignored": 0, "language_tied_head_ignored": 0,
              "legacy_position_placeholder_ignored": 0, "separator_embedding_ignored": 0}
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()):
        raise ValueError("VLM conversion output must be an empty staging directory")
    writer = _ShardWriter(output_path, max_shard_bytes)
    quantized: list[str] = []
    source_precision: dict[str, dict] = {}
    qwen_tensors: dict[str, Any] = {}

    def emit(name: str, array: np.ndarray, category: str, source_tensor: Any = None) -> None:
        if name in seen:
            raise ValueError(f"Duplicate/colliding VLM tensor: {name}")
        if name not in expected:
            raise ValueError(f"Unexpected VLM tensor: {name}")
        if tuple(array.shape) != expected[name]:
            raise ValueError(
                f"VLM shape mismatch for {name}: got {tuple(array.shape)}, "
                f"expected {expected[name]}"
            )
        seen.add(name)
        counts[category] += 1
        if direct_quant and category == "language" and core._is_direct_quantizable_linear_weight(
            name, array
        ):
            stem = name.removesuffix(".weight")
            if preserve_source_quant:
                if source_tensor is None:
                    raise ValueError(f"Missing source tensor for precision-preserving output: {name}")
                source_type = source_tensor.tensor_type
                repacked = source_type == GGMLQuantizationType.Q4_K
                source_precision[stem] = {
                    "source_type": GGMLQuantizationType(source_type).name,
                    "method": "repack_q4_k" if repacked else "keep_float",
                    "bits": 4 if repacked else None,
                    "group_size": 32 if repacked else None,
                    "floating_dtype": dtype,
                }
                if not repacked:
                    writer.add(name, _cast_finite(array, name, dtype))
                    return
                packed, scales, biases = _repack_q4_k(source_tensor, dtype)
            else:
                packed, scales, biases = core._quantize_affine_4bit(array, q_group_size)
            writer.add(name, packed)
            writer.add(stem + ".scales", _cast_finite(scales, stem + ".scales", dtype))
            writer.add(stem + ".biases", _cast_finite(biases, stem + ".biases", dtype))
            quantized.append(stem)
        else:
            writer.add(name, _cast_finite(array, name, dtype))

    embeddings = None
    tied_head = None
    separator_embedding_seen = False
    validate_tied_head = text.get("tie_word_embeddings", False) and any(
        tensor.name == "output.weight" for tensor in reader.tensors
    )
    for index, current in enumerate(readers):
        for tensor in current.tensors:
            name = tensor.name
            if name in raw_seen:
                raise ValueError(f"Duplicate GGUF tensor across inputs: {name}")
            raw_seen.add(name)
            if qwen and name.startswith(("v.", "mm.")):
                if qwen.legacy_merger and name == "v.position_embd.weight":
                    placeholder = _decode(tensor, "float32")
                    if placeholder.shape != (10, 10) or np.any(placeholder):
                        raise ValueError("Invalid legacy Qwen2-VL zero position placeholder")
                    counts["legacy_position_placeholder_ignored"] += 1
                    continue
                if name not in qwen.source_names:
                    raise ValueError(f"Unsupported Qwen2-VL vision/projector tensor: {name}")
                qwen_tensors[name] = tensor
            elif name.startswith("v."):
                if minicpm and name == "v.tok_embd_sep":
                    if separator_embedding_seen:
                        raise ValueError("Duplicate MiniCPM-V separator embedding tensor")
                    separator_rows = _decode(tensor, "float32")
                    if separator_rows.shape != (4, text["hidden_size"]):
                        raise ValueError("MiniCPM-V v.tok_embd_sep must contain four text rows")
                    separator_embedding_seen = True
                    counts["separator_embedding_ignored"] += 1
                    continue
                mapped = minicpm.map_vision(name) if minicpm else _map_vision(name)
                if mapped not in expected:
                    raise ValueError(f"Unexpected vision layer/tensor: {name}")
                if hf_fallback_vision:
                    counts["vision_gguf_ignored"] += 1
                    continue
                array = _decode(tensor, dtype)
                if mapped.endswith("class_embedding") and array.shape in (
                    (vision["hidden_size"], 1), (1, vision["hidden_size"]),
                ):
                    array = array.reshape(vision["hidden_size"])
                emit(mapped, array, "vision_gguf")
            elif name.startswith("mm."):
                mapped = minicpm.map_projector(name) if minicpm else _map_projector(name)
                emit(mapped, _decode(tensor, dtype), "projector")
            else:
                if index:
                    raise ValueError(f"Unexpected non-vision tensor in --mmproj: {name}")
                if name == "output.weight" and text.get("tie_word_embeddings", False):
                    tied_head = _decode(tensor, "float32")
                    if tuple(tied_head.shape) != expected["language_model.model.embed_tokens.weight"]:
                        raise ValueError("Tied language output.weight shape differs from token embeddings")
                    counts["language_tied_head_ignored"] += 1
                    continue
                text_arch = "qwen2" if qwen else "qwen35" if minicpm else "llama"
                mapped = "language_model." + core._map_tensor_name(name, text_arch)
                if mapped not in expected or not mapped.startswith("language_model."):
                    family = "Qwen2-VL" if qwen else "MiniCPM-V" if minicpm else "Llama"
                    raise ValueError(f"Unsupported {family} GGUF tensor: {name}")
                decode_dtype = "float32" if direct_quant else dtype
                array = _decode(tensor, decode_dtype)
                if minicpm:
                    array = core._restore_architecture_tensor(name, array, "qwen35")
                elif not qwen:
                    array = _restore_llama(name, array, text)
                emit(mapped, array, "language", tensor)
                if name == "token_embd.weight" and validate_tied_head:
                    embeddings = _decode(tensor, "float32") if dtype == "float16" else array
    if minicpm and not separator_embedding_seen:
        raise ValueError("MiniCPM-V 4.7 GGUF is missing required v.tok_embd_sep")
    if tied_head is not None and (
        embeddings is None or not np.array_equal(tied_head, embeddings)
    ):
        raise ValueError(
            "HF tie_word_embeddings=True conflicts with GGUF output.weight; "
            "supply the matching original HF config rather than discarding a distinct LM head."
        )
    if qwen:
        for name, sources, axis in qwen.vision_plan:
            category = "projector" if ".merger." in name else "vision_gguf"
            if category == "vision_gguf" and hf_fallback_vision:
                counts["vision_gguf_ignored"] += sum(src in qwen_tensors for src in sources)
                continue
            missing_sources = [src for src in sources if src not in qwen_tensors]
            if missing_sources:
                raise ValueError(f"Missing required Qwen2-VL tensors: {', '.join(missing_sources)}")
            arrays = [_decode(qwen_tensors[src], dtype) for src in sources]
            for src, array in zip(sources, arrays):
                if array.shape != qwen.source_shapes[src]:
                    raise ValueError(f"Qwen2-VL shape mismatch for {src}: {array.shape}")
            if len(arrays) == 1:
                array = arrays[0]
            elif name.endswith("patch_embed.proj.weight"):
                array = np.stack(arrays, axis=2)
            else:
                array = np.concatenate(arrays, axis=axis)
            emit(name, array, category)
    if hf_fallback_vision:
        # Pin subsequent downloads to the resolved asset revision when available.
        files, vision_index = _vision_files(
            root, source, resolved or hf_revision, offline, architecture, minicpm
        )
        found_index_names: set[str] = set()
        for file in files:
            with safe_open(str(file), framework="np") as weights:
                for name in list(weights.keys()):
                    hf_name = _hf_vision_name(name, architecture, minicpm)
                    if hf_name is None:
                        continue
                    if vision_index is not None and (
                        name not in vision_index
                        or not file.as_posix().endswith("/" + vision_index[name])
                    ):
                        raise ValueError(f"HF vision tensor/index mismatch: {name}")
                    found_index_names.add(name)
                    emit(hf_name, _read_hf_tensor(weights, file, name), "vision_hf")
        if vision_index is not None and found_index_names != set(vision_index):
            raise ValueError("HF vision safetensors index references missing tensors")
    missing = sorted(set(expected) - seen)
    if missing:
        hint = (
            " GGUF CLIP exports can omit the final layer/post-norm; use "
            "--hf-fallback-vision with the original matching HF model."
            if any(name.startswith(VISION_PREFIX) for name in missing) else ""
        )
        raise ValueError(f"Missing required VLM tensors: {', '.join(missing[:8])}.{hint}")
    writer.finish()
    for asset in ASSETS:
        if asset == "config.json":
            continue
        path = root / asset
        if path.is_file():
            shutil.copyfile(path, output_path / asset)
    config["text_config"] = text
    config["torch_dtype"] = dtype
    config["text_config"]["torch_dtype"] = dtype
    config["vision_config"]["torch_dtype"] = dtype
    if qwen:
        config.update(text)
        config["model_type"] = architecture
        config["vision_config"] = {**vision, "torch_dtype": dtype}
    elif minicpm:
        config["text_config"] = {**text, "torch_dtype": dtype}
        config["vision_config"] = {**vision, "torch_dtype": dtype}
    if direct_quant:
        if not quantized:
            raise ValueError(
                "Source-preserving conversion requires Q4_K language projections"
                if preserve_source_quant else
                "VLM direct quantization found no supported language projections"
            )
        config["quantization"] = {
            "group_size": 32 if preserve_source_quant else q_group_size,
            "bits": q_bits, "mode": q_mode,
        }
        config["vision_config"]["skip_vision"] = True
    _write_json(output_path / "config.json", config)
    _write_json(output_path / "vlm_conversion_report.json", {
        "architecture": architecture, "adapter": ARCHITECTURE_REGISTRY[architecture],
        "mode": "hybrid" if hf_fallback_vision else "direct",
        "gguf": Path(gguf_path).name,
        "mmproj": Path(mmproj).name if mmproj else None,
        "hf_source": root.name if Path(source).expanduser().is_dir() else source,
        "hf_revision": hf_revision, "hf_resolved_revision": resolved,
        "gguf_vision_layout": (
            "qwen2vl_merger" if qwen and qwen.legacy_merger else architecture
        ),
        "dtype": dtype, "tensor_counts": counts, "tensor_count": len(seen),
        "vision_patch_layout": (
            "OITHW (mlx-vlm sanitize converts to OTHWI)" if qwen
            else "OIHW (mlx-vlm CLIP sanitize converts to OHWI)"
        ),
        "quantization": {
            "method": (
                "preserve_source_q4_k" if preserve_source_quant else
                "dequantize_then_affine" if direct_quant else "none"
            ),
            "language_modules": quantized, "vision": dtype, "projector": dtype,
            "source_precision": source_precision,
        },
        "stored_tensor_count": len(writer.weight_map),
        "weight_bytes": writer.total, "shard_count": len(writer.files),
    })
