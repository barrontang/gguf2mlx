"""Synthetic LLaVA-NeXT (anyres) releases: a text GGUF, a ``mmproj``, HF assets.

The fixture exists because the interesting cases are the ones a real file cannot
cover on demand: a tower that stores fewer blocks than the HF config declares, an
export that omits ``v.post_ln``, and the fail-closed variants (SigLIP tower, wrong
projector type, mismatched grid pinpoints). Each is one keyword away here and a
multi-gigabyte search on the Hub.

``stored_blocks`` defaults to ``hf_layers - 1`` because that is what llama.cpp
actually ships: the last CLIP block only feeds the pooler, which LLaVA discards.

Planning needs the GGUF and the GGUF needs the plan, so the module breaks the
cycle with :class:`MetadataReader` -- a stand-in that answers ``get_field`` the way
``GGUFReader`` does. The same metadata builders feed both the stand-in and the
written file, so the two cannot drift.
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
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "<image>",
]

DEFAULT_DIMS = {
    "hidden": 8,
    "ffn": 12,
    "heads": 2,
    "kv_heads": 2,
    "layers": 1,
    "vocab": 16,
    "image_token_index": 15,
    "vision_hidden": 8,
    "vision_ffn": 12,
    "vision_heads": 2,
    "image_size": 8,
    "patch": 4,
    "hf_layers": 3,
}

GRID = [[8, 16], [16, 8]]

_LAYER_NAMES = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
_ROOTS = {
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.lm_head.weight": "output.weight",
    "language_model.model.norm.weight": "output_norm.weight",
}


def _dims(dims: dict[str, int] | None) -> dict[str, int]:
    return {**DEFAULT_DIMS, **(dims or {})}


def language_gguf_name(name: str) -> str:
    if name in _ROOTS:
        return _ROOTS[name]
    prefix = "language_model.model.layers.0."
    if not name.startswith(prefix):
        raise AssertionError(f"fixture cannot express {name}")
    stem, _, kind = name.removeprefix(prefix).rpartition(".")
    if stem not in _LAYER_NAMES:
        raise AssertionError(f"fixture cannot express {name}")
    return f"blk.0.{_LAYER_NAMES[stem]}.{kind}"


def llava_next_config(dims: dict[str, int] | None = None) -> dict:
    dims = _dims(dims)
    return {
        "model_type": "llava_next",
        "architectures": ["LlavaNextForConditionalGeneration"],
        "image_token_index": dims["image_token_index"],
        "image_grid_pinpoints": [list(pair) for pair in GRID],
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "projector_hidden_act": "gelu",
        "use_image_newline_parameter": True,
        "vocab_size": dims["vocab"],
        # A real llava_next config.json leaves the language dimensions to
        # dataclass defaults; the fixture keeps that shape so the "fill from the
        # GGUF" path is the one under test.
        "text_config": {
            "model_type": "llama",
            "vocab_size": dims["vocab"],
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 64,
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "num_hidden_layers": dims["hf_layers"],
            "hidden_size": dims["vision_hidden"],
            "intermediate_size": dims["vision_ffn"],
            "num_attention_heads": dims["vision_heads"],
            "image_size": dims["image_size"],
            "patch_size": dims["patch"],
            "projection_dim": dims["vision_hidden"],
            "vocab_size": 32,
        },
    }


def text_metadata(dims: dict[str, int] | None = None) -> dict[str, Any]:
    dims = _dims(dims)
    return {
        "general.architecture": "llama",
        "llama.embedding_length": dims["hidden"],
        "llama.feed_forward_length": dims["ffn"],
        "llama.block_count": dims["layers"],
        "llama.attention.head_count": dims["heads"],
        "llama.attention.head_count_kv": dims["kv_heads"],
        "llama.vocab_size": dims["vocab"],
        "llama.rope.dimension_count": dims["hidden"] // dims["heads"],
        "llama.context_length": 64,
        "llama.rope.freq_base": 10000.0,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
    }


def clip_metadata(
    dims: dict[str, int] | None = None,
    *,
    stored_blocks: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dims = _dims(dims)
    stored = dims["hf_layers"] - 1 if stored_blocks is None else stored_blocks
    metadata: dict[str, Any] = {
        "general.architecture": "clip",
        "clip.has_vision_encoder": True,
        "clip.has_llava_projector": True,
        "clip.projector_type": "mlp",
        "clip.use_gelu": False,
        "clip.vision.image_size": dims["image_size"],
        "clip.vision.patch_size": dims["patch"],
        "clip.vision.embedding_length": dims["vision_hidden"],
        "clip.vision.feed_forward_length": dims["vision_ffn"],
        "clip.vision.projection_dim": dims["vision_hidden"],
        "clip.vision.attention.head_count": dims["vision_heads"],
        "clip.vision.attention.layer_norm_epsilon": 1e-5,
        "clip.vision.block_count": stored,
        "clip.vision.image_grid_pinpoints": [v for pair in GRID for v in pair],
        "clip.vision.image_aspect_ratio": "anyres",
        "clip.vision.mm_patch_merge_type": "spatial_unpad",
        "clip.vision.mm_projector_type": "mlp2x_gelu",
        "clip.vision.image_crop_resolution": dims["image_size"],
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
    }
    for key, value in (overrides or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


class _Field:
    def __init__(self, value: Any) -> None:
        self._value = value

    def contents(self) -> Any:
        return self._value


class MetadataReader:
    """The ``GGUFReader`` surface the adapters use, backed by a dict.

    ``tensors`` entries are either a name or a ``(name, ne_shape)`` pair. The shape
    is only needed by readers that resolve a tensor role from its dimensions -- the
    CLIP MLP does, because the two exporters swap ``ffn_up``/``ffn_down``.
    """

    def __init__(self, metadata: dict[str, Any], tensors: tuple[Any, ...] = ()) -> None:
        self._metadata = metadata
        self.tensors = [
            SimpleNamespace(name=entry[0], shape=entry[1]) if isinstance(entry, tuple)
            else SimpleNamespace(name=entry)
            for entry in tensors
        ]

    def get_field(self, key: str) -> _Field | None:
        if key not in self._metadata:
            return None
        return _Field(self._metadata[key])


def plan_sources(
    config: dict,
    *,
    dims: dict[str, int] | None = None,
    stored_blocks: int | None = None,
    clip_overrides: dict[str, Any] | None = None,
    include_post_norm: bool = False,
    include_newline: bool = True,
    rng: np.random.Generator | None = None,
    ffn_convention: str = "mtmd",
):
    """Configure an adapter against stand-in metadata and return its sources.

    ``include_post_norm`` is the only thing that has to be declared up front: the
    adapter decides between planning ``v.post_ln`` and filling the identity by
    asking the reader whether those tensors exist, exactly as it does against a
    real ``mmproj``.

    ``ffn_convention`` picks which ``ffn_*`` name carries the CLIP expansion. The
    stand-in cannot know either, so it declares the shapes that convention implies
    and lets the adapter resolve the roles from them, as it does on a real file.
    """
    from gguf2mlx.vision_llava import LlavaNextAdapter

    dims = _dims(dims)
    adapter = LlavaNextAdapter(config)
    adapter.validate_text_metadata(MetadataReader(text_metadata(dims)))
    metadata = clip_metadata(dims, stored_blocks=stored_blocks, overrides=clip_overrides)
    width, feed_forward = dims["vision_hidden"], dims["vision_ffn"]
    expansion, output = (
        ("ffn_up", "ffn_down") if ffn_convention == "mtmd"
        else ("ffn_down", "ffn_up")
    )
    tensors: list[Any] = [
        (f"v.blk.0.{expansion}.weight", (width, feed_forward)),
        (f"v.blk.0.{output}.weight", (feed_forward, width)),
    ]
    if include_post_norm:
        tensors += [("v.post_ln.weight", (width,)), ("v.post_ln.bias", (width,))]
    adapter.configure_layout([MetadataReader(metadata, tuple(tensors))])

    sources = vision_plan_arrays(adapter, rng or np.random.default_rng(7))
    if include_post_norm:
        sources["v.post_ln.weight"] = np.full(width, 1.5, np.float32)
        sources["v.post_ln.bias"] = np.full(width, 0.25, np.float32)
    if not include_newline:
        sources.pop("model.image_newline", None)
    return adapter, sources


def vision_plan_arrays(adapter: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build the expected outputs, then split them back into GGUF sources."""
    expected: dict[str, np.ndarray] = {}
    for name, shape in adapter.expected.items():
        expected[name] = rng.normal(0, 0.08, shape).astype(np.float32)
    sources: dict[str, np.ndarray] = {}
    for target, inputs, _axis in adapter.vision_plan:
        array = expected[target]
        (source,) = inputs
        sources[source] = array.reshape(adapter.source_shapes[source])
    for name, array in expected.items():
        if name.startswith("language_model."):
            sources[language_gguf_name(name)] = array
    return sources


def write_llava_next_fixture(
    tmp_path: Path,
    config: dict,
    sources: dict[str, np.ndarray],
    *,
    dims: dict[str, int] | None = None,
    stored_blocks: int | None = None,
    clip_overrides: dict[str, Any] | None = None,
    drop_sources: tuple[str, ...] = (),
    gguf_tokens: list[str] | None = None,
) -> SimpleNamespace:
    dims = _dims(dims)
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "LlavaNextImageProcessor",
        "processor_class": "LlavaNextProcessor",
        "aspect_ratio_setting": "anyres",
        "crop_size": {"height": dims["image_size"], "width": dims["image_size"]},
        "image_grid_pinpoints": [list(pair) for pair in GRID],
        "size": {"shortest_edge": dims["image_size"]},
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }))
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {token: i for i, token in enumerate(TOKENS)}},
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ msgs }}"}))
    (root / "processor_config.json").write_text(json.dumps({"chat_template": "{{ msgs }}"}))
    (root / "special_tokens_map.json").write_text(json.dumps({}))

    main, mmproj = tmp_path / "language.gguf", tmp_path / "mmproj.gguf"
    text = text_metadata(dims)
    writer = GGUFWriter(str(main), "llama")
    for key, value in text.items():
        if key == "general.architecture":
            continue
        if isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_uint32(key, value)
    writer.add_token_list(gguf_tokens if gguf_tokens is not None else TOKENS)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 0)
    for name, array in sources.items():
        if not _is_vision_source(name):
            writer.add_tensor(name, np.ascontiguousarray(array))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    writer = GGUFWriter(str(mmproj), "clip")
    for key, value in clip_metadata(
        dims, stored_blocks=stored_blocks, overrides=clip_overrides
    ).items():
        if key == "general.architecture":
            continue
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list) and value and isinstance(value[0], float):
            writer.add_array(key, [float(v) for v in value])
        elif isinstance(value, list):
            writer.add_array(key, [int(v) for v in value])
        else:
            writer.add_uint32(key, value)
    for name, array in sources.items():
        if _is_vision_source(name) and name not in drop_sources:
            writer.add_tensor(name, np.ascontiguousarray(array))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return SimpleNamespace(
        root=root, main=main, mmproj=mmproj, config=config,
        sources=sources, output=tmp_path / "out",
    )


def _is_vision_source(name: str) -> bool:
    return name.startswith(("v.", "mm.", "model.image_newline"))
