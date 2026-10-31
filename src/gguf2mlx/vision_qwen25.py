"""Qwen2.5-VL's split GGUF ViT -> mlx-vlm ``vision_tower`` contract.

llama.cpp exports Qwen2-VL and Qwen2.5-VL through one graph builder, so the GGUF
tensor names are identical and a name-only check cannot tell them apart. What
differs is inside the weights, and all of it is visible from the tensors:

* the ViT norms are RMSNorm (``ln1.weight`` with no ``ln1.bias``), where Qwen2-VL
  uses LayerNorm;
* the ViT MLP is gated (``ffn_gate``/``ffn_up``/``ffn_down``) and SiLU-activated,
  where Qwen2-VL has a two-layer ``fc1``/``fc2`` MLP;
* the merger norm is RMSNorm.

Treating the two as aliases therefore produces a model that loads and emits
finite logits while computing the wrong function, which is why this adapter
validates the norms, the gated MLP and the projector type before it emits a
single tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import gguf2mlx as core
from . import vlm


class Qwen25VisionAdapter(vlm.MropePlanAdapter):
    name = "Qwen2.5-VL"
    text_arch = "qwen2"
    gguf_text_architectures = ("qwen2vl",)
    processor_type = "Qwen2VLImageProcessor"
    nested_text_config = False

    def __init__(self, config: dict):
        if config.get("architectures") not in (None, ["Qwen2_5_VLForConditionalGeneration"]):
            raise ValueError("Unsupported Qwen2.5-VL HF architecture")
        raw_text = config.get("text_config", config)
        raw_vision = config.get("vision_config")
        if not isinstance(raw_text, dict) or not isinstance(raw_vision, dict):
            raise ValueError("Qwen2.5-VL requires text and vision configuration objects")  # noqa: TRY004
        self.text = {
            key: value for key, value in raw_text.items()
            if key not in ("vision_config", "text_config", "architectures")
        }
        self.vision = dict(raw_vision)
        text, vision = self.text, self.vision
        if text.get("model_type") != "qwen2_5_vl":
            raise ValueError("Qwen2.5-VL requires HF model_type='qwen2_5_vl'")
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size"):
            vlm._positive(text, key)
        hidden, heads, kv = (
            text["hidden_size"], text["num_attention_heads"], text["num_key_value_heads"]
        )
        if hidden % heads or heads % kv or (hidden // heads) % 2:
            raise ValueError("Invalid Qwen2.5-VL language head dimensions")
        head_dim = hidden // heads
        if text.get("head_dim", head_dim) != head_dim:
            raise ValueError("Nonstandard Qwen2.5-VL head_dim is unsupported")
        if text.get("hidden_act", "silu") != "silu" or text.get("mlp_bias", False):
            raise ValueError("Qwen2.5-VL requires an unbiased SiLU gated language MLP")
        if text.get("attention_bias", True) is not True:
            raise ValueError("Qwen2.5-VL requires biased Q/K/V projections")
        if text.get("use_sliding_window", False) or text.get("rope_traditional", False):
            raise ValueError("Sliding-window/traditional RoPE Qwen2.5-VL is unsupported")
        if not isinstance(text.get("tie_word_embeddings", False), bool):
            raise ValueError("Qwen2.5-VL tie_word_embeddings must be boolean")  # noqa: TRY004
        for key, default in (("rope_theta", 1e6), ("rms_norm_eps", 1e-6)):
            value = text.setdefault(key, default)
            if (
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Invalid Qwen2.5-VL {key}")
        rope = text.get("rope_scaling")
        if (
            not isinstance(rope, dict)
            or set(rope) - {"type", "rope_type", "mrope_section"}
            or rope.get("type", rope.get("rope_type")) not in ("mrope", "default")
            or rope.get("rope_type", "default") not in ("mrope", "default")
        ):
            raise ValueError("Qwen2.5-VL requires explicit mrope_section and default/mrope scaling")
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list) or len(sections) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in sections)
            or sum(sections) * 2 != head_dim
        ):
            raise ValueError("Qwen2.5-VL mrope_section must partition the rotary head dimension")
        text["rope_scaling"] = {"type": "mrope", "mrope_section": list(sections)}

        if vision.get("model_type", "qwen2_5_vl") != "qwen2_5_vl":
            raise ValueError("Only the Qwen2.5-VL ViT is supported by this adapter")
        vision["model_type"] = "qwen2_5_vl"
        for key in ("depth", "hidden_size", "intermediate_size", "num_heads", "patch_size",
                    "spatial_merge_size", "temporal_patch_size", "out_hidden_size",
                    "window_size"):
            vlm._positive(vision, key)
        if (
            vision["temporal_patch_size"] != 2 or vision["spatial_merge_size"] != 2
            or vision["hidden_size"] % vision["num_heads"]
            or (vision["hidden_size"] // vision["num_heads"]) % 4
            or vision.get("in_chans", 3) != 3 or vision.get("in_channels", 3) != 3
            or vision.get("spatial_patch_size", vision["patch_size"]) != vision["patch_size"]
        ):
            raise ValueError("Unsupported Qwen2.5-VL patch/merge/channel/head dimensions")
        if vision["out_hidden_size"] != hidden:
            raise ValueError(
                "Qwen2.5-VL vision out_hidden_size must equal the text hidden_size"
            )
        # Qwen2.5-VL activates its ViT MLP with SiLU; GELU here would be Qwen3-VL.
        if vision.get("hidden_act", "silu") != "silu":
            raise ValueError("Qwen2.5-VL requires the SiLU vision activation")
        if vision.get("layer_norm_eps", 1e-6) != 1e-6:
            raise ValueError("Qwen2.5-VL vision layer_norm_eps must be 1e-6")
        indexes = vision.get("fullatt_block_indexes")
        if (
            not isinstance(indexes, list) or not indexes
            or any(isinstance(x, bool) or not isinstance(x, int) for x in indexes)
            or any(not 0 <= x < vision["depth"] for x in indexes)
        ):
            raise ValueError("Qwen2.5-VL fullatt_block_indexes must index existing ViT blocks")
        vision["in_channels"] = 3
        vision["spatial_patch_size"] = vision["patch_size"]
        vision["fullatt_block_indexes"] = list(indexes)

        self.expected: dict[str, tuple[int, ...]] = vlm._language_shapes(
            text, attention_bias=True
        )
        self.vision_plan: list[tuple[str, tuple[str, ...], int]] = []
        self.source_shapes: dict[str, tuple[int, ...]] = {}
        self._build_vision_plan()
        self.source_names = set(self.source_shapes)

    def _add(
        self, target: str, shape: tuple[int, ...], sources: tuple[str, ...],
        source_shape: tuple[int, ...] | None = None, axis: int = 0,
    ) -> None:
        target = "vision_tower." + target
        self.expected[target] = shape
        self.vision_plan.append((target, sources, axis))
        for source in sources:
            self.source_shapes[source] = source_shape or shape

    def _build_vision_plan(self) -> None:
        vision = self.vision
        width = vision["hidden_size"]
        hidden = vision["intermediate_size"]
        patch = vision["patch_size"]
        merged = width * vision["spatial_merge_size"] ** 2
        out = vision["out_hidden_size"]
        self._add(
            "patch_embed.proj.weight", (width, 3, 2, patch, patch),
            ("v.patch_embd.weight", "v.patch_embd.weight.1"), (width, 3, patch, patch),
            axis=2,
        )
        for i in range(vision["depth"]):
            # RMSNorm: a gain with no bias. Declaring a bias here would leave 65
            # sources permanently missing on every real export.
            for norm in ("norm1", "norm2"):
                self._add(
                    f"blocks.{i}.{norm}.weight", (width,),
                    (f"v.blk.{i}.ln{norm[-1]}.weight",),
                )
            for suffix in ("weight", "bias"):
                self._add(
                    f"blocks.{i}.attn.qkv.{suffix}",
                    (3 * width, width) if suffix == "weight" else (3 * width,),
                    tuple(f"v.blk.{i}.attn_{part}.{suffix}" for part in ("q", "k", "v")),
                    (width, width) if suffix == "weight" else (width,),
                )
                self._add(
                    f"blocks.{i}.attn.proj.{suffix}",
                    (width, width) if suffix == "weight" else (width,),
                    (f"v.blk.{i}.attn_out.{suffix}",),
                )
                for source, target, shape in (
                    ("ffn_gate", "gate_proj", (hidden, width)),
                    ("ffn_up", "up_proj", (hidden, width)),
                    ("ffn_down", "down_proj", (width, hidden)),
                ):
                    self._add(
                        f"blocks.{i}.mlp.{target}.{suffix}",
                        shape if suffix == "weight" else (shape[0],),
                        (f"v.blk.{i}.{source}.{suffix}",),
                    )
        for suffix in ("weight", "bias"):
            for index, shape in ((0, (merged, merged)), (2, (out, merged))):
                self._add(
                    f"merger.mlp.{index}.{suffix}",
                    shape if suffix == "weight" else (shape[0],),
                    (f"mm.{index}.{suffix}",),
                )
        # The merger norm is RMSNorm too, so it has no bias source.
        self._add("merger.ln_q.weight", (width,), ("v.post_ln.weight",))

    def validate_vision_metadata(self, reader: Any) -> None:
        projector = core.get_metadata_str(reader, "clip.projector_type")
        if projector not in (None, "qwen2.5vl_merger"):
            raise ValueError(
                f"Unsupported Qwen2.5-VL projector type {projector!r}; "
                "the qwen2vl and qwen3vl mergers are different architectures"
            )
        for src, dst in {
            "embedding_length": "hidden_size", "block_count": "depth",
            "attention.head_count": "num_heads", "patch_size": "patch_size",
            "spatial_merge_size": "spatial_merge_size",
        }.items():
            value = core.get_metadata_int(reader, f"clip.vision.{src}")
            if value is not None and value != self.vision[dst]:
                raise ValueError(f"GGUF Qwen2.5-VL {src} is incompatible with HF vision {dst}")
        feed_forward = core.get_metadata_int(reader, "clip.vision.feed_forward_length")
        if feed_forward is not None and feed_forward != self.vision["intermediate_size"]:
            raise ValueError(
                "GGUF Qwen2.5-VL feed_forward_length is incompatible with HF vision"
            )
        projection = core.get_metadata_int(reader, "clip.vision.projection_dim")
        if projection is not None and projection != self.vision["out_hidden_size"]:
            raise ValueError("GGUF Qwen2.5-VL projection_dim is incompatible with HF vision")
        epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
        if epsilon is not None and not np.isclose(epsilon, 1e-6, rtol=1e-5, atol=0):
            raise ValueError("GGUF Qwen2.5-VL vision layer norm epsilon must be 1e-6")
        if core.get_metadata_bool(reader, "clip.use_gelu") is True:
            raise ValueError("Qwen2.5-VL uses a SiLU ViT MLP, not GELU")
        if any(core.get_metadata_bool(reader, key) is True for key in (
            "clip.has_siglip", "clip.has_audio_encoder",
        )):
            raise ValueError("Unsupported Qwen2.5-VL vision encoder")
