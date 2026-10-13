"""Qwen2-VL's split GGUF ViT -> HF Conv3D/QKV/patch-merger contract.

Qwen2.5-VL and Qwen3.x use different vision blocks and are not aliases.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import gguf2mlx as core
from . import vlm


class Qwen2VisionAdapter:
    def __init__(self, config: dict):
        if config.get("architectures") not in (None, ["Qwen2VLForConditionalGeneration"]):
            raise ValueError("Unsupported Qwen2-VL HF architecture")
        raw_text = config.get("text_config", config)
        raw_vision = config.get("vision_config")
        if not isinstance(raw_text, dict) or not isinstance(raw_vision, dict):
            raise ValueError("Qwen2-VL requires text and vision configuration objects")  # noqa: TRY004
        self.text = {
            key: value for key, value in raw_text.items()
            if key not in ("vision_config", "text_config", "architectures")
        }
        self.vision = dict(raw_vision)
        text, vision = self.text, self.vision
        if text.get("model_type") not in ("qwen2_vl", "qwen2_vl_text"):
            raise ValueError("Qwen2-VL requires the matching Qwen2-VL language architecture")
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size"):
            vlm._positive(text, key)
        hidden, heads, kv = (
            text["hidden_size"], text["num_attention_heads"], text["num_key_value_heads"]
        )
        if hidden % heads or heads % kv or (hidden // heads) % 2:
            raise ValueError("Invalid Qwen2-VL language head dimensions")
        if text.get("head_dim", hidden // heads) != hidden // heads:
            raise ValueError("Nonstandard Qwen2-VL head_dim is unsupported")
        if text.get("hidden_act", "silu") != "silu" or text.get("mlp_bias", False):
            raise ValueError("Qwen2-VL requires an unbiased SiLU gated language MLP")
        if text.get("attention_bias", True) is not True:
            raise ValueError("Qwen2-VL requires biased Q/K/V projections")
        if text.get("use_sliding_window", False) or text.get("rope_traditional", False):
            raise ValueError("Sliding-window/traditional RoPE Qwen2-VL is unsupported")
        if not isinstance(text.get("tie_word_embeddings", False), bool):
            raise ValueError("Qwen2-VL tie_word_embeddings must be boolean")  # noqa: TRY004
        for key, default in (("rope_theta", 1e6), ("rms_norm_eps", 1e-6)):
            value = text.setdefault(key, default)
            if (
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Invalid Qwen2-VL {key}")
        rope = text.get("rope_scaling")
        if (
            not isinstance(rope, dict)
            or set(rope) - {"type", "rope_type", "mrope_section"}
            or rope.get("type", rope.get("rope_type")) not in ("mrope", "default")
            or rope.get("rope_type", "default") not in ("mrope", "default")
        ):
            raise ValueError("Qwen2-VL requires explicit mrope_section and default/mrope scaling")
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list) or len(sections) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in sections)
            or sum(sections) * 2 != hidden // heads
        ):
            raise ValueError("Qwen2-VL mrope_section must partition the rotary head dimension")
        text["rope_scaling"] = {"type": "mrope", "mrope_section": list(sections)}
        if vision.get("model_type", "qwen2_vl") != "qwen2_vl":
            raise ValueError("Only the Qwen2-VL ViT is supported, not Qwen2.5/3.x")
        vision["model_type"] = "qwen2_vl"
        for key in ("depth", "embed_dim", "hidden_size", "num_heads", "patch_size",
                    "spatial_merge_size", "temporal_patch_size"):
            vlm._positive(vision, key)
        if (
            vision["temporal_patch_size"] != 2 or vision["spatial_merge_size"] != 2
            or vision["hidden_size"] != hidden
            or vision["embed_dim"] % vision["num_heads"]
            or (vision["embed_dim"] // vision["num_heads"]) % 4
            or vision.get("in_chans", 3) != 3 or vision.get("in_channels", 3) != 3
            or vision.get("spatial_patch_size", vision["patch_size"]) != vision["patch_size"]
        ):
            raise ValueError("Unsupported Qwen2-VL patch/merge/channel/head dimensions")
        vision["in_channels"] = 3
        vision["spatial_patch_size"] = vision["patch_size"]
        if vision.get("hidden_act", "quick_gelu") != "quick_gelu":
            raise ValueError("Qwen2-VL requires quick_gelu vision activation")
        if vision.get("layer_norm_eps", 1e-6) != 1e-6:
            raise ValueError("Qwen2-VL vision layer_norm_eps must be 1e-6")
        ratio = vision.get("mlp_ratio", 4)
        if (
            isinstance(ratio, bool) or not isinstance(ratio, (float, int))
            or not np.isfinite(ratio) or ratio <= 0
            or not float(vision["embed_dim"] * ratio).is_integer()
        ):
            raise ValueError("Invalid Qwen2-VL vision mlp_ratio")
        vision["mlp_ratio"] = ratio
        self.expected = vlm._language_shapes(text, attention_bias=True)
        self.legacy_merger = False
        self.vision_plan: list[tuple[str, tuple[str, ...], int]] = []
        self.source_shapes: dict[str, tuple[int, ...]] = {}
        self._build_vision_plan()
        self.source_names = set(self.source_shapes)

    def configure_layout(self, readers: list[Any]) -> None:
        types = {
            core.get_metadata_str(reader, "clip.projector_type") for reader in readers
        } - {None}
        if len(types) > 1:
            raise ValueError("Conflicting Qwen2-VL projector layouts across GGUF inputs")
        if types == {"qwen2vl_merger"}:
            self.legacy_merger = True
            # The historical llava exporter used down=fc1, up=fc2.
            def legacy_name(name: str) -> str:
                if ".ffn_up." in name:
                    return name.replace(".ffn_up.", ".ffn_down.")
                return name.replace(".ffn_down.", ".ffn_up.")

            self.source_shapes = {legacy_name(k): v for k, v in self.source_shapes.items()}
            self.vision_plan = [
                (target, tuple(legacy_name(src) for src in sources), axis)
                for target, sources, axis in self.vision_plan
            ]
            self.source_names = set(self.source_shapes)

    def _add(
        self, target: str, shape: tuple[int, ...], sources: tuple[str, ...],
        source_shape: tuple[int, ...] | None = None,
    ) -> None:
        target = "vision_tower." + target
        self.expected[target] = shape
        self.vision_plan.append((target, sources, 0))
        for source in sources:
            self.source_shapes[source] = source_shape or shape

    def _build_vision_plan(self) -> None:
        v, p = self.vision["embed_dim"], self.vision["patch_size"]
        f = int(v * self.vision["mlp_ratio"])
        merged = v * self.vision["spatial_merge_size"] ** 2
        self._add(
            "patch_embed.proj.weight", (v, 3, 2, p, p),
            ("v.patch_embd.weight", "v.patch_embd.weight.1"), (v, 3, p, p),
        )
        for i in range(self.vision["depth"]):
            for suffix in ("weight", "bias"):
                self._add(
                    f"blocks.{i}.attn.qkv.{suffix}",
                    (3 * v, v) if suffix == "weight" else (3 * v,),
                    tuple(f"v.blk.{i}.attn_{part}.{suffix}" for part in ("q", "k", "v")),
                    (v, v) if suffix == "weight" else (v,),
                )
            for src, dst, shape in (
                ("attn_out", "attn.proj", (v, v)), ("ln1", "norm1", (v,)),
                ("ln2", "norm2", (v,)), ("ffn_up", "mlp.fc1", (f, v)),
                ("ffn_down", "mlp.fc2", (v, f)),
            ):
                for suffix in ("weight", "bias"):
                    self._add(
                        f"blocks.{i}.{dst}.{suffix}",
                        shape if suffix == "weight" else (shape[0],),
                        (f"v.blk.{i}.{src}.{suffix}",),
                    )
        for suffix in ("weight", "bias"):
            self._add(f"merger.ln_q.{suffix}", (v,), (f"v.post_ln.{suffix}",))
            for i, shape in ((0, (merged, merged)), (2, (self.text["hidden_size"], merged))):
                self._add(
                    f"merger.mlp.{i}.{suffix}",
                    shape if suffix == "weight" else (shape[0],),
                    (f"mm.{i}.{suffix}",),
                )

    def validate_text_metadata(self, reader: Any) -> None:
        if core.get_metadata_str(reader, "general.architecture") != "qwen2vl":
            raise ValueError("Qwen2-VL requires GGUF architecture='qwen2vl'")
        for src, dst in {
            "embedding_length": "hidden_size", "feed_forward_length": "intermediate_size",
            "block_count": "num_hidden_layers", "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
        }.items():
            if core.get_metadata_int(reader, f"qwen2vl.{src}") != self.text[dst]:
                raise ValueError(f"GGUF qwen2vl.{src} missing or incompatible with HF {dst}")
        for src, expected in {
            "vocab_size": self.text["vocab_size"],
            "rope.dimension_count": self.text["hidden_size"] // self.text["num_attention_heads"],
        }.items():
            value = core.get_metadata_int(reader, f"qwen2vl.{src}")
            if value is not None and value != expected:
                raise ValueError(f"GGUF qwen2vl.{src} is incompatible with HF config")
        for src, dst in {
            "rope.freq_base": "rope_theta", "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        }.items():
            float_value = core.get_metadata_float(reader, f"qwen2vl.{src}")
            if float_value is not None and not np.isclose(
                float_value, self.text[dst], rtol=1e-5, atol=0
            ):
                raise ValueError(f"GGUF qwen2vl.{src} is incompatible with HF {dst}")
        sections = core.get_metadata_array_int(reader, "qwen2vl.rope.dimension_sections")
        expected_sections = self.text["rope_scaling"]["mrope_section"]
        if sections not in (expected_sections, expected_sections + [0]):
            raise ValueError("GGUF Qwen2-VL mrope dimension_sections missing or incompatible")
        scaling = core.get_metadata_str(reader, "qwen2vl.rope.scaling.type")
        if scaling not in (None, "none", "mrope"):
            raise ValueError("Unsupported GGUF Qwen2-VL RoPE scaling")

    def validate_vision_metadata(self, reader: Any) -> None:
        projector = core.get_metadata_str(reader, "clip.projector_type")
        if projector not in (None, "qwen2vl", "qwen2vl_merger"):
            raise ValueError(f"Unsupported Qwen2-VL projector type {projector!r}")
        for src, dst in {
            "embedding_length": "embed_dim", "block_count": "depth",
            "attention.head_count": "num_heads", "patch_size": "patch_size",
        }.items():
            value = core.get_metadata_int(reader, f"clip.vision.{src}")
            if value is not None and value != self.vision[dst]:
                raise ValueError(f"GGUF Qwen2-VL {src} is incompatible with HF vision {dst}")
        feed_forward = core.get_metadata_int(reader, "clip.vision.feed_forward_length")
        expected_ff = 0 if self.legacy_merger else self.vision["hidden_size"]
        if feed_forward is not None and feed_forward != expected_ff:
            raise ValueError("GGUF Qwen2-VL feed_forward_length is incompatible with exporter layout")
        projection = core.get_metadata_int(reader, "clip.vision.projection_dim")
        if projection is not None and projection != self.vision["hidden_size"]:
            raise ValueError("GGUF Qwen2-VL projection_dim is incompatible with HF vision")
        epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
        if epsilon is not None and not np.isclose(epsilon, 1e-6, rtol=1e-5, atol=0):
            raise ValueError("GGUF Qwen2-VL vision layer norm epsilon must be 1e-6")
        if any(core.get_metadata_bool(reader, key) is True for key in (
            "clip.has_siglip", "clip.use_silu", "clip.use_gelu", "clip.has_audio_encoder",
        )):
            raise ValueError("Unsupported Qwen2-VL vision activation/encoder")
