"""Qwen3-VL's split GGUF ViT -> mlx-vlm ``vision_tower`` contract.

Qwen3-VL is not a scaled Qwen2.5-VL ViT. It adds four things that change the
tensor set rather than just the values, and each one is checked here instead of
assumed:

* a fused ``attn_qkv`` projection with a bias, where Qwen2.5-VL exports separate
  ``attn_q``/``attn_k``/``attn_v`` tensors;
* a learned, bilinearly resized ``pos_embed``, which has no Qwen2.x equivalent;
* a convolutional patch bias, because mlx-vlm builds ``Conv3d(bias=True)`` here;
* deepstack mergers on selected layers. The GGUF names those by *layer* index
  (``v.deepstack.5``), while the target is a compact list ordered by occurrence
  (``deepstack_merger_list.0``), so the mapping is by rank, not by name.

The last one is why a name-only mapping is unsafe: ``v.deepstack.11`` would land
on ``deepstack_merger_list.11`` and fail only at load time on a 24-layer model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import gguf2mlx as core
from . import vlm


class Qwen3VisionAdapter(vlm.MropePlanAdapter):
    name = "Qwen3-VL"
    text_arch = "qwen3"
    gguf_text_architectures = ("qwen3vl",)
    processor_type = "Qwen2VLImageProcessor"
    nested_text_config = True
    qk_norm = True
    # Qwen3-VL's processor carries the merge size but not a temporal patch size
    # in every release, so only the geometry mlx-vlm actually reads is asserted.
    processor_geometry = (("patch_size", "patch_size"), ("merge_size", "spatial_merge_size"))

    def __init__(self, config: dict):
        if config.get("architectures") not in (None, ["Qwen3VLForConditionalGeneration"]):
            raise ValueError("Unsupported Qwen3-VL HF architecture")
        raw_text, raw_vision = config.get("text_config"), config.get("vision_config")
        if not isinstance(raw_text, dict) or not isinstance(raw_vision, dict):
            raise ValueError("Qwen3-VL requires text and vision configuration objects")  # noqa: TRY004
        self.text = dict(raw_text)
        self.vision = dict(raw_vision)
        text, vision = self.text, self.vision
        if not str(text.get("model_type", "")).startswith("qwen3_vl"):
            raise ValueError("Qwen3-VL requires HF text model_type='qwen3_vl_text'")
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size"):
            vlm._positive(text, key)
        hidden, heads, kv = (
            text["hidden_size"], text["num_attention_heads"], text["num_key_value_heads"]
        )
        if hidden % heads or heads % kv:
            raise ValueError("Invalid Qwen3-VL language head dimensions")
        head_dim = text.setdefault("head_dim", hidden // heads)
        if (
            isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0
            or head_dim % 2
        ):
            raise ValueError("Invalid Qwen3-VL head_dim")
        if text.get("hidden_act", "silu") != "silu" or text.get("mlp_bias", False):
            raise ValueError("Qwen3-VL requires an unbiased SiLU gated language MLP")
        # Qwen3 dropped Qwen2's Q/K/V biases; a biased checkpoint is a different model.
        if text.get("attention_bias", False) is not False:
            raise ValueError("Qwen3-VL requires unbiased Q/K/V projections")
        if not isinstance(text.get("tie_word_embeddings", False), bool):
            raise ValueError("Qwen3-VL tie_word_embeddings must be boolean")  # noqa: TRY004
        for key, default in (("rope_theta", 5e6), ("rms_norm_eps", 1e-6)):
            value = text.setdefault(key, default)
            if (
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Invalid Qwen3-VL {key}")
        rope = text.get("rope_scaling")
        if (
            not isinstance(rope, dict)
            or set(rope) - {"type", "rope_type", "mrope_section", "mrope_interleaved"}
            or rope.get("type", rope.get("rope_type", "default")) not in ("mrope", "default")
        ):
            raise ValueError("Qwen3-VL requires explicit mrope_section and default/mrope scaling")
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list) or len(sections) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in sections)
            or sum(sections) * 2 != head_dim
        ):
            raise ValueError("Qwen3-VL mrope_section must partition the rotary head dimension")
        if rope.get("mrope_interleaved", True) is not True:
            raise ValueError(
                "Interleaved mRoPE is required by mlx-vlm's Qwen3-VL rotary embedding; "
                "refusing to emit a model that would rotate the wrong axes"
            )
        text["rope_scaling"] = {
            **{key: rope[key] for key in ("type", "rope_type") if key in rope},
            "mrope_section": list(sections), "mrope_interleaved": True,
        }

        if vision.get("model_type") != "qwen3_vl":
            raise ValueError("Only the Qwen3-VL ViT is supported by this adapter")
        for key in ("depth", "hidden_size", "intermediate_size", "num_heads", "patch_size",
                    "spatial_merge_size", "temporal_patch_size", "out_hidden_size",
                    "num_position_embeddings"):
            vlm._positive(vision, key)
        positions = vision["num_position_embeddings"]
        if int(positions**0.5) ** 2 != positions:
            raise ValueError(
                "Qwen3-VL num_position_embeddings must be a perfect square: "
                "mlx-vlm interpolates the learned table as a square grid"
            )
        if (
            vision["temporal_patch_size"] != 2 or vision["spatial_merge_size"] != 2
            or vision["hidden_size"] % vision["num_heads"]
            or (vision["hidden_size"] // vision["num_heads"]) % 4
            or vision.get("in_chans", 3) != 3 or vision.get("in_channels", 3) != 3
        ):
            raise ValueError("Unsupported Qwen3-VL patch/merge/channel/head dimensions")
        if vision["out_hidden_size"] != hidden:
            raise ValueError("Qwen3-VL vision out_hidden_size must equal the text hidden_size")
        if vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Qwen3-VL requires the tanh-approximate GELU vision activation")
        if vision.get("layer_norm_eps", 1e-6) != 1e-6:
            raise ValueError("Qwen3-VL vision layer_norm_eps must be 1e-6")
        indexes = vision.get("deepstack_visual_indexes", [])
        if (
            not isinstance(indexes, list)
            or any(isinstance(x, bool) or not isinstance(x, int) for x in indexes)
            or any(not 0 <= x < vision["depth"] for x in indexes)
            or len(set(indexes)) != len(indexes)
        ):
            raise ValueError("Qwen3-VL deepstack_visual_indexes must index distinct ViT blocks")
        vision["in_channels"] = 3
        vision["deepstack_visual_indexes"] = list(indexes)

        self.expected: dict[str, tuple[int, ...]] = vlm._language_shapes(
            text, attention_bias=False, qk_norm=True
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
        self._add(
            "patch_embed.proj.bias", (width,), ("v.patch_embd.bias",),
        )
        self._add(
            "pos_embed.weight", (vision["num_position_embeddings"], width),
            ("v.position_embd.weight",),
        )
        for i in range(vision["depth"]):
            for suffix in ("weight", "bias"):
                for norm in ("norm1", "norm2"):
                    self._add(
                        f"blocks.{i}.{norm}.{suffix}", (width,),
                        (f"v.blk.{i}.ln{norm[-1]}.{suffix}",),
                    )
                self._add(
                    f"blocks.{i}.attn.qkv.{suffix}",
                    (3 * width, width) if suffix == "weight" else (3 * width,),
                    (f"v.blk.{i}.attn_qkv.{suffix}",),
                )
                self._add(
                    f"blocks.{i}.attn.proj.{suffix}",
                    (width, width) if suffix == "weight" else (width,),
                    (f"v.blk.{i}.attn_out.{suffix}",),
                )
                for source, target, shape in (
                    ("ffn_up", "linear_fc1", (hidden, width)),
                    ("ffn_down", "linear_fc2", (width, hidden)),
                ):
                    self._add(
                        f"blocks.{i}.mlp.{target}.{suffix}",
                        shape if suffix == "weight" else (shape[0],),
                        (f"v.blk.{i}.{source}.{suffix}",),
                    )
        for suffix in ("weight", "bias"):
            self._add(f"merger.norm.{suffix}", (width,), (f"v.post_ln.{suffix}",))
            for index, shape in ((0, (merged, merged)), (2, (out, merged))):
                self._add(
                    f"merger.linear_fc{index // 2 + 1}.{suffix}",
                    shape if suffix == "weight" else (shape[0],),
                    (f"mm.{index}.{suffix}",),
                )
        # deepstack mergers are indexed by rank in the source list, not by layer.
        for rank, layer in enumerate(vision["deepstack_visual_indexes"]):
            for suffix in ("weight", "bias"):
                self._add(
                    f"deepstack_merger_list.{rank}.norm.{suffix}", (merged,),
                    (f"v.deepstack.{layer}.norm.{suffix}",),
                )
                for source, target, shape in (
                    ("fc1", "linear_fc1", (merged, merged)),
                    ("fc2", "linear_fc2", (out, merged)),
                ):
                    self._add(
                        f"deepstack_merger_list.{rank}.{target}.{suffix}",
                        shape if suffix == "weight" else (shape[0],),
                        (f"v.deepstack.{layer}.{source}.{suffix}",),
                    )

    def validate_vision_metadata(self, reader: Any) -> None:
        if core.get_metadata_int(reader, "clip.vision.block_count") is None:
            # A split export's language GGUF carries no CLIP metadata; only the
            # mmproj (or a fused single-file VLM GGUF) describes a vision tower.
            return
        projector = core.get_metadata_str(reader, "clip.projector_type")
        if projector not in (None, "qwen3vl_merger"):
            raise ValueError(
                f"Unsupported Qwen3-VL projector type {projector!r}; only qwen3vl_merger "
                "is this architecture"
            )
        for src, dst in {
            "embedding_length": "hidden_size", "block_count": "depth",
            "attention.head_count": "num_heads", "patch_size": "patch_size",
            "spatial_merge_size": "spatial_merge_size",
            "temporal_patch_size": "temporal_patch_size",
        }.items():
            value = core.get_metadata_int(reader, f"clip.vision.{src}")
            if value is not None and value != self.vision[dst]:
                raise ValueError(f"GGUF Qwen3-VL {src} is incompatible with HF vision {dst}")
        feed_forward = core.get_metadata_int(reader, "clip.vision.feed_forward_length")
        if feed_forward is not None and feed_forward != self.vision["intermediate_size"]:
            raise ValueError("GGUF Qwen3-VL feed_forward_length is incompatible with HF vision")
        projection = core.get_metadata_int(reader, "clip.vision.projection_dim")
        if projection is not None and projection != self.vision["out_hidden_size"]:
            raise ValueError("GGUF Qwen3-VL projection_dim is incompatible with HF vision")
        epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
        if epsilon is not None and not np.isclose(epsilon, 1e-6, rtol=1e-5, atol=0):
            raise ValueError("GGUF Qwen3-VL vision layer norm epsilon must be 1e-6")
        # The GGUF states which layers carry a deepstack merger. Trusting the HF
        # config alone would put the mergers on the wrong layers when the two
        # checkpoints were exported from different revisions.
        flags = core.get_metadata_bool_array(reader, "clip.vision.is_deepstack_layers")
        if flags is None:
            raise ValueError(
                "Qwen3-VL requires clip.vision.is_deepstack_layers to locate the "
                "deepstack mergers"
            )
        layers = [i for i, flag in enumerate(flags) if flag]
        if layers != list(self.vision["deepstack_visual_indexes"]):
            raise ValueError(
                "GGUF deepstack layers "
                f"{layers} disagree with HF deepstack_visual_indexes "
                f"{self.vision['deepstack_visual_indexes']}"
            )
        if core.get_metadata_bool(reader, "clip.use_silu") is True:
            raise ValueError("Qwen3-VL uses a GELU ViT MLP, not SiLU")
        if any(core.get_metadata_bool(reader, key) is True for key in (
            "clip.has_siglip", "clip.has_audio_encoder",
        )):
            raise ValueError("Unsupported Qwen3-VL vision encoder")
