"""LLaVA-NeXT (anyres) -> mlx-vlm ``llava_next``, with no HF vision weights.

LLaVA-NeXT is not a new vision tower. It is the same CLIP-L/336 encoder LLaVA-1.5
uses, plus three things that decide whether the converted directory loads and
computes the right function:

* **The tiled projector parameter.** ``model.image_newline`` is a top-level
  ``[hidden]`` tensor that mlx-vlm keeps outside ``vision_tower``; a converter that
  drops it (because it is neither ``v.*`` nor ``mm.*``) produces a model whose
  image features are missing the separator row that anyres stitching depends on.
* **A tower with one fewer block than HF reports.** llama.cpp stores only the
  blocks LLaVA reads: a real ``llava-v1.6`` mmproj declares
  ``clip.vision.block_count = 23`` for a 24-block CLIP, because HF's
  ``vision_feature_layer = -2`` with 24 blocks *is* the output of block 22, and
  block 23 only feeds the pooler, which LLaVA discards. Emitting
  ``num_hidden_layers = 24`` therefore asks mlx-vlm for 32 tensors that no GGUF
  contains, and emitting 23 while keeping ``-2`` silently selects block 21 -- one
  layer early, still finite, still plausible. The stored count is authoritative,
  and the feature layer moves with it.
* **A post-norm the export omits.** ``v.post_ln`` is absent from real mmproj
  files. mlx-vlm still declares ``post_layernorm`` and loads strictly, so the
  parameter has to exist; it is written as the identity and recorded, because the
  only consumer is ``pooler_output``, which every LLaVA code path discards.

So this adapter refuses to treat LLaVA-NeXT as a rename of LLaVA-1.5: it checks
the projector type, the anyres marker and the block arithmetic before it emits a
single tensor.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from . import gguf2mlx as core
from . import vlm

#: The two CLIP MLP conventions in the wild, as ``(expansion, output)`` tensor
#: name fragments. llama.cpp's LLaVA-era loader reads ``ffn_down`` as the GELU
#: input (``ff_i``) and ``ffn_up`` as the output projection (``ff_o``); the modern
#: mtmd graph reads ``ffn_up`` as the expansion. The file never says which one it
#: is, so :meth:`LlavaNextAdapter._ffn_order` asks the shapes instead.
_FFN_ORDERS = {
    "mtmd": ("ffn_up", "ffn_down"),
    "llava-tools": ("ffn_down", "ffn_up"),
}

#: ``clip.vision.*`` key -> HF ``vision_config`` key.
_VISION_METADATA = {
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "attention.head_count": "num_attention_heads",
    "image_size": "image_size",
    "patch_size": "patch_size",
    "projection_dim": "projection_dim",
}

#: ``llama.*`` key -> HF ``text_config`` key, for the dimensions a
#: ``llava_next`` config.json routinely leaves to dataclass defaults.
_TEXT_METADATA = {
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "block_count": "num_hidden_layers",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "vocab_size": "vocab_size",
}

_POST_NORM = "vision_tower.vision_model.post_layernorm"


class LlavaNextAdapter(vlm.PlanAdapter):
    """Anyres LLaVA (LLaVA-1.6/NeXT) with a GGUF-authoritative vision tower."""

    name = "LLaVA-NeXT"
    text_arch = "llama"
    gguf_text_architectures = ("llama",)
    processor_type = "LlavaNextImageProcessor"
    nested_text_config = True
    #: `preserve` reuses language codes only; the anyres path has no pinned
    #: real-model preserve case, so the mode stays closed for this family.
    supports_source_preserve = False
    special_tokens: ClassVar[dict[str, str]] = {"image_token_index": "<image>"}

    def __init__(self, config: dict):
        if config.get("model_type") != "llava_next":
            raise ValueError("LlavaNextAdapter requires HF model_type='llava_next'")
        if config.get("architectures") not in (
            None, ["LlavaNextForConditionalGeneration"],
        ):
            raise ValueError(
                "Unsupported LLaVA-NeXT HF architecture; expected "
                "LlavaNextForConditionalGeneration"
            )
        grid = config.get("image_grid_pinpoints")
        if (
            not isinstance(grid, list) or not grid
            or any(
                not isinstance(pair, list) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in pair)
                for pair in grid
            )
        ):
            raise ValueError(
                "LLaVA-NeXT requires image_grid_pinpoints (the anyres marker); "
                "without it the model is LLaVA-1.5 and belongs to the 'llava' adapter"
            )
        if config.get("vision_feature_select_strategy", "default") != "default":
            raise ValueError("Only the LLaVA default patch feature selection is supported")
        feature_layer = config.get("vision_feature_layer", -2)
        if isinstance(feature_layer, bool) or feature_layer not in (-1, -2):
            raise ValueError("LLaVA-NeXT vision_feature_layer must be -1 or -2")
        if config.get("projector_hidden_act", "gelu") != "gelu":
            raise ValueError("Only the two-linear GELU LLaVA-NeXT projector is supported")
        if config.get("mm_projector_type", "mlp2x_gelu") != "mlp2x_gelu":
            raise ValueError("Only an mlp2x_gelu LLaVA-NeXT projector is supported")
        if config.get("multimodal_projector_bias", True) is not True:
            raise ValueError("The supported two-linear projector requires bias tensors")
        if config.get("use_image_newline_parameter", True) is not True:
            raise ValueError(
                "LLaVA-NeXT without an image_newline parameter is not a layout "
                "mlx-vlm's llava_next module implements"
            )
        token_index = config.get("image_token_index")
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise ValueError(  # noqa: TRY004 - a config value, not an argument
                "LLaVA-NeXT config requires an integer image_token_index"
            )

        raw_text = config.get("text_config")
        raw_vision = config.get("vision_config")
        if not isinstance(raw_text, dict) or not isinstance(raw_vision, dict):
            raise ValueError(  # noqa: TRY004 - a config value, not an argument
                "LLaVA-NeXT requires nested text_config and vision_config objects; "
                "legacy flat llava-1.6 configs must be re-saved by transformers"
            )
        self.text = {
            key: value for key, value in raw_text.items()
            if key not in ("vision_config", "text_config", "architectures")
        }
        self.vision = dict(raw_vision)
        if self.text.get("model_type") != "llama":
            raise ValueError(
                "LLaVA-NeXT text_config.model_type must be 'llama'; the mistral "
                "variant needs sliding-window support this converter does not have"
            )
        if self.vision.get("model_type") != "clip_vision_model":
            raise ValueError(
                "LLaVA-NeXT requires CLIP vision (clip_vision_model); the SigLIP "
                "variant is a different adapter"
            )
        self._grid = [list(pair) for pair in grid]
        self._hf_feature_layer = int(feature_layer)
        self._hf_layers = self.vision.get("num_hidden_layers")
        self._built = False
        self.feature_layer = int(feature_layer)
        self.output_overrides: dict[str, Any] = {}
        self.layout_report: dict[str, Any] = {}
        self.identity_fills: dict[str, np.ndarray] = {}
        self.identity_reasons: dict[str, str] = {}

    # -- metadata -------------------------------------------------------------

    def validate_text_metadata(self, reader: Any) -> None:
        """Fill the dimensions ``llava_next`` configs omit, then cross-check."""
        architecture = core.get_metadata_str(reader, "general.architecture")
        if architecture not in self.gguf_text_architectures:
            raise ValueError(
                f"{self.name} requires GGUF architecture in "
                f"{list(self.gguf_text_architectures)}, not {architecture!r}"
            )
        text = self.text
        for source, key in _TEXT_METADATA.items():
            value = core.get_metadata_int(reader, f"{architecture}.{source}")
            if value is None:
                if key not in text:
                    raise ValueError(
                        f"GGUF {architecture}.{source} is missing and the HF "
                        f"text_config does not declare {key}"
                    )
                continue
            if key in text and text[key] != value:
                raise ValueError(f"GGUF {source} is incompatible with HF {key}")
            text[key] = value
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size"):
            vlm._positive(text, key)
        text.setdefault("head_dim", text["hidden_size"] // text["num_attention_heads"])
        for source, key, default in (
            ("rope.freq_base", "rope_theta", 10000.0),
            ("attention.layer_norm_rms_epsilon", "rms_norm_eps", 1e-6),
        ):
            # Named separately from the integer loop above so the two types do not
            # get unified into one inferred local.
            measured = core.get_metadata_float(reader, f"{architecture}.{source}")
            if measured is not None:
                if key in text and not np.isclose(text[key], measured, rtol=1e-5, atol=0):
                    raise ValueError(f"GGUF {source} is incompatible with HF {key}")
                text[key] = measured
            else:
                text.setdefault(key, default)
        rotary = core.get_metadata_int(reader, f"{architecture}.rope.dimension_count")
        head_dim = text["head_dim"]
        if rotary is not None and rotary != head_dim:
            raise ValueError("Partial rotary Llama embeddings are unsupported")
        if head_dim % 2 or text["hidden_size"] % text["num_attention_heads"]:
            raise ValueError("Llama head dimensions must support the GGUF RoPE permutation")
        scaling = core.get_metadata_str(reader, f"{architecture}.rope.scaling.type")
        if scaling not in (None, "none"):
            raise ValueError(
                "Scaled Llama RoPE is unsupported for LLaVA-NeXT; only the "
                "unscaled vicuna/Llama-3 text halves are covered"
            )
        if core.get_metadata_int(reader, f"{architecture}.attention.sliding_window"):
            raise ValueError("Sliding-window Llama attention is unsupported")
        if core.get_metadata_bool(reader, f"{architecture}.attention.qkv_bias") is True:
            raise ValueError("Biased Llama attention is unsupported")
        if text.get("hidden_act", "silu") != "silu" or text.get("mlp_bias", False):
            raise ValueError("Only the Llama SiLU gated MLP is supported")
        if text.get("attention_bias", False):
            raise ValueError("Biased Llama attention is unsupported")
        if not isinstance(text.get("tie_word_embeddings", False), bool):
            raise ValueError("HF tie_word_embeddings must be a boolean")  # noqa: TRY004
        text["rope_scaling"] = None
        text.setdefault("max_position_embeddings", 4096)

    def configure_layout(self, readers: list[Any]) -> None:
        """Reconcile the stored block count, then build the tensor plan."""
        clips = [reader for reader in readers if _is_clip(reader)]
        if len(clips) > 1:
            raise ValueError("LLaVA-NeXT accepts at most one vision GGUF")
        if clips:
            self._read_vision_metadata(clips[0])
        else:
            for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                        "image_size", "patch_size", "projection_dim"):
                vlm._positive(self.vision, key)
            if self.vision.get("num_hidden_layers") is None:
                raise ValueError(
                    "LLaVA-NeXT needs either a vision GGUF (--mmproj) or a "
                    "vision_config that declares num_hidden_layers"
                )
        self._build_plan(clips[0] if clips else None)

    def _read_vision_metadata(self, reader: Any) -> None:
        """Take the tower geometry from the mmproj, which is what it contains."""
        vision = self.vision
        for source, key in _VISION_METADATA.items():
            value = core.get_metadata_int(reader, f"clip.vision.{source}")
            if value is None:
                if vision.get(key) is None:
                    raise ValueError(
                        f"GGUF clip.vision.{source} is missing and the HF "
                        f"vision_config does not declare {key}"
                    )
                continue
            if vision.get(key) is not None and vision[key] != value:
                raise ValueError(
                    f"GGUF clip.vision.{source} is incompatible with HF vision {key}"
                )
            vision[key] = value
        epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
        if epsilon is not None:
            if vision.get("layer_norm_eps") not in (None, epsilon) and not np.isclose(
                vision["layer_norm_eps"], epsilon, rtol=1e-5, atol=0
            ):
                raise ValueError("GGUF CLIP layer norm epsilon is incompatible with HF vision")
            vision["layer_norm_eps"] = epsilon
        vision.setdefault("num_channels", 3)
        if vision["num_channels"] != 3:
            raise ValueError("Unsupported CLIP channel count")
        if vision["hidden_size"] % vision["num_attention_heads"]:
            raise ValueError("Unsupported CLIP hidden/head dimensions")
        if vision["image_size"] % vision["patch_size"]:
            raise ValueError("CLIP image_size must be a multiple of patch_size")

        stored = core.get_metadata_int(reader, "clip.vision.block_count")
        if stored is None:
            raise ValueError("GGUF clip.vision.block_count is required")
        if self._hf_layers is not None and stored not in (self._hf_layers, self._hf_layers - 1):
            raise ValueError(
                f"GGUF stores {stored} CLIP blocks but HF vision_config declares "
                f"{self._hf_layers}; LLaVA-NeXT may omit only the pooler-only final block"
            )
        vision["num_hidden_layers"] = stored
        self.layout_report = {
            "hf_vision_layers": self._hf_layers,
            "stored_vision_blocks": stored,
        }
        if self.vision_from_hf:
            # The tower comes from the HF checkpoint, so the HF geometry wins and
            # nothing is reconciled; the plan only declares the names.
            if self._hf_layers is not None:
                vision["num_hidden_layers"] = self._hf_layers
            self.feature_layer = self._hf_feature_layer
            self.output_overrides = {}
            self.layout_report["feature_source"] = "hf checkpoint"
        elif self._hf_layers is not None and stored == self._hf_layers - 1:
            # The GGUF dropped the block HF only feeds to the (discarded) pooler,
            # so the feature is now the tower's last stored output.
            self.feature_layer = -1
            self.output_overrides = {"vision_feature_layer": -1}
            self.layout_report.update({
                "feature_source": "last stored block",
                "layers_omitted": 1,
                "hf_feature_layer": self._hf_feature_layer,
            })
        else:
            self.feature_layer = self._hf_feature_layer
            self.output_overrides = {}
            self.layout_report["feature_source"] = "hf checkpoint"

    def validate_vision_metadata(self, reader: Any) -> None:
        if not _is_clip(reader):
            return
        projector = core.get_metadata_str(reader, "clip.projector_type")
        if projector not in (None, "mlp"):
            raise ValueError(
                f"Unsupported LLaVA-NeXT projector type {projector!r}; this adapter "
                "implements the two-linear mlp2x_gelu projector only"
            )
        if core.get_metadata_bool(reader, "clip.has_llava_projector") is False:
            raise ValueError("GGUF is missing the LLaVA projector this layout requires")
        if core.get_metadata_bool(reader, "clip.use_gelu") is not False:
            raise ValueError(
                "LLaVA-NeXT's CLIP activates with quick_gelu; a GELU or SiLU "
                "tower is a different vision encoder"
            )
        for key in ("clip.has_siglip", "clip.has_audio_encoder", "clip.has_text_encoder"):
            if core.get_metadata_bool(reader, key) is True:
                raise ValueError(f"Unsupported LLaVA-NeXT vision encoder ({key})")
        aspect = core.get_metadata_str(reader, "clip.vision.image_aspect_ratio")
        if aspect not in (None, "anyres"):
            raise ValueError(
                f"GGUF image_aspect_ratio {aspect!r} is not the anyres layout this "
                "adapter implements"
            )
        merge = core.get_metadata_str(reader, "clip.vision.mm_patch_merge_type")
        if merge not in (None, "spatial_unpad"):
            raise ValueError(
                f"GGUF mm_patch_merge_type {merge!r} is unsupported; only "
                "spatial_unpad anyres stitching is implemented"
            )
        pinpoints = core.get_metadata_array_int(
            reader, "clip.vision.image_grid_pinpoints"
        )
        if pinpoints is not None:
            flat = [value for pair in self._grid for value in pair]
            if pinpoints != flat:
                raise ValueError(
                    "GGUF image_grid_pinpoints disagree with HF image_grid_pinpoints; "
                    "the processor would tile differently from the trained model"
                )
        # ``image_crop_resolution``/``image_split_resolution`` are LLaVA-1.5
        # leftovers: a real llava-v1.6 export still declares 224 there while the
        # anyres processor crops every tile to ``image_size`` 336. Neither value
        # reaches mlx-vlm -- tiling comes from ``preprocessor_config.json``, which
        # ``validate_processor`` already pins to the config grid -- so they are
        # recorded as evidence rather than enforced.
        crop = core.get_metadata_int(reader, "clip.vision.image_crop_resolution")
        split = core.get_metadata_int(reader, "clip.vision.image_split_resolution")
        if (crop is not None and crop <= 0) or (split is not None and split <= 0):
            raise ValueError("GGUF image crop/split resolution must be positive")
        if crop is not None or split is not None:
            self.layout_report["gguf_image_crop"] = {
                "crop_resolution": crop,
                "split_resolution": split,
                "processor_tile": self.vision["image_size"],
            }

    def validate_processor(self, processor: dict, readers: list[Any]) -> None:
        setting = processor.get("aspect_ratio_setting")
        if setting not in (None, "anyres", "anyres_max_9"):
            raise ValueError(
                f"LLaVA-NeXT processor aspect_ratio_setting {setting!r} is unsupported"
            )
        if setting is None:
            raise ValueError(
                "LLaVA-NeXT processor config must declare aspect_ratio_setting=anyres"
            )
        crop = processor.get("crop_size")
        size = self.vision["image_size"]
        if crop is not None and crop != {"height": size, "width": size}:
            raise ValueError(
                "LLaVA-NeXT processor crop_size must equal the CLIP image_size"
            )
        pinpoints = processor.get("image_grid_pinpoints")
        if pinpoints is not None and pinpoints != self._grid:
            raise ValueError(
                "HF processor image_grid_pinpoints disagree with config.json"
            )
        grid = processor.get("size")
        if isinstance(grid, dict) and grid.get("shortest_edge") not in (None, size):
            raise ValueError(
                "LLaVA-NeXT processor size.shortest_edge must equal image_size"
            )

    # -- planning -------------------------------------------------------------

    def _add(
        self, target: str, shape: tuple[int, ...], source: str,
        source_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.expected[target] = shape
        self.vision_plan.append((target, (source,), 0))
        self.source_shapes[source] = source_shape or shape

    def _build_plan(self, clip: Any) -> None:
        if self._built:
            return
        vision, hidden = self.vision, self.text["hidden_size"]
        width = vision["hidden_size"]
        feed_forward = vision["intermediate_size"]
        patch = vision["patch_size"]
        positions = (vision["image_size"] // patch) ** 2 + 1
        self.expected = vlm._language_shapes(self.text)
        self.vision_plan = []
        self.source_shapes = {}
        self.identity_fills = {}
        self.identity_reasons = {}
        self.output_overrides = dict(getattr(self, "output_overrides", {}))

        prefix = "vision_tower.vision_model."
        self._add(
            prefix + "embeddings.class_embedding", (width,), "v.class_embd",
        )
        self._add(
            prefix + "embeddings.patch_embedding.weight",
            (width, 3, patch, patch), "v.patch_embd.weight",
        )
        self._add(
            prefix + "embeddings.position_embedding.weight",
            (positions, width), "v.position_embd.weight",
        )
        for suffix in ("weight", "bias"):
            self._add(prefix + f"pre_layrnorm.{suffix}", (width,), f"v.pre_ln.{suffix}")
        self._plan_post_norm(prefix, width, clip)
        expansion, output = self._ffn_order(clip, width, feed_forward)
        for index in range(vision["num_hidden_layers"]):
            block = prefix + f"encoder.layers.{index}."
            for part, source in (
                ("q_proj", "attn_q"), ("k_proj", "attn_k"),
                ("v_proj", "attn_v"), ("out_proj", "attn_out"),
            ):
                self._add(block + f"self_attn.{part}.weight", (width, width),
                          f"v.blk.{index}.{source}.weight")
                self._add(block + f"self_attn.{part}.bias", (width,),
                          f"v.blk.{index}.{source}.bias")
            for norm in ("layer_norm1", "layer_norm2"):
                key = "ln1" if norm == "layer_norm1" else "ln2"
                for suffix in ("weight", "bias"):
                    self._add(
                        prefix + f"encoder.layers.{index}.{norm}.{suffix}",
                        (width,), f"v.blk.{index}.{key}.{suffix}",
                    )
            for part, source, shape in (
                ("fc1", expansion, (feed_forward, width)),
                ("fc2", output, (width, feed_forward)),
            ):
                self._add(
                    prefix + f"encoder.layers.{index}.mlp.{part}.weight",
                    shape, f"v.blk.{index}.{source}.weight",
                )
                self._add(
                    prefix + f"encoder.layers.{index}.mlp.{part}.bias",
                    (shape[0],), f"v.blk.{index}.{source}.bias",
                )
        for part, index, shape in (
            ("linear_1", 0, (hidden, width)), ("linear_2", 2, (hidden, hidden)),
        ):
            self._add(
                f"multi_modal_projector.{part}.weight", shape, f"mm.{index}.weight",
            )
            self._add(
                f"multi_modal_projector.{part}.bias", (hidden,), f"mm.{index}.bias",
            )
        # The anyres separator row: a top-level parameter in mlx-vlm, not part of
        # the tower, which is why a name-only "v./mm." filter drops it.
        self._add("image_newline", (hidden,), "model.image_newline")
        self.source_names = set(self.source_shapes)
        self._built = True

    def _ffn_order(self, clip: Any, width: int, feed_forward: int) -> tuple[str, str]:
        """Resolve which ``ffn_*`` tensor is the MLP expansion, from the shapes.

        A CLIP MLP is ``width -> feed_forward -> width``, and both exporters ship
        under the same two names in opposite roles. Getting it backwards is silent
        -- the tensors are the right size for *a* linear layer, the directory loads,
        and every image is scored by a swapped MLP -- so the shape decides:

        * ``ffn_up`` shaped ``(feed_forward, width)`` is the mtmd convention;
        * ``ffn_down`` shaped like that is the LLaVA-era ``clip.cpp`` convention,
          which reads ``ffn_down`` as ``ff_i`` (the GELU input) and ``ffn_up`` as
          ``ff_o``. Every pre-mtmd ``mmproj`` (llava-1.5 and llava-1.6 exports
          alike) is of this kind.
        """
        if clip is None:
            # Nothing to inspect: the tower comes from HF under these names only.
            return _FFN_ORDERS["mtmd"]
        shapes: dict[str, tuple[int, ...] | None] = {}
        for fragment in ("ffn_up", "ffn_down"):
            name = f"v.blk.0.{fragment}.weight"
            tensor = next((item for item in clip.tensors if item.name == name), None)
            shapes[fragment] = (
                None if tensor is None
                else tuple(int(dim) for dim in reversed(tensor.shape))
            )
        if shapes["ffn_up"] is None or shapes["ffn_down"] is None:
            missing = "ffn_up" if shapes["ffn_up"] is None else "ffn_down"
            raise ValueError(f"LLaVA-NeXT vision GGUF is missing v.blk.0.{missing}.weight")
        if feed_forward == width:
            raise ValueError(
                "LLaVA-NeXT cannot tell the CLIP ffn tensors apart when "
                "feed_forward_length equals embedding_length"
            )
        for convention, order in _FFN_ORDERS.items():
            expansion, output = order
            if shapes[expansion] == (feed_forward, width) and shapes[output] == (
                width, feed_forward
            ):
                self.layout_report["clip_ffn_convention"] = convention
                return order
        raise ValueError(
            "LLaVA-NeXT CLIP ffn shapes match neither known convention: "
            f"ffn_up={shapes['ffn_up']}, ffn_down={shapes['ffn_down']} for a "
            f"{width}->{feed_forward}->{width} MLP"
        )

    def _plan_post_norm(self, prefix: str, width: int, clip: Any) -> None:
        """Declare ``post_layernorm`` from the GGUF, the identity, or the HF tower.

        Three cases, and only the second writes a value that did not come from a
        file: the export has the weights, the export omits them (real LLaVA mmproj
        files do), or the tower is coming from HF in hybrid mode.
        """
        target_w = prefix + "post_layernorm.weight"
        target_b = prefix + "post_layernorm.bias"
        if clip is not None and _has_tensors(clip, "v.post_ln.weight"):
            self._add(target_w, (width,), "v.post_ln.weight")
            self._add(target_b, (width,), "v.post_ln.bias")
            return
        self.expected[target_w] = (width,)
        self.expected[target_b] = (width,)
        if self.vision_from_hf:
            return
        self.identity_fills[target_w] = np.ones(width, dtype=np.float32)
        self.identity_fills[target_b] = np.zeros(width, dtype=np.float32)
        self.identity_reasons[target_w] = (
            "llama.cpp omits CLIP post_layernorm; mlx-vlm computes it only on the "
            "pooler output, which llava_next discards"
        )
        self.identity_reasons[target_b] = self.identity_reasons[target_w]

    # -- assembly -------------------------------------------------------------

    def normalize_source(self, name: str, array: np.ndarray) -> np.ndarray:
        """Accept the ``[hidden, 1]`` class-embedding variants some exports write."""
        if name == "v.class_embd" and array.ndim == 2 and 1 in array.shape:
            return array.reshape(-1)
        return array

    def restore_text(self, name: str, array: np.ndarray) -> np.ndarray:
        return vlm._restore_llama(name, array, self.text)


def _is_clip(reader: Any) -> bool:
    return (
        core.get_metadata_str(reader, "general.architecture") == "clip"
        and core.get_metadata_bool(reader, "clip.has_vision_encoder") is not False
    )


def _has_tensors(reader: Any, prefix: str) -> bool:
    return any(tensor.name.startswith(prefix) for tensor in reader.tensors)
