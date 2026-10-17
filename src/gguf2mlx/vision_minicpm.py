"""Strict adapter for the dense MiniCPM-V 4.7 1B GGUF layout."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from . import gguf2mlx as core
from . import vlm

_VISION_ROOTS = {
    "v.patch_embd.weight": "vision_tower.embeddings.patch_embedding.weight",
    "v.patch_embd.bias": "vision_tower.embeddings.patch_embedding.bias",
    "v.position_embd.weight": "vision_tower.embeddings.position_embedding.weight",
    "v.post_ln.weight": "vision_tower.post_layernorm.weight",
    "v.post_ln.bias": "vision_tower.post_layernorm.bias",
}
_VISION_BLOCKS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
}
_VIT_MERGER = {
    "ln1": "layer_norm1",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ds_ln": "pre_norm",
    "ds_ffn_up": "linear_1",
    "ds_ffn_down": "linear_2",
}
_LINEAR_ATTENTION = (
    "attn_qkv", "attn_gate", "ssm_conv1d", "ssm_dt", "ssm_a",
    "ssm_beta", "ssm_alpha", "ssm_norm", "ssm_out",
)


class MiniCPMV47Adapter:
    """Validate and map the 4.7 dense text and SigLIP/merger GGUF components."""

    def __init__(self, config: dict):
        if config.get("model_type") != "minicpmv4_7":
            raise ValueError("MiniCPM-V adapter requires HF model_type='minicpmv4_7'")
        if config.get("architectures") != ["MiniCPMV4_7ForConditionalGeneration"]:
            raise ValueError("MiniCPM-V adapter requires MiniCPMV4_7ForConditionalGeneration")
        raw_text, raw_vision = config.get("text_config"), config.get("vision_config")
        if not isinstance(raw_text, dict) or not isinstance(raw_vision, dict):
            raise ValueError(  # noqa: TRY004
                "MiniCPM-V 4.7 requires text_config and vision_config objects"
            )
        if raw_text.get("model_type") != "qwen3_5_text":
            raise ValueError("MiniCPM-V 4.7 1B requires dense qwen3_5_text")
        if raw_vision.get("model_type") not in (
            "siglip", "siglip_vision_model", "minicpmv4_7_vision",
        ):
            raise ValueError("MiniCPM-V 4.7 requires its SigLIP vision configuration")
        if config.get("mrope_mode", "canvas") != "canvas":
            raise ValueError("MiniCPM-V 4.7 requires mrope_mode='canvas'")

        self.text = dict(raw_text)
        self.vision = dict(raw_vision)
        self.downsample_mode = config.get("downsample_mode", "16x")
        if self.downsample_mode not in ("4x", "16x"):
            raise ValueError("MiniCPM-V 4.7 downsample_mode must be '4x' or '16x'")
        if config.get("merger_times", 1) != 1:
            raise ValueError("MiniCPM-V 4.7 adapter supports merger_times=1 only")
        if tuple(config.get("merge_kernel_size", (2, 2))) != (2, 2):
            raise ValueError("MiniCPM-V 4.7 adapter supports a 2x2 merger kernel only")
        if self.text.get("tie_word_embeddings", False) or self.text.get("attention_bias", False):
            raise ValueError("MiniCPM-V 4.7 dense text requires untied, bias-free Qwen3.5 weights")
        if self.text.get("hidden_act", "silu") != "silu":
            raise ValueError("MiniCPM-V 4.7 dense text requires the Qwen3.5 SiLU MLP")

        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
        ):
            vlm._positive(self.text, key)
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "image_size", "patch_size",
        ):
            vlm._positive(self.vision, key)
        vision_size = self.vision["hidden_size"]
        heads = self.vision["num_attention_heads"]
        if (
            vision_size % heads or self.vision.get("num_channels", 3) != 3
            or self.vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
            or self.vision.get("layer_norm_eps", 1e-6) != 1e-6
        ):
            raise ValueError("Unsupported MiniCPM-V 4.7 SigLIP dimensions or activation")
        if self.vision["image_size"] % self.vision["patch_size"]:
            raise ValueError("MiniCPM-V vision image_size must be divisible by patch_size")
        self.insert_layer_id = config.get(
            "insert_layer_id", self.vision.get("insert_layer_id", 6)
        )
        if (
            isinstance(self.insert_layer_id, bool)
            or not isinstance(self.insert_layer_id, int)
            or not 0 <= self.insert_layer_id < self.vision["num_hidden_layers"]
        ):
            raise ValueError("MiniCPM-V insert_layer_id is outside the vision encoder")

        self.expected: dict[str, tuple[int, ...]] = {}
        self.source_shapes: dict[str, tuple[int, ...]] = {}
        self.vision_plan: list[tuple[str, str]] = []
        self._build_vision_plan()

    @staticmethod
    def _add(
        expected: dict[str, tuple[int, ...]],
        source_shapes: dict[str, tuple[int, ...]],
        plan: list[tuple[str, str]],
        source: str,
        target: str,
        shape: tuple[int, ...],
    ) -> None:
        expected[target] = shape
        source_shapes[source] = shape
        plan.append((source, target))

    def _build_vision_plan(self) -> None:
        v = self.vision["hidden_size"]
        f = self.vision["intermediate_size"]
        p = self.vision["patch_size"]
        h = self.text["hidden_size"]
        grid = self.vision["image_size"] // p
        expected, source_shapes, plan = self.expected, self.source_shapes, self.vision_plan

        for source, target in _VISION_ROOTS.items():
            shape = {
                "v.patch_embd.weight": (v, 3, p, p),
                "v.patch_embd.bias": (v,),
                "v.position_embd.weight": (grid * grid, v),
                "v.post_ln.weight": (v,),
                "v.post_ln.bias": (v,),
            }[source]
            self._add(expected, source_shapes, plan, source, target, shape)
        for index in range(self.vision["num_hidden_layers"]):
            for source_fragment, target_fragment in _VISION_BLOCKS.items():
                for suffix in ("weight", "bias"):
                    if source_fragment in ("attn_q", "attn_k", "attn_v", "attn_out"):
                        shape = (v, v) if suffix == "weight" else (v,)
                    elif source_fragment in ("ln1", "ln2"):
                        shape = (v,)
                    else:
                        output = f if source_fragment == "ffn_up" else v
                        input_size = v if source_fragment == "ffn_up" else f
                        shape = (output, input_size) if suffix == "weight" else (output,)
                    self._add(
                        expected, source_shapes, plan,
                        f"v.blk.{index}.{source_fragment}.{suffix}",
                        f"vision_tower.encoder.layers.{index}.{target_fragment}.{suffix}",
                        shape,
                    )

        if self.downsample_mode == "16x":
            for suffix in ("weight", "bias"):
                self._add(
                    expected, source_shapes, plan,
                    f"v.vit_merger.ln1.{suffix}",
                    f"vit_merger.layer_norm1.{suffix}", (v,),
                )
                for source_fragment, target_fragment in _VIT_MERGER.items():
                    if source_fragment == "ln1":
                        continue
                    if source_fragment.startswith("attn_"):
                        shape = (v, v) if suffix == "weight" else (v,)
                    elif source_fragment == "ds_ln":
                        shape = (4 * v,)
                    elif source_fragment == "ds_ffn_up":
                        shape = (4 * f, 4 * v) if suffix == "weight" else (4 * f,)
                    else:
                        shape = (v, 4 * f) if suffix == "weight" else (v,)
                    self._add(
                        expected, source_shapes, plan,
                        f"v.vit_merger.{source_fragment}.{suffix}",
                        f"vit_merger.{target_fragment}.{suffix}", shape,
                    )

        for suffix in ("weight", "bias"):
            for source, target, shape in (
                ("mm.input_norm", "pre_norm", (4 * v,)),
                ("mm.up", "linear_1", (4 * v, 4 * v) if suffix == "weight" else (4 * v,)),
                ("mm.down", "linear_2", (h, 4 * v) if suffix == "weight" else (h,)),
            ):
                self._add(
                    expected, source_shapes, plan,
                    f"{source}.{suffix}", f"merger.mlp.0.{target}.{suffix}", shape,
                )

    def validate_text_metadata(self, reader: Any) -> None:
        architecture = core.get_metadata_str(reader, "general.architecture")
        if architecture != "qwen35":
            raise ValueError("MiniCPM-V 4.7 dense text requires GGUF architecture='qwen35'")
        variant_error = core.validate_architecture_variant(reader, "qwen35")
        if variant_error:
            raise ValueError(variant_error)
        for suffix, config_key in (
            ("embedding_length", "hidden_size"),
            ("feed_forward_length", "intermediate_size"),
            ("block_count", "num_hidden_layers"),
            ("attention.head_count", "num_attention_heads"),
            ("attention.head_count_kv", "num_key_value_heads"),
            ("vocab_size", "vocab_size"),
        ):
            value = core.get_metadata_int(reader, f"qwen35.{suffix}")
            if value is None or value != self.text[config_key]:
                raise ValueError(f"GGUF qwen35.{suffix} is missing or incompatible with HF")
        head_dim = core.get_metadata_int(reader, "qwen35.attention.key_length")
        if head_dim is None or head_dim != self.text.get(
            "head_dim", self.text["hidden_size"] // self.text["num_attention_heads"]
        ):
            raise ValueError("GGUF Qwen3.5 head dimension is incompatible with HF config")
        for suffix, config_key in (
            ("ssm.group_count", "linear_num_key_heads"),
            ("ssm.time_step_rank", "linear_num_value_heads"),
            ("ssm.state_size", "linear_key_head_dim"),
            ("ssm.conv_kernel", "linear_conv_kernel_dim"),
            ("full_attention_interval", "full_attention_interval"),
        ):
            value = core.get_metadata_int(reader, f"qwen35.{suffix}")
            if value is None or value != self.text.get(config_key):
                raise ValueError(f"GGUF qwen35.{suffix} is incompatible with HF config")
        inner_size = core.get_metadata_int(reader, "qwen35.ssm.inner_size")
        expected_inner_size = (
            self.text["linear_num_value_heads"] * self.text["linear_value_head_dim"]
        )
        if inner_size is None or inner_size != expected_inner_size:
            raise ValueError("GGUF qwen35.ssm.inner_size is incompatible with HF config")
        for suffix, expected in (
            ("rope.freq_base", self.text.get("rope_theta")),
            ("attention.layer_norm_rms_epsilon", self.text.get("rms_norm_eps")),
        ):
            value = core.get_metadata_float(reader, f"qwen35.{suffix}")
            if (
                expected is None or value is None
                or not np.isclose(value, expected, rtol=1e-5, atol=0)
            ):
                raise ValueError(f"GGUF qwen35.{suffix} is incompatible with HF config")
        sections = core.get_metadata_array_int(reader, "qwen35.rope.dimension_sections")
        rope = self.text.get("rope_parameters", self.text.get("rope_scaling"))
        if len(sections) == 4 and sections[-1] == 0:
            sections.pop()
        expected = rope.get("mrope_section") if isinstance(rope, dict) else None
        if (
            len(sections) != 3 or any(section <= 0 for section in sections)
            or expected is None or sections != expected
        ):
            raise ValueError("GGUF Qwen3.5 mRoPE sections are missing or incompatible with HF")

    def validate_text_tensors(self, reader: Any) -> dict[str, tuple[int, ...]]:
        text_expected: dict[str, tuple[int, ...]] = {}
        block_tensors: dict[int, set[str]] = {}
        roots: set[str] = set()
        for tensor in reader.tensors:
            name = tensor.name
            if name.startswith(("v.", "mm.")):
                raise ValueError(f"Vision tensor found in MiniCPM-V text GGUF: {name}")
            if name.startswith("blk."):
                match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
                if match is None:
                    raise ValueError(f"Unsupported MiniCPM-V text tensor {name}")
                block_tensors.setdefault(int(match.group(1)), set()).add(match.group(2))
            else:
                roots.add(name)
            mapped = core._map_tensor_name(name, "qwen35")
            if mapped == name:
                raise ValueError(f"Unsupported dense Qwen3.5 text tensor: {name}")
            target = "language_model." + mapped
            shape = tuple(int(dim) for dim in reversed(tensor.shape))
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(f"Invalid MiniCPM-V text tensor shape: {name}")
            if target in text_expected:
                raise ValueError(f"Colliding MiniCPM-V text mapping: {target}")
            text_expected[target] = shape

        required_roots = {"token_embd.weight", "output_norm.weight", "output.weight"}
        if not required_roots.issubset(roots):
            raise ValueError("MiniCPM-V dense text GGUF is missing required embedding/head tensors")
        if set(block_tensors) != set(range(self.text["num_hidden_layers"])):
            raise ValueError("MiniCPM-V text GGUF layer IDs do not match HF num_hidden_layers")
        dense_mlp = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
        for index, names in block_tensors.items():
            if not {"attn_norm.weight", "ffn_norm.weight"}.issubset(names):
                raise ValueError(f"MiniCPM-V text layer {index} is missing required norms")
            if not dense_mlp.issubset(names):
                raise ValueError(f"MiniCPM-V text layer {index} is missing dense MLP weights")
            full_attention = {
                f"attn_{part}.weight" for part in ("q", "k", "v")
            } | {"attn_output.weight"}
            linear_attention = {
                item + (".weight" if item not in ("ssm_dt", "ssm_a") else "")
                for item in _LINEAR_ATTENTION
            }
            if not full_attention.issubset(names) and not linear_attention.issubset(names):
                raise ValueError(f"MiniCPM-V text layer {index} has no complete attention family")
        return text_expected

    def validate_vision_metadata(self, reader: Any, preprocessor: dict) -> None:
        if core.get_metadata_str(reader, "clip.projector_type") != "minicpmv4_7":
            raise ValueError("MiniCPM-V 4.7 requires clip.projector_type='minicpmv4_7'")
        for suffix, expected in (
            ("embedding_length", self.vision["hidden_size"]),
            ("feed_forward_length", self.vision["intermediate_size"]),
            ("block_count", self.vision["num_hidden_layers"]),
            ("attention.head_count", self.vision["num_attention_heads"]),
            ("patch_size", self.vision["patch_size"]),
        ):
            value = core.get_metadata_int(reader, f"clip.vision.{suffix}")
            if value is None or value != expected:
                raise ValueError(f"GGUF CLIP {suffix} is missing or incompatible with MiniCPM-V")
        scale_resolution = preprocessor.get("scale_resolution", 448)
        image_size = core.get_metadata_int(reader, "clip.vision.image_size")
        if image_size is None or image_size != scale_resolution:
            raise ValueError("GGUF image_size does not match MiniCPM-V processor scale_resolution")
        scale_factor = core.get_metadata_int(reader, "clip.vision.projector.scale_factor")
        expected_scale = 2 if self.downsample_mode == "4x" else 4
        if scale_factor != expected_scale:
            raise ValueError("GGUF projector scale factor does not match downsample_mode")
        if core.get_metadata_bool(reader, "clip.use_gelu") is not True:
            raise ValueError("MiniCPM-V 4.7 requires the GGUF SigLIP GELU activation")
        epsilon = core.get_metadata_float(reader, "clip.vision.attention.layer_norm_epsilon")
        if epsilon is None or not np.isclose(epsilon, 1e-6, rtol=1e-5, atol=0):
            raise ValueError("MiniCPM-V 4.7 vision layer norm epsilon must be 1e-6")
        for key in ("image_mean", "image_std"):
            value = preprocessor.get(key)
            if reader.get_field(f"clip.vision.{key}") is not None:
                gguf_value = core.get_metadata_array_float(reader, f"clip.vision.{key}")
                if (
                    not isinstance(value, list) or len(value) != 3
                    or len(gguf_value) != 3
                    or not np.allclose(gguf_value, value, rtol=1e-5, atol=1e-7)
                ):
                    raise ValueError(f"GGUF/HF MiniCPM-V processor {key} mismatch")
        indexes = core.get_metadata_array_int(reader, "clip.vision.wa_layer_indexes")
        if indexes != [self.insert_layer_id]:
            raise ValueError("GGUF vit_merger insertion layer differs from HF config")

    def validate_processor(self, root: Any) -> dict:
        processor = vlm._read_json(root / "preprocessor_config.json")
        if processor.get("image_processor_type") not in {
            "MiniCPMV4_7ImageProcessorPil",
            "MiniCPMV4_7ImageProcessor",
        }:
            raise ValueError("MiniCPM-V 4.7 requires its matching MiniCPMV4_7 image processor")
        mode = processor.get("downsample_mode", self.downsample_mode)
        if mode != self.downsample_mode:
            raise ValueError("HF processor downsample_mode differs from the model config")
        scale_resolution = processor.get("scale_resolution", 448)
        if (
            isinstance(scale_resolution, bool) or not isinstance(scale_resolution, int)
            or scale_resolution <= 0
        ):
            raise ValueError("Invalid MiniCPM-V processor scale_resolution")
        return processor

    def validate_tokenizer(self, reader: Any, root: Any) -> None:
        document = vlm._read_json(root / "tokenizer.json")
        tokenizer_config = vlm._read_json(root / "tokenizer_config.json")
        vocab = document.get("model", {}).get("vocab", {})
        token_ids = (
            dict(vocab) if isinstance(vocab, dict)
            else {entry[0]: index for index, entry in enumerate(vocab)}
            if isinstance(vocab, list) and all(
                isinstance(entry, list) and entry and isinstance(entry[0], str)
                for entry in vocab
            )
            else {}
        )
        if not token_ids:
            raise ValueError("MiniCPM-V tokenizer must contain a verifiable vocabulary")
        for entry in document.get("added_tokens", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
                raise ValueError("Invalid MiniCPM-V tokenizer added token")  # noqa: TRY004
            token_ids[entry["content"]] = entry.get("id")
        decoder = tokenizer_config.get("added_tokens_decoder", {})
        if isinstance(decoder, dict):
            for key, entry in decoder.items():
                if key.isdecimal() and isinstance(entry, dict) and isinstance(
                    entry.get("content"), str
                ):
                    token_ids[entry["content"]] = int(key)
        by_id = {index: token for token, index in token_ids.items()
                 if isinstance(index, int) and not isinstance(index, bool)}
        tokens = core.get_metadata_array_str(reader, "tokenizer.ggml.tokens")
        if len(tokens) != self.text["vocab_size"] or len(tokens) != len(by_id):
            raise ValueError("MiniCPM-V HF/GGUF tokenizer vocabulary size mismatch")
        if any(by_id.get(index) != token for index, token in enumerate(tokens)):
            raise ValueError("MiniCPM-V HF/GGUF tokenizer IDs differ")
        config = vlm._read_json(root / "config.json")
        for key in (
            "image_token_id", "video_token_id", "image_start_id", "image_end_id",
            "slice_start_id", "slice_end_id", "newline_id",
        ):
            token_id = config.get(key)
            if token_id is None and key != "image_token_id":
                continue
            if (
                isinstance(token_id, bool) or not isinstance(token_id, int)
                or not 0 <= token_id < self.text["vocab_size"]
            ):
                raise ValueError(f"MiniCPM-V config requires a valid {key}")
        if not (
            tokenizer_config.get("chat_template")
            or (root / "chat_template.jinja").is_file()
            or vlm._read_optional_json(root / "processor_config.json").get("chat_template")
        ):
            raise ValueError("MiniCPM-V HF assets require a chat template")

    def map_vision(self, name: str) -> str:
        if name in _VISION_ROOTS:
            return _VISION_ROOTS[name]
        match = re.fullmatch(r"v\.blk\.(\d+)\.([a-z0-9_]+)\.(weight|bias)", name)
        if match and match[2] in _VISION_BLOCKS:
            return f"vision_tower.encoder.layers.{match[1]}.{_VISION_BLOCKS[match[2]]}.{match[3]}"
        match = re.fullmatch(r"v\.vit_merger\.([a-z0-9_]+)\.(weight|bias)", name)
        if match and self.downsample_mode == "16x" and match[1] in _VIT_MERGER:
            return f"vit_merger.{_VIT_MERGER[match[1]]}.{match[2]}"
        raise ValueError(f"Unsupported MiniCPM-V 4.7 vision tensor {name}")

    def map_projector(self, name: str) -> str:
        match = re.fullmatch(r"mm\.(input_norm|up|down)\.(weight|bias)", name)
        if match:
            component = {"input_norm": "pre_norm", "up": "linear_1", "down": "linear_2"}[
                match[1]
            ]
            return f"merger.mlp.0.{component}.{match[2]}"
        raise ValueError(f"Unsupported MiniCPM-V 4.7 merger tensor {name}")

    def hf_vision_name(self, name: str) -> str | None:
        if name.startswith(("model.merger.", "merger.")):
            return None
        prefixes = (
            ("model.vision_tower.", "vision_tower."),
            ("vision_tower.", "vision_tower."),
            ("model.vit_merger.", "vit_merger."),
            ("vit_merger.", "vit_merger."),
        )
        for prefix, target in prefixes:
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.startswith("vit_merger.") and self.downsample_mode == "4x":
                    return None
                return target + suffix
        return None
