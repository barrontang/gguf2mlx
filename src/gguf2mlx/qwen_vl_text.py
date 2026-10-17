"""Text-only Qwen-VL conditioning artifacts, not a complete vision-language model."""

from __future__ import annotations

import re

import numpy as np

QWEN_VL_TEXT_ARCHES = {"qwen2vl", "qwen2_vl", "qwen3vl", "qwen3_vl"}


class Qwen3VLTextEncoderConverter:
    """Reuse the Qwen2 language layout while preserving the multimodal contract."""

    @staticmethod
    def is_visual_tensor(name: str) -> bool:
        return bool(re.search(
            r"(^|\.)(vision_tower|vision_model|visual|vision|v|merger|mm|mmproj|"
            r"multi_modal_projector)(\.|$)", name,
        ))

    @staticmethod
    def map_tensor_name(name: str) -> str:
        from .gguf2mlx import _map_tensor_name

        roots = {"token_embd.weight", "output_norm.weight", "output.weight"}
        layer = re.fullmatch(
            r"blk\.\d+\.(attn_[qkv]|attn_output|attn_[qk]_norm|"
            r"ffn_gate|ffn_up|ffn_down|attn_norm|ffn_norm)\.weight", name,
        )
        bias = re.fullmatch(r"blk\.\d+\.attn_[qkv]\.bias", name)
        if name not in roots and layer is None and bias is None:
            raise ValueError(f"Unsupported Qwen-VL text tensor: {name}")
        return _map_tensor_name(name, "qwen2")

    @staticmethod
    def build_config(reader, arch: str, dtype: str) -> dict:
        from .gguf2mlx import (
            build_config,
            get_metadata_array_str,
            get_metadata_float,
            get_metadata_int,
            get_metadata_str,
        )

        # Exports differ in architecture spelling and metadata namespace.
        prefixes = [arch, *sorted(QWEN_VL_TEXT_ARCHES - {arch}), "qwen2"]

        class TextMetadata:
            tensors = reader.tensors

            def get_field(self, key):
                if key.startswith("qwen2."):
                    suffix = key[len("qwen2."):]
                    for prefix in prefixes:
                        field = reader.get_field(f"{prefix}.{suffix}")
                        if field is not None:
                            return field
                return reader.get_field(key)

        metadata = TextMetadata()
        scaling_type = get_metadata_str(metadata, "qwen2.rope.scaling.type")
        scaling_factor = get_metadata_float(metadata, "qwen2.rope.scaling.factor")
        if scaling_type not in (None, "none", "default") or scaling_factor not in (None, 1.0):
            raise ValueError(
                "Nondefault Qwen-VL RoPE scaling is unsupported; "
                "refusing to discard source scaling parameters"
            )
        for suffix in (
            "embedding_length", "block_count", "attention.head_count",
            "attention.head_count_kv", "feed_forward_length", "context_length",
        ):
            value = get_metadata_int(metadata, f"qwen2.{suffix}")
            if value is None or value <= 0:
                raise ValueError(f"Missing or invalid Qwen-VL {suffix}")
        config = build_config(metadata, "qwen2", dtype)
        name = get_metadata_str(reader, "general.name") or ""
        is_qwen3 = (
            arch in {"qwen3vl", "qwen3_vl"}
            or re.search(r"qwen3[-_ ]?vl", name, re.IGNORECASE) is not None
            or any(t.name.endswith(".attn_q_norm.weight") for t in reader.tensors)
        )
        family = "qwen3_vl" if is_qwen3 else "qwen2_vl"
        config.update({
            "model_type": f"{family}_text",
            "architectures": ["Qwen3VLTextModel" if is_qwen3 else "Qwen2VLTextModel"],
            "_gguf_architecture": arch,
            "gguf2mlx_component": "text_encoder",
            "attention_bias": not is_qwen3,
            "tie_word_embeddings": not any(t.name == "output.weight" for t in reader.tensors),
        })
        head_dim = get_metadata_int(metadata, "qwen2.attention.key_length")
        config["head_dim"] = head_dim or config["hidden_size"] // config["num_attention_heads"]
        field = metadata.get_field("qwen2.rope.dimension_sections")
        values = [] if field is None else field.contents()
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("Qwen-VL rope.dimension_sections must be an integer array")
        sections = list(values)
        # GGUF reserves a fourth, unused section for these three-axis models.
        if len(sections) == 4 and sections[-1] == 0:
            sections.pop()
        if (
            len(sections) != 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or v <= 0 for v in sections)
            or sum(sections) * 2 != config["head_dim"]
        ):
            raise ValueError(
                "Qwen-VL requires three positive rope.dimension_sections "
                "whose sum is half the attention head dimension"
            )
        config["rope_scaling"] = {
            "rope_type": "default",
            "mrope_section": [int(v) for v in sections],
        }
        if is_qwen3:
            config["rope_scaling"]["mrope_interleaved"] = True

        tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
        for key, token in {
            "image_token_id": "<|image_pad|>",
            "video_token_id": "<|video_pad|>",
            "vision_start_token_id": "<|vision_start|>",
            "vision_end_token_id": "<|vision_end|>",
        }.items():
            token_id = get_metadata_int(metadata, f"qwen2.{key}")
            if token_id is None:
                token_id = get_metadata_int(reader, f"tokenizer.ggml.{key}")
            if token_id is None and token in tokens:
                token_id = tokens.index(token)
            if token_id is None or not 0 <= token_id < config["vocab_size"]:
                raise ValueError(f"Missing or invalid Qwen-VL {key}; provide matching GGUF tokens")
            config[key] = token_id
        return config
