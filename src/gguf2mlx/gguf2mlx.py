#!/usr/bin/env python3
"""
GGUF to MLX Converter v2.0
Converts GGUF models to MLX format (safetensors) for Apple Silicon inference.

Phase 1: Real weight extraction, safetensors output, architecture detection,
         real tokenizer extraction.
"""

import argparse
import gc
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import numpy as np

# ---------------------------------------------------------------------------
# Required imports with friendly error messages
# ---------------------------------------------------------------------------

try:
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("❌ gguf library required. Install: pip install gguf>=0.18.0")
    sys.exit(1)

try:
    from safetensors.numpy import save_file as save_safetensors
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("❌ safetensors library required. Install: pip install safetensors")
    sys.exit(1)

# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------


def get_metadata_str(reader: GGUFReader, key: str) -> Optional[str]:
    """Extract a string metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val) if val is not None else None


def get_metadata_int(reader: GGUFReader, key: str) -> Optional[int]:
    """Extract an integer metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return int(val.flat[0]) if val.size > 0 else None
    if isinstance(val, (list, tuple)):
        return int(val[0]) if len(val) > 0 else None
    return int(val)


def get_metadata_float(reader: GGUFReader, key: str) -> Optional[float]:
    """Extract a float metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 else None
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else None
    return float(val)


def get_metadata_array_str(reader: GGUFReader, key: str) -> list[str]:
    """Extract a string array from GGUF fields (e.g., tokenizer tokens)."""
    field = reader.get_field(key)
    if field is None:
        return []
    try:
        vals = field.contents()
        if isinstance(vals, (list, np.ndarray)):
            return [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                for v in vals
            ]
        return []
    except Exception:
        return []


def get_metadata_array_int(reader: GGUFReader, key: str) -> list[int]:
    """Extract an integer array from GGUF fields (e.g., token types)."""
    field = reader.get_field(key)
    if field is None:
        return []
    try:
        vals = field.contents()
        if isinstance(vals, (list, np.ndarray)):
            return [int(v) for v in vals]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Architecture detection & config building
# ---------------------------------------------------------------------------

# Map GGUF architecture names to HuggingFace model types
ARCH_MAP: dict[str, str] = {
    "llama": "llama",
    "mistral": "mistral",
    "falcon": "falcon",
    "mpt": "mpt",
    "gptneox": "gpt_neox",
    "gpt2": "gpt2",
    "bert": "bert",
    "bloom": "bloom",
    "starcoder": "gpt_bigcode",
    "refact": "refact",
    "command-r": "cohere",
    "command-r-plus": "cohere",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "qwen3moe": "qwen3_moe",
    "phi3": "phi3",
    "phi": "phi",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "stablelm": "stablelm",
    "deepseek2": "deepseek_v2",
    "deepseek3": "deepseek_v3",
    "chatglm": "chatglm",
    "glm-dsa": "glm_moe_dsa",  # GLM-5.2: MLA + DSA + MoE + MTP (experimental)
    "baichuan": "baichuan",
    "xverse": "xverse",
    "orion": "orion",
    "bitnet": "bitnet",
    "plamo": "plamo",
    "codeshell": "codeshell",
    "minicpm": "minicpm",
    "minicpm3": "minicpm3",
    "t5": "t5",
    "jais": "jais",
    "olmo": "olmo",
    "olmo2": "olmo2",
    "openelm": "openelm",
    "dbrx": "dbrx",
    "grok-1": "grok",
    "arctic": "arctic",
    "nemotron": "nemotron",
    "exaone": "exaone",
    "granite": "granite",
    "smolm": "smolm",
    "chameleon": "chameleon",
}

# Popular GGUF naming patterns that do not directly include a GGUF architecture key.
# These are used only when `general.architecture` is missing.
MODEL_NAME_ARCH_FALLBACKS: list[tuple[str, str]] = [
    (r"\bdeepseek-r1-distill-qwen\b", "qwen2"),
    (r"\bdeepseek-r1-distill-llama\b", "llama"),
    (r"\bdeepseek-v3\b", "deepseek3"),
    (r"\bdeepseek-r1\b", "deepseek3"),
    (r"\bdeepseek-v2\b", "deepseek2"),
    (r"\bmixtral\b", "mistral"),
    (r"\bcommand-r\+", "command-r-plus"),
    (r"\bcommand-r\b", "command-r"),
    # Keep Yi matching anchored to the start to avoid generic false positives.
    (r"^\s*yi\b", "llama"),
]


def detect_architecture(reader: GGUFReader) -> str:
    """Detect model architecture from GGUF metadata."""
    arch = get_metadata_str(reader, "general.architecture")
    if arch:
        return arch
    # Try fallback based on model name
    name = get_metadata_str(reader, "general.name")
    if name:
        name_lower = name.lower()
        # Check specific popular-model fallbacks before generic substring matching.
        # Order matters: this preserves intended routing for names like
        # `deepseek-r1-distill-qwen` before broad ARCH_MAP substring checks.
        for pattern, mapped_arch in MODEL_NAME_ARCH_FALLBACKS:
            if re.search(pattern, name_lower):
                return mapped_arch
        for gguf_arch, hf_name in ARCH_MAP.items():
            if gguf_arch in name_lower:
                return gguf_arch
    return "llama"  # Safe default


def _build_glm_dsa_config(reader: GGUFReader, config: dict[str, Any]) -> dict[str, Any]:
    """Add GLM-5.2 (glm-dsa) specific fields: MLA + DSA indexer + MoE + MTP + IndexShare.

    Values default to the published zai-org/GLM-5.2 config when a metadata key is
    absent (the canonical GGUF producer does not exist yet, so keys may be partial).
    """
    arch = "glm-dsa"

    hidden = (get_metadata_int(reader, f"{arch}.embedding_length")
              or get_metadata_int(reader, "llama.embedding_length") or config["hidden_size"])
    n_heads = (get_metadata_int(reader, f"{arch}.attention.head_count")
               or get_metadata_int(reader, "llama.attention.head_count") or config["num_attention_heads"])
    config["hidden_size"] = hidden
    config["num_attention_heads"] = n_heads

    # --- Block-count arithmetic: GGUF block_count includes the MTP/NextN block(s) ---
    total_blocks = (get_metadata_int(reader, f"{arch}.block_count")
                    or get_metadata_int(reader, "llama.block_count") or config["num_hidden_layers"])
    num_nextn = (get_metadata_int(reader, f"{arch}.nextn_predict_layers")
                 or get_metadata_int(reader, f"{arch}.num_nextn_predict_layers") or 1)
    num_hidden = total_blocks - num_nextn
    if num_hidden < 1:  # be defensive if GGUF reports the transformer count directly
        num_hidden = total_blocks
        num_nextn = 0
    config["num_hidden_layers"] = num_hidden
    config["num_nextn_predict_layers"] = num_nextn

    # --- MLA (Multi-head Latent Attention, DeepSeek-V2/V3 family) ---
    qk_nope = get_metadata_int(reader, f"{arch}.attention.qk_nope_head_dim") or 192
    qk_rope = get_metadata_int(reader, f"{arch}.attention.qk_rope_head_dim") or 64
    v_head = (get_metadata_int(reader, f"{arch}.attention.value_length")
              or get_metadata_int(reader, f"{arch}.attention.v_head_dim") or qk_nope)
    config["q_lora_rank"] = get_metadata_int(reader, f"{arch}.attention.q_lora_rank") or 2048
    config["kv_lora_rank"] = get_metadata_int(reader, f"{arch}.attention.kv_lora_rank") or 512
    config["qk_nope_head_dim"] = qk_nope
    config["qk_rope_head_dim"] = qk_rope
    config["qk_head_dim"] = qk_nope + qk_rope
    config["v_head_dim"] = v_head
    config["head_dim"] = qk_nope  # GLM-5.2 sets head_dim = qk_nope_head_dim
    config["rope_interleave"] = True
    config["partial_rotary_factor"] = qk_rope / (qk_nope + qk_rope)

    # --- MoE: 3 dense + 75 sparse layers, 256 routed + 1 shared expert ---
    n_experts = get_metadata_int(reader, f"{arch}.expert_count") or 256
    n_shared = get_metadata_int(reader, f"{arch}.expert_shared_count") or 1
    n_per_tok = get_metadata_int(reader, f"{arch}.expert_used_count") or 8
    moe_ffn = (get_metadata_int(reader, f"{arch}.expert_feed_forward_length")
               or get_metadata_int(reader, f"{arch}.moe_intermediate_size") or 2048)
    dense_ffn = (get_metadata_int(reader, f"{arch}.feed_forward_length") or (hidden * 4))
    config["num_experts"] = n_experts
    config["num_experts_per_tok"] = n_per_tok
    config["n_shared_experts"] = n_shared
    config["moe_intermediate_size"] = moe_ffn
    config["intermediate_size"] = dense_ffn
    config["first_k_dense_replace"] = get_metadata_int(reader, f"{arch}.first_k_dense_replace") or 3
    config["topk_method"] = "noaux_tc"
    config["scoring_func"] = "sigmoid"
    config["norm_topk_prob"] = True
    config["routed_scaling_factor"] = get_metadata_float(reader, f"{arch}.routed_scaling_factor") or 2.5
    config["n_group"] = get_metadata_int(reader, f"{arch}.n_group") or 1
    config["topk_group"] = get_metadata_int(reader, f"{arch}.topk_group") or 1
    config["moe_layer_freq"] = 1
    config["decoder_sparse_step"] = 1
    config["mlp_only_layers"] = []

    # --- DSA lightning indexer + IndexShare (1-in-4 F/S pattern) ---
    config["index_head_dim"] = (get_metadata_int(reader, f"{arch}.attention.index_head_dim") or 128)
    config["index_n_heads"] = (get_metadata_int(reader, f"{arch}.attention.index_head_count")
                               or get_metadata_int(reader, f"{arch}.attention.index_n_heads") or 32)
    config["index_topk"] = (get_metadata_int(reader, f"{arch}.attention.index_top_k")
                            or get_metadata_int(reader, f"{arch}.attention.index_topk") or 2048)
    indexer_types = get_metadata_array_str(reader, f"{arch}.attention.indexer_types")
    if indexer_types:
        config["indexer_types"] = indexer_types
    mlp_layer_types = get_metadata_array_str(reader, f"{arch}.mlp_layer_types")
    if mlp_layer_types:
        config["mlp_layer_types"] = mlp_layer_types
    config["index_share_for_mtp_iteration"] = bool(
        get_metadata_int(reader, f"{arch}.index_share_for_mtp_iteration") or 0)

    config["tie_word_embeddings"] = False
    config["attention_bias"] = False
    config["model_type"] = "glm_moe_dsa"
    config["architectures"] = ["GlmMoeDsaForCausalLM"]
    return config


def build_config(reader: GGUFReader, arch: str) -> dict[str, Any]:
    """Build MLX-compatible config.json from GGUF metadata."""

    def _warn(key: str, value: Any) -> None:
        warnings.warn(f"  ⚠ '{key}' not found in GGUF metadata, using default: {value}")

    # --- Basic params ---
    vocab_size = get_metadata_int(reader, "llama.vocab_size") or get_metadata_int(
        reader, f"{arch}.vocab_size"
    )

    # If vocab_size not in metadata, infer from tokenizer tokens
    if vocab_size is None:
        tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
        if tokens:
            vocab_size = len(tokens)
        else:
            vocab_size = 32000
            _warn("vocab_size", 32000)

    hidden_size = get_metadata_int(reader, "llama.embedding_length") or get_metadata_int(
        reader, f"{arch}.embedding_length"
    )
    if hidden_size is None:
        hidden_size = 4096
        _warn("embedding_length", 4096)

    num_layers = get_metadata_int(reader, "llama.block_count") or get_metadata_int(
        reader, f"{arch}.block_count"
    )
    if num_layers is None:
        num_layers = 32
        _warn("block_count", 32)

    num_heads = get_metadata_int(reader, "llama.attention.head_count") or get_metadata_int(
        reader, f"{arch}.attention.head_count"
    )
    if num_heads is None:
        num_heads = 32
        _warn("head_count", 32)

    num_kv_heads = get_metadata_int(
        reader, "llama.attention.head_count_kv"
    ) or get_metadata_int(reader, f"{arch}.attention.head_count_kv") or num_heads

    # MoE: expert feed-forward length may differ from shared FFN
    if arch in ("qwen2moe", "qwen3moe", "deepseek2", "deepseek3", "dbrx", "grok-1"):
        ffn_size = get_metadata_int(
            reader, f"{arch}.expert_feed_forward_length"
        ) or get_metadata_int(reader, f"{arch}.feed_forward_length") or (hidden_size * 4)
        # Shared expert FFN (if present)
        shared_ffn_size = get_metadata_int(
            reader, f"{arch}.expert_shared_feed_forward_length"
        ) or ffn_size
    else:
        ffn_size = get_metadata_int(reader, "llama.feed_forward_length") or get_metadata_int(
            reader, f"{arch}.feed_forward_length"
        )
        if ffn_size is None:
            ffn_size = hidden_size * 4
        shared_ffn_size = ffn_size

    ctx_length = get_metadata_int(reader, "llama.context_length") or get_metadata_int(
        reader, f"{arch}.context_length"
    )
    if ctx_length is None:
        ctx_length = 4096
        _warn("context_length", 4096)

    rope_theta = get_metadata_float(reader, "llama.rope.freq_base") or get_metadata_float(
        reader, f"{arch}.rope.freq_base"
    )
    if rope_theta is None:
        rope_theta = 10000.0

    norm_eps = get_metadata_float(
        reader, "llama.attention.layer_norm_rms_epsilon"
    ) or get_metadata_float(reader, f"{arch}.attention.layer_norm_rms_epsilon")
    if norm_eps is None:
        norm_eps = 1e-6

    file_type = get_metadata_int(reader, "general.file_type") or 1
    model_name = get_metadata_str(reader, "general.name") or "unknown"
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id") or 1
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id") or 2

    hf_model_type = ARCH_MAP.get(arch, arch)

    # --- Build config ---
    # Detect tied embeddings (many Qwen, Gemma, etc. models have this)
    tie_embeddings = arch in (
        "qwen2", "qwen2moe", "gemma", "gemma2", "gemma3",
        "olmo", "olmo2", "openelm",
    )

    # Detect attention bias (Qwen, Gemma, etc.)
    attention_bias = arch in (
        "qwen2", "qwen2moe", "qwen3moe",
        "gemma", "gemma2", "gemma3",
    )

    # Convert model_type to CamelCase architecture class name
    arch_class = "".join(part.capitalize() for part in hf_model_type.split("_")) + "ForCausalLM"

    config = {
        "architectures": [arch_class],
        "model_type": hf_model_type,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "max_position_embeddings": ctx_length,
        "rms_norm_eps": norm_eps,
        "rope_theta": rope_theta,
        "vocab_size": vocab_size,
        "hidden_act": "silu",
        "tie_word_embeddings": tie_embeddings,
        "attention_bias": attention_bias,
        "torch_dtype": "float16",
        "transformers_version": "4.50.0",
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        # Extra metadata from source GGUF
        "_gguf_architecture": arch,
        "_gguf_file_type": file_type,
        "_original_name": model_name,
    }

    # --- Architecture-specific overrides ---
    if arch in ("qwen2moe", "deepseek2", "deepseek3", "qwen3moe", "dbrx", "grok-1"):
        num_experts = get_metadata_int(reader, f"{arch}.expert_count") or 8
        num_experts_per_tok = get_metadata_int(
            reader, f"{arch}.expert_used_count"
        ) or 2
        config["num_experts"] = num_experts
        config["num_experts_per_tok"] = num_experts_per_tok
        config["model_type"] = {
            "qwen2moe": "qwen2_moe",
            "qwen3moe": "qwen3_moe",
            "deepseek2": "deepseek_v2",
            "deepseek3": "deepseek_v3",
        }.get(arch, hf_model_type)

        # MoE-specific config fields
        config["moe_intermediate_size"] = ffn_size
        config["norm_topk_prob"] = arch in ("qwen3moe",)
        config["decoder_sparse_step"] = 1
        config["mlp_only_layers"] = []

        # Head dim (Qwen3, DeepSeek-V3 style)
        head_dim = get_metadata_int(
            reader, f"{arch}.attention.key_length"
        ) or get_metadata_int(reader, f"{arch}.attention.value_length") or (hidden_size // num_heads)
        if arch in ("qwen3moe", "deepseek3"):
            config["head_dim"] = head_dim

        # Shared expert config (Qwen3MoE, DeepSeek-V3)
        if arch in ("qwen3moe", "deepseek3"):
            config["shared_expert_intermediate_size"] = shared_ffn_size
            config["output_router_logits"] = False
            config["router_aux_loss_coef"] = 0.001

    # --- GLM-5.2 (glm-dsa): MLA + DSA + MoE + MTP + IndexShare ---
    if arch == "glm-dsa":
        _build_glm_dsa_config(reader, config)

    return config


# ---------------------------------------------------------------------------
# Tensor name mapping: GGUF → MLX/HuggingFace
# ---------------------------------------------------------------------------


def _map_llama_tensor_name(gguf_name: str) -> str:
    """Map a Llama-architecture GGUF tensor name to HuggingFace format."""
    # Embedding
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"

    # Output
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"

    # Blocks: blk.N.xxx → model.layers.N.xxx
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)
        if len(parts) < 3:
            return gguf_name
        layer_idx = parts[1]
        rest = parts[2]

        # Attention weights
        if rest == "attn_q.weight":
            return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        if rest == "attn_k.weight":
            return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        if rest == "attn_v.weight":
            return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if rest == "attn_output.weight":
            return f"model.layers.{layer_idx}.self_attn.o_proj.weight"

        # Attention biases (some architectures have these)
        if rest == "attn_q.bias":
            return f"model.layers.{layer_idx}.self_attn.q_proj.bias"
        if rest == "attn_k.bias":
            return f"model.layers.{layer_idx}.self_attn.k_proj.bias"
        if rest == "attn_v.bias":
            return f"model.layers.{layer_idx}.self_attn.v_proj.bias"
        if rest == "attn_output.bias":
            return f"model.layers.{layer_idx}.self_attn.o_proj.bias"

        # FFN
        if rest == "ffn_gate.weight":
            return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if rest == "ffn_up.weight":
            return f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if rest == "ffn_down.weight":
            return f"model.layers.{layer_idx}.mlp.down_proj.weight"

        # MoE: expert router gate
        if rest == "ffn_gate_inp.weight":
            return f"model.layers.{layer_idx}.mlp.gate.weight"

        # MoE: stacked expert weights (3D: [num_experts, out, in])
        # mlx-lm expects switch_mlp.*.weight for stacked format
        if rest == "ffn_gate_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.weight"
        if rest == "ffn_down_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.weight"
        if rest == "ffn_up_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.up_proj.weight"

        # QK normalization (Qwen3, Qwen3MoE)
        if rest == "attn_q_norm.weight":
            return f"model.layers.{layer_idx}.self_attn.q_norm.weight"
        if rest == "attn_k_norm.weight":
            return f"model.layers.{layer_idx}.self_attn.k_norm.weight"

        # Norms
        if rest == "attn_norm.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        if rest == "attn_norm_2.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"

        # Norm biases (rare but possible)
        if rest == "attn_norm.bias":
            return f"model.layers.{layer_idx}.input_layernorm.bias"
        if rest == "ffn_norm.bias":
            return f"model.layers.{layer_idx}.post_attention_layernorm.bias"

    return gguf_name


# MLA-family tensor fragments shared by DeepSeek-V2/V3 and GLM-DSA.
# Maps the GGUF block-relative fragment -> HF self_attn.* suffix.
# NOTE: split attn_k_b / attn_v_b are NOT here — they require concatenation and
# are handled in the convert loop (_plan_tensor_emit / _reconstruct_kv_b).
_MLA_ATTN_MAP = {
    "attn_q_a": "q_a_proj",
    "attn_q_a_norm": "q_a_layernorm",
    "attn_q_b": "q_b_proj",
    "attn_kv_a_mqa": "kv_a_proj_with_mqa",
    "attn_kv_a_norm": "kv_a_layernorm",
    "attn_kv_b": "kv_b_proj",  # combined MLA kv_b (GLM-DSA primary form)
    "attn_out": "o_proj",
}
# Indexer submodule: GGUF blk.N.indexer.<frag> -> self_attn.indexer.<dst>
_INDEXER_FRAG_MAP = {
    "attn_q_b": "wq_b",
    "attn_k": "wk",
    "proj": "weights_proj",
}
# Archetypes that use the MLA tensor family.
_MLA_ARCHES = ("glm-dsa", "deepseek2", "deepseek3", "glm4moe")


def _map_mla_tensor_name(gguf_name: str) -> Optional[str]:
    """Map MLA-family tensor names (DeepSeek-V2/V3, GLM-DSA) to HF format.

    Returns None for anything it does not specifically own, so the caller can
    fall back to the llama mapper for shared concepts (norms, dense FFN,
    ffn_gate_inp router, embeddings, output).
    """
    # Root tensors (own them so MLA arches don't depend on llama for these)
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"

    if not gguf_name.startswith("blk."):
        return None
    parts = gguf_name.split(".", 2)
    if len(parts) < 3:
        return None
    layer_idx, rest = parts[1], parts[2]

    # MLA attention projections (combined kv_b form)
    for src, dst in _MLA_ATTN_MAP.items():
        if rest == f"{src}.weight":
            return f"model.layers.{layer_idx}.self_attn.{dst}.weight"

    # DSA lightning indexer submodule: blk.N.indexer.<frag>.[weight|bias]
    if rest.startswith("indexer."):
        sub = rest[len("indexer."):]
        suffix = None
        for sfx in (".weight", ".bias"):
            if sub.endswith(sfx):
                frag, suffix = sub[: -len(sfx)], sfx[1:]
                break
        if suffix is None:
            return None
        if frag == "k_norm":
            return f"model.layers.{layer_idx}.self_attn.indexer.k_norm.{suffix}"
        if frag in _INDEXER_FRAG_MAP:
            return (f"model.layers.{layer_idx}.self_attn.indexer."
                    f"{_INDEXER_FRAG_MAP[frag]}.{suffix}")
        return None

    # Shared expert: blk.N.ffn_{gate,up,down}_shexp.weight
    m = re.match(r"ffn_(gate|up|down)_shexp\.weight$", rest)
    if m:
        return f"model.layers.{layer_idx}.mlp.shared_experts.{m.group(1)}_proj.weight"

    # noaux_tc gate correction bias: blk.N.exp_probs_b -> mlp.gate.e_score_correction_bias
    if rest == "exp_probs_b":
        return f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"

    # NextN/MTP shared head norm: blk.N.nextn_shared_head_norm.weight
    if rest == "nextn_shared_head_norm.weight":
        return f"model.layers.{layer_idx}.shared_head.norm.weight"

    return None


def _map_tensor_name(gguf_name: str, arch: str) -> str:
    """Map a GGUF tensor name to HuggingFace format based on architecture.

    MLA-family arches (deepseek2/3, glm4moe, glm-dsa) get the MLA mapper first;
    anything it does not own falls through to the llama mapper. Multi-tensor
    transforms (kv_b concat, per-expert split) are handled separately in the
    convert loop by _plan_tensor_emit.
    """
    if arch in _MLA_ARCHES:
        mapped = _map_mla_tensor_name(gguf_name)
        if mapped is not None:
            return mapped
    return _map_llama_tensor_name(gguf_name)


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------


_GLM_DSA_TEMPLATE_PATH = Path(__file__).parent / "data" / "glm_dsa_chat_template.jinja"


def _get_chat_template(reader: GGUFReader, arch: str) -> Optional[str]:
    """Return a chat template: GGUF metadata first, then a canonical GLM fallback."""
    tmpl = get_metadata_str(reader, "tokenizer.chat_template")
    if tmpl:
        return tmpl
    if arch in ("glm-dsa", "chatglm") and _GLM_DSA_TEMPLATE_PATH.exists():
        return _GLM_DSA_TEMPLATE_PATH.read_text(encoding="utf-8")
    return None


def extract_tokenizer(reader: GGUFReader, output_dir: Path, arch: str = "llama") -> None:
    """Extract tokenizer from GGUF metadata and save standard files."""
    model_type = get_metadata_str(reader, "tokenizer.ggml.model") or "bpe"
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id") or 1
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id") or 2
    pad_id = get_metadata_int(reader, "tokenizer.ggml.padding_token_id") or 0

    tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
    token_types = get_metadata_array_int(reader, "tokenizer.ggml.token_type")
    merges = get_metadata_array_str(reader, "tokenizer.ggml.merges")
    scores = [
        float(s)
        for s in get_metadata_array_str(reader, "tokenizer.ggml.scores")
    ] if reader.get_field("tokenizer.ggml.scores") else []

    if not tokens:
        print("  ⚠ No tokenizer tokens found in GGUF — creating minimal tokenizer")
        tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        token_types = [0, 3, 3, 3]
        bos_id, eos_id, pad_id = 1, 2, 3

    # --- Fix BOS/EOS for Qwen, DeepSeek, and similar families ---
    # These models use special tokens like <|endoftext|> (BOS) and <|im_end|> (EOS)
    # but GGUF files often omit bos_token_id / eos_token_id, or set them to wrong defaults (1, 2).
    #
    # Known special token names by role:
    SPECIAL_BOS_CANDIDATES = [
        "<|endoftext|>", "<s>", "<|begin_of_text|>", "<|startoftext|>",
    ]
    SPECIAL_EOS_CANDIDATES = [
        "<|im_end|>", "</s>", "<|end_of_text|>", "<|eot_id|>", "<|end|>",
    ]

    if tokens and (bos_id in (0, 1, 2, 3)):
        found = False
        for candidate in SPECIAL_BOS_CANDIDATES:
            if candidate in tokens:
                bos_id = tokens.index(candidate)
                print(f"  ✓ Fixed bos_token_id: {bos_id} ({candidate})")
                found = True
                break
        if not found:
            # Try uppercase versions
            for candidate in SPECIAL_BOS_CANDIDATES:
                for i, tok in enumerate(tokens):
                    if tok.upper() == candidate.upper():
                        bos_id = i
                        print(f"  ✓ Fixed bos_token_id: {bos_id} ({tok})")
                        found = True
                        break
                if found:
                    break

    if tokens and (eos_id in (0, 1, 2, 3)):
        found = False
        for candidate in SPECIAL_EOS_CANDIDATES:
            if candidate in tokens:
                eos_id = tokens.index(candidate)
                print(f"  ✓ Fixed eos_token_id: {eos_id} ({candidate})")
                found = True
                break
        if not found:
            for candidate in SPECIAL_EOS_CANDIDATES:
                for i, tok in enumerate(tokens):
                    if tok.upper() == candidate.upper():
                        eos_id = i
                        print(f"  ✓ Fixed eos_token_id: {eos_id} ({tok})")
                        found = True
                        break
                if found:
                    break

    vocab_size = len(tokens)
    print(f"  Extracted tokenizer: {vocab_size} tokens, model={model_type}")

    # --- tokenizer_config.json ---
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": tokens[bos_id] if bos_id < vocab_size else "<s>",
        "eos_token": tokens[eos_id] if eos_id < vocab_size else "</s>",
        "unk_token": tokens[0] if tokens else "<unk>",
        "pad_token": tokens[pad_id] if pad_id < vocab_size else "<pad>",
        "model_max_length": 131072,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }

    if model_type == "llama" or model_type == "bpe":
        tokenizer_config.update(
            {
                "model_type": "bpe",
                "tokenizer_class": "LlamaTokenizerFast"
                if "llama" in model_type
                else "PreTrainedTokenizerFast",
            }
        )

    # --- Chat template (GGUF metadata first, canonical GLM fallback for glm-dsa/chatglm) ---
    chat_template = _get_chat_template(reader, arch)
    if chat_template:
        tokenizer_config["chat_template"] = chat_template
        with open(output_dir / "chat_template.jinja", "w", encoding="utf-8") as f:
            f.write(chat_template)
        print("  ✓ Saved chat_template.jinja")

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved tokenizer_config.json")

    # --- special_tokens_map.json ---
    special_tokens = {
        "bos_token": tokens[bos_id] if bos_id < vocab_size else "<s>",
        "eos_token": tokens[eos_id] if eos_id < vocab_size else "</s>",
        "unk_token": tokens[0] if tokens else "<unk>",
    }
    if pad_id < vocab_size and tokens[pad_id]:
        special_tokens["pad_token"] = tokens[pad_id]

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved special_tokens_map.json")

    # --- vocab.json (word → id mapping) ---
    vocab = {}
    for i, token in enumerate(tokens):
        if i < vocab_size:
            # Normalize token: GGUF stores bytes, HF expects strings
            if isinstance(token, str):
                vocab[token] = i

    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved vocab.json ({len(vocab)} entries)")

    # --- merges.txt (for BPE tokenizers) ---
    if model_type in ("bpe", "gpt2") and merges:
        # GGUF stores merges with space characters
        merges_path = output_dir / "merges.txt"
        with open(merges_path, "w") as f:
            # No version header — HuggingFace GPT-2 tokenizer expects raw merges
            for merge in merges:
                if isinstance(merge, bytes):
                    merge = merge.decode("utf-8", errors="replace")
                f.write(merge + "\n")
        print(f"  ✓ Saved merges.txt ({len(merges)} merges)")

    # --- tokenizer.json (for fast tokenizers) ---
    tokenizer_json = _build_tokenizer_json(
        tokens, token_types, merges, scores, model_type, bos_id, eos_id, pad_id
    )
    if tokenizer_json:
        with open(output_dir / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        print("  ✓ Saved tokenizer.json")


def _build_tokenizer_json(
    tokens: list[str],
    token_types: list[int],
    merges: list[str],
    scores: list[float],
    model_type: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> dict:
    """Build a complete tokenizer.json for HuggingFace tokenizers."""
    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    # Token type codes: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
    # (mapping kept as a comment; token_types per-token is consumed below)
    added_tokens = []
    normal_tokens = []
    for i, token in enumerate(tokens):
        tt = token_types[i] if i < len(token_types) else 1
        # Control/special tokens go in added_tokens
        if tt in (3,) or i in (bos_id, eos_id, pad_id):
            special = i in (bos_id, eos_id, pad_id)
            added_tokens.append(
                {
                    "id": i,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": special,
                }
            )
        else:
            normal_tokens.append(token)

    # Build model block
    if model_type in ("bpe", "gpt2"):
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[0] if tokens else "<unk>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    elif model_type == "llama":
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "▁",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    else:
        # Generic fallback
        model_block = {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges if merges else [],
        }

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": model_block,
    }

    return tokenizer_json


# ---------------------------------------------------------------------------
# Weight extraction & conversion
# ---------------------------------------------------------------------------


def _read_mla_dims(reader: GGUFReader, arch: str) -> dict[str, int]:
    """Read MLA dims needed to reconstruct kv_b_proj from split k_b/v_b."""
    def _g(key: str) -> Optional[int]:
        return get_metadata_int(reader, f"{arch}.{key}") or get_metadata_int(reader, f"llama.{key}")
    n_heads = _g("attention.head_count") or 64
    qk_nope = _g("attention.qk_nope_head_dim") or 192
    v_head = _g("attention.value_length") or _g("attention.v_head_dim") or qk_nope
    return {"num_heads": n_heads, "qk_nope_head_dim": qk_nope, "v_head_dim": v_head}


def _reconstruct_kv_b(k_b: np.ndarray, v_b: np.ndarray,
                      n_heads: int, dk_nope: int, dv: int) -> np.ndarray:
    """Reconstruct HF kv_b_proj from GGUF split attn_k_b / attn_v_b.

    HF kv_b_proj.weight is [n_heads*(dk_nope+dv), kv_lora] and is consumed by
    reshaping per-head to [n_heads, dk_nope+dv, kv_lora], then splitting nope-K
    (first dk_nope) from V (last dv) within each head. GGUF stores k_b
    ([n_heads*dk_nope, kv_lora]) and v_b ([n_heads*dv, kv_lora]) head-major, so
    the inverse is: reshape each to [n_heads, dim, kv_lora], concat along dim 1,
    flatten back to [n_heads*(dk_nope+dv), kv_lora].
    """
    if k_b.ndim != 2 or v_b.ndim != 2:
        raise ValueError(f"kv_b reconstruction expects 2D k_b/v_b, got {k_b.ndim}D/{v_b.ndim}D")
    kv_lora = k_b.shape[1]
    if v_b.shape[1] != kv_lora:
        raise ValueError(f"kv_lora mismatch: k_b has {kv_lora}, v_b has {v_b.shape[1]}")
    k = k_b.reshape(n_heads, dk_nope, kv_lora)
    v = v_b.reshape(n_heads, dv, kv_lora)
    kv = np.concatenate([k, v], axis=1)  # [n_heads, dk_nope+dv, kv_lora]
    return kv.reshape(n_heads * (dk_nope + dv), kv_lora).astype(k_b.dtype)


def _plan_tensor_emit(gguf_name: str, arr: np.ndarray, arch: str,
                      mla_dims: dict[str, int],
                      pending_kv_b: dict[str, dict[str, np.ndarray]]
                      ) -> list[tuple[str, np.ndarray]]:
    """Decide output (hf_name, arr) pairs for one source GGUF tensor.

    Handles arch-specific multi-tensor transforms:
      * Split attn_k_b/attn_v_b (deepseek2/3, glm4moe, and glm-dsa fallback when
        the combined attn_kv_b is absent) -> combined kv_b_proj via per-head
        interleave (_reconstruct_kv_b).
      * glm-dsa stacked ffn_*_exps -> per-expert mlp.experts.{e}.{kind}_proj.

    Returns [] when the tensor is buffered (waiting for its kv_b pair).
    """
    # --- Split kv_b -> combined kv_b_proj ---
    m = re.match(r"blk\.(\d+)\.attn_(k_b|v_b)(?:\.weight)?$", gguf_name)
    if m and arch in ("deepseek2", "deepseek3", "glm-dsa", "glm4moe"):
        layer_idx = m.group(1)
        which = "k" if m.group(2) == "k_b" else "v"
        buf = pending_kv_b.setdefault(layer_idx, {})
        buf[which] = arr
        if "k" in buf and "v" in buf:
            combined = _reconstruct_kv_b(
                buf["k"], buf["v"], mla_dims["num_heads"],
                mla_dims["qk_nope_head_dim"], mla_dims["v_head_dim"])
            del pending_kv_b[layer_idx]
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", combined)]
        return []  # buffered until the pair arrives

    # --- glm-dsa per-expert split: stacked ffn_*_exps [n_exp, out, in] ---
    if arch == "glm-dsa":
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps(?:\.weight)?$", gguf_name)
        if m and arr.ndim == 3:
            layer_idx, kind = m.group(1), m.group(2)
            return [
                (f"model.layers.{layer_idx}.mlp.experts.{e}.{kind}_proj.weight", arr[e])
                for e in range(arr.shape[0])
            ]

    # --- default 1:1 rename ---
    return [(_map_tensor_name(gguf_name, arch), arr)]


def _detect_full_indexer_layers(all_keys: list[str]) -> list[int]:
    """Layers that own a DSA indexer (Full layers in the IndexShare F/S pattern).

    A layer is 'Full' iff it emitted at least one ``self_attn.indexer.*`` tensor;
    'Shared' layers have none (they reuse a preceding Full layer's top-k indices).
    """
    return sorted({
        int(k.split(".")[2]) for k in all_keys
        if k.startswith("model.layers.") and ".self_attn.indexer." in k
    })


def extract_and_convert_weights(
    reader: GGUFReader, arch: str, output_dir: Path, dtype: str = "float16"
) -> dict[str, np.ndarray]:
    """Extract GGUF tensors, dequantize, rename, and save as safetensors."""

    print(f"\n  Converting {len(reader.tensors)} tensors...")
    print(f"  Output dtype: {dtype}")

    np_dtype = np.float16 if dtype == "float16" else np.float32
    weights: dict[str, np.ndarray] = {}
    all_keys: list[str] = []

    skipped = 0
    total_bytes_in = 0
    total_bytes_out = 0
    shard_idx = 1
    current_shard_bytes = 0
    max_shard_bytes = int(4.5 * 1e9)  # 4.5 GB per shard max for safetensors

    # Progress bar
    pbar = tqdm(total=len(reader.tensors), desc="  Converting", unit="tensor")

    def _shard_filename(idx: int, total_final: int | None = None) -> str:
        """Generate shard filename. When total is unknown, use NNNNN placeholder."""
        if total_final is None:
            return f"model-{idx:05d}-of-NNNNN.safetensors"
        return f"model-{idx:05d}-of-{total_final:05d}.safetensors"

    def _flush_shard(
        shard_weights: dict[str, np.ndarray], shard_idx: int, total_shards: int | None
    ) -> int:
        """Write current shard to disk, clear dict, return bytes written. Returns byte count."""
        if not shard_weights:
            return 0
        path = output_dir / _shard_filename(shard_idx, total_shards)
        save_safetensors(shard_weights, str(path))
        n_bytes = sum(arr.nbytes for arr in shard_weights.values())
        shard_weights.clear()
        gc.collect()
        return n_bytes

    # MLA dims needed for kv_b reconstruction (split k_b/v_b -> combined kv_b_proj)
    mla_dims = _read_mla_dims(reader, arch)
    # Buffer for split kv_b pairs (deepseek2/3; glm-dsa fallback when combined absent)
    pending_kv_b: dict[str, dict[str, np.ndarray]] = {}

    for i, tensor in enumerate(reader.tensors):
        gguf_name = tensor.name
        qtype = tensor.tensor_type
        logical_shape = tuple(tensor.shape)
        n_bytes = tensor.n_bytes
        total_bytes_in += n_bytes

        # Progress indicator
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i + 1}/{len(reader.tensors)}] Processing...")

        try:
            # Dequantize if needed
            qtype_val = int(qtype) if hasattr(qtype, "value") else int(qtype)
            raw_data = tensor.data

            if qtype_val == 0:  # F32
                arr = np.array(raw_data, dtype=np.float32).reshape(logical_shape)
                if dtype == "float16":
                    arr = arr.astype(np.float16)
                # F32/F16 tensors use GGUF layout [in_features, out_features],
                # need transpose to HF layout [out_features, in_features]
                if arr.ndim == 2:
                    arr = arr.T
            elif qtype_val == 1:  # F16
                arr = np.array(raw_data, dtype=np.float16).reshape(logical_shape)
                if arr.ndim == 2:
                    arr = arr.T
            elif qtype_val == 28:  # F64
                arr = np.array(raw_data, dtype=np.float64).reshape(logical_shape)
                if arr.ndim == 2:
                    arr = arr.T
                arr = arr.astype(np_dtype)
            elif qtype_val in (24, 25, 26, 27):  # I8, I16, I32, I64
                int_dtype_map = {24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64}
                arr = np.array(raw_data, dtype=int_dtype_map.get(qtype_val, np.int32))
                arr = arr.reshape(logical_shape).astype(np_dtype)
                if arr.ndim == 2:
                    arr = arr.T
            else:
                # Quantized: gguf's dequantize already returns [out_features, in_features]
                # (HF layout), so don't reshape or transpose
                try:
                    ggml_qtype = (
                        qtype if isinstance(qtype, GGMLQuantizationType) else GGMLQuantizationType(qtype_val)
                    )
                    arr = dequantize(raw_data, ggml_qtype)
                    # dequantize already returns correct shape, no .reshape needed
                    arr = arr.astype(np_dtype)
                except Exception as e:
                    print(f"    ⚠ Failed to dequantize {gguf_name} ({qtype}): {e}")
                    skipped += 1
                    continue

            # Determine output tensors (may be 0, 1, or many for arch-specific
            # transforms such as kv_b concat and per-expert split).
            emit_pairs = _plan_tensor_emit(gguf_name, arr, arch, mla_dims, pending_kv_b)
            for hf_name, out_arr in emit_pairs:
                weights[hf_name] = out_arr
                all_keys.append(hf_name)
                total_bytes_out += out_arr.nbytes
                current_shard_bytes += out_arr.nbytes

                # Shard when approaching the per-shard byte limit
                if current_shard_bytes >= max_shard_bytes:
                    n_bytes = _flush_shard(weights, shard_idx, None)
                    print(f"\n    ✓ Shard {shard_idx}: {len(all_keys)} tensors so far, {n_bytes / 1e9:.2f} GB")
                    shard_idx += 1
                    current_shard_bytes = 0
                    weights = {}
            pbar.update(1)

        except Exception as e:
            print(f"    ⚠ Error processing {gguf_name}: {e}")
            skipped += 1
            continue

    pbar.close()

    # --- Flush any orphaned split-kv_b pairs (defensive; shouldn't happen on valid GGUF) ---
    if pending_kv_b:
        for layer_idx, buf in pending_kv_b.items():
            if "k" in buf and "v" in buf:
                combined = _reconstruct_kv_b(
                    buf["k"], buf["v"], mla_dims["num_heads"],
                    mla_dims["qk_nope_head_dim"], mla_dims["v_head_dim"])
                hf_name = f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"
                weights[hf_name] = combined
                all_keys.append(hf_name)
                total_bytes_out += combined.nbytes
                print(f"    ⚠ Flushed orphaned kv_b for layer {layer_idx}")
            else:
                missing = "v" if "k" in buf else "k"
                print(f"    ⚠ Unpaired {missing}_b at layer {layer_idx} — discarded")

    # --- IndexShare invariant: log Full-indexer layer count (glm-dsa) ---
    if arch == "glm-dsa":
        f_layers = _detect_full_indexer_layers(all_keys)
        if f_layers:
            print(f"    ℹ IndexShare: {len(f_layers)} Full-indexer layer(s): {f_layers}")
        else:
            print("    ℹ IndexShare: no indexer tensors found (all-Shared or non-DSA GGUF)")

    # --- Save remaining tensors as final shard ---
    if weights:
        n_bytes = _flush_shard(weights, shard_idx, None)
        if n_bytes:
            print(f"    ✓ Shard {shard_idx}: {len(all_keys)} tensors total, {n_bytes / 1e9:.2f} GB")
            shard_idx += 1

    if not all_keys:
        raise RuntimeError("No weights extracted!")

    # --- Rename shards with correct total-count filenames ---
    shard_files = sorted(
        output_dir.glob("model-*-of-NNNNN.safetensors"),
        key=lambda p: int(p.stem.split("-")[1])
    )
    total_shards = len(shard_files)
    weight_map: dict[str, str] = {}

    for i, old_path in enumerate(shard_files, 1):
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        new_path = output_dir / new_name
        old_path.rename(new_path)

        # Read keys from shard to build weight map
        with safe_open(str(new_path), framework="np") as f:
            for key in f.keys():
                weight_map[key] = new_name

    # --- Save index ---
    index_json = {
        "metadata": {"total_size": total_bytes_out},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)

    print(f"\n  ✓ Saved {len(all_keys)} weight tensors ({total_shards} shards)")
    print(f"    Total input:  {total_bytes_in / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {total_bytes_out / 1e9:.2f} GB (safetensors)")
    if skipped:
        print(f"    ⚠ Skipped {skipped} tensors due to errors")

    return weights


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(gguf_path: str, output_dir: str, dtype: str = "float16") -> bool:
    """Convert a GGUF file to MLX-compatible safetensors format."""

    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = gguf_file.stem

    print("=" * 60)
    print("GGUF → MLX Converter v2.0")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Step 1: Open GGUF
    print("\n[1/5] Reading GGUF file...")
    reader = GGUFReader(str(gguf_path))
    print(
        f"  ✓ GGUF version {reader.fields['GGUF.version'].contents()}, "
        f"{len(reader.tensors)} tensors, "
        f"{len(reader.fields)} metadata fields"
    )
    print(f"  File size: {gguf_file.stat().st_size / 1e9:.2f} GB")

    # Step 2: Detect architecture & build config
    print("\n[2/5] Detecting architecture...")
    arch = detect_architecture(reader)
    hf_type = ARCH_MAP.get(arch, arch)
    model_name_full = get_metadata_str(reader, "general.name") or model_name
    print(f"  Architecture: {arch} (HF type: {hf_type})")
    print(f"  Model name:   {model_name_full}")

    config = build_config(reader, arch)
    print(
        f"  Config: {config['num_hidden_layers']} layers, "
        f"{config['hidden_size']} hidden, "
        f"{config['num_attention_heads']} heads, "
        f"{config['vocab_size']} vocab"
    )
    if "num_experts" in config:
        print(f"  MoE: {config['num_experts']} experts, top-{config['num_experts_per_tok']}")

    try:
        file_type = reader.get_field("general.file_type")
        if file_type:
            ft = file_type.contents()
            ft_names = {
                1: "F16", 2: "Q4_0", 3: "Q4_1",
                7: "Q8_0", 10: "Q2_K", 12: "Q4_K",
                13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
                17: "IQ2_XS", 19: "IQ1_S", 20: "IQ4_NL",
            }
            print(f"  Source quantization: {ft_names.get(int(ft), f'unknown({ft})')}")
    except Exception:
        pass

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ Saved config.json")

    # GLM-5.2: write generation_config.json with the multi-EOS array (matches
    # the published zai-org/GLM-5.2 generation_config.json).
    if arch == "glm-dsa":
        gen_cfg = {
            "_from_model_config": True,
            "eos_token_id": [154820, 154827, 154829],
            "pad_token_id": 154820,
            "temperature": 1.0,
            "top_p": 0.95,
            "transformers_version": "5.12.0",
        }
        with open(output_path / "generation_config.json", "w") as f:
            json.dump(gen_cfg, f, indent=2)
        print("  ✓ Saved generation_config.json (GLM-5.2 multi-EOS)")

    # Step 3: Extract tokenizer
    print("\n[3/5] Extracting tokenizer...")
    extract_tokenizer(reader, output_path, arch)

    # Step 4: Extract, dequantize, and convert weights
    print("\n[4/5] Extracting and converting weights...")
    try:
        extract_and_convert_weights(reader, arch, output_path, dtype)
    except Exception as e:
        print(f"❌ Weight extraction failed: {e}")
        return False

    # Step 5: Verify output
    print("\n[5/5] Finalizing...")
    # Index file already created by extract_and_convert_weights
    index_path = output_path / "model.safetensors.index.json"
    if not index_path.exists():
        print("❌ model.safetensors.index.json was not created")
        return False

    with open(index_path) as f:
        index_data = json.load(f)
    num_keys = len(index_data.get("weight_map", {}))
    print(f"  ✓ Index file: {num_keys} keys across {len(set(index_data['weight_map'].values()))} shards")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Architecture:     {arch} → {hf_type}")
    print("  Files generated:")
    for f_path in sorted(output_path.iterdir()):
        size = f_path.stat().st_size
        if size > 1_000_000_000:
            size_str = f"{size / 1e9:.2f} GB"
        elif size > 1_000_000:
            size_str = f"{size / 1e6:.1f} MB"
        elif size > 1000:
            size_str = f"{size / 1000:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"    - {f_path.name} ({size_str})")
    print("=" * 60)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GGUF to MLX Converter — Convert GGUF models to MLX safetensors format"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input GGUF file path"
    )
    parser.add_argument(
        "--output", "-o", help="Output MLX directory"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Output data type (default: float16)",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight extraction (metadata + tokenizer only, for inspection)",
    )

    args = parser.parse_args()

    # Auto-derive output directory from input filename if not specified
    if args.output is None:
        args.output = Path(args.input).stem + "-mlx"

    if args.skip_weights:
        # Just dump info
        reader = GGUFReader(args.input)
        print(f"Architecture: {detect_architecture(reader)}")
        print(f"Tensors: {len(reader.tensors)}")
        print(f"Fields: {len(reader.fields)}")
        for name in sorted(reader.fields.keys()):
            print(f"  {name}")
        return

    success = convert(args.input, args.output, args.dtype)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
