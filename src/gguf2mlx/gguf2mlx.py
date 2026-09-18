"""
GGUF to MLX Converter v2.0
Converts GGUF models to MLX format (safetensors) for Apple Silicon inference.

Phase 1: Real weight extraction, safetensors output, architecture detection,
         real tokenizer extraction.
"""

import argparse
import gc
import hashlib
import json
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from tqdm import tqdm

from .rust_backend import detect_architecture as rust_detect_architecture

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
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("❌ safetensors library required. Install: pip install safetensors")
    sys.exit(1)

try:
    from mlx_lm import convert as mlx_lm_convert
except ImportError:
    mlx_lm_convert = None

# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------


def get_metadata_str(reader: GGUFReader, key: str) -> str | None:
    """Extract a string metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val) if val is not None else None


def get_metadata_int(reader: GGUFReader, key: str) -> int | None:
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


def get_metadata_float(reader: GGUFReader, key: str) -> float | None:
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


def get_metadata_bool(reader: GGUFReader, key: str) -> bool | None:
    """Extract a boolean metadata value while preserving an explicit false value."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return bool(val.flat[0]) if val.size > 0 else None
    if isinstance(val, (list, tuple)):
        return bool(val[0]) if len(val) > 0 else None
    return bool(val)


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
    except (AttributeError, TypeError, ValueError):
        return []


def get_metadata_array_float(reader: GGUFReader, key: str) -> list[float]:
    """Extract a float array from GGUF metadata."""
    field = reader.get_field(key)
    if field is None:
        return []
    try:
        vals = field.contents()
        if isinstance(vals, (list, np.ndarray)):
            return [float(v) for v in vals]
        return []
    except (AttributeError, TypeError, ValueError):
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
    except (AttributeError, TypeError, ValueError):
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
    "phi2": "phi",
    "phi": "phi",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "stablelm": "stablelm",
    "deepseek2": "deepseek_v2",
    "deepseek3": "deepseek_v3",
    "chatglm": "chatglm",
    "glm-dsa": "glm_moe_dsa",  # GLM-5.2: MLA + DSA + MoE + MTP (experimental)
    "glm4moe": "glm4_moe",
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

# Architectures whose tensor layouts are explicitly handled below. Other names
# may be detected for inspection, but conversion must not silently emit an
# invalid Llama-layout model.
CONVERTIBLE_ARCHES = {
    "deepseek2",
    "deepseek3",
    "glm-dsa",
    "glm4moe",
    "gemma",
    "llama",
    "mistral",
    "phi3",
    "qwen2",
    "qwen2moe",
    "qwen3moe",
    "stablelm",
}

STRICT_ADAPTER_ARCHES = {"gemma", "phi3"}

SUPPORTED_MLX_LM_Q_GROUP_SIZES = {32, 64, 128}
DIRECT_QUANT_SUPPORTED_ARCHES = {"llama", "gemma", "mistral", "qwen2", "stablelm"}
DIRECT_QUANT_SUPPORTED_SOURCE_QTYPES = {
    2,   # Q4_0
    3,   # Q4_1
    6,   # Q5_0
    7,   # Q5_1
    8,   # Q8_0
    10,  # Q2_K
    11,  # Q3_K
    12,  # Q4_K
    13,  # Q5_K
    14,  # Q6_K
    15,  # Q8_K
    30,  # BF16
}
DIRECT_QUANT_DEFAULT_MAX_SHARD_BYTES = 256 * 1024 * 1024

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
    name = get_metadata_str(reader, "general.name")

    rust_arch = rust_detect_architecture(arch, name)
    if rust_arch:
        return rust_arch

    if arch:
        return arch
    # Try fallback based on model name
    if name:
        name_lower = name.lower()
        # Check specific popular-model fallbacks before generic substring matching.
        # Order matters: this preserves intended routing for names like
        # `deepseek-r1-distill-qwen` before broad ARCH_MAP substring checks.
        for pattern, mapped_arch in MODEL_NAME_ARCH_FALLBACKS:
            if re.search(pattern, name_lower):
                return mapped_arch
        for gguf_arch in sorted(ARCH_MAP, key=len, reverse=True):
            if gguf_arch in name_lower:
                return gguf_arch
    return "unknown"


def validate_architecture_variant(reader: GGUFReader, arch: str) -> str | None:
    """Return an actionable error when a recognized architecture variant is unverified."""
    if arch == "phi3" and any(
        tensor.name in {"rope_factors_long", "rope_factors_short"}
        for tensor in reader.tensors
    ):
        return (
            "Phi-3 LongRoPE tensors are not supported yet; the verified adapter "
            "currently targets standard Phi-3 4K checkpoints."
        )
    return None


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


def build_config(reader: GGUFReader, arch: str, dtype: str = "float16") -> dict[str, Any]:
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
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id")
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id")
    bos_id = 1 if bos_id is None else bos_id
    eos_id = 2 if eos_id is None else eos_id

    hf_model_type = ARCH_MAP.get(arch, arch)

    # --- Build config ---
    # Detect tied embeddings (many Qwen, Gemma, etc. models have this)
    tie_embeddings = arch in (
        "qwen2", "qwen2moe", "gemma", "gemma2", "gemma3",
        "olmo", "olmo2", "openelm",
    )

    # Detect attention bias. Gemma projections are bias-free; Phi-3 uses a fused,
    # bias-free QKV projection in MLX-LM.
    attention_bias = arch in (
        "qwen2", "qwen2moe", "qwen3moe",
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
        "torch_dtype": dtype,
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

    if arch == "gemma":
        config["head_dim"] = (
            get_metadata_int(reader, "gemma.attention.key_length")
            or hidden_size // num_heads
        )
        config["hidden_activation"] = "gelu_pytorch_tanh"
        config["attention_bias"] = False
        config["tie_word_embeddings"] = True

    if arch == "phi3":
        head_dim = hidden_size // num_heads
        rope_dim = get_metadata_int(reader, "phi3.rope.dimension_count") or head_dim
        original_context = get_metadata_int(
            reader, "phi3.rope.scaling.original_context_length"
        ) or min(ctx_length, 4096)
        config["partial_rotary_factor"] = rope_dim / head_dim
        config["original_max_position_embeddings"] = original_context
        config["attention_bias"] = False
        config["tie_word_embeddings"] = False

        rope_scaling_type = get_metadata_str(reader, "phi3.rope.scaling.type")
        rope_scaling_factor = get_metadata_float(reader, "phi3.rope.scaling.factor")
        if rope_scaling_type == "linear" and rope_scaling_factor is not None:
            config["rope_scaling"] = {
                "type": "linear",
                "factor": rope_scaling_factor,
            }

    # --- StableLM ---
    if arch == "stablelm":
        # StableLM uses LayerNorm (not RMSNorm); the HF config key is norm_eps.
        # The GGUF metadata key is stablelm.attention.layer_norm_epsilon (no "rms_").
        # The generic build above may have silently fallen back to a default — re-read
        # explicitly and replace the rms key with the correct one.
        stablelm_norm_eps = get_metadata_float(
            reader, "stablelm.attention.layer_norm_epsilon"
        ) or norm_eps
        config.pop("rms_norm_eps", None)
        config["norm_eps"] = stablelm_norm_eps

        # partial_rotary_factor = rope_dim / (hidden_size // num_heads)
        rope_dim = get_metadata_int(reader, "stablelm.rope.dimension_count")
        if rope_dim is not None and num_heads > 0:
            head_dim = hidden_size // num_heads
            config["partial_rotary_factor"] = rope_dim / head_dim

        # qk_layernorm: present when per-head Q/K norm tensors exist in the file.
        tensor_names = {t.name for t in reader.tensors}
        config["qk_layernorm"] = any(
            "attn_q_norm" in name or "attn_k_norm" in name for name in tensor_names
        )

        # use_parallel_residual: False when blk.N.ffn_norm tensors are present.
        config["use_parallel_residual"] = not any(
            "ffn_norm" in name for name in tensor_names
        )

        # Fix CamelCase: StableLmForCausalLM (capital L, not Stablelm…)
        config["architectures"] = ["StableLmForCausalLM"]

        config["tie_word_embeddings"] = False
        config["attention_bias"] = any(
            "attn_q.bias" in name or "attn_k.bias" in name or "attn_v.bias" in name
            for name in tensor_names
        )

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


def _map_phi3_tensor_name(gguf_name: str) -> str:
    """Map Phi-3 GGUF tensors to the fused layout expected by MLX-LM."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"

    match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if match is None:
        return gguf_name

    layer_idx, rest = match.groups()
    tensor_map = {
        "attn_qkv.weight": "self_attn.qkv_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_up.weight": "mlp.gate_up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    suffix = tensor_map.get(rest)
    if suffix is None:
        return gguf_name
    return f"model.layers.{layer_idx}.{suffix}"


def _restore_architecture_tensor(gguf_name: str, arr: np.ndarray, arch: str) -> np.ndarray:
    """Undo architecture-specific transformations applied during HF-to-GGUF export."""
    if arch == "gemma" and (
        gguf_name == "output_norm.weight"
        or re.fullmatch(r"blk\.\d+\.(attn_norm|ffn_norm)\.weight", gguf_name)
    ):
        return arr - np.array(1, dtype=arr.dtype)
    return arr


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


def _map_mla_tensor_name(gguf_name: str) -> str | None:
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
    if arch == "phi3":
        mapped = _map_phi3_tensor_name(gguf_name)
    else:
        mapped = _map_llama_tensor_name(gguf_name)
    if arch in STRICT_ADAPTER_ARCHES and mapped == gguf_name:
        raise ValueError(f"Unsupported {arch} tensor: {gguf_name}")
    return mapped


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------


_GLM_DSA_TEMPLATE_PATH = Path(__file__).parent / "data" / "glm_dsa_chat_template.jinja"


def _get_chat_template(reader: GGUFReader, arch: str) -> str | None:
    """Return a chat template: GGUF metadata first, then a canonical GLM fallback."""
    tmpl = get_metadata_str(reader, "tokenizer.chat_template")
    if tmpl:
        return tmpl
    if arch in ("glm-dsa", "chatglm") and _GLM_DSA_TEMPLATE_PATH.exists():
        return _GLM_DSA_TEMPLATE_PATH.read_text(encoding="utf-8")
    return None


def extract_tokenizer(
    reader: GGUFReader,
    output_dir: Path,
    arch: str = "llama",
    model_max_length: int | None = None,
) -> None:
    """Extract tokenizer from GGUF metadata and save standard files."""
    model_type = get_metadata_str(reader, "tokenizer.ggml.model") or "bpe"
    tokenizer_pre = get_metadata_str(reader, "tokenizer.ggml.pre")
    embedded_hf_json = get_metadata_str(reader, "tokenizer.huggingface.json")
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id")
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id")
    unk_id = get_metadata_int(reader, "tokenizer.ggml.unknown_token_id")
    pad_id = get_metadata_int(reader, "tokenizer.ggml.padding_token_id")
    bos_was_missing = bos_id is None
    eos_was_missing = eos_id is None
    bos_id = 1 if bos_id is None else bos_id
    eos_id = 2 if eos_id is None else eos_id
    unk_id = 0 if unk_id is None else unk_id
    pad_id = 0 if pad_id is None else pad_id
    add_bos_token = get_metadata_bool(reader, "tokenizer.ggml.add_bos_token")
    add_eos_token = get_metadata_bool(reader, "tokenizer.ggml.add_eos_token")
    add_space_prefix = get_metadata_bool(reader, "tokenizer.ggml.add_space_prefix")
    add_bos_token = True if add_bos_token is None else add_bos_token
    add_eos_token = False if add_eos_token is None else add_eos_token
    add_space_prefix = True if add_space_prefix is None else add_space_prefix

    tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
    token_types = get_metadata_array_int(reader, "tokenizer.ggml.token_type")
    merges = get_metadata_array_str(reader, "tokenizer.ggml.merges")
    scores = get_metadata_array_float(reader, "tokenizer.ggml.scores")

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

    if tokens and bos_was_missing:
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

    if tokens and eos_was_missing:
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
        "add_bos_token": add_bos_token,
        "add_eos_token": add_eos_token,
        "bos_token": tokens[bos_id] if bos_id < vocab_size else "<s>",
        "eos_token": tokens[eos_id] if eos_id < vocab_size else "</s>",
        "unk_token": tokens[unk_id] if unk_id < vocab_size else "<unk>",
        "pad_token": tokens[pad_id] if pad_id < vocab_size else "<pad>",
        "model_max_length": model_max_length or 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }
    if tokenizer_pre:
        tokenizer_config["gguf_tokenizer_pre"] = tokenizer_pre

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
        "unk_token": tokens[unk_id] if unk_id < vocab_size else "<unk>",
    }
    if pad_id < vocab_size and tokens[pad_id]:
        special_tokens["pad_token"] = tokens[pad_id]

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved special_tokens_map.json")

    # --- vocab.json (word → id mapping) ---
    vocab = {}
    for i, token in enumerate(tokens):
        if i < vocab_size and isinstance(token, str):
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
    tokenizer_json = None
    if embedded_hf_json:
        try:
            parsed_hf_json = json.loads(embedded_hf_json)
            if not isinstance(parsed_hf_json, dict) or "model" not in parsed_hf_json:
                raise ValueError("embedded tokenizer JSON is not a tokenizer object")
            tokenizer_json = parsed_hf_json
            print("  ✓ Preserved embedded tokenizer.huggingface.json")
        except (json.JSONDecodeError, ValueError) as error:
            warnings.warn(f"Invalid tokenizer.huggingface.json; rebuilding tokenizer: {error}")

    if tokenizer_json is None:
        tokenizer_json = _build_tokenizer_json(
            tokens,
            token_types,
            merges,
            scores,
            model_type,
            bos_id,
            eos_id,
            pad_id,
            unk_id=unk_id,
            add_space_prefix=add_space_prefix,
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
    unk_id: int = 0,
    add_space_prefix: bool = True,
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
        # Preserve GGUF control and user-defined token semantics.
        if tt in (3, 4) or i in (bos_id, eos_id, pad_id):
            special = tt == 3 or i in (bos_id, eos_id, pad_id)
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

    normalized_model_type = model_type.lower()

    # Build model block
    if normalized_model_type in ("bpe", "gpt2"):
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[unk_id] if unk_id < len(tokens) else "<unk>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    elif normalized_model_type in ("llama", "spm", "sentencepiece", "unigram"):
        vocab_scores = []
        for i, token in enumerate(tokens):
            score = scores[i] if i < len(scores) else 0.0
            vocab_scores.append([token, score])
        model_block = {
            "type": "Unigram",
            "unk_id": unk_id,
            "vocab": vocab_scores,
            "byte_fallback": any(token.startswith("<0x") and token.endswith(">") for token in tokens),
        }
    elif normalized_model_type == "wordpiece":
        model_block = {
            "type": "WordPiece",
            "unk_token": tokens[unk_id] if unk_id < len(tokens) else "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab,
        }
    else:
        raise ValueError(f"Unsupported GGUF tokenizer model: {model_type}")

    normalizer = {"type": "NFC"}
    pre_tokenizer = {
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
    }
    post_processor = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    decoder = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }

    if normalized_model_type in ("spm", "sentencepiece", "unigram", "llama"):
        pre_tokenizer = {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always" if add_space_prefix else "never",
            "split": True,
        }
        post_processor = None
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        }
    elif normalized_model_type == "wordpiece":
        normalizer = {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        }
        pre_tokenizer = {"type": "BertPreTokenizer"}
        post_processor = None
        decoder = {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": True,
        }

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": model_block,
    }

    return tokenizer_json


# ---------------------------------------------------------------------------
# Weight extraction & conversion
# ---------------------------------------------------------------------------


def _read_mla_dims(reader: GGUFReader, arch: str) -> dict[str, int]:
    """Read MLA dims needed to reconstruct kv_b_proj from split k_b/v_b."""
    def _g(key: str) -> int | None:
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
                      pending_kv_b: dict[str, dict[str, np.ndarray]],
                      pending_qkv: dict[str, dict[str, np.ndarray]],
                      ) -> list[tuple[str, np.ndarray]]:
    """Decide output (hf_name, arr) pairs for one source GGUF tensor.

    Handles arch-specific multi-tensor transforms:
      * Split attn_k_b/attn_v_b (deepseek2/3, glm4moe, and glm-dsa fallback when
        the combined attn_kv_b is absent) -> combined kv_b_proj via per-head
        interleave (_reconstruct_kv_b).
      * glm-dsa stacked ffn_*_exps -> per-expert mlp.experts.{e}.{kind}_proj.

    Returns [] when the tensor is buffered (waiting for its kv_b pair).
    """
    arr = _restore_architecture_tensor(gguf_name, arr, arch)

    # --- Phi-3 split Q/K/V -> fused qkv_proj ---
    match = re.fullmatch(r"blk\.(\d+)\.attn_(q|k|v)\.weight", gguf_name)
    if match and arch == "phi3":
        layer_idx, projection = match.groups()
        buf = pending_qkv.setdefault(layer_idx, {})
        buf[projection] = arr
        if all(name in buf for name in ("q", "k", "v")):
            fused = np.concatenate([buf["q"], buf["k"], buf["v"]], axis=0)
            del pending_qkv[layer_idx]
            return [(f"model.layers.{layer_idx}.self_attn.qkv_proj.weight", fused)]
        return []

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


def _decode_tensor_to_array(
    tensor: Any,
    dtype: str,
    gguf_name: str,
) -> np.ndarray | None:
    """Decode one GGUF tensor into HF-layout ndarray."""
    np_dtype = np.float16 if dtype == "float16" else np.float32
    qtype = tensor.tensor_type
    qtype_val = int(qtype)
    logical_shape = tuple(tensor.shape)
    raw_data = tensor.data

    if qtype_val == 0:  # F32
        arr = np.array(raw_data, dtype=np.float32).reshape(logical_shape)
        if dtype == "float16":
            arr = arr.astype(np.float16)
        if arr.ndim == 2:
            arr = arr.T
        return arr
    if qtype_val == 1:  # F16
        arr = np.array(raw_data, dtype=np.float16).reshape(logical_shape)
        if arr.ndim == 2:
            arr = arr.T
        return arr.astype(np_dtype)
    if qtype_val == 28:  # F64
        arr = np.array(raw_data, dtype=np.float64).reshape(logical_shape)
        if arr.ndim == 2:
            arr = arr.T
        return arr.astype(np_dtype)
    if qtype_val in (24, 25, 26, 27):  # I8, I16, I32, I64
        int_dtype_map = {24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64}
        arr = np.array(raw_data, dtype=int_dtype_map.get(qtype_val, np.int32))
        arr = arr.reshape(logical_shape).astype(np_dtype)
        if arr.ndim == 2:
            arr = arr.T
        return arr

    try:
        ggml_qtype = (
            qtype if isinstance(qtype, GGMLQuantizationType) else GGMLQuantizationType(qtype_val)
        )
        arr = dequantize(raw_data, ggml_qtype)
        return arr.astype(np_dtype)
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠ Failed to dequantize {gguf_name} ({qtype}): {e}")
        return None


def _is_direct_quantizable_linear_weight(hf_name: str, arr: np.ndarray) -> bool:
    """Only quantize 2D linear projection weights in the first direct pipeline release."""
    return arr.ndim == 2 and hf_name.endswith("_proj.weight")


def _quantize_affine_4bit(
    arr: np.ndarray, q_group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a 2D array into affine 4-bit groups."""
    if arr.ndim != 2:
        raise ValueError(f"Direct quantization expects 2D tensors, got {arr.ndim}D")
    rows, cols = arr.shape
    if cols % q_group_size != 0:
        raise ValueError(
            f"Column size {cols} is not divisible by q_group_size={q_group_size}"
        )

    grouped = arr.astype(np.float32).reshape(rows, cols // q_group_size, q_group_size)
    mins = grouped.min(axis=-1)
    maxs = grouped.max(axis=-1)
    scales = (maxs - mins) / 15.0
    zero_scale = scales == 0.0
    safe_scales = scales.copy()
    safe_scales[zero_scale] = 1.0

    q = np.round((grouped - mins[..., None]) / safe_scales[..., None])
    q = np.clip(q, 0, 15).astype(np.uint8)
    q[zero_scale[..., None].repeat(q_group_size, axis=-1)] = 0

    low = q[..., 0::2]
    high = q[..., 1::2] << 4
    packed = (low | high).reshape(rows, cols // 2).astype(np.uint8)
    return packed, safe_scales.astype(np.float16), mins.astype(np.float16)


def _finalize_safetensor_shards(output_dir: Path, total_bytes_out: int) -> tuple[int, int]:
    """Rename placeholder shard filenames and write model.safetensors.index.json."""
    shard_files = sorted(
        output_dir.glob("model-*-of-NNNNN.safetensors"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    total_shards = len(shard_files)
    if total_shards == 0:
        raise RuntimeError("No safetensor shards were written")

    weight_map: dict[str, str] = {}
    for i, old_path in enumerate(shard_files, 1):
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        new_path = output_dir / new_name
        old_path.rename(new_path)
        with safe_open(str(new_path), framework="np") as f:
            for key in f:
                weight_map[key] = new_name

    index_json = {
        "metadata": {"total_size": total_bytes_out},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)
    return len(weight_map), total_shards


def extract_and_convert_weights(
    reader: GGUFReader, arch: str, output_dir: Path, dtype: str = "float16"
) -> None:
    """Extract GGUF tensors, dequantize, rename, and save as safetensors."""

    print(f"\n  Converting {len(reader.tensors)} tensors...")
    print(f"  Output dtype: {dtype}")

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
    pending_qkv: dict[str, dict[str, np.ndarray]] = {}

    for i, tensor in enumerate(reader.tensors):
        gguf_name = tensor.name
        n_bytes = tensor.n_bytes
        total_bytes_in += n_bytes

        # Progress indicator
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i + 1}/{len(reader.tensors)}] Processing...")

        try:
            arr = _decode_tensor_to_array(tensor, dtype, gguf_name)
            if arr is None:
                skipped += 1
                continue

            # Determine output tensors (may be 0, 1, or many for arch-specific
            # transforms such as kv_b concat and per-expert split).
            emit_pairs = _plan_tensor_emit(
                gguf_name,
                arr,
                arch,
                mla_dims,
                pending_kv_b,
                pending_qkv,
            )
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

        except Exception as e:  # noqa: BLE001
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

    if pending_qkv:
        incomplete = ", ".join(
            f"layer {layer_idx} ({'/'.join(sorted(parts))})"
            for layer_idx, parts in sorted(pending_qkv.items())
        )
        raise RuntimeError(f"Incomplete Phi-3 QKV tensor groups: {incomplete}")

    if skipped:
        raise RuntimeError(f"Failed to convert {skipped} tensor(s); no partial model was published")

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

    indexed_keys, total_shards = _finalize_safetensor_shards(output_dir, total_bytes_out)

    print(f"\n  ✓ Saved {indexed_keys} weight tensors ({total_shards} shards)")
    print(f"    Total input:  {total_bytes_in / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {total_bytes_out / 1e9:.2f} GB (safetensors)")


def extract_and_convert_weights_direct_quant(
    reader: GGUFReader,
    arch: str,
    output_dir: Path,
    dtype: str,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> None:
    """Bounded-memory direct quantization path for supported linear weights."""
    if q_bits != 4 or q_group_size != 64 or q_mode != "affine":
        raise RuntimeError(
            "Direct quantization currently supports only affine 4-bit with q_group_size=64"
        )

    np_dtype = np.float16 if dtype == "float16" else np.float32
    weights: dict[str, np.ndarray] = {}
    total_bytes_in = 0
    total_bytes_out = 0
    current_shard_bytes = 0
    shard_idx = 1
    skipped = 0
    quantized_tensors = 0
    fallback_tensors = 0
    max_shard_bytes = DIRECT_QUANT_DEFAULT_MAX_SHARD_BYTES

    pbar = tqdm(total=len(reader.tensors), desc="  Direct quantizing", unit="tensor")

    def _shard_filename(idx: int, total_final: int | None = None) -> str:
        if total_final is None:
            return f"model-{idx:05d}-of-NNNNN.safetensors"
        return f"model-{idx:05d}-of-{total_final:05d}.safetensors"

    def _flush_shard(shard_weights: dict[str, np.ndarray], index: int) -> int:
        if not shard_weights:
            return 0
        path = output_dir / _shard_filename(index, None)
        save_safetensors(shard_weights, str(path))
        n_bytes = sum(arr.nbytes for arr in shard_weights.values())
        shard_weights.clear()
        gc.collect()
        return n_bytes

    mla_dims = _read_mla_dims(reader, arch)
    pending_kv_b: dict[str, dict[str, np.ndarray]] = {}
    pending_qkv: dict[str, dict[str, np.ndarray]] = {}

    for tensor in reader.tensors:
        gguf_name = tensor.name
        total_bytes_in += tensor.n_bytes
        try:
            arr = _decode_tensor_to_array(tensor, dtype, gguf_name)
            if arr is None:
                skipped += 1
                continue

            emit_pairs = _plan_tensor_emit(
                gguf_name,
                arr,
                arch,
                mla_dims,
                pending_kv_b,
                pending_qkv,
            )
            qtype_val = int(tensor.tensor_type)
            for hf_name, out_arr in emit_pairs:
                if _is_direct_quantizable_linear_weight(hf_name, out_arr):
                    if qtype_val not in DIRECT_QUANT_SUPPORTED_SOURCE_QTYPES:
                        raise RuntimeError(
                            f"Unsupported source quantization for direct quantization: "
                            f"{gguf_name} has qtype={qtype_val}; supported: "
                            f"{sorted(DIRECT_QUANT_SUPPORTED_SOURCE_QTYPES)}"
                        )
                    packed, scales, biases = _quantize_affine_4bit(out_arr, q_group_size)
                    base = hf_name.removesuffix(".weight")
                    quantized_entries = {
                        f"{base}.weight": packed,
                        f"{base}.scales": scales.astype(np_dtype),
                        f"{base}.biases": biases.astype(np_dtype),
                    }
                    for name, value in quantized_entries.items():
                        weights[name] = value
                        total_bytes_out += value.nbytes
                        current_shard_bytes += value.nbytes
                    quantized_tensors += 1
                else:
                    weights[hf_name] = out_arr
                    total_bytes_out += out_arr.nbytes
                    current_shard_bytes += out_arr.nbytes
                    fallback_tensors += 1

                if current_shard_bytes >= max_shard_bytes:
                    _flush_shard(weights, shard_idx)
                    shard_idx += 1
                    current_shard_bytes = 0
            pbar.update(1)
        except Exception as error:  # noqa: BLE001
            print(f"    ⚠ Error processing {gguf_name}: {error}")
            skipped += 1

    pbar.close()

    if pending_qkv:
        incomplete = ", ".join(
            f"layer {layer_idx} ({'/'.join(sorted(parts))})"
            for layer_idx, parts in sorted(pending_qkv.items())
        )
        raise RuntimeError(f"Incomplete Phi-3 QKV tensor groups: {incomplete}")

    if pending_kv_b:
        raise RuntimeError("Incomplete split kv_b tensor groups in direct quantization path")

    if skipped:
        raise RuntimeError(
            f"Failed to convert {skipped} tensor(s); no partial model was published"
        )

    if weights:
        _flush_shard(weights, shard_idx)

    indexed_keys, total_shards = _finalize_safetensor_shards(output_dir, total_bytes_out)
    print(f"\n  ✓ Direct quantization wrote {indexed_keys} tensors across {total_shards} shards")
    print(f"    Quantized projection tensors: {quantized_tensors}")
    print(f"    Fallback non-projection tensors: {fallback_tensors}")
    print(f"    Total input:  {total_bytes_in / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {total_bytes_out / 1e9:.2f} GB (safetensors)")


def _convert_direct_quantized(
    gguf_path: str,
    output_dir: str,
    dtype: str,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> bool:
    """Convert GGUF directly into quantized MLX-LM-compatible shards."""
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(str(gguf_path))

    arch = detect_architecture(reader)
    if arch not in DIRECT_QUANT_SUPPORTED_ARCHES:
        print(
            "❌ Direct quantization currently supports only: "
            f"{', '.join(sorted(DIRECT_QUANT_SUPPORTED_ARCHES))}. Detected: {arch}"
        )
        return False
    variant_error = validate_architecture_variant(reader, arch)
    if variant_error:
        print(f"❌ {variant_error}")
        return False

    print("\n[direct-quant] Building config and tokenizer...")
    config = build_config(reader, arch, dtype)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    extract_tokenizer(reader, output_path, arch, config["max_position_embeddings"])

    print("\n[direct-quant] Converting tensors with bounded memory...")
    try:
        extract_and_convert_weights_direct_quant(
            reader,
            arch,
            output_path,
            dtype=dtype,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
        )
    except Exception as error:  # noqa: BLE001
        print(f"❌ Direct quantization failed: {error}")
        return False

    config["quantization"] = {
        "bits": q_bits,
        "group_size": q_group_size,
        "mode": q_mode,
        "scheme": "direct_affine_4bit_v1",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return True
# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def _convert(gguf_path: str, output_dir: str, dtype: str = "float16") -> bool:
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
    if arch not in CONVERTIBLE_ARCHES:
        print(f"❌ Unsupported GGUF architecture: {arch}")
        return False
    variant_error = validate_architecture_variant(reader, arch)
    if variant_error:
        print(f"❌ {variant_error}")
        return False
    hf_type = ARCH_MAP.get(arch, arch)
    model_name_full = get_metadata_str(reader, "general.name") or model_name
    print(f"  Architecture: {arch} (HF type: {hf_type})")
    print(f"  Model name:   {model_name_full}")

    config = build_config(reader, arch, dtype)
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
    except (AttributeError, KeyError, TypeError, ValueError):
        print("  ⚠ Could not read source quantization metadata")

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
    extract_tokenizer(reader, output_path, arch, config["max_position_embeddings"])

    # Step 4: Extract, dequantize, and convert weights
    print("\n[4/5] Extracting and converting weights...")
    try:
        extract_and_convert_weights(reader, arch, output_path, dtype)
    except Exception as e:  # noqa: BLE001
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


def _quantize_output_with_mlx_lm(
    model_path: Path,
    output_path: Path,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> None:
    """Quantize an MLX-LM-loadable directory into a new MLX output directory."""
    if mlx_lm_convert is None:
        raise RuntimeError(
            "Quantization requested but mlx-lm is not installed. "
            "Install with `pip install -e '.[mlx]'` or `pip install mlx-lm`."
        )

    mlx_lm_convert(
        str(model_path),
        mlx_path=str(output_path),
        quantize=True,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )


def convert(
    gguf_path: str,
    output_dir: str,
    dtype: str = "float16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    direct_quant: bool = False,
) -> bool:
    """Convert into a staging directory so failed runs never leave partial output."""
    if quantize and q_group_size not in SUPPORTED_MLX_LM_Q_GROUP_SIZES:
        supported = ", ".join(str(size) for size in sorted(SUPPORTED_MLX_LM_Q_GROUP_SIZES))
        print(f"❌ Unsupported q_group_size={q_group_size}; supported values: {supported}")
        return False

    output_path = Path(output_dir)
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        print(f"❌ Output path already exists and is not empty: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    fp_output_path = staging_path / "fp"
    quantized_output_path = staging_path / "quantized"
    direct_quantized_output_path = staging_path / "direct-quantized"

    try:
        if quantize and direct_quant:
            try:
                succeeded = _convert_direct_quantized(
                    gguf_path,
                    str(direct_quantized_output_path),
                    dtype=dtype,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    q_mode=q_mode,
                )
            except Exception as error:  # noqa: BLE001
                print(f"❌ Conversion failed: {error}")
                return False
            if not succeeded:
                return False
            final_stage_path = direct_quantized_output_path
        else:
            try:
                succeeded = _convert(gguf_path, str(fp_output_path), dtype)
            except Exception as error:  # noqa: BLE001
                print(f"❌ Conversion failed: {error}")
                return False

            if not succeeded:
                return False

            final_stage_path = fp_output_path
            if quantize:
                print("\n[mlx-lm] Quantizing converted output...")
                try:
                    _quantize_output_with_mlx_lm(
                        fp_output_path,
                        quantized_output_path,
                        q_bits=q_bits,
                        q_group_size=q_group_size,
                        q_mode=q_mode,
                    )
                except Exception as error:  # noqa: BLE001
                    print(f"❌ Quantization failed: {error}")
                    return False
                final_stage_path = quantized_output_path

        if output_path.exists():
            output_path.rmdir()
        final_stage_path.replace(output_path)
        return True
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_safe_bundle_path(path: str) -> bool:
    if "\\" in path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute():
        return False
    if str(pure) != path:
        return False
    if any(part in {".", ".."} for part in pure.parts):
        return False
    return len(pure.parts) > 0


def package_mlx_directory(model_dir: str, output_bundle: str | None = None) -> bool:
    """Package an MLX model directory into a .mlx bundle with manifest hashes."""
    model_path = Path(model_dir)
    if not model_path.exists() or not model_path.is_dir():
        print(f"❌ MLX model directory not found: {model_dir}")
        return False

    bundle_path = Path(output_bundle) if output_bundle else model_path.with_suffix(".mlx")
    if bundle_path.suffix != ".mlx":
        bundle_path = bundle_path.with_suffix(".mlx")

    if bundle_path.exists():
        print(f"❌ Output bundle already exists: {bundle_path}")
        return False

    files = sorted(p for p in model_path.rglob("*") if p.is_file())
    if not files:
        print(f"❌ MLX model directory is empty: {model_path}")
        return False

    manifest_files = []
    for file_path in files:
        rel_path = file_path.relative_to(model_path).as_posix()
        manifest_files.append(
            {
                "path": rel_path,
                "size": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
            }
        )

    manifest = {
        "format": "gguf2mlx.mlx_bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_dir": model_path.name,
        "file_count": len(manifest_files),
        "files": manifest_files,
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in manifest_files:
            zf.write(model_path / entry["path"], arcname=entry["path"])
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    bundle_sha = _sha256_file(bundle_path)
    sha_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    sha_path.write_text(f"{bundle_sha}  {bundle_path.name}\n", encoding="utf-8")

    print(f"✅ Packed MLX bundle: {bundle_path}")
    print(f"  ✓ Manifest: {manifest['file_count']} files with SHA-256 hashes")
    print(f"  ✓ Bundle SHA-256: {bundle_sha}")
    print(f"  ✓ Sidecar hash file: {sha_path}")
    verified = verify_mlx_bundle(str(bundle_path))
    if not verified:
        bundle_path.unlink(missing_ok=True)
        sha_path.unlink(missing_ok=True)
        print("❌ Verification failed; removed generated bundle artifacts")
        return False
    return True


def verify_mlx_bundle(bundle_path: str) -> bool:
    """Verify a .mlx bundle against its embedded manifest and optional sidecar SHA."""
    path = Path(bundle_path)
    if not path.exists() or not path.is_file():
        print(f"❌ Bundle not found: {bundle_path}")
        return False

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "manifest.json" not in names:
                print("❌ Missing manifest.json in bundle")
                return False

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            if not isinstance(manifest, dict):
                print("❌ Invalid manifest format: manifest.json must be an object")
                return False
            expected_files = manifest.get("files", [])
            if not isinstance(expected_files, list):
                print("❌ Invalid manifest format: files must be a list")
                return False

            manifest_paths = set()
            for entry in expected_files:
                if not isinstance(entry, dict):
                    print("❌ Invalid manifest entry: each file entry must be an object")
                    return False
                rel_path = entry.get("path")
                if not isinstance(rel_path, str):
                    print("❌ Invalid manifest entry: path must be a string")
                    return False
                if not _is_safe_bundle_path(rel_path):
                    print(f"❌ Invalid manifest entry path: {rel_path}")
                    return False
                if rel_path in manifest_paths:
                    print(f"❌ Duplicate manifest entry path: {rel_path}")
                    return False
                manifest_paths.add(rel_path)

            zip_entries = {name for name in names if not name.endswith("/")}
            if len(zip_entries) != len(names):
                print("❌ Archive contains directory entries; only files are allowed")
                return False
            for name in zip_entries:
                if name == "manifest.json":
                    continue
                if not _is_safe_bundle_path(name):
                    print(f"❌ Invalid archive member path: {name}")
                    return False

            allowed_entries = set(manifest_paths)
            allowed_entries.add("manifest.json")
            if zip_entries != allowed_entries:
                missing = sorted(allowed_entries - zip_entries)
                extra = sorted(zip_entries - allowed_entries)
                if missing:
                    print(f"❌ Missing files in bundle: {missing}")
                if extra:
                    print(f"❌ Extra files in bundle not in manifest: {extra}")
                return False

            for entry in expected_files:
                rel_path = entry.get("path")
                expected_sha = entry.get("sha256")
                expected_size = entry.get("size")
                if not rel_path:
                    print("❌ Invalid manifest entry: missing path")
                    return False
                if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                    print(f"❌ Invalid manifest entry sha256 for {rel_path}: {expected_sha}")
                    return False
                with zf.open(rel_path) as f:
                    data = f.read()
                actual_sha = _sha256_bytes(data)
                actual_size = len(data)
                if actual_sha != expected_sha:
                    print(f"❌ SHA-256 mismatch for {rel_path}")
                    print(f"   expected: {expected_sha}")
                    print(f"   actual:   {actual_sha}")
                    return False
                if expected_size is not None:
                    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
                        print(f"❌ Invalid manifest entry size for {rel_path}: {expected_size}")
                        return False
                    if expected_size != actual_size:
                        print(f"❌ Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}")
                        return False
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"❌ Bundle verification failed: {error}")
        return False

    sidecar_path = path.with_suffix(path.suffix + ".sha256")
    if sidecar_path.exists():
        line = sidecar_path.read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"([0-9a-f]{64})(?:\s{2}.+)?", line)
        if not match:
            print(f"❌ Invalid sidecar hash format: {sidecar_path}")
            return False
        expected_bundle_sha = match.group(1)
        actual_bundle_sha = _sha256_file(path)
        if expected_bundle_sha != actual_bundle_sha:
            print("❌ Bundle SHA-256 mismatch against sidecar file")
            print(f"   expected: {expected_bundle_sha}")
            print(f"   actual:   {actual_bundle_sha}")
            return False

    print(f"✅ Bundle verification passed: {path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--quantize",
        action="store_true",
        help="Run mlx-lm quantization after conversion to produce a quantized MLX output directory",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bit-width for mlx-lm (default: 4)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        choices=sorted(SUPPORTED_MLX_LM_Q_GROUP_SIZES),
        help="Quantization group size for mlx-lm (default: 64)",
    )
    parser.add_argument(
        "--q-mode",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        help="Quantization mode for mlx-lm (default: affine)",
    )
    parser.add_argument(
        "--direct-quant",
        action="store_true",
        help=(
            "Experimental bounded-memory direct quantization path "
            "(currently affine 4-bit, group size 64, llama/gemma only)"
        ),
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight extraction (metadata + tokenizer only, for inspection)",
    )


def _run_convert_command(args: argparse.Namespace) -> int:
    if args.output is None:
        args.output = Path(args.input).stem + "-mlx"

    if args.skip_weights:
        reader = GGUFReader(args.input)
        print(f"Architecture: {detect_architecture(reader)}")
        print(f"Tensors: {len(reader.tensors)}")
        print(f"Fields: {len(reader.fields)}")
        for name in sorted(reader.fields.keys()):
            print(f"  {name}")
        return 0

    success = convert(
        args.input,
        args.output,
        args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        q_mode=args.q_mode,
        direct_quant=args.direct_quant,
    )
    return 0 if success else 1


def main(argv: list[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    command = argv[0] if argv else None

    if command in {"convert", "package", "verify"}:
        parser = argparse.ArgumentParser(
            description="gguf2mlx command suite for conversion, packaging, and verification"
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        convert_parser = subparsers.add_parser("convert", help="Convert GGUF to MLX directory")
        _add_convert_arguments(convert_parser)

        package_parser = subparsers.add_parser(
            "package", help="Package an MLX model directory into a .mlx bundle"
        )
        package_parser.add_argument(
            "--model-dir", "-m", required=True, help="Input MLX model directory"
        )
        package_parser.add_argument(
            "--output", "-o", help="Output .mlx bundle path (default: <model-dir>.mlx)"
        )

        verify_parser = subparsers.add_parser(
            "verify", help="Verify .mlx bundle manifest and SHA-256 integrity"
        )
        verify_parser.add_argument("--bundle", "-b", required=True, help="Path to .mlx bundle")

        args = parser.parse_args(argv)
        if args.command == "convert":
            exit_code = _run_convert_command(args)
        elif args.command == "package":
            exit_code = 0 if package_mlx_directory(args.model_dir, args.output) else 1
        else:
            exit_code = 0 if verify_mlx_bundle(args.bundle) else 1
        if exit_code:
            sys.exit(exit_code)
        return

    parser = argparse.ArgumentParser(
        description="GGUF to MLX Converter — Convert GGUF models to MLX safetensors format"
    )
    _add_convert_arguments(parser)

    args = parser.parse_args(argv)
    if _run_convert_command(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
