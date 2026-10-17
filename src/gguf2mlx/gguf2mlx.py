"""
GGUF to MLX Converter v2.0
Converts GGUF models to MLX format (safetensors) for Apple Silicon inference.

Phase 1: Real weight extraction, safetensors output, architecture detection,
         real tokenizer extraction.
"""

import argparse
import base64
import gc  # noqa: F401
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

from .qwen_vl_text import QWEN_VL_TEXT_ARCHES, Qwen3VLTextEncoderConverter
from .rust_backend import classify_architecture as rust_classify_architecture
from .rust_backend import detect_architecture as rust_detect_architecture
from .streaming import StreamingShardWriter

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
    "qwen3": "qwen3",
    **{
        arch: "qwen3_5"
        for arch in (
            "qwen35", "qwen3_5", "qwen3.5",
            "qwen36", "qwen3_6", "qwen3.6",
            "qwen38", "qwen3_8", "qwen3.8",
        )
    },
    **{arch: "qwen3_vl_text" if "3" in arch else "qwen2_vl_text"
       for arch in QWEN_VL_TEXT_ARCHES},
    "qwen2moe": "qwen2_moe",
    "qwen3moe": "qwen3_moe",
    "phi3": "phi3",
    "phi4": "phi3",
    "phi2": "phi",
    "phi": "phi",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "gemma4": "gemma4_text",
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
    "gemma2",
    "gemma3",
    "gemma4",
    "llama",
    "mistral",
    "phi3",
    "phi4",
    "qwen2",
    "qwen3",
    "qwen35",
    "qwen3_5",
    "qwen3.5",
    "qwen36",
    "qwen3_6",
    "qwen3.6",
    "qwen38",
    "qwen3_8",
    "qwen3.8",
    "qwen2moe",
    "qwen3moe",
    "stablelm",
    *QWEN_VL_TEXT_ARCHES,
}

STRICT_ADAPTER_ARCHES = {
    "gemma", "gemma2", "gemma3", "gemma4", "phi3", "phi4", "qwen3",
    "qwen35", "qwen3_5", "qwen3.5", "qwen36", "qwen3_6", "qwen3.6",
    "qwen38", "qwen3_8", "qwen3.8",
}

QWEN35_ARCHES = {
    "qwen35", "qwen3_5", "qwen3.5", "qwen36", "qwen3_6", "qwen3.6",
    "qwen38", "qwen3_8", "qwen3.8",
}
GEMMA_ARCHES = {"gemma", "gemma2", "gemma3", "gemma4"}
PHI_ARCHES = {"phi3", "phi4"}
SUPPORTED_MLX_LM_Q_GROUP_SIZES = {32, 64, 128}
MOE_ARCHES = {"qwen2moe", "qwen3moe", "deepseek2", "deepseek3", "glm4moe", "glm-dsa"}
DIRECT_QUANT_SUPPORTED_ARCHES = {
    "llama",
    "gemma",
    "mistral",
    "qwen2",
    "stablelm",
    *QWEN_VL_TEXT_ARCHES,
    *MOE_ARCHES,
}
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
    20,  # IQ4_NL (also used in mixed Q2_K GGUF files)
    30,  # BF16
}
DIRECT_QUANT_DEFAULT_MAX_SHARD_BYTES = 256 * 1024 * 1024

_ROUTER_PATTERNS = (
    "ffn_gate_inp.weight",
    ".mlp.gate.weight",
    ".router.weight",
    "router.weight",
)
_EXPERT_PATTERNS = (
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    ".switch_mlp.",
    ".experts.",
)
_ATTENTION_PATTERNS = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_out",
    ".self_attn.",
)

# Popular GGUF naming patterns that do not directly include a GGUF architecture key.
# These are used only when `general.architecture` is missing.
MODEL_NAME_ARCH_FALLBACKS: list[tuple[str, str]] = [
    (r"\bdeepseek-r1-distill-qwen\b", "qwen2"),
    (r"\bdeepseek-r1-distill-llama\b", "llama"),
    (r"\bdeepseek-v3\b", "deepseek3"),
    (r"\bdeepseek-r1\b", "deepseek3"),
    (r"\bdeepseek-v2\b", "deepseek2"),
    (r"\bqwen\s*3[._ -]?5\b", "qwen35"),
    (r"\bqwen\s*3[._ -]?6\b", "qwen36"),
    (r"\bqwen\s*3[._ -]?8\b", "qwen38"),
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

    if arch == "phi4" or (
        arch in PHI_ARCHES
        and get_metadata_int(reader, f"{arch}.attention.sliding_window") == 0
    ) or (name and re.search(r"\bphi[\s._-]*4\b", name, re.IGNORECASE)):
        return "phi4"

    if arch in QWEN_VL_TEXT_ARCHES:
        return arch
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


def classify_architecture(reader: GGUFReader, arch: str | None = None) -> str:
    """Classify a GGUF tensor layout as dense, MoE, or hybrid."""
    arch = arch or detect_architecture(reader)
    tensor_names = [tensor.name for tensor in reader.tensors]
    rust_kind = rust_classify_architecture(arch, tensor_names)
    if rust_kind:
        return rust_kind

    has_experts = any(
        any(pattern in name for pattern in _EXPERT_PATTERNS) for name in tensor_names
    )
    has_dense_mlp = any(
        re.search(r"ffn_(gate|up|down)\.weight$", name) and "_exps" not in name
        for name in tensor_names
    )
    if (arch in MOE_ARCHES or has_experts) and has_dense_mlp:
        return "hybrid"
    if arch in MOE_ARCHES or has_experts:
        return "moe"
    return "dense"


def classify_tensor(gguf_name: str, hf_name: str | None = None) -> str:
    """Return the precision-policy category for a tensor."""
    names = (gguf_name, hf_name or "")
    if any(pattern in name for name in names for pattern in _ROUTER_PATTERNS):
        return "router"
    if any(pattern in name for name in names for pattern in _EXPERT_PATTERNS):
        return "expert_mlp"
    if any(pattern in name for name in names for pattern in _ATTENTION_PATTERNS):
        return "attention"
    if gguf_name.startswith("blk.") and gguf_name.endswith(".weight"):
        return "unknown"
    return "other"


def validate_moe_layout(reader: GGUFReader, arch: str, architecture_kind: str) -> None:
    """Reject MoE layouts whose routing tensor cannot be identified."""
    if architecture_kind not in {"moe", "hybrid"}:
        return
    router_names = [
        tensor.name
        for tensor in reader.tensors
        if classify_tensor(tensor.name) == "router"
    ]
    if not router_names:
        raise RuntimeError(
            f"Detected {architecture_kind} architecture {arch!r}, but no recognized "
            "router tensor was found; refusing conversion to avoid corrupt routing"
        )


def validate_architecture_variant(reader: GGUFReader, arch: str) -> str | None:
    """Return an actionable error when a recognized architecture variant is unverified."""
    if (arch == "qwen3" or arch in QWEN35_ARCHES) and any(
        any(pattern in tensor.name for pattern in _EXPERT_PATTERNS)
        for tensor in reader.tensors
    ):
        return (
            f"The {arch} adapter supports dense layouts only; MoE expert tensors "
            "are not supported."
        )
    if arch in PHI_ARCHES:
        factors = _find_phi3_rope_factor_tensors(reader)
        if bool(factors.get("long")) != bool(factors.get("short")):
            return "Phi LongRoPE requires both rope_factors_long and rope_factors_short tensors."
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
    if arch == "phi4":
        original_reader = reader

        class Phi4Metadata:
            tensors = original_reader.tensors

            def get_field(self, key):
                field = original_reader.get_field(key)
                if field is None and key.startswith("phi4."):
                    field = original_reader.get_field("phi3." + key[len("phi4."):])
                return field

        reader = Phi4Metadata()
    if arch in QWEN_VL_TEXT_ARCHES:
        return Qwen3VLTextEncoderConverter.build_config(reader, arch, dtype)
    if arch in QWEN35_ARCHES and arch != "qwen35":
        original_reader = reader

        class Qwen35Metadata:
            tensors = original_reader.tensors

            def get_field(self, key):
                field = original_reader.get_field(key)
                if field is None and key.startswith(f"{arch}."):
                    field = original_reader.get_field(
                        "qwen35." + key[len(arch) + 1:]
                    )
                return field

        reader = Qwen35Metadata()

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
    if arch in (
        "qwen2moe",
        "qwen3moe",
        "deepseek2",
        "deepseek3",
        "glm4moe",
        "dbrx",
        "grok-1",
    ):
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
    attention_bias = arch in ("qwen2", "qwen2moe")

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
    if arch in (
        "qwen2moe",
        "deepseek2",
        "deepseek3",
        "qwen3moe",
        "glm4moe",
        "dbrx",
        "grok-1",
    ):
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
            "glm4moe": "glm4_moe",
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
        if arch in ("qwen2moe", "qwen3moe", "deepseek3"):
            config["shared_expert_intermediate_size"] = shared_ffn_size
            config["output_router_logits"] = False
            config["router_aux_loss_coef"] = 0.001

        if arch == "qwen2moe":
            for suffix, config_key, metadata_key, stacked in (
                ("exps", "moe_intermediate_size", "expert_feed_forward_length", True),
                ("shexp", "shared_expert_intermediate_size", "expert_shared_feed_forward_length", False),
            ):
                dimensions = set()
                for tensor in reader.tensors:
                    match = re.fullmatch(rf"blk\.\d+\.ffn_(gate|up|down)_{suffix}\.weight", tensor.name)
                    if match is None:
                        continue
                    shape = tuple(int(dim) for dim in tensor.shape)
                    if len(shape) != (3 if stacked else 2):
                        raise ValueError(f"Invalid Qwen2MoE tensor shape: {tensor.name}: {shape}")
                    dimensions.add(shape[0] if match.group(1) == "down" else shape[1])
                if len(dimensions) > 1:
                    raise ValueError(f"Inconsistent Qwen2MoE {config_key}: {sorted(dimensions)}")
                if dimensions:
                    inferred = dimensions.pop()
                    declared = get_metadata_int(reader, f"{arch}.{metadata_key}")
                    if declared is not None and declared != inferred:
                        raise ValueError(f"Qwen2MoE {metadata_key} disagrees with tensor shapes")
                    config[config_key] = inferred

    if arch in {"gemma", "gemma2", "gemma3"}:
        config["head_dim"] = (
            get_metadata_int(reader, f"{arch}.attention.key_length")
            or hidden_size // num_heads
        )
        config["hidden_activation"] = "gelu_pytorch_tanh"
        config["attention_bias"] = False
        config["tie_word_embeddings"] = True

    if arch == "gemma2":
        for metadata_key, config_key in (
            ("attn_logit_softcapping", "attn_logit_softcapping"),
            ("final_logit_softcapping", "final_logit_softcapping"),
            ("attention.sliding_window", "sliding_window"),
        ):
            value = (
                get_metadata_int(reader, f"{arch}.{metadata_key}")
                if metadata_key == "attention.sliding_window"
                else get_metadata_float(reader, f"{arch}.{metadata_key}")
            )
            if value is not None:
                config[config_key] = value
        config["query_pre_attn_scalar"] = get_metadata_int(
            reader, f"{arch}.attention.query_pre_attn_scalar"
        ) or config["head_dim"]

    if arch == "gemma3":
        for metadata_key, config_key in (
            ("attention.sliding_window", "sliding_window"),
            ("attention.sliding_window_pattern", "sliding_window_pattern"),
        ):
            value = get_metadata_int(reader, f"{arch}.{metadata_key}")
            if value is not None:
                config[config_key] = value
        final_softcap = get_metadata_float(reader, f"{arch}.final_logit_softcapping")
        if final_softcap is not None:
            config["final_logit_softcapping"] = final_softcap
        config["query_pre_attn_scalar"] = get_metadata_int(
            reader, f"{arch}.attention.query_pre_attn_scalar"
        ) or config["head_dim"]
        rope_theta = get_metadata_float(reader, f"{arch}.rope.freq_base")
        if rope_theta is not None:
            config["rope_theta"] = rope_theta

    if arch == "gemma4":
        gemma4_sliding_dim = get_metadata_int(
            reader, f"{arch}.attention.key_length_swa"
        )
        gemma4_global_dim = get_metadata_int(
            reader, f"{arch}.attention.key_length"
        )
        layer_pattern = get_metadata_array_int(
            reader, f"{arch}.attention.sliding_window_pattern"
        )
        sliding_window_pattern = (
            next((index + 1 for index, is_sliding in enumerate(layer_pattern)
                  if not is_sliding), len(layer_pattern))
            if layer_pattern else get_metadata_int(
                reader, f"{arch}.attention.sliding_window_pattern"
            ) or 5
        )
        config.update({
            "model_type": "gemma4_text",
            "architectures": ["Gemma4TextModel"],
            "head_dim": gemma4_sliding_dim or hidden_size // num_heads,
            "global_head_dim": gemma4_global_dim or hidden_size // num_heads,
            "hidden_size_per_layer_input": get_metadata_int(
                reader, f"{arch}.embedding_length_per_layer_input"
            ) or 0,
            "vocab_size_per_layer_input": vocab_size,
            "num_kv_shared_layers": get_metadata_int(
                reader, f"{arch}.attention.shared_kv_layers"
            ) or 0,
            "sliding_window": get_metadata_int(
                reader, f"{arch}.attention.sliding_window"
            ) or 512,
            "sliding_window_pattern": sliding_window_pattern,
            "use_double_wide_mlp": True,
            "tie_word_embeddings": True,
        })
        global_rope_theta = get_metadata_float(reader, f"{arch}.rope.freq_base")
        sliding_rope_theta = get_metadata_float(reader, f"{arch}.rope.freq_base_swa")
        config["rope_parameters"] = {
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": global_rope_theta or 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": sliding_rope_theta or 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        }
        if layer_pattern:
            if len(layer_pattern) != num_layers:
                raise ValueError(
                    "Gemma4 sliding-window pattern length does not match block_count"
                )
            config["layer_types"] = [
                "sliding_attention" if is_sliding else "full_attention"
                for is_sliding in layer_pattern
            ]
        elif any(t.name.startswith("blk.") and ".attn_qkv." in t.name for t in reader.tensors):
            raise ValueError("Gemma4 linear-attention layer layout is unsupported")
        for suffix, config_key in (
            ("attention.head_count_kv", "num_key_value_heads"),
        ):
            value = get_metadata_int(reader, f"{arch}.{suffix}")
            if value is not None:
                config[config_key] = value
        final_softcap = get_metadata_float(reader, f"{arch}.final_logit_softcapping")
        if final_softcap is not None:
            config["final_logit_softcapping"] = final_softcap
        num_experts = get_metadata_int(reader, f"{arch}.expert_count")
        if num_experts:
            config.update({
                "enable_moe_block": True,
                "num_experts": num_experts,
                "top_k_experts": get_metadata_int(reader, f"{arch}.expert_used_count") or 1,
                "moe_intermediate_size": (
                    get_metadata_int(reader, f"{arch}.expert_feed_forward_length")
                    or config["intermediate_size"]
                ),
            })

    if arch == "qwen3":
        config["head_dim"] = (
            get_metadata_int(reader, f"{arch}.attention.key_length")
            or hidden_size // num_heads
        )
        config["attention_bias"] = False
        config["tie_word_embeddings"] = False

    if arch in QWEN35_ARCHES:
        config["model_type"] = "qwen3_5"
        config["architectures"] = ["Qwen3_5ForCausalLM"]
        config["attention_bias"] = False
        config["tie_word_embeddings"] = False
        config["head_dim"] = (
            get_metadata_int(reader, f"{arch}.attention.key_length")
            or hidden_size // num_heads
        )
        for suffix, config_key in (
            ("ssm.group_count", "linear_num_key_heads"),
            ("ssm.time_step_rank", "linear_num_value_heads"),
            ("ssm.state_size", "linear_key_head_dim"),
            ("ssm.conv_kernel", "linear_conv_kernel_dim"),
            ("full_attention_interval", "full_attention_interval"),
        ):
            value = get_metadata_int(reader, f"{arch}.{suffix}")
            if value is not None:
                config[config_key] = value
        inner_size = get_metadata_int(reader, f"{arch}.ssm.inner_size")
        value_heads = config.get("linear_num_value_heads")
        if inner_size is not None and value_heads:
            if inner_size % value_heads:
                raise ValueError("Qwen3.5 SSM inner size is not divisible by value-head count")
            config["linear_value_head_dim"] = inner_size // value_heads
        mrope = get_metadata_array_int(reader, f"{arch}.rope.dimension_sections")
        if len(mrope) == 4 and mrope[-1] == 0:
            mrope.pop()
        if mrope:
            if len(mrope) != 3 or any(section <= 0 for section in mrope):
                raise ValueError("Qwen3.5 requires three positive RoPE dimension sections")
            config["rope_parameters"] = {
                "type": "default",
                "mrope_section": mrope,
                "rope_theta": rope_theta,
                "partial_rotary_factor": 0.25,
            }
        config["linear_num_key_heads"] = config.get("linear_num_key_heads", 16)
        config["linear_num_value_heads"] = config.get("linear_num_value_heads", 64)
        config["linear_key_head_dim"] = config.get("linear_key_head_dim", 128)
        config["linear_value_head_dim"] = config.get("linear_value_head_dim", 128)
        config["linear_conv_kernel_dim"] = config.get("linear_conv_kernel_dim", 4)
        config["full_attention_interval"] = config.get("full_attention_interval", 4)

    if arch in PHI_ARCHES:
        head_dim = hidden_size // num_heads
        rope_dim = get_metadata_int(reader, f"{arch}.rope.dimension_count") or head_dim
        original_context = get_metadata_int(
            reader, f"{arch}.rope.scaling.original_context_length"
        ) or min(ctx_length, 4096)
        config["partial_rotary_factor"] = rope_dim / head_dim
        config["original_max_position_embeddings"] = original_context
        config["attention_bias"] = False
        config["tie_word_embeddings"] = False

        rope_scaling_type = get_metadata_str(reader, f"{arch}.rope.scaling.type")
        rope_scaling_factor = get_metadata_float(reader, f"{arch}.rope.scaling.factor")
        if rope_scaling_type == "linear" and rope_scaling_factor is not None:
            config["rope_scaling"] = {
                "type": "linear",
                "factor": rope_scaling_factor,
            }
        rope_factors = _find_phi3_rope_factor_tensors(reader)
        if rope_factors:
            if set(rope_factors) != {"long", "short"}:
                raise ValueError(
                    "Phi LongRoPE requires both rope_factors_long and rope_factors_short tensors"
                )
            long_factor = rope_factors["long"]
            short_factor = rope_factors["short"]
            expected_factor_count = rope_dim // 2
            if (
                rope_dim % 2
                or len(long_factor) != expected_factor_count
                or len(short_factor) != expected_factor_count
            ):
                raise ValueError(
                    "Phi LongRoPE factor lengths must equal half the rotary dimension "
                    f"({expected_factor_count})"
                )
            if rope_scaling_type not in (None, "longrope", "su"):
                raise ValueError(
                    f"Phi rope scaling type {rope_scaling_type!r} conflicts with LongRoPE tensors"
                )
            config["rope_scaling"] = {
                "type": "longrope",
                "long_factor": long_factor,
                "short_factor": short_factor,
            }
        elif rope_scaling_type in ("longrope", "su"):
            raise ValueError(
                "Phi LongRoPE config is missing rope_factors_long and rope_factors_short tensors"
            )

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


def _find_phi3_rope_factor_tensors(reader: GGUFReader) -> dict[str, list[float]]:
    """Read and validate the shared LongRoPE factor tensors from a Phi GGUF."""
    factors: dict[str, list[float]] = {}
    for tensor in reader.tensors:
        match = re.fullmatch(
            r"(?:blk\.\d+\.)?rope_factors_(long|short)(?:\.weight)?",
            tensor.name,
        )
        if match is None:
            continue
        which = match.group(1)
        values = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
        if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid Phi LongRoPE {which} factor tensor")
        factor_values = values.tolist()
        if which in factors and factors[which] != factor_values:
            raise ValueError(f"Conflicting Phi LongRoPE {which} factor tensors")
        factors[which] = factor_values
    return factors


def _map_phi3_tensor_name(gguf_name: str) -> str:
    """Map Phi-3/Phi-4 GGUF tensors to the layout expected by MLX-LM."""
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
    if arch in {"gemma", "gemma2", "gemma3"} and (
        gguf_name == "output_norm.weight"
        or re.fullmatch(r"blk\.\d+\.(attn_norm|ffn_norm)\.weight", gguf_name)
    ):
        return arr - np.array(1, dtype=arr.dtype)
    if arch in QWEN35_ARCHES:
        if gguf_name == "ssm_a":
            if np.any(~np.isfinite(arr)) or np.any(arr >= 0):
                raise ValueError("Qwen3.5 ssm_a must contain finite negative values")
            return np.log(-arr).astype(arr.dtype)
        if re.fullmatch(r"blk\.\d+\.ssm_norm\.weight", gguf_name):
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
    if arch in QWEN_VL_TEXT_ARCHES:
        return Qwen3VLTextEncoderConverter.map_tensor_name(gguf_name)
    if arch in _MLA_ARCHES:
        mapped = _map_mla_tensor_name(gguf_name)
        if mapped is not None:
            return mapped
    if arch in QWEN35_ARCHES and (match := re.fullmatch(
        r"blk\.(\d+)\.(attn_qkv|attn_gate|ssm_conv1d|ssm_dt|ssm_a|"
        r"ssm_beta|ssm_alpha|ssm_norm|ssm_out)(?:\.weight|\.bias)?", gguf_name
    )):
        layer, source = match.groups()
        suffix = {
            "attn_qkv": "linear_attn.in_proj_qkv.weight",
            "attn_gate": "linear_attn.in_proj_z.weight",
            "ssm_conv1d": "linear_attn.conv1d.weight",
            "ssm_dt": "linear_attn.dt_bias",
            "ssm_a": "linear_attn.A_log",
            "ssm_beta": "linear_attn.in_proj_b.weight",
            "ssm_alpha": "linear_attn.in_proj_a.weight",
            "ssm_norm": "linear_attn.norm.weight",
            "ssm_out": "linear_attn.out_proj.weight",
        }[source]
        mapped = f"model.layers.{layer}.{suffix}"
    elif arch == "gemma4" and (match := re.fullmatch(
        r"blk\.(\d+)\.(attn_post_norm|ffn_post_norm|ffn_pre_norm|"
        r"ffn_pre_norm_2|ffn_post_norm_1|ffn_post_norm_2|"
        r"ffn_gate_inp|ffn_gate_inp_s|ffn_gate_exps|ffn_up_exps|"
        r"ffn_down_exps|layer_output_scale|per_layer_inp_gate|"
        r"per_layer_proj|per_layer_post_norm)(?:\.weight|\.scale)?", gguf_name
    )):
        layer, source = match.groups()
        suffix = {
            "attn_post_norm": "post_attention_layernorm.weight",
            "ffn_pre_norm": "pre_feedforward_layernorm.weight",
            "ffn_pre_norm_2": "pre_feedforward_layernorm_2.weight",
            "ffn_post_norm_1": "post_feedforward_layernorm_1.weight",
            "ffn_post_norm_2": "post_feedforward_layernorm_2.weight",
            "ffn_post_norm": "post_feedforward_layernorm.weight",
            "ffn_gate_inp": "router.proj.weight",
            "ffn_gate_inp_s": "router.scale",
            "ffn_gate_exps": "experts.switch_glu.gate_proj.weight",
            "ffn_up_exps": "experts.switch_glu.up_proj.weight",
            "ffn_down_exps": "experts.switch_glu.down_proj.weight",
            "layer_output_scale": "layer_scalar",
            "per_layer_inp_gate": "per_layer_input_gate.weight",
            "per_layer_proj": "per_layer_projection.weight",
            "per_layer_post_norm": "post_per_layer_input_norm.weight",
        }[source]
        mapped = f"model.layers.{layer}.{suffix}"
    elif arch == "gemma4" and gguf_name in {
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        "per_layer_proj_norm.weight",
    }:
        mapped = {
            "per_layer_token_embd.weight": "model.embed_tokens_per_layer.weight",
            "per_layer_model_proj.weight": "model.per_layer_model_projection.weight",
            "per_layer_proj_norm.weight": "model.per_layer_projection_norm.weight",
        }[gguf_name]
    elif arch in PHI_ARCHES:
        mapped = _map_phi3_tensor_name(gguf_name)
    elif arch == "qwen2moe" and (
        match := re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down|gate_inp)_shexp\.weight", gguf_name)
    ):
        layer, projection = match.groups()
        suffix = (
            "shared_expert_gate.weight" if projection == "gate_inp"
            else f"shared_expert.{projection}_proj.weight"
        )
        mapped = f"model.layers.{layer}.mlp.{suffix}"
    else:
        mapped = _map_llama_tensor_name(gguf_name)
    if arch == "gemma4" and mapped == gguf_name and gguf_name.startswith("blk."):
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
        if match is not None:
            layer, rest = match.groups()
            gemma4_layer_names = {
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "attn_norm.weight": "input_layernorm.weight",
                "ffn_norm.weight": "pre_feedforward_layernorm.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight",
            }
            if rest in gemma4_layer_names:
                mapped = f"model.layers.{layer}.{gemma4_layer_names[rest]}"
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
    additional_special_token_ids: list[int] | None = None,
) -> None:
    """Extract tokenizer from GGUF metadata and save standard files."""
    source_model = (get_metadata_str(reader, "tokenizer.ggml.model") or "bpe").lower()
    model_type = {"bert": "wordpiece", "t5": "unigram"}.get(source_model, source_model)
    tokenizer_pre = get_metadata_str(reader, "tokenizer.ggml.pre")
    embedded_hf_json = get_metadata_str(reader, "tokenizer.huggingface.json")
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id")
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id")
    unk_id = get_metadata_int(reader, "tokenizer.ggml.unknown_token_id")
    pad_id = get_metadata_int(reader, "tokenizer.ggml.padding_token_id")
    bos_was_missing = bos_id is None
    eos_was_missing = eos_id is None
    bos_id = -1 if bos_id is None else bos_id
    eos_id = -1 if eos_id is None else eos_id
    unk_id = -1 if unk_id is None else unk_id
    pad_id = -1 if pad_id is None else pad_id
    add_bos_token = get_metadata_bool(reader, "tokenizer.ggml.add_bos_token")
    add_eos_token = get_metadata_bool(reader, "tokenizer.ggml.add_eos_token")
    add_sep_token = get_metadata_bool(reader, "tokenizer.ggml.add_sep_token")
    add_space_prefix = get_metadata_bool(reader, "tokenizer.ggml.add_space_prefix")
    add_bos_token = (
        model_type in ("llama", "spm", "sentencepiece", "wordpiece")
        or tokenizer_pre in ("llama-bpe", "llama3", "llama-v3")
        if add_bos_token is None else add_bos_token
    )
    add_eos_token = source_model == "t5" if add_eos_token is None else add_eos_token
    add_sep_token = model_type == "wordpiece" if add_sep_token is None else add_sep_token
    add_space_prefix = (
        model_type in ("llama", "spm", "sentencepiece", "unigram") and source_model != "t5"
        if add_space_prefix is None else add_space_prefix
    )

    tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
    token_types = get_metadata_array_int(reader, "tokenizer.ggml.token_type")
    merges = get_metadata_array_str(reader, "tokenizer.ggml.merges")
    scores = get_metadata_array_float(reader, "tokenizer.ggml.scores")
    sep_id = get_metadata_int(reader, "tokenizer.ggml.seperator_token_id")
    if sep_id is None:
        sep_id = get_metadata_int(reader, "tokenizer.ggml.separator_token_id")
    if source_model == "bert":
        tokens = [
            token[1:] if token.startswith("▁") else (
                token if token.startswith("[") or (
                    i < len(token_types) and token_types[i] in (2, 3, 4)
                ) else "##" + token
            )
            for i, token in enumerate(tokens)
        ]

    if not tokens:
        raise ValueError("GGUF has no tokenizer.ggml.tokens; cannot reconstruct a tokenizer")
    if len(set(tokens)) != len(tokens):
        raise ValueError("GGUF tokenizer vocabulary contains duplicate tokens")
    if any(not 0 <= token_id < len(tokens) for token_id in (additional_special_token_ids or [])):
        raise ValueError("Additional special token ID is outside the vocabulary")
    if token_types and len(token_types) != len(tokens):
        raise ValueError("GGUF tokenizer requires a type for every token")
    if scores and (len(scores) != len(tokens) or not np.isfinite(scores).all()):
        raise ValueError("GGUF tokenizer requires a finite score for every token")
    for role in ("bos", "eos", "unknown", "padding", "separator", "seperator", "mask"):
        token_id = get_metadata_int(reader, f"tokenizer.ggml.{role}_token_id")
        if token_id is not None and not 0 <= token_id < len(tokens):
            raise ValueError(f"GGUF tokenizer {role}_token_id is outside the vocabulary")
    if unk_id == -1:
        unk_id = next((i for i, kind in enumerate(token_types) if kind == 2), -1)
        if unk_id == -1:
            unk_id = next((tokens.index(t) for t in ("<unk>", "[UNK]") if t in tokens), -1)
    if pad_id == -1:
        pad_id = next((tokens.index(t) for t in ("<pad>", "[PAD]") if t in tokens), -1)
    if sep_id is None and model_type == "wordpiece" and "[SEP]" in tokens:
        sep_id = tokens.index("[SEP]")

    # --- Fix BOS/EOS for Qwen, DeepSeek, and similar families ---
    # These models use special tokens like <|endoftext|> (BOS) and <|im_end|> (EOS)
    # but GGUF files often omit bos_token_id / eos_token_id, or set them to wrong defaults (1, 2).
    #
    # Known special token names by role:
    SPECIAL_BOS_CANDIDATES = [
        "<|endoftext|>", "<s>", "<|begin_of_text|>", "<|startoftext|>", "[CLS]",
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
        "add_sep_token": add_sep_token,
        "model_max_length": model_max_length or 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }
    if tokenizer_pre:
        tokenizer_config["gguf_tokenizer_pre"] = tokenizer_pre
    special_tokens = {
        role: tokens[token_id]
        for role, token_id in (
            ("bos_token", bos_id), ("eos_token", eos_id),
            ("unk_token", unk_id), ("pad_token", pad_id),
            ("sep_token", sep_id),
            ("mask_token", get_metadata_int(reader, "tokenizer.ggml.mask_token_id")),
        )
        if token_id is not None and 0 <= token_id < vocab_size
    }
    if additional_special_token_ids:
        special_tokens["additional_special_tokens"] = [
            tokens[token_id] for token_id in additional_special_token_ids
        ]
    tokenizer_config.update(special_tokens)

    if arch not in GEMMA_ARCHES and (model_type == "llama" or model_type == "bpe"):
        tokenizer_config.update(
            {
                "model_type": "bpe",
                "tokenizer_class": "LlamaTokenizerFast"
                if "llama" in model_type
                else "PreTrainedTokenizerFast",
            }
        )

    # --- tokenizer.json (for fast tokenizers) ---
    tokenizer_json = None
    if embedded_hf_json:
        try:
            parsed_hf_json = json.loads(embedded_hf_json)
            if not isinstance(parsed_hf_json, dict) or not isinstance(
                parsed_hf_json.get("model"), dict,
            ):
                raise TypeError("embedded tokenizer JSON is not a tokenizer object")
            embedded_vocab = parsed_hf_json["model"].get("vocab")
            if isinstance(embedded_vocab, list):
                embedded_vocab = {row[0]: i for i, row in enumerate(embedded_vocab)}
            if not isinstance(embedded_vocab, dict) or not embedded_vocab:
                raise ValueError("embedded tokenizer JSON has no vocabulary")
            embedded_vocab = dict(embedded_vocab)
            for added in parsed_hf_json.get("added_tokens", []):
                embedded_vocab[added["content"]] = added["id"]
            if embedded_vocab != {token: i for i, token in enumerate(tokens)}:
                raise ValueError("embedded tokenizer vocabulary does not match GGUF token IDs")
            tokenizer_json = parsed_hf_json
            print("  ✓ Preserved embedded tokenizer.huggingface.json")
        except (ValueError, TypeError, KeyError, IndexError) as error:
            warnings.warn(f"Invalid tokenizer.huggingface.json; rebuilding tokenizer: {error}")

    if tokenizer_json is None:
        charsmap = reader.get_field("tokenizer.ggml.precompiled_charsmap")
        encoded_charsmap = None
        if charsmap is not None:
            encoded_charsmap = base64.b64encode(bytes(charsmap.contents())).decode("ascii")
        tokenizer_json = _build_tokenizer_json(
            tokens,
            token_types,
            merges,
            scores,
            source_model,
            bos_id,
            eos_id,
            pad_id,
            unk_id=unk_id,
            add_space_prefix=add_space_prefix,
            arch=arch,
            tokenizer_pre=tokenizer_pre,
            do_lower_case=get_metadata_bool(reader, "tokenizer.ggml.normalizer.lowercase"),
            strip_accents=get_metadata_bool(reader, "tokenizer.ggml.normalizer.strip_accents"),
            precompiled_charsmap=encoded_charsmap,
            special_token_ids=(additional_special_token_ids or []) + [
                token_id for token_id in (
                    sep_id, get_metadata_int(reader, "tokenizer.ggml.mask_token_id"),
                ) if token_id is not None
            ],
            remove_extra_whitespaces=get_metadata_bool(
                reader, "tokenizer.ggml.remove_extra_whitespaces",
            ),
        )
        end_id = sep_id if model_type == "wordpiece" and add_sep_token else eos_id
        add_end_token = add_sep_token if model_type == "wordpiece" else add_eos_token
        if add_bos_token or add_end_token:
            special_ids = [
                (tokens[token_id], token_id)
                for enabled, token_id in ((add_bos_token, bos_id), (add_end_token, end_id))
                if enabled and token_id is not None and 0 <= token_id < vocab_size
            ]
            prefix = (
                [{"SpecialToken": {"id": tokens[bos_id], "type_id": 0}}]
                if add_bos_token and bos_id >= 0 else []
            )
            suffix = (
                [{"SpecialToken": {"id": tokens[end_id], "type_id": 0}}]
                if add_end_token and end_id is not None and end_id >= 0 else []
            )
            pair_suffix = (
                [{"SpecialToken": {"id": tokens[end_id], "type_id": 1}}]
                if model_type == "wordpiece" and suffix else suffix
            )
            tokenizer_json["post_processor"] = {
                "type": "TemplateProcessing",
                "single": prefix + [{"Sequence": {"id": "A", "type_id": 0}}] + suffix,
                "pair": (
                    prefix + [{"Sequence": {"id": "A", "type_id": 0}}] + suffix
                    + ([] if model_type == "wordpiece" else prefix)
                    + [{"Sequence": {"id": "B", "type_id": 1}}]
                    + pair_suffix
                ),
                "special_tokens": {
                    token: {"id": token, "ids": [token_id], "tokens": [token]}
                    for token, token_id in special_ids
                },
            }
    chat_template = _get_chat_template(reader, arch)
    if chat_template:
        tokenizer_config["chat_template"] = chat_template
        (output_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")
        print("  ✓ Saved chat_template.jinja")
    vocab = {token: i for i, token in enumerate(tokens)}
    for filename, document in (
        ("tokenizer_config.json", tokenizer_config),
        ("special_tokens_map.json", special_tokens),
        ("vocab.json", vocab),
    ):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {filename}")
    if model_type in ("bpe", "gpt2") and merges:
        (output_dir / "merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8")
        print(f"  ✓ Saved merges.txt ({len(merges)} merges)")
    if tokenizer_json:
        with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        print("  ✓ Saved tokenizer.json")


def _sentencepiece_bpe_merges(
    tokens: list[str], token_types: list[int], scores: list[float],
) -> list[list[str]]:
    if len(scores) != len(tokens) or not np.isfinite(scores).all():
        raise ValueError("SentencePiece BPE requires a finite score for every token")
    if token_types and len(token_types) != len(tokens):
        raise ValueError("SentencePiece BPE requires a type for every token")
    vocab = {
        token: i for i, token in enumerate(tokens)
        if not token_types or token_types[i] == 1
    }
    candidates = []
    for token, token_id in vocab.items():
        splits = [
            (token[:i], token[i:]) for i in range(1, len(token))
            if token[:i] in vocab and token[i:] in vocab
        ]
        splits.sort(key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        candidates.extend((left, right, scores[token_id]) for left, right in splits)
    # SentencePiece selects the highest-scored adjacent piece, not Unigram's best path.
    candidates.sort(key=lambda item: (item[2], len(item[0]), len(item[1])), reverse=True)
    return [[left, right] for left, right, _ in candidates]


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
    arch: str = "llama",
    tokenizer_pre: str | None = None,
    do_lower_case: bool | None = None,
    remove_extra_whitespaces: bool | None = None,
    strip_accents: bool | None = None,
    precompiled_charsmap: str | None = None,
    special_token_ids: list[int] | None = None,
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
        role_token = i in (bos_id, eos_id, pad_id, unk_id) or i in (special_token_ids or [])
        if tt in (2, 3, 4) or role_token:
            special = tt in (2, 3) or role_token
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

    normalized_model_type = {"bert": "wordpiece", "t5": "unigram"}.get(
        model_type.lower(), model_type.lower(),
    )

    # Build model block
    has_byte_fallback = any(
        (i < len(token_types) and token_types[i] == 6)
        or (token.startswith("<0x") and token.endswith(">"))
        for i, token in enumerate(tokens)
    )
    sentencepiece_bpe = normalized_model_type == "llama" or (
        arch == "gemma" and normalized_model_type in ("spm", "sentencepiece", "unigram")
    )
    if sentencepiece_bpe:
        if not 0 <= unk_id < len(tokens):
            raise ValueError("SentencePiece BPE reconstruction requires a valid unknown_token_id")
        model_block = {
            "type": "BPE",
            "vocab": vocab,
            "merges": _sentencepiece_bpe_merges(tokens, token_types, scores),
            "unk_token": tokens[unk_id],
            "byte_fallback": has_byte_fallback,
            "fuse_unk": True,
        }
    elif normalized_model_type in ("bpe", "gpt2"):
        for merge in merges:
            pieces = merge.split(" ")
            if len(pieces) != 2 or any(piece not in vocab for piece in pieces):
                raise ValueError(f"Invalid GGUF BPE merge: {merge!r}")
            if "".join(pieces) not in vocab:
                raise ValueError(f"GGUF BPE merge result is missing from the vocabulary: {merge!r}")
        byte_fallback = any(
            (i < len(token_types) and token_types[i] == 6)
            or (token.startswith("<0x") and token.endswith(">"))
            for i, token in enumerate(tokens)
        )
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[unk_id] if 0 <= unk_id < len(tokens) else None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": byte_fallback,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    elif normalized_model_type in ("llama", "spm", "sentencepiece", "unigram"):
        if not 0 <= unk_id < len(tokens):
            raise ValueError("Unigram reconstruction requires a valid unknown_token_id")
        if len(scores) != len(tokens) or not np.isfinite(scores).all():
            raise ValueError("Unigram reconstruction requires a finite score for every token")
        vocab_scores = []
        for i, token in enumerate(tokens):
            score = scores[i] if i < len(scores) else 0.0
            vocab_scores.append([token, score])
        model_block = {
            "type": "Unigram",
            "unk_id": unk_id,
            "vocab": vocab_scores,
            "byte_fallback": has_byte_fallback,
        }
    elif normalized_model_type == "wordpiece":
        if not 0 <= unk_id < len(tokens):
            raise ValueError("WordPiece reconstruction requires a valid unknown_token_id")
        model_block = {
            "type": "WordPiece",
            "unk_token": tokens[unk_id] if 0 <= unk_id < len(tokens) else "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab,
        }
    else:
        raise ValueError(f"Unsupported GGUF tokenizer model: {model_type}")

    normalizer = (
        None if arch in {"qwen2", "qwen2moe", "qwen3moe", *QWEN_VL_TEXT_ARCHES}
        or model_type == "t5"
        else {"type": "NFC"}
    )
    if precompiled_charsmap:
        normalizer = {"type": "Precompiled", "precompiled_charsmap": precompiled_charsmap}
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
    if normalized_model_type in ("bpe", "gpt2") and tokenizer_pre:
        if tokenizer_pre in ("gpt-2", "gpt2"):
            normalizer = None
            pre_tokenizer = {
                "type": "ByteLevel", "add_prefix_space": False,
                "trim_offsets": False, "use_regex": True,
            }
        elif tokenizer_pre in ("llama-bpe", "llama3", "llama-v3"):
            model_block["ignore_merges"] = True
            normalizer = None
            pre_tokenizer["pretokenizers"][0]["pattern"]["Regex"] = (
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            )
        elif tokenizer_pre in ("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2"):
            normalizer = None
        else:
            warnings.warn(
                f"Unsupported tokenizer.ggml.pre={tokenizer_pre!r}; "
                "using legacy BPE splitting, exact tokenization parity is not guaranteed",
            )
    if normalized_model_type in ("bpe", "gpt2") and has_byte_fallback:
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "ByteFallback"},
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
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
        if not add_space_prefix:
            decoder["decoders"].pop()
        if remove_extra_whitespaces:
            normalizer = {
                "type": "Sequence",
                "normalizers": [
                    *([normalizer] if normalizer else []),
                    {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                    {"type": "Strip", "strip_left": True, "strip_right": True},
                ],
            }
        if model_type == "t5":
            normalizers = []
            if precompiled_charsmap:
                normalizers = [{
                    "type": "Precompiled", "precompiled_charsmap": precompiled_charsmap,
                }]
            if remove_extra_whitespaces:
                normalizers.extend([
                    {"type": "Replace", "pattern": {"Regex": " +"}, "content": " "},
                    {"type": "Replace", "pattern": {"Regex": "^ | $"}, "content": ""},
                    {
                        "type": "Replace", "pattern": {"Regex": "^(?=[\\s\\S])"},
                        "content": "▁",
                    },
                ])
            elif add_space_prefix:
                normalizers.append({
                    "type": "Replace", "pattern": {"Regex": "^(?=[\\s\\S]*[^ ])"},
                    "content": "▁",
                })
            normalizers.append({
                "type": "Replace", "pattern": {"String": " "}, "content": "▁",
            })
            normalizer = {"type": "Sequence", "normalizers": normalizers}
            pre_tokenizer = None
        if sentencepiece_bpe:
            normalizers = []
            if remove_extra_whitespaces:
                normalizers.extend([
                    {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                    {"type": "Strip", "strip_left": True, "strip_right": True},
                ])
            if add_space_prefix:
                normalizers.append({"type": "Prepend", "prepend": "▁"})
            normalizers.append({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})
            normalizer = {"type": "Sequence", "normalizers": normalizers}
            pre_tokenizer = None
    elif normalized_model_type == "wordpiece":
        normalizer = {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": strip_accents,
            "lowercase": True if do_lower_case is None else do_lower_case,
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
    dims = {"num_heads": n_heads, "qk_nope_head_dim": qk_nope, "v_head_dim": v_head}
    if arch == "gemma4":
        dims["num_attention_heads"] = (
            get_metadata_int(reader, "gemma4.attention.head_count") or 8
        )
        dims["num_key_value_heads"] = (
            get_metadata_int(reader, "gemma4.attention.head_count_kv") or 1
        )
        dims["gemma4_head_dim"] = (
            get_metadata_int(reader, "gemma4.attention.key_length")
            or get_metadata_int(reader, "gemma4.attention.key_length_swa")
            or 256
        )
    return dims


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
    if arch == "qwen2moe" and re.fullmatch(r"blk\.\d+\.ffn_gate_inp_shexp\.weight", gguf_name):
        arr = arr.reshape(1, -1)

    # Some Gemma4 exporters fuse full-attention Q/K/V projections. MLX-LM's
    # Gemma4 attention module keeps these as independent linear layers.
    match = re.fullmatch(r"blk\.(\d+)\.attn_qkv\.weight", gguf_name)
    if match and arch == "gemma4":
        layer_idx = match.group(1)
        heads = mla_dims["num_attention_heads"]
        kv_heads = mla_dims["num_key_value_heads"]
        head_dim = mla_dims["gemma4_head_dim"]
        q_size = heads * head_dim
        kv_size = kv_heads * head_dim
        if arr.shape[0] == q_size + 2 * kv_size:
            q, k, v = np.split(arr, [q_size, q_size + kv_size], axis=0)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v),
            ]
        if arr.shape[0] == q_size + kv_size:
            q, k = np.split(arr, [q_size], axis=0)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k),
            ]
        raise ValueError(f"Invalid Gemma4 fused QKV shape: {gguf_name}: {arr.shape}")

    # --- Phi-3 split Q/K/V -> fused qkv_proj ---
    match = re.fullmatch(r"blk\.(\d+)\.attn_(q|k|v)\.weight", gguf_name)
    if match and arch in PHI_ARCHES:
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
    logical_shape = tuple(reversed(tensor.shape))
    raw_data = tensor.data

    if qtype_val == 0:  # F32
        arr = np.array(raw_data, dtype=np.float32).reshape(logical_shape)
        if dtype == "float16":
            arr = arr.astype(np.float16)
        return arr
    if qtype_val == 1:  # F16
        arr = np.array(raw_data, dtype=np.float16).reshape(logical_shape)
        return arr.astype(np_dtype)
    if qtype_val == 28:  # F64
        arr = np.array(raw_data, dtype=np.float64).reshape(logical_shape)
        return arr.astype(np_dtype)
    if qtype_val in (24, 25, 26, 27):  # I8, I16, I32, I64
        int_dtype_map = {24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64}
        arr = np.array(raw_data, dtype=int_dtype_map.get(qtype_val, np.int32))
        arr = arr.reshape(logical_shape).astype(np_dtype)
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
    """Quantize dense or stacked-expert linear projection weights."""
    return arr.ndim >= 2 and hf_name.endswith("_proj.weight")


def _quantize_affine_4bit(
    arr: np.ndarray, q_group_size: int, *, floating_dtype: str = "float16",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize an array of linear weights into affine 4-bit groups."""
    if arr.ndim < 2:
        raise ValueError(f"Direct quantization expects at least 2D tensors, got {arr.ndim}D")
    leading_shape, cols = arr.shape[:-1], arr.shape[-1]
    if cols % q_group_size != 0:
        raise ValueError(
            f"Column size {cols} is not divisible by q_group_size={q_group_size}"
        )

    grouped = arr.astype(np.float32).reshape(
        *leading_shape, cols // q_group_size, q_group_size
    )
    mins = grouped.min(axis=-1)
    maxs = grouped.max(axis=-1)
    scales = (maxs - mins) / 15.0
    zero_scale = scales == 0.0
    safe_scales = scales.copy()
    safe_scales[zero_scale] = 1.0

    q = np.round((grouped - mins[..., None]) / safe_scales[..., None])
    q = np.clip(q, 0, 15).astype(np.uint8)
    q[zero_scale[..., None].repeat(q_group_size, axis=-1)] = 0

    codes = q.reshape(*leading_shape, cols // 8, 8).astype(np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = np.bitwise_or.reduce(codes << shifts, axis=-1)
    return packed, safe_scales.astype(floating_dtype), mins.astype(floating_dtype)


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
            weight_map.update(dict.fromkeys(f.keys(), new_name))

    index_json = {
        "metadata": {"total_size": total_bytes_out},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)
    return len(weight_map), total_shards


def _iter_weight_arrays(reader: GGUFReader, arch: str, dtype: str):
    """Decode lazily, visiting related headers together to bound fusion buffers."""
    groups: dict[str, list[Any]] = {}
    for tensor in reader.tensors:
        if arch in PHI_ARCHES and re.fullmatch(
            r"(?:blk\.\d+\.)?rope_factors_(?:long|short)(?:\.weight)?",
            tensor.name,
        ):
            continue
        if arch in QWEN_VL_TEXT_ARCHES:
            if Qwen3VLTextEncoderConverter.is_visual_tensor(tensor.name):
                print(f"    Skipping vision/projector tensor: {tensor.name}")
                continue
            _map_tensor_name(tensor.name, arch)
        group = tensor.name
        if arch in PHI_ARCHES:
            match = re.fullmatch(r"(blk\.\d+)\.attn_[qkv]\.weight", tensor.name)
            if match:
                group = match[1] + ".attn_qkv.weight"
        elif arch in ("deepseek2", "deepseek3", "glm-dsa", "glm4moe"):
            match = re.fullmatch(r"(blk\.\d+)\.attn_[kv]_b(?:\.weight)?", tensor.name)
            if match:
                group = match[1] + ".attn_kv_b.weight"
        groups.setdefault(group, []).append(tensor)

    mla_dims = _read_mla_dims(reader, arch)
    pending_kv_b: dict[str, dict[str, np.ndarray]] = {}
    pending_qkv: dict[str, dict[str, np.ndarray]] = {}
    with tqdm(total=len(reader.tensors), desc="  Converting", unit="tensor") as pbar:
        for tensors in groups.values():
            for tensor in tensors:
                arr = _decode_tensor_to_array(tensor, dtype, tensor.name)
                if arr is None:
                    raise RuntimeError(f"Failed to convert tensor {tensor.name}")
                pairs = _plan_tensor_emit(
                    tensor.name, arr, arch, mla_dims, pending_kv_b, pending_qkv,
                )
                for name, output in pairs:
                    yield tensor.name, tensor.tensor_type, name, output
                    del output
                del arr, pairs
                pbar.update(1)
            if pending_qkv:
                raise RuntimeError("Incomplete Phi-3 QKV tensor groups")
            if pending_kv_b:
                raise RuntimeError("Incomplete split kv_b tensor groups")


def extract_and_convert_weights(
    reader: GGUFReader, arch: str, output_dir: Path, dtype: str = "float16",
    max_shard_bytes: int = int(4.5 * 1e9),
) -> None:
    """Extract, transform, and stream tensors without buffering a shard in RAM."""
    print(f"\n  Converting {len(reader.tensors)} tensors...")
    print(f"  Output dtype: {dtype}")
    with StreamingShardWriter(output_dir, max_shard_bytes) as writer:
        for _, _, name, array in _iter_weight_arrays(reader, arch, dtype):
            writer.add(name, array, **({"dtype": "bfloat16"} if dtype == "bfloat16" else {}))
            del array
        if arch == "glm-dsa":
            layers = _detect_full_indexer_layers(list(writer.weight_map))
            print(f"    ℹ IndexShare: {len(layers)} Full-indexer layer(s): {layers}")
        writer.finish()
    print(f"\n  ✓ Saved {len(writer.weight_map)} weight tensors ({len(writer.files)} shards)")
    print(f"    Total input:  {sum(t.n_bytes for t in reader.tensors) / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {writer.total / 1e9:.2f} GB (safetensors)")


def extract_and_convert_weights_direct_quant(
    reader: GGUFReader,
    arch: str,
    output_dir: Path,
    dtype: str,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
    architecture_kind: str = "dense",
    moe_router_protect: bool = True,
    mixed_precision: bool = False,
    max_shard_bytes: int | None = None,
) -> dict[str, Any]:
    """Bounded-memory direct quantization path for supported linear weights."""
    if q_bits != 4 or q_group_size != 64 or q_mode != "affine":
        raise RuntimeError(
            "Direct quantization currently supports only affine 4-bit with q_group_size=64"
        )

    np_dtype = np.float16 if dtype == "float16" else np.float32
    quantized_tensors = 0
    fallback_tensors = 0
    decisions: list[dict[str, Any]] = []
    protected_routers: list[dict[str, Any]] = []
    compressed_experts: list[dict[str, Any]] = []
    unknown_tensors: list[str] = []
    if max_shard_bytes is None:
        max_shard_bytes = DIRECT_QUANT_DEFAULT_MAX_SHARD_BYTES

    with StreamingShardWriter(output_dir, max_shard_bytes) as writer:
        for gguf_name, qtype, hf_name, out_arr in _iter_weight_arrays(reader, arch, dtype):
            category = classify_tensor(gguf_name, hf_name)
            protected = (
                architecture_kind in {"moe", "hybrid"}
                and moe_router_protect
                and category == "router"
            ) or (mixed_precision and category in {"router", "attention"})
            quantizable = _is_direct_quantizable_linear_weight(hf_name, out_arr)
            should_quantize = quantizable and not protected
            precision: int | str = 4 if should_quantize else dtype

            if should_quantize:
                qtype_val = int(qtype)
                supported_qtypes = DIRECT_QUANT_SUPPORTED_SOURCE_QTYPES
                if arch in QWEN_VL_TEXT_ARCHES:
                    supported_qtypes = supported_qtypes | {0, 1}
                if qtype_val not in supported_qtypes:
                    raise RuntimeError(
                        f"Unsupported source quantization for direct quantization: "
                        f"{gguf_name} has qtype={qtype_val}; supported: "
                        f"{sorted(supported_qtypes)}"
                    )
                preserve_float_range = dtype == "bfloat16" or (
                    arch in QWEN_VL_TEXT_ARCHES and dtype == "float32"
                )
                quant_options = {"floating_dtype": "float32"} if preserve_float_range else {}
                packed, scales, biases = _quantize_affine_4bit(
                    out_arr, q_group_size, **quant_options,
                )
                base = hf_name.removesuffix(".weight")
                writer.add(f"{base}.weight", packed)
                bf16_options = {"dtype": "bfloat16"} if dtype == "bfloat16" else {}
                writer.add(f"{base}.scales", scales.astype(np_dtype), **bf16_options)
                writer.add(f"{base}.biases", biases.astype(np_dtype), **bf16_options)
                del packed, scales, biases
                quantized_tensors += 1
            else:
                writer.add(
                    hf_name, out_arr,
                    **({"dtype": "bfloat16"} if dtype == "bfloat16" else {}),
                )
                fallback_tensors += 1
            del out_arr

            decision = {
                "source_name": gguf_name,
                "output_name": hf_name,
                "category": category,
                "precision": precision,
                "protected": protected,
            }
            decisions.append(decision)
            if category == "router" and protected:
                protected_routers.append(decision)
            if category == "expert_mlp" and should_quantize:
                compressed_experts.append(decision)
            if category == "unknown":
                unknown_tensors.append(gguf_name)
        writer.finish()

    print(f"\n  ✓ Direct quantization wrote {len(writer.weight_map)} tensors "
          f"across {len(writer.files)} shards")
    print(f"    Quantized projection tensors: {quantized_tensors}")
    print(f"    Fallback non-projection tensors: {fallback_tensors}")
    print(f"    Total input:  {sum(t.n_bytes for t in reader.tensors) / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {writer.total / 1e9:.2f} GB (safetensors)")
    if unknown_tensors:
        print(
            "    ⚠ Unclassified block tensors retained without policy assumptions: "
            + ", ".join(sorted(set(unknown_tensors)))
        )
    return {
        "schema_version": 1,
        "architecture": arch,
        "architecture_kind": architecture_kind,
        "mixed_precision": mixed_precision,
        "router_protection": moe_router_protect,
        "protected_router_tensors": protected_routers,
        "compressed_expert_tensors": compressed_experts,
        "unknown_tensors": sorted(set(unknown_tensors)),
        "tensor_decisions": decisions,
    }


def _write_runtime_adapter(output_path: Path, arch: str, config: dict[str, Any]) -> None:
    if arch == "gemma":
        filename = "gemma_model.py"
        shutil.copyfile(Path(__file__).parent / "data" / filename, output_path / filename)
        config["model_file"] = filename


def _convert_direct_quantized(
    gguf_path: str,
    output_dir: str,
    dtype: str,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
    moe_router_protect: bool = True,
    mixed_precision: bool = False,
    max_shard_bytes: int | None = None,
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
    architecture_kind = classify_architecture(reader, arch)
    try:
        validate_moe_layout(reader, arch, architecture_kind)
    except RuntimeError as error:
        print(f"❌ {error}")
        return False
    if mixed_precision and architecture_kind == "dense":
        print("❌ --mixed-precision currently requires a detected MoE or hybrid architecture")
        return False
    variant_error = validate_architecture_variant(reader, arch)
    if variant_error:
        print(f"❌ {variant_error}")
        return False

    print("\n[direct-quant] Building config and tokenizer...")
    config = build_config(reader, arch, dtype)
    _write_runtime_adapter(output_path, arch, config)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    token_options = (
        {"additional_special_token_ids": [
            config[key] for key in (
                "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
            )
        ]} if arch in QWEN_VL_TEXT_ARCHES else {}
    )
    extract_tokenizer(reader, output_path, arch, config["max_position_embeddings"], **token_options)

    print("\n[direct-quant] Converting tensors with bounded memory...")
    try:
        report = extract_and_convert_weights_direct_quant(
            reader,
            arch,
            output_path,
            dtype=dtype,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
            architecture_kind=architecture_kind,
            moe_router_protect=moe_router_protect,
            mixed_precision=mixed_precision,
            max_shard_bytes=max_shard_bytes,
        )
    except Exception as error:  # noqa: BLE001
        print(f"❌ Direct quantization failed: {error}")
        return False

    if mixed_precision:
        jang_config = {
            "format": "gguf2mlx-mixed-v1",
            "group_size": q_group_size,
            "mode": q_mode,
            "default_bits": q_bits,
            "high_precision_categories": ["attention", "router"],
            "low_precision_categories": ["expert_mlp"],
            "tensor_decisions": {
                item["output_name"]: item["precision"]
                for item in report["tensor_decisions"]
            },
        }
        with open(output_path / "jang_config.json", "w") as f:
            json.dump(jang_config, f, indent=2)
        config["gguf2mlx_mixed_precision"] = {
            "format": "gguf2mlx-mixed-v1",
            "runtime": "JANG-compatible runtime required",
        }
    else:
        config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
            "mode": q_mode,
            "scheme": "direct_affine_4bit_v1",
        }
    with open(output_path / "conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return True
# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def _convert(
    gguf_path: str, output_dir: str, dtype: str = "float16",
    max_shard_bytes: int = int(4.5 * 1e9),
) -> bool:
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
    architecture_kind = classify_architecture(reader, arch)
    try:
        validate_moe_layout(reader, arch, architecture_kind)
    except RuntimeError as error:
        print(f"❌ {error}")
        return False
    variant_error = validate_architecture_variant(reader, arch)
    if variant_error:
        print(f"❌ {variant_error}")
        return False
    hf_type = ARCH_MAP.get(arch, arch)
    model_name_full = get_metadata_str(reader, "general.name") or model_name
    print(f"  Architecture: {arch} ({architecture_kind}; HF type: {hf_type})")
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
    _write_runtime_adapter(output_path, arch, config)
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
    token_options = (
        {"additional_special_token_ids": [
            config[key] for key in (
                "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
            )
        ]} if arch in QWEN_VL_TEXT_ARCHES else {}
    )
    extract_tokenizer(reader, output_path, arch, config["max_position_embeddings"], **token_options)

    # Step 4: Extract, dequantize, and convert weights
    print("\n[4/5] Extracting and converting weights...")
    try:
        extract_and_convert_weights(reader, arch, output_path, dtype, max_shard_bytes)
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
    dtype: str | None = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    direct_quant: bool = False,
    moe_router_protect: bool = True,
    mixed_precision: bool = False,
    model_type: str = "auto",
    mmproj: str | None = None,
    hf_model: str | None = None,
    hf_fallback_vision: str | None = None,
    hf_revision: str = "main",
    offline: bool = False,
    preserve_source_quant: bool = False,
    max_shard_gb: float | None = None,
) -> bool:
    """Convert into a staging directory so failed runs never leave partial output."""
    shard_options = {}
    if max_shard_gb is not None:
        if not np.isfinite(max_shard_gb) or max_shard_gb <= 0:
            print("❌ max_shard_gb must be finite and positive")
            return False
        byte_limit = max_shard_gb * 1024**3
        if not np.isfinite(byte_limit):
            print("❌ max_shard_gb is too large")
            return False
        max_shard_bytes = int(byte_limit)
        if max_shard_bytes < 1:
            print("❌ max_shard_gb must specify at least one byte")
            return False
        shard_options["max_shard_bytes"] = max_shard_bytes
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
        from .vlm import convert_vlm, is_vlm

        if model_type not in {"auto", "llm", "vlm", "text-encoder"}:
            print(f"❌ Unsupported conversion type: {model_type}")
            return False
        vlm_options = bool(mmproj or hf_model or hf_fallback_vision or preserve_source_quant)
        if model_type == "llm" and vlm_options:
            print("❌ VLM source options cannot be used with --type llm")
            return False
        detected_vlm = False
        detected_arch = None
        if Path(gguf_path).is_file():
            try:
                reader = GGUFReader(gguf_path)
                if model_type == "text-encoder":
                    detected_arch = detect_architecture(reader)
                detected_vlm = is_vlm(reader)
                del reader
            except Exception as error:  # noqa: BLE001
                print(f"❌ Could not inspect GGUF: {error}")
                return False
        if model_type == "text-encoder":
            if detected_arch not in QWEN_VL_TEXT_ARCHES:
                print("❌ --type text-encoder requires a Qwen2-VL or Qwen3-VL GGUF")
                return False
            if vlm_options or mixed_precision:
                print("❌ Text-only conversion cannot use VLM source or MoE precision options")
                return False
            if direct_quant != quantize:
                print("❌ Text-encoder quantization requires --quantize --direct-quant together")
                return False
            dtype = dtype or "bfloat16"
            if dtype == "float16" and not direct_quant:
                print("❌ Text conditioning requires BF16 or FP32; FP16 requires --direct-quant")
                return False
            detected_vlm = False
        else:
            dtype = dtype or "float16"
            if dtype == "bfloat16":
                print("❌ --dtype bfloat16 currently requires --type text-encoder")
                return False
        if dtype not in {"float16", "float32", "bfloat16"}:
            print(f"❌ Unsupported output dtype: {dtype}")
            return False
        if model_type == "llm" and detected_vlm:
            print("❌ Multimodal GGUF detected; use --type vlm with HF assets")
            return False
        if model_type == "vlm" or vlm_options or detected_vlm:
            if preserve_source_quant and not (quantize and direct_quant):
                print("❌ --preserve-source-quant requires --quantize --direct-quant")
                return False
            if mixed_precision or quantize != direct_quant:
                print(
                    "❌ VLM quantization requires --quantize --direct-quant together; "
                    "--mixed-precision is MoE-only"
                )
                return False
            try:
                vlm_quant_options = (
                    {"direct_quant": True, "q_bits": q_bits,
                     "q_group_size": q_group_size, "q_mode": q_mode}
                    if direct_quant else {}
                )
                if preserve_source_quant:
                    vlm_quant_options["preserve_source_quant"] = True
                convert_vlm(
                    gguf_path,
                    fp_output_path,
                    dtype,
                    mmproj=mmproj,
                    hf_model=hf_model,
                    hf_fallback_vision=hf_fallback_vision,
                    hf_revision=hf_revision,
                    offline=offline,
                    **shard_options,
                    **vlm_quant_options,
                )
            except Exception as error:  # noqa: BLE001
                print(f"❌ VLM conversion failed: {error}")
                return False
            if output_path.exists():
                output_path.rmdir()
            fp_output_path.replace(output_path)
            print(f"✅ VLM conversion complete: {output_path}")
            return True
        if mixed_precision and not (quantize and direct_quant):
            print("❌ --mixed-precision requires --quantize and --direct-quant")
            return False
        if quantize and direct_quant:
            try:
                succeeded = _convert_direct_quantized(
                    gguf_path,
                    str(direct_quantized_output_path),
                    dtype=dtype,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    q_mode=q_mode,
                    moe_router_protect=moe_router_protect,
                    mixed_precision=mixed_precision,
                    **shard_options,
                )
            except Exception as error:  # noqa: BLE001
                print(f"❌ Conversion failed: {error}")
                return False
            if not succeeded:
                return False
            final_stage_path = direct_quantized_output_path
        else:
            try:
                succeeded = _convert(gguf_path, str(fp_output_path), dtype, **shard_options)
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
        "--type", dest="model_type", choices=["auto", "llm", "vlm", "text-encoder"],
        default="auto",
        help="Conversion family (default: auto-detect multimodal metadata/tensors)",
    )
    parser.add_argument(
        "--mmproj", help="Companion vision/projector GGUF for split VLM releases",
    )
    parser.add_argument(
        "--hf-model", help="Matching full HF VLM repository or local directory for config/assets",
    )
    parser.add_argument(
        "--hf-fallback-vision",
        help="Use original vision weights from this matching HF VLM repository/local directory",
    )
    parser.add_argument(
        "--hf-revision", default="main", help="HF source revision; pin a commit for reproducibility",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use only local/cached HF VLM assets",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input GGUF file path"
    )
    parser.add_argument(
        "--output", "-o", help="Output MLX directory"
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "float32", "bfloat16"],
        help="Output dtype (default: BF16 for text-encoder, FP16 otherwise)",
    )
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        help=(
            "Maximum streamed shard payload in GiB (oversized tensors remain whole; "
            "defaults: float LLM 4.5 GB, direct-quant/VLM 256 MiB)"
        ),
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize MLX output (VLM requires --direct-quant; quantizes language projections only)",
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
            "(affine 4-bit, group size 64; VLM keeps vision/projector floating-point)"
        ),
    )
    parser.add_argument(
        "--preserve-source-quant",
        action="store_true",
        help=(
            "Qwen2-VL only, with --quantize --direct-quant: preserve Q4_K codes "
            "and their 32-value groups; keep other projections floating-point "
            "(larger mixed-precision output, avoids lossy uniform re-quantization)"
        ),
    )
    parser.add_argument(
        "--moe-router-protect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep recognized MoE router tensors at FP16 (default: enabled)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use FP16 attention/router tensors and 4-bit expert MLP tensors (experimental)",
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
        from .vlm import is_vlm

        reader = GGUFReader(args.input)
        print(f"Architecture: {detect_architecture(reader)}")
        print(f"Conversion type: {'vlm' if is_vlm(reader) else 'llm'}")
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
        moe_router_protect=args.moe_router_protect,
        mixed_precision=args.mixed_precision,
        model_type=args.model_type,
        mmproj=args.mmproj,
        hf_model=args.hf_model,
        hf_fallback_vision=args.hf_fallback_vision,
        hf_revision=args.hf_revision,
        offline=args.offline,
        preserve_source_quant=args.preserve_source_quant,
        max_shard_gb=args.max_shard_gb,
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
