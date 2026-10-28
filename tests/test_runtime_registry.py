"""Runtime-registry conformance for every artifact this project emits.

The weight and load gates answer "did these bytes survive?" and "does this one
real checkpoint run?". This suite answers the question between them: can the
*declared* MLX runtime even resolve the ``model_type`` written into every
converted config?

This file is intentionally run on Apple Silicon in the Runtime contract
workflow. A portable test guards the derived model-type inventory so adding an
adapter without extending the contract cannot silently reduce coverage.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from typing import Any

import pytest

from gguf2mlx import gguf2mlx as core
from gguf2mlx import vlm

MINIMUM_MLX_LM = (0, 31, 3)
MINIMUM_MLX_VLM = (0, 7, 10)

# Explicit inventory on purpose: adding an adapter must update this list and
# therefore force a conscious decision about the runtime that can load it.
EXPECTED_LLM_MODEL_TYPES = frozenset({
    "deepseek_v2",
    "deepseek_v3",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4_text",
    "glm4_moe",
    "glm_moe_dsa",
    "llama",
    "mistral",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_5",
    "qwen3_moe",
    "stablelm",
})
EXPECTED_VLM_MODEL_TYPES = frozenset(vlm.ARCHITECTURE_REGISTRY)
VLM_TEXT_HOSTS = {
    "qwen2_vl_text": "qwen2_vl",
    "qwen3_vl_text": "qwen3_vl",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(part) for part in numbers[:3])


def _converted_llm_types() -> set[str]:
    text_arches = set(core.QWEN_VL_TEXT_ARCHES)
    return {
        core.ARCH_MAP.get(arch, arch)
        for arch in core.CONVERTIBLE_ARCHES
        if arch not in text_arches
    }


def _converted_vlm_text_types() -> set[str]:
    return {
        core.ARCH_MAP.get(arch, arch)
        for arch in core.QWEN_VL_TEXT_ARCHES
    }


def test_llm_registry_inventory_is_explicit() -> None:
    """Every LLM adapter must be represented in the runtime contract."""
    assert _converted_llm_types() == set(EXPECTED_LLM_MODEL_TYPES), (
        "the emitted LLM model_type inventory changed; add the new type to "
        "EXPECTED_LLM_MODEL_TYPES and verify it in the pinned runtime"
    )


def test_vlm_registry_inventory_is_explicit() -> None:
    """Every full-VLM adapter must be represented in the runtime contract."""
    assert set(EXPECTED_VLM_MODEL_TYPES) == set(vlm.ARCHITECTURE_REGISTRY)
    assert EXPECTED_VLM_MODEL_TYPES == {"llava", "qwen2_vl", "minicpmv4_7"}


def test_text_encoder_runtime_hosts_are_explicit() -> None:
    """Text-only Qwen-VL outputs must have a named runtime package."""
    assert _converted_vlm_text_types() == set(VLM_TEXT_HOSTS)


def _require_mlx_lm() -> Any:
    mlx_lm = pytest.importorskip(
        "mlx_lm", reason="runtime registry checks need Apple Silicon and mlx-lm"
    )
    lm_version = _version_tuple(importlib.metadata.version("mlx-lm"))
    assert lm_version >= MINIMUM_MLX_LM, (
        f"mlx-lm {lm_version} is older than the declared floor {MINIMUM_MLX_LM}"
    )
    return mlx_lm


def _require_mlx_vlm() -> Any:
    mlx_vlm = pytest.importorskip(
        "mlx_vlm", reason="runtime registry checks need Apple Silicon and mlx-vlm"
    )
    vlm_version = _version_tuple(importlib.metadata.version("mlx-vlm"))
    assert vlm_version >= MINIMUM_MLX_VLM, (
        f"mlx-vlm {vlm_version} is older than the declared floor {MINIMUM_MLX_VLM}"
    )
    return mlx_vlm


@pytest.mark.parametrize("model_type", sorted(EXPECTED_LLM_MODEL_TYPES))
def test_supported_llm_model_type_resolves(model_type: str) -> None:
    """Every LLM config this converter writes must resolve in mlx-lm.

    Only mlx-lm is required: a text-only environment still has to answer this,
    and coupling it to mlx-vlm made the check skip wherever the VLM runtime was
    absent, which is most developer machines.
    """
    mlx_lm = _require_mlx_lm()
    utils = importlib.import_module(f"{mlx_lm.__name__}.utils")
    model_class, args_class = utils._get_classes({"model_type": model_type})
    assert model_class is not None
    assert args_class is not None


@pytest.mark.parametrize("model_type", sorted(EXPECTED_VLM_MODEL_TYPES))
def test_supported_vlm_model_type_resolves(model_type: str) -> None:
    """Every full-VLM config this converter writes must resolve in mlx-vlm."""
    mlx_vlm = _require_mlx_vlm()
    utils = importlib.import_module(f"{mlx_vlm.__name__}.utils")
    module, resolved = utils.get_model_and_args({"model_type": model_type})
    assert module is not None
    assert isinstance(resolved, str) and resolved


@pytest.mark.parametrize("text_type,host", sorted(VLM_TEXT_HOSTS.items()))
def test_supported_vlm_text_encoder_host_imports(
    text_type: str, host: str
) -> None:
    """Text-encoder outputs must target a real mlx-vlm family module."""
    mlx_vlm = _require_mlx_vlm()
    module = importlib.import_module(f"{mlx_vlm.__name__}.models.{host}")
    assert module is not None, text_type
