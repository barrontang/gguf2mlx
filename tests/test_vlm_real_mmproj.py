"""Plan-versus-reality checks against real llama.cpp ``mmproj`` exports.

``tests/vlm_plan_fixtures.py`` proves the converter *assembles* what an adapter
declares. It cannot prove the declaration is correct, because the fixture writes
exactly the sources the plan asks for: a plan that invents ``ln1.bias`` would
still round-trip through it. This module closes that gap the only way that counts
— it reads a real ``mmproj`` written by llama.cpp and checks the plan against the
tensors that were actually shipped.

What each case asserts:

1. the declared source set and the file's ``v.*``/``mm.*`` tensor set are equal,
   in both directions (a missing source means a broken export; an undeclared
   source means the converter would silently drop weights);
2. every declared source shape matches the real tensor, decoded;
3. concatenating/stacking the real shapes along the plan's axis reproduces each
   declared target shape — this is what catches a fused QKV whose axis is off or
   a patch embedding stacked along the wrong dimension;
4. ``validate_vision_metadata`` *accepts* the real file, so a fail-closed check
   that is merely too strict cannot ship unnoticed.

The third point is the reason a shape bug cannot hide: a fused QKV concatenated
along the wrong axis changes shape, so the replay catches it. Tensor *ordering*
inside a correct shape does not, and is covered separately by the element-wise
comparison against the original HF checkpoint recorded in
``docs/vlm-conversion.md``.

Run with::

    GGUF2MLX_RUN_E2E=1 pytest tests/test_vlm_real_mmproj.py -q

Fixture files are looked up in ``$GGUF2MLX_VLM_MMPROJ_DIR`` (default
``~/.cache/gguf2mlx-vlm``) and the case is skipped when a file is absent, so a
checkout without the ~2 GB of downloads stays green.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from gguf2mlx import vlm


@dataclass(frozen=True)
class Case:
    """One real ``mmproj`` file and the adapter that must reproduce it."""

    label: str
    filename: str
    hf_repo: str
    module: str
    adapter_class: str
    #: number of ``v.*``/``mm.*`` tensors in the real file; pinning the count
    #: makes a silently regenerated fixture visible instead of quietly passing.
    tensor_count: int
    #: language-side dimensions the *plan* needs. Only adapters that build their
    #: plan from a caller-supplied text config need them: LLaVA-NeXT reads its
    #: text half from a separate GGUF, so the values are pinned here from that
    #: release's ``llama.*`` metadata (4096/11008, 32 blocks, 32 heads).
    text_dims: dict | None = None


CASES = (
    Case(
        label="qwen25vl",
        filename="qwen25vl-mmproj-F16.gguf",
        hf_repo="Qwen/Qwen2.5-VL-3B-Instruct",
        module="gguf2mlx.vision_qwen25",
        adapter_class="Qwen25VisionAdapter",
        tensor_count=519,
    ),
    Case(
        label="qwen3vl",
        filename="qwen3vl-mmproj-F16.gguf",
        hf_repo="Qwen/Qwen3-VL-2B-Instruct",
        module="gguf2mlx.vision_qwen3",
        adapter_class="Qwen3VisionAdapter",
        tensor_count=316,
    ),
    Case(
        label="llava_next",
        filename="mmproj-model-f16.gguf",
        hf_repo="llava-hf/llava-v1.6-vicuna-7b-hf",
        module="gguf2mlx.vision_llava",
        adapter_class="LlavaNextAdapter",
        tensor_count=378,
        text_dims={
            "hidden_size": 4096, "intermediate_size": 11008,
            "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 32, "vocab_size": 32000,
            "tie_word_embeddings": False,
        },
    ),
)


def _fixture_root() -> Path:
    return Path(os.environ.get("GGUF2MLX_VLM_MMPROJ_DIR", "~/.cache/gguf2mlx-vlm")).expanduser()


def _adapter_module(case: Case) -> ModuleType:
    return importlib.import_module(case.module)


def _hf_config(case: Case) -> dict:
    """Return the original HF ``config.json``, preferring the local HF cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover - optional dependency
        pytest.skip("huggingface-hub is required to read the HF config")

    for local_only in (True, False):
        try:
            path = hf_hub_download(
                case.hf_repo, "config.json", local_files_only=local_only
            )
        except Exception:  # noqa: BLE001, S112 - any failure means "try the next mode"
            continue
        return json.loads(Path(path).read_text())
    pytest.skip(f"config.json for {case.hf_repo} is neither cached nor downloadable")


@pytest.fixture(scope="module")
def cases() -> tuple[Case, ...]:
    if os.getenv("GGUF2MLX_RUN_E2E") != "1":
        pytest.skip("set GGUF2MLX_RUN_E2E=1 to run checks against real mmproj files")
    root = _fixture_root()
    available = tuple(c for c in CASES if (root / c.filename).is_file())
    if not available:
        pytest.skip(f"no mmproj fixtures in {root}")
    return available


@pytest.fixture(scope="module")
def loaded(cases: tuple[Case, ...]) -> dict[str, tuple]:
    """Build one adapter per case and read its real ``mmproj`` tensor table."""
    from gguf import GGUFReader

    root = _fixture_root()
    built: dict[str, tuple] = {}
    for case in cases:
        adapter = getattr(_adapter_module(case), case.adapter_class)(_hf_config(case))
        reader = GGUFReader(str(root / case.filename))
        if case.text_dims is not None:
            # An adapter whose plan spans both halves cannot be planned from the
            # mmproj alone; seed the language dimensions and let it plan the tower.
            adapter.text.update(case.text_dims)
            adapter.configure_layout([reader])
        built[case.label] = (case, adapter, reader, _vision_tensors(reader))
    return built


def _vision_tensors(reader) -> dict[str, object]:
    """The tensors a plan may consume: ``v.*``, ``mm.*`` and the stray roots.

    LLaVA-NeXT keeps its anyres separator outside the tower as
    ``model.image_newline``, which is neither prefix -- the reason a converter that
    filters by ``v.``/``mm.`` loses it.
    """
    return {
        tensor.name: tensor
        for tensor in reader.tensors
        if tensor.name.startswith(("v.", "mm.", "model."))
    }


def _plan_entries(adapter) -> list[tuple[str, tuple[str, ...], int]]:
    return list(adapter.vision_plan)


def test_real_file_has_the_pinned_tensor_count(loaded: dict[str, tuple]) -> None:
    for case, _adapter, _reader, tensors in loaded.values():
        assert len(tensors) == case.tensor_count, (
            f"{case.filename} has {len(tensors)} vision tensors, "
            f"the pinned count is {case.tensor_count}; the fixture file changed"
        )


def test_declared_sources_match_the_real_mmproj(loaded: dict[str, tuple]) -> None:
    for label, (_case, adapter, _reader, tensors) in loaded.items():
        real = set(tensors)
        missing = real - adapter.source_names
        undeclared = adapter.source_names - real
        assert not missing, (
            f"{label}: the plan does not consume {len(missing)} real tensors, "
            f"so they would be dropped: {sorted(missing)[:8]}"
        )
        assert not undeclared, (
            f"{label}: the plan declares {len(undeclared)} tensors the file does not "
            f"have, so conversion would fail on every real export: "
            f"{sorted(undeclared)[:8]}"
        )


def test_declared_source_shapes_match_the_real_mmproj(loaded: dict[str, tuple]) -> None:
    for label, (_case, adapter, _reader, tensors) in loaded.items():
        mismatched = []
        for name, expected in adapter.source_shapes.items():
            real = tuple(vlm._decode(tensors[name], "float16").shape)
            if real != tuple(expected):
                mismatched.append((name, real, tuple(expected)))
        assert not mismatched, (
            f"{label}: {len(mismatched)} source shapes disagree with the real file: "
            f"{mismatched[:5]}"
        )


def test_plan_assembles_the_declared_target_shapes(loaded: dict[str, tuple]) -> None:
    """Replay the converter's assembly on real shapes, not on the fixture's."""
    for label, (_case, adapter, _reader, tensors) in loaded.items():
        for target, sources, axis in _plan_entries(adapter):
            arrays = [
                np.zeros(tuple(vlm._decode(tensors[src], "float16").shape), np.float32)
                for src in sources
            ]
            if len(arrays) == 1:
                assembled = arrays[0]
            elif target.removeprefix("vision_tower.") in adapter.stacked_targets:
                assembled = np.stack(arrays, axis=axis)
            else:
                assembled = np.concatenate(arrays, axis=axis)
            assert assembled.shape == tuple(adapter.expected[target]), (
                f"{label}: assembling {target} from {sources} along axis {axis} gives "
                f"{assembled.shape}, declared {adapter.expected[target]}"
            )


def test_real_metadata_passes_validation(loaded: dict[str, tuple]) -> None:
    """A fail-closed check that rejects a genuine export is also a bug."""
    for _case, adapter, reader, _tensors in loaded.values():
        adapter.validate_vision_metadata(reader)


def test_every_declared_target_is_reachable_from_the_file(loaded: dict[str, tuple]) -> None:
    """No target may depend on a source that only the synthetic fixture provides."""
    for label, (_case, adapter, _reader, tensors) in loaded.items():
        for target, sources, _axis in _plan_entries(adapter):
            absent = [src for src in sources if src not in tensors]
            assert not absent, f"{label}: {target} needs absent sources {absent}"


def test_the_real_llava_next_export_is_the_legacy_clip_mlp_layout(
    loaded: dict[str, tuple],
) -> None:
    """Pin the two facts about LLaVA-NeXT that only a real file can settle.

    Every pre-mtmd ``mmproj`` names the CLIP MLP expansion ``ffn_down``; llama.cpp's
    LLaVA-era loader reads it that way, and the tensor shapes here (4096->1024)
    confirm it for this file. The anyres separator is also present, outside the
    tower, which is exactly what a name-filtered converter drops.
    """
    built = loaded.get("llava_next")
    if built is None:
        pytest.skip("the llava_next mmproj fixture is not installed")
    _case, adapter, _reader, _tensors = built

    assert adapter.layout_report["clip_ffn_convention"] == "llava-tools"
    assert adapter.layout_report["stored_vision_blocks"] == 23
    assert adapter.feature_layer == -1
    assert "image_newline" in adapter.expected
    assert adapter.source_shapes["model.image_newline"] == (adapter.text["hidden_size"],)
