"""End-to-end load matrix: real GGUF in, loadable MLX directory out.

This is the answer to "declared supported but never tested". Every entry is a
pinned public GGUF whose SHA-256 is checked before conversion, so a silent
upstream re-upload cannot turn a green run into a false one.

The suite is opt-in because it downloads 0.4-4 GB per case and needs Apple
Silicon for the ``mlx_lm`` load:

    GGUF2MLX_LOAD_MATRIX=1 pytest tests/test_load_matrix.py -q

Restrict the quantization types with:

    GGUF2MLX_LOAD_MATRIX=1 GGUF2MLX_LOAD_MATRIX_QTYPES=q4_k_m \
      pytest tests/test_load_matrix.py -q

or run a single pinned case (this is how CI fans the matrix out, one runner per
entry):

    GGUF2MLX_LOAD_MATRIX=1 GGUF2MLX_LOAD_MATRIX_ENTRY=llama-q4_k_m \
      pytest tests/test_load_matrix.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from gguf2mlx import fidelity
from gguf2mlx import gguf2mlx as core

MATRIX_PATH = Path(__file__).with_name("load_matrix.json")
ALL_ENTRIES: list[dict] = json.loads(MATRIX_PATH.read_text())


def _selected_qtypes() -> set[str]:
    raw = os.environ.get("GGUF2MLX_LOAD_MATRIX_QTYPES", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _selected() -> list[dict]:
    wanted = _selected_qtypes()
    only = os.environ.get("GGUF2MLX_LOAD_MATRIX_ENTRY", "").strip()
    entries = [entry for entry in ALL_ENTRIES if not wanted or entry["qtype"] in wanted]
    if only:
        entries = [
            entry for entry in entries
            if f"{entry['arch']}-{entry['qtype']}" == only
        ]
        if not entries:
            known = [f"{entry['arch']}-{entry['qtype']}" for entry in ALL_ENTRIES]
            raise RuntimeError(
                f"GGUF2MLX_LOAD_MATRIX_ENTRY={only} matches no pinned entry; "
                f"known ids: {known}"
            )
    return entries


ENTRIES = _selected()
IDS = [f"{entry['arch']}-{entry['qtype']}" for entry in ENTRIES]


def _enabled() -> bool:
    return os.environ.get("GGUF2MLX_LOAD_MATRIX") == "1"


def test_matrix_entries_are_complete_and_unique():
    """Portable: the matrix itself must stay well-formed as it is edited."""
    assert ALL_ENTRIES, "load matrix is empty"
    seen: set[tuple[str, str]] = set()
    for entry in ALL_ENTRIES:
        missing = {"arch", "qtype", "repo_id", "filename", "sha256", "size"} - entry.keys()
        assert not missing, f"{entry.get('filename')} is missing {sorted(missing)}"
        assert len(entry["sha256"]) == 64, f"{entry['filename']} has a truncated sha256"
        assert entry["filename"].endswith(".gguf")
        assert entry.get("quant_source", "requant") in {"requant", "preserve"}
        key = (entry["arch"], entry["qtype"])
        assert key not in seen, f"duplicate matrix entry for {key}"
        seen.add(key)


def test_matrix_covers_declared_architectures():
    """Every matrix architecture must be a conversion-enabled one."""
    unknown = {entry["arch"] for entry in ALL_ENTRIES} - set(core.CONVERTIBLE_ARCHES)
    assert not unknown, f"matrix references architectures that cannot be converted: {unknown}"


# Alternative GGUF spellings of one architecture. They share a conversion path,
# so one pinned case proves all of the aliases and they must not each demand a
# separate entry.
_ALIASES = {
    "qwen3_5": "qwen35",
    "qwen3.5": "qwen35",
    "qwen3_6": "qwen36",
    "qwen3.6": "qwen36",
    "qwen3_8": "qwen38",
    "qwen3.8": "qwen38",
    "qwen2vl": "qwen2_vl",
    "qwen3vl": "qwen3_vl",
}

# Declared, reasoned coverage gaps. `CONVERTIBLE_ARCHES` is a claim about what
# the converters accept; this matrix is the evidence for what actually loads.
# They are 30 vs 5, so the difference is written down here rather than implied:
# adding an architecture to `CONVERTIBLE_ARCHES` without pinning a case (or
# listing it below) fails `test_every_convertible_architecture_is_covered_or_declared`.
DECLARED_COVERAGE_GAPS: dict[str, str] = {
    "deepseek2": "MoE with MLA; the smallest public GGUF far exceeds the per-runner budget",
    "deepseek3": "MoE with MLA; same download budget as deepseek2",
    "gemma": "gemma2 covers the shared adapter; no small pinned gemma GGUF yet",
    "gemma3": "text twin of a multimodal release; no small pinned GGUF yet",
    "gemma4": "adapter is fixture-tested only; no pinned public GGUF yet",
    "glm-dsa": "MoE with a full-indexer path; needs a dedicated router-protect case",
    "glm4moe": "MoE; same download budget as the other expert models",
    "mistral": "shares llama's tensor map, which is pinned; no pinned mistral GGUF yet",
    "phi4": "phi3 pins the same sliding-window adapter; phi4 is fixture-tested",
    "qwen2_vl": "needs the matching HF repo beside the GGUF; vision.yml owns that path",
    "qwen2moe": "MoE; weekly-moe.yml exercises two families, not this matrix",
    "qwen3_vl": "needs the matching HF repo beside the GGUF; vision.yml owns that path",
    "qwen36": "hybrid adapter, fixture-tested; no small pinned GGUF yet",
    "qwen38": "hybrid adapter, fixture-tested; no small pinned GGUF yet",
    "qwen3moe": "MoE; weekly-moe.yml exercises two families, not this matrix",
    "stablelm": "adapter ships; no pinned public GGUF yet",
}


def _canonical(arch: str) -> str:
    return _ALIASES.get(arch, arch)


def test_every_convertible_architecture_is_covered_or_declared():
    """No architecture may be 'supported' without evidence or a stated gap."""
    covered = {_canonical(entry["arch"]) for entry in ALL_ENTRIES}
    declared = {_canonical(arch) for arch in DECLARED_COVERAGE_GAPS}
    declared_arches = {_canonical(arch) for arch in core.CONVERTIBLE_ARCHES}

    undeclared = declared_arches - covered - declared
    assert not undeclared, (
        "these architectures are convertible but have no pinned load case and no "
        f"reason in DECLARED_COVERAGE_GAPS: {sorted(undeclared)}"
    )
    stale = declared - (declared_arches - covered)
    assert not stale, (
        "DECLARED_COVERAGE_GAPS lists architectures that are now covered or no "
        f"longer convertible: {sorted(stale)}"
    )



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_converted_model_loads_and_runs(entry, tmp_path):
    """Convert a real GGUF, then prove the result actually loads and runs."""
    if not _enabled():
        pytest.skip("set GGUF2MLX_LOAD_MATRIX=1 to run the real-model load matrix")

    huggingface_hub = pytest.importorskip("huggingface_hub")
    source = Path(
        huggingface_hub.hf_hub_download(
            repo_id=entry["repo_id"], filename=entry["filename"]
        )
    )
    assert _sha256(source) == entry["sha256"], (
        f"{entry['filename']} does not match its pinned SHA-256"
    )

    output = tmp_path / "converted"
    preserve = entry.get("quant_source") == "preserve"
    try:
        options = {"quant_source": "preserve"} if preserve else {}
        assert core.convert(str(source), str(output), **options) is True

        report = json.loads((output / fidelity.REPORT_NAME).read_text())
        weight = report["fidelity"]["weight"]
        assert weight["status"] == "pass", (
            f"weight fidelity failed: {weight['failures']} "
            f"(max relative L2 {weight['max_relative_l2']})"
        )
        assert weight["tensors_compared"] > 0
        if preserve:
            preserved = report["preserve"]
            assert preserved["gate"] == "pass", preserved["gate"]
            assert preserved["tiers"]["exact"]["max_relative_l2"] <= 1e-3
            assert preserved["size"]["ratio"] <= 1.1, preserved["size"]
            assert preserved["preserved_by_qtype"], preserved

        result = fidelity.load_and_run(output)
        assert result["status"] == "pass", result
        assert result["logits_finite"] is True
    finally:
        # Keep the (small) report for the artifact and the failure message, drop
        # the multi-GB weights: one runner has ~14 GB total and the converted
        # FP16 tree is 3-5x the source GGUF.
        report_path = output / fidelity.REPORT_NAME
        if report_path.is_file():
            shutil.copy2(report_path, tmp_path / "conversion_report.json")
        shutil.rmtree(output, ignore_errors=True)
