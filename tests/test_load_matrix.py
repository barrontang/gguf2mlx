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
"""

from __future__ import annotations

import hashlib
import json
import os
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
    return [entry for entry in ALL_ENTRIES if not wanted or entry["qtype"] in wanted]


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
        key = (entry["arch"], entry["qtype"])
        assert key not in seen, f"duplicate matrix entry for {key}"
        seen.add(key)


def test_matrix_covers_declared_architectures():
    """Every matrix architecture must be a conversion-enabled one."""
    unknown = {entry["arch"] for entry in ALL_ENTRIES} - set(core.CONVERTIBLE_ARCHES)
    assert not unknown, f"matrix references architectures that cannot be converted: {unknown}"


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
    assert core.convert(str(source), str(output)) is True

    report = json.loads((output / fidelity.REPORT_NAME).read_text())
    weight = report["fidelity"]["weight"]
    assert weight["status"] == "pass", (
        f"weight fidelity failed: {weight['failures']} "
        f"(max relative L2 {weight['max_relative_l2']})"
    )
    assert weight["tensors_compared"] > 0

    result = fidelity.load_and_run(output)
    assert result["status"] == "pass", result
    assert result["logits_finite"] is True
