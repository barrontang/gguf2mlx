"""CLI contract for ``--force``, ``--json`` and the logging flags.

These are cheap, offline checks: the overwrite guard runs before any GGUF is
opened, so a missing input exercises the guard without building a fixture.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from gguf2mlx import fidelity
from gguf2mlx import gguf2mlx as core


def _non_empty(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "stale.safetensors").write_bytes(b"old")
    return target


def test_convert_refuses_a_non_empty_output_directory(tmp_path):
    target = _non_empty(tmp_path)

    assert core.convert(str(tmp_path / "missing.gguf"), str(target)) is False
    assert (target / "stale.safetensors").is_file(), "output was touched without --force"


def test_convert_force_clears_the_directory_before_starting(tmp_path):
    target = _non_empty(tmp_path)
    (target / "nested").mkdir()
    (target / "nested" / "deep.bin").write_bytes(b"old")

    # The conversion itself fails (no such GGUF); only the guard behaviour matters.
    assert core.convert(str(tmp_path / "missing.gguf"), str(target), force=True) is False
    assert list(target.iterdir()) == [], "--force did not clear the directory"


def test_convert_force_still_rejects_a_file_path(tmp_path):
    target = tmp_path / "out"
    target.write_bytes(b"not a directory")

    assert core.convert(str(tmp_path / "missing.gguf"), str(target), force=True) is False
    assert target.is_file(), "--force must not delete a file sitting at --output"


def test_json_summary_reads_the_report_back(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    report = {
        "schema_version": 3,
        "architecture": "llama",
        "dtype": "float16",
        "source": "/models/x.gguf",
        "fidelity": {
            "weight": {
                "status": "pass",
                "tensors_compared": 9,
                "max_relative_l2": 1.5e-7,
                "failures": [],
            }
        },
        "preserve": {"gate": "pass"},
    }
    (output / fidelity.REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")

    summary = core._convert_json_summary(output, True)

    assert summary["ok"] is True
    assert summary["report"] == str(output / fidelity.REPORT_NAME)
    assert summary["gates"] == {"weight": "pass", "preserve": "pass"}
    assert summary["weight"]["tensors_compared"] == 9
    assert summary["architecture"] == "llama"


def test_json_summary_tolerates_a_missing_report(tmp_path):
    summary = core._convert_json_summary(tmp_path / "absent", False)

    assert summary == {"ok": False, "output": str(tmp_path / "absent"), "report": None}


def test_json_mode_keeps_stdout_parseable(capsys, tmp_path):
    """Failures are printed with `❌` — they must not land in the JSON stream."""
    with pytest.raises(SystemExit) as exit_info:
        core.main(
            [
                "convert",
                "--input", str(tmp_path / "missing.gguf"),
                "--output", str(tmp_path / "out"),
                "--json",
            ]
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # the whole point: this must not raise
    assert payload["ok"] is False
    assert exit_info.value.code == 1
    assert "not found" in captured.err


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, logging.INFO),
        ({"verbose": True}, logging.DEBUG),
        ({"quiet": True}, logging.WARNING),
    ],
)
def test_logging_flags_set_the_level(flags, expected):
    core._configure_logging(**flags)

    assert core._LOG.level == expected
    assert core._LOG.handlers, "the logger needs a handler or records vanish"
    # stdout carries --json, so every record must go to stderr.
    assert core._LOG.handlers[0].stream is sys.stderr
