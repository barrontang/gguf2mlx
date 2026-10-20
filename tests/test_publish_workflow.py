"""Contract tests for the reusable publish workflow (a thin shell over the CLI)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-mlx.yml"
)
WORKFLOW = WORKFLOW_PATH.read_text()
PARSED = yaml.safe_load(WORKFLOW)
JOB = PARSED["jobs"]["convert-publish"]
STEPS = JOB["steps"]


def _step(name: str) -> dict:
    for step in STEPS:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found")


def _run_bodies() -> list[str]:
    return [step["run"] for step in STEPS if "run" in step]


def test_workflow_delegates_to_the_publish_subcommand():
    publish_step = _step("Download, convert, and publish")
    body = publish_step["run"]
    assert "gguf2mlx publish" in body
    # The pipeline lives in Python, not in a heredoc inside the workflow.
    assert "python - <<" not in WORKFLOW
    assert "import " not in WORKFLOW
    for flag in ("--source", "--gguf", "--dest", "--source-revision"):
        assert flag in body


def test_untrusted_inputs_reach_the_shell_only_through_the_environment():
    for body in _run_bodies():
        assert "${{ inputs." not in body
        assert "${{ secrets." not in body
    # ``runs-on`` is a workflow expression, not a shell body, so it is exempt.
    publish_step = _step("Download, convert, and publish")
    env = publish_step["env"]
    assert env["SOURCE_REPO"] == "${{ inputs.source_repo }}"
    assert env["GGUF_FILE"] == "${{ inputs.gguf_file }}"
    assert env["DESTINATION_REPO"] == "${{ inputs.destination_repo }}"
    assert env["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"


def test_workflow_declares_the_expected_call_interface():
    inputs = PARSED[True]["workflow_call"]["inputs"]
    assert set(inputs) == {
        "source_repo",
        "gguf_file",
        "source_revision",
        "destination_repo",
        "private",
        "quantize",
        "runner",
        "converter_version",
    }
    assert inputs["quantize"]["default"] is False
    assert inputs["private"]["default"] is True
    assert PARSED[True]["workflow_call"]["secrets"]["HF_TOKEN"]["required"] is True


def test_float16_runs_on_linux_and_only_quantization_requires_apple_silicon():
    assert JOB["runs-on"] == "${{ inputs.quantize && inputs.runner || 'ubuntu-latest' }}"
    guard = _step("Require Apple Silicon for quantization")
    assert guard["if"] == "inputs.quantize"
    assert 'test "$(uname -m)" = arm64' in guard["run"]
    # The guard must be conditional; an unconditional check would block Linux.
    unconditional = [
        step for step in STEPS if 'test "$(uname -m)" = arm64' in step.get("run", "")
    ]
    assert all(step.get("if") for step in unconditional)


def test_converter_version_is_pinned():
    install = _step("Install converter and Hugging Face client")
    assert install["env"]["CONVERTER_VERSION"] == "${{ inputs.converter_version }}"
    assert re.search(r'pip install "gguf2mlx\[vlm\]==\$\{CONVERTER_VERSION\}"', install["run"])
    default = PARSED[True]["workflow_call"]["inputs"]["converter_version"]["default"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(default))


def test_publication_is_serialised_per_destination():
    assert PARSED["concurrency"]["group"] == "publish-mlx-${{ inputs.destination_repo }}"
    assert PARSED["concurrency"]["cancel-in-progress"] is False
    assert PARSED["permissions"] == {"contents": "read"}


def test_shell_fails_fast_and_does_not_word_split_caller_text():
    body = _step("Download, convert, and publish")["run"]
    assert body.startswith("set -euo pipefail")
    # Only the literal flag accumulation is intentionally unquoted.
    for token in re.findall(r"\$\w[\w]*", body):
        if token == "$extra":
            continue
        assert f'"{token}' in body, f"{token} must be quoted"
