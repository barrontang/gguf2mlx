"""Exercise the reusable workflow's Python payload without MLX or HF access."""

import importlib.metadata
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-mlx.yml"
).read_text()
PAYLOAD = textwrap.dedent(WORKFLOW.split("python - <<'PY'\n", 1)[1].rsplit("          PY", 1)[0])


@pytest.fixture
def publishing(monkeypatch, tmp_path):
    environment = {
        "SOURCE_REPO": "publisher/model-GGUF",
        "DESTINATION_REPO": "mlx-community/model-MLX",
        "GGUF_FILE": "weights/model.gguf",
        "SOURCE_REVISION": "uploaded-commit",
        "HF_TOKEN": "test-placeholder",
        "PRIVATE": "true",
        "QUANTIZE": "false",
        "RUNNER_TEMP": str(tmp_path),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(
        downloads=[], commands=[], creates=[], uploads=[], events=[], fail=False, incomplete=False,
    )
    card = SimpleNamespace(
        data=SimpleNamespace(tags=["gguf", "text-generation"], license="apache-2.0"),
        text="Original model card and license information.",
    )
    card.save = lambda path: path.write_text(card.text)
    state.card = card

    def model_info(repo, revision):
        assert (repo, revision) == ("publisher/model-GGUF", "uploaded-commit")
        return SimpleNamespace(sha="immutable-source-commit")

    def download(repo, filename, revision):
        assert repo == "publisher/model-GGUF"
        state.downloads.append((filename, revision))
        path = tmp_path / "cache" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source content")
        return str(path)

    def convert(command, check):
        assert check is True
        state.commands.append(command)
        state.events.append("convert")
        if state.fail:
            raise subprocess.CalledProcessError(1, command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "config.json").write_text("{}")
        if not state.incomplete:
            (output / "model.safetensors").write_bytes(b"weights")

    def create_repo(repo, **kwargs):
        state.creates.append((repo, kwargs))
        state.events.append("create")

    def upload_folder(**kwargs):
        state.events.append("upload")
        output = Path(kwargs["folder_path"])
        state.uploads.append({
            "arguments": kwargs,
            "files": {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*")
                      if p.is_file()},
        })
        return SimpleNamespace(commit_url="https://huggingface.co/commit/example")

    api = SimpleNamespace(
        model_info=model_info, create_repo=create_repo, upload_folder=upload_folder,
        list_repo_files=lambda *args, **kwargs: ["README.md", "LICENSE", "legal/NOTICE.txt"],
    )
    hub = ModuleType("huggingface_hub")
    hub.HfApi = lambda token: api
    hub.ModelCard = SimpleNamespace(load=lambda path: card)
    hub.hf_hub_download = download
    utils = ModuleType("huggingface_hub.utils")

    def validate_repo_id(repo):
        if repo.count("/") != 1:
            raise ValueError("Invalid repository")

    utils.validate_repo_id = validate_repo_id
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    monkeypatch.setattr(subprocess, "run", convert)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.1.4")
    payload_path = tmp_path / "workflow_payload.py"
    payload_path.write_text(PAYLOAD)
    state.run = lambda: runpy.run_path(str(payload_path))
    return state


def test_publish_pins_downloads_and_preserves_license(publishing):
    import json

    publishing.run()
    assert publishing.events == ["convert", "create", "upload"]
    assert all(revision == "immutable-source-commit" for _, revision in publishing.downloads)
    assert publishing.creates == [
        ("mlx-community/model-MLX", {"repo_type": "model", "private": True, "exist_ok": True})
    ]
    files = publishing.uploads[0]["files"]
    assert set(files) == {
        "config.json", "model.safetensors", "README.md",
        "gguf2mlx-provenance.json", "LICENSE", "legal/NOTICE.txt",
    }
    provenance = json.loads(files["gguf2mlx-provenance.json"])
    assert provenance["source_revision"] == "immutable-source-commit"
    assert provenance["gguf_file"] == "weights/model.gguf"
    assert provenance["precision"] == "float16"
    assert b"Original model card" in files["README.md"]
    assert publishing.card.data.license == "apache-2.0"
    assert publishing.card.data.library_name == "mlx"
    assert publishing.card.data.base_model == "publisher/model-GGUF"
    assert publishing.card.data.tags == ["gguf2mlx", "mlx", "text-generation"]
    assert "--quantize" not in publishing.commands[0]
    assert not Path(publishing.uploads[0]["arguments"]["folder_path"]).exists()


def test_quantization_is_explicit(publishing, monkeypatch):
    monkeypatch.setenv("QUANTIZE", "true")
    monkeypatch.setenv("PRIVATE", "false")
    publishing.run()
    assert publishing.commands[0][-8:] == [
        "--quantize", "--direct-quant", "--q-bits", "4",
        "--q-group-size", "64", "--q-mode", "affine",
    ]
    assert publishing.creates[0][1]["private"] is False
    assert b"affine-4bit-group64" in publishing.uploads[0]["files"]["gguf2mlx-provenance.json"]


@pytest.mark.parametrize("incomplete", [False, True])
def test_failed_conversion_never_creates_or_uploads(publishing, incomplete):
    publishing.fail = not incomplete
    publishing.incomplete = incomplete
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        publishing.run()
    assert not publishing.creates
    assert not publishing.uploads


@pytest.mark.parametrize("filename", ["../model.gguf", "/model.gguf", "model.bin"])
def test_invalid_paths_fail_before_download(publishing, monkeypatch, filename):
    monkeypatch.setenv("GGUF_FILE", filename)
    with pytest.raises(ValueError):
        publishing.run()
    assert not publishing.downloads
    assert not publishing.uploads


def test_source_cannot_be_destination(publishing, monkeypatch):
    monkeypatch.setenv("DESTINATION_REPO", "PUBLISHER/model-GGUF")
    with pytest.raises(ValueError, match="different repositories"):
        publishing.run()
    assert not publishing.downloads


def test_untrusted_inputs_are_not_interpolated_in_scripts():
    for block in WORKFLOW.split("run: ")[1:]:
        script = block.split("\n      - ", 1)[0]
        assert "${{ inputs." not in script
        assert "${{ secrets." not in script
    assert "cancel-in-progress: false" in WORKFLOW
    assert 'test "$(uname -m)" = arm64' in WORKFLOW
