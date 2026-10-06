"""CLI routing and atomic publication tests, independent of VLM tensor adapters."""

import json
import sys
from types import SimpleNamespace

import pytest

from gguf2mlx import gguf2mlx as core


@pytest.fixture
def routing(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.touch()
    state = {"detected": False, "calls": []}

    def convert_vlm(path, output, dtype, **kwargs):
        state["calls"].append((path, output, dtype, kwargs))
        output.mkdir()
        (output / "config.json").write_text(json.dumps({"model_type": "llava"}))

    module = SimpleNamespace(
        is_vlm=lambda reader: state["detected"],
        convert_vlm=convert_vlm,
    )
    monkeypatch.setitem(sys.modules, "gguf2mlx.vlm", module)
    monkeypatch.setattr(core, "GGUFReader", lambda path: object())
    return source, tmp_path / "out", state, module


def test_auto_vlm_detection_routes_without_llm_conversion(routing, monkeypatch):
    source, output, state, _ = routing
    state["detected"] = True
    monkeypatch.setattr(core, "_convert", lambda *args: pytest.fail("LLM route used"))
    assert core.convert(str(source), str(output))
    assert json.loads((output / "config.json").read_text())["model_type"] == "llava"
    assert len(state["calls"]) == 1
    assert not list(output.parent.glob(".out.*"))


def test_hybrid_and_companion_options_imply_vlm(routing):
    source, output, state, _ = routing
    assert core.convert(
        str(source), str(output), "float32",
        mmproj="mmproj.gguf", hf_fallback_vision="org/llava",
        hf_revision="commit-id", offline=True,
    )
    assert state["calls"][0][2:] == (
        "float32",
        {
            "mmproj": "mmproj.gguf",
            "hf_model": None,
            "hf_fallback_vision": "org/llava",
            "hf_revision": "commit-id",
            "offline": True,
        },
    )


def test_vlm_failure_removes_staging_and_preserves_empty_destination(routing):
    source, output, _, module = routing
    output.mkdir()

    def fail(path, staging, dtype, **kwargs):
        staging.mkdir()
        (staging / "partial.safetensors").touch()
        raise ValueError("missing projector")

    module.convert_vlm = fail
    assert not core.convert(str(source), str(output), model_type="vlm")
    assert output.is_dir()
    assert not list(output.iterdir())
    assert not list(output.parent.glob(".out.*"))


@pytest.mark.parametrize("options", [
    {"quantize": True},
    {"direct_quant": True},
    {"mixed_precision": True},
])
def test_vlm_rejects_llm_quantization(routing, options):
    source, output, state, _ = routing
    state["detected"] = True
    assert not core.convert(str(source), str(output), **options)
    assert not output.exists()
    assert not state["calls"]


@pytest.mark.parametrize("options", [
    {"hf_model": "org/llava"},
    {"mmproj": "mmproj.gguf"},
    {"hf_fallback_vision": "org/llava"},
])
def test_explicit_llm_rejects_vlm_source_options(routing, options):
    source, output, state, _ = routing
    assert not core.convert(str(source), str(output), model_type="llm", **options)
    assert not state["calls"]
    assert not output.exists()


def test_explicit_llm_cannot_drop_multimodal_weights(routing):
    source, output, state, _ = routing
    state["detected"] = True
    assert not core.convert(str(source), str(output), model_type="llm")
    assert not state["calls"]
    assert not output.exists()


def test_existing_llm_conversion_still_routes_to_llm(routing, monkeypatch):
    source, output, state, _ = routing

    def convert_llm(path, stage, dtype):
        assert path == str(source)
        assert dtype == "float16"
        stage = core.Path(stage)
        stage.mkdir()
        (stage / "llm.txt").write_text("unchanged route")
        return True

    monkeypatch.setattr(core, "_convert", convert_llm)
    assert core.convert(str(source), str(output))
    assert (output / "llm.txt").read_text() == "unchanged route"
    assert not state["calls"]


def test_existing_destination_never_overwritten(routing):
    source, output, state, _ = routing
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing model")
    assert not core.convert(str(source), str(output), model_type="vlm")
    assert marker.read_text() == "existing model"
    assert not state["calls"]


@pytest.mark.parametrize("command", [[], ["convert"]])
def test_both_cli_forms_forward_vlm_flags(monkeypatch, command):
    calls = []
    monkeypatch.setattr(core, "convert", lambda *args, **kwargs: calls.append(kwargs) or True)
    core.main(command + [
        "--input", "language.gguf", "--output", "out", "--type", "vlm",
        "--mmproj", "mmproj.gguf", "--hf-model", "org/llava",
        "--hf-fallback-vision", "org/llava", "--hf-revision", "pinned", "--offline",
    ])
    assert calls[0]["model_type"] == "vlm"
    assert calls[0]["mmproj"] == "mmproj.gguf"
    assert calls[0]["hf_model"] == calls[0]["hf_fallback_vision"] == "org/llava"
    assert calls[0]["hf_revision"] == "pinned"
    assert calls[0]["offline"] is True
