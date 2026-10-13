"""Fail-closed checks for the real-checkpoint acceptance runner."""

from types import SimpleNamespace
from urllib.error import URLError

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from benchmarks import validate_vlm as benchmark
from gguf2mlx import gguf2mlx as core
from tests.test_vlm_qwen2 import write_fixture


@pytest.mark.parametrize("text,accepted,expected", [
    (" Red.\n", ["red"], True),
    ("space suit", ["spacesuit", "space suit"], True),
    ("The person is wearing an orange space suit.", ["space suit"], False),
    ("The square might be red or blue", ["red"], False),
    ("", ["red"], False),
])
def test_answer_scoring_does_not_accept_substrings(text, accepted, expected):
    assert benchmark.correct_answer(text, accepted) is expected


@pytest.mark.parametrize("text,semantic", [
    ("space suit", True),
    ("The person is wearing an orange space suit.", True),
    ("The person is wearing an orange spacesuit.", True),
    ("an orange astronaut suit", True),
    ("The person is wearing a blue space suit.", False),
    ("a white spacesuit", False),
    ("The person is not wearing an orange space suit.", False),
    ("orange or blue space suit", False),
    ("The person is wearing an orange space suit and a blue helmet.", False),
    ("", False),
])
def test_photo_semantics_are_not_exact_format_or_keyword_presence(text, semantic):
    case = {"accepted": ["space suit"],
            "semantic": {"garment": "space suit", "color": "orange"}}
    assert benchmark.semantic_answer(text, case) is semantic
    if text.startswith("The person is wearing an orange"):
        assert not benchmark.correct_answer(text, case["accepted"])


@pytest.mark.parametrize("correct,passed", [(12, True), (11, True), (10, False), (0, False)])
def test_gate_keeps_predeclared_reference_tolerance(correct, passed):
    results = {label: {"correct": 12, "count": 12}
               for label in ("hf", "converted", "direct_quant", "llama_cpp")}
    results["direct_quant"]["correct"] = correct
    assert benchmark.quality_gate(results)["passed"] is passed


def test_gate_requires_absolute_accuracy_too():
    results = {label: {"correct": 8, "count": 12}
               for label in ("hf", "converted", "direct_quant", "llama_cpp")}
    assert not benchmark.quality_gate(results)["passed"]


def test_strict_regression_gate_rejects_original_fourbit_failure():
    results = {label: {"correct": 12, "count": 12}
               for label in ("hf", "converted", "direct_quant", "llama_cpp")}
    assert benchmark.quality_gate(results, require_all_correct=True)["passed"]
    results["direct_quant"]["correct"] = 11
    assert benchmark.quality_gate(results)["passed"]
    assert not benchmark.quality_gate(results, require_all_correct=True)["passed"]
    results["direct_quant"].update(correct=12, semantic_correct=11)
    assert not benchmark.quality_gate(results, require_all_correct=True)["passed"]


def test_metrics_reject_nonfinite_or_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        benchmark.metrics(np.zeros((2, 1)), np.zeros((2,)))
    with pytest.raises(ValueError, match="Nonfinite"):
        benchmark.metrics(np.array([np.nan]), np.array([1]))


@pytest.mark.parametrize("damaged", [None, "vision_tower.merger.mlp.0.weight",
                                    "language_model.model.layers.0.self_attn.q_proj.weight"])
@pytest.mark.parametrize("preserve", [False, True])
def test_independent_weight_audit(tmp_path, damaged, preserve):
    fixture = write_fixture(tmp_path, quantized_source=not preserve, legacy=True,
                            k_quantized_source=preserve)
    direct = tmp_path / "direct-quant"
    for output, quantized in ((fixture.output, False), (direct, True)):
        assert core.convert(
            str(fixture.main), str(output), "float16", mmproj=str(fixture.mmproj),
            hf_model=str(fixture.root), offline=True, quantize=quantized, direct_quant=quantized,
            preserve_source_quant=preserve and quantized,
        )
    original = {key.replace("vision_tower.", "visual."): value
                for key, value in fixture.arrays.items()}
    save_file(original, str(fixture.root / "model.safetensors"))
    if damaged:
        path = fixture.output / "model.safetensors"
        weights = load_file(str(path))
        weights[damaged] += 1
        save_file(weights, str(path))
        with pytest.raises(ValueError, match="mismatch"):
            benchmark.audit_weights(fixture.main, fixture.root, fixture.output, direct)
    else:
        result = benchmark.audit_weights(fixture.main, fixture.root, fixture.output, direct)
        assert result["passed"]
        assert result["tensor_count"] == len(fixture.arrays)


def test_source_preserving_audit_detects_corrupted_quantized_codes(tmp_path):
    fixture = write_fixture(tmp_path, k_quantized_source=True)
    direct = tmp_path / "preserved"
    for output, quantized in ((fixture.output, False), (direct, True)):
        assert core.convert(
            str(fixture.main), str(output), "float16", mmproj=str(fixture.mmproj),
            hf_model=str(fixture.root), offline=True, quantize=quantized, direct_quant=quantized,
            preserve_source_quant=quantized,
        )
    save_file({k.replace("vision_tower.", "visual."): v for k, v in fixture.arrays.items()},
              str(fixture.root / "model.safetensors"))
    path = direct / "model.safetensors"
    weights = load_file(str(path))
    weights["language_model.model.layers.0.self_attn.q_proj.weight"] ^= np.uint32(0xFFFFFFFF)
    save_file(weights, str(path))
    with pytest.raises(ValueError, match="coefficient-rounding bound"):
        benchmark.audit_weights(fixture.main, fixture.root, fixture.output, direct)

def test_native_server_does_not_swallow_inference_errors(tmp_path, monkeypatch):
    from pathlib import Path

    calls = []
    process = SimpleNamespace(
        poll=lambda: None, terminate=lambda: calls.append("terminate"),
        wait=lambda **kwargs: calls.append("wait"),
    )
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(benchmark, "_request", lambda url: {"status": "ok"})
    with (
        pytest.raises(URLError, match="inference failed"),
        benchmark.native_server(Path("server"), Path("model"), Path("mmproj"), tmp_path),
    ):
        raise URLError("inference failed")
    assert calls == ["terminate", "wait"]


def test_fixed_cases_are_paired_and_hashed(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    photo = tmp_path / "photo.png"
    Image.new("RGB", (224, 224), "white").save(photo)
    cases = benchmark.prepare_cases(tmp_path / "images", photo)
    assert len(cases) == 12
    colors = [case for case in cases if case["pair"] == "color"]
    assert len({case["question"] for case in colors}) == 1
    assert len({case["image_sha256"] for case in colors}) == 4
    assert all(case["accepted"] for case in cases)
