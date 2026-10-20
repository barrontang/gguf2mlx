"""Tests for the three-tier quality gate in :mod:`gguf2mlx.fidelity`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter
from safetensors.numpy import load_file, save_file

from gguf2mlx import fidelity
from gguf2mlx import gguf2mlx as core


def _write_gguf(path, arch, arrays):
    writer = GGUFWriter(path, arch)
    for name, array in arrays.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GGUFReader(path)


@pytest.fixture()
def converted(tmp_path):
    rng = np.random.default_rng(7)
    arrays = {
        "token_embd.weight": rng.standard_normal((8, 16)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.standard_normal((4, 4)).astype(np.float32),
        "output_norm.weight": rng.standard_normal((16,)).astype(np.float32),
    }
    reader = _write_gguf(tmp_path / "source.gguf", "llama", arrays)
    output = tmp_path / "out"
    output.mkdir()
    report = core.extract_and_convert_weights(
        reader, "llama", output, "float16", max_shard_bytes=1 << 20
    )
    return tmp_path / "source.gguf", output, report


def test_compare_reports_zero_for_identical_values():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = fidelity.compare(array, array)
    assert result["relative_l2"] == 0.0
    assert result["max_abs_error"] == 0.0
    assert result["bit_exact"] is True
    assert result["elements"] == 12


def test_compare_relative_l2_is_scale_free():
    reference = np.array([3.0, 4.0], dtype=np.float32)
    actual = np.array([3.0, 4.0], dtype=np.float32) + np.float32(0.5)
    result = fidelity.compare(reference, actual)
    # ||diff|| = sqrt(2) * 0.5, ||reference|| = 5
    assert result["relative_l2"] == pytest.approx(np.sqrt(2) * 0.5 / 5.0, rel=1e-6)
    assert result["max_abs_error"] == pytest.approx(0.5)
    assert result["bit_exact"] is False


def test_compare_handles_empty_and_mismatched_sizes():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert fidelity.compare(empty, empty)["elements"] == 0
    with pytest.raises(ValueError, match="cannot compare"):
        fidelity.compare(np.zeros(2), np.zeros(3))


def test_compare_treats_non_finite_output_as_not_finite():
    reference = np.ones(4, dtype=np.float32)
    actual = np.array([1.0, np.nan, 1.0, 1.0], dtype=np.float32)
    assert fidelity.compare(reference, actual)["finite"] is False


def test_weight_gate_passes_within_dtype_tolerance():
    records = [
        {"name": "a", "relative_l2": 1e-6, "bit_exact": True, "finite": True},
        {"name": "b", "relative_l2": 2e-4, "bit_exact": False, "finite": True},
    ]
    gate = fidelity.evaluate_weight_gate(records, "float16")
    assert gate["status"] == "pass"
    assert gate["tolerance"] == pytest.approx(6e-4)
    assert gate["max_relative_l2"] == pytest.approx(2e-4)
    assert gate["worst_tensor"] == "b"
    assert gate["failure_count"] == 0


def test_weight_gate_fails_beyond_tolerance_and_on_non_finite():
    records = [
        {"name": "moved", "relative_l2": 1e-2, "bit_exact": False, "finite": True},
        {"name": "nan", "relative_l2": 0.0, "bit_exact": False, "finite": False},
        {"name": "inf", "relative_l2": float("inf"), "bit_exact": False, "finite": True},
    ]
    gate = fidelity.evaluate_weight_gate(records, "float16")
    assert gate["status"] == "fail"
    assert gate["failure_count"] == 3
    assert "moved" in gate["failures"]


def test_weight_gate_fails_when_a_declared_identity_tensor_changed_shape():
    gate = fidelity.evaluate_weight_gate(
        [{"name": "ok", "relative_l2": 0.0, "bit_exact": True, "finite": True}],
        "float16",
        mismatches=["misplaced"],
    )
    assert gate["status"] == "fail"
    assert gate["failures"] == ["misplaced"]


def test_weight_gate_is_not_run_without_comparisons():
    assert fidelity.evaluate_weight_gate([], "float16")["status"] == "not_run"


def test_bf16_payloads_are_widened_before_comparison():
    reference = np.array([1.5, -2.25], dtype=np.float32)
    bf16 = (reference.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_allclose(
        fidelity.stored_tensor_to_float32(bf16, "bfloat16"), reference
    )
    # Float16 payloads must not be reinterpreted as bfloat16.
    np.testing.assert_allclose(
        fidelity.stored_tensor_to_float32(np.array([1.5], dtype=np.float16), "float16"),
        np.array([1.5], dtype=np.float32),
    )


def test_conversion_writes_a_report_with_a_passing_weight_gate(converted):
    _, output, report = converted
    on_disk = json.loads((output / fidelity.REPORT_NAME).read_text())
    assert on_disk == report
    assert on_disk["schema_version"] == fidelity.SCHEMA_VERSION
    assert on_disk["gates"]["weight"] == "pass"
    assert on_disk["fidelity"]["weight"]["tensors_compared"] == 3
    assert on_disk["fidelity"]["numerical"]["status"] == "not_run"
    assert on_disk["fidelity"]["perplexity"]["gated"] is False
    assert set(on_disk["tensor_layout"]) >= {"model.embed_tokens.weight"}


def test_audit_skips_transformed_tensors_and_flags_shape_changes(converted):
    _, output, _report = converted
    layout = {
        "model.embed_tokens.weight": {"source": "token_embd.weight", "layout": "identity"},
        "rotated": {"source": "blk.0.attn_q.weight", "layout": "transformed"},
    }
    reader = GGUFReader(str(output.parent / "source.gguf"))
    audit = fidelity.audit_weights(output, reader, layout, "float16")
    assert audit["compared"] == 1
    assert audit["skipped"]["transformed"] == 1

    # Claiming "identity" for a tensor whose stored shape differs is a defect.
    broken = {"model.embed_tokens.weight": {"source": "blk.0.ffn_down.weight",
                                            "layout": "identity"}}
    audit = fidelity.audit_weights(output, reader, broken, "float16")
    assert audit["mismatches"] == ["model.embed_tokens.weight"]


def test_verify_weights_rejects_a_corrupted_shard(converted):
    source, output, _ = converted
    assert fidelity.verify_weights(output, source)["status"] == "pass"

    index = json.loads((output / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["model.embed_tokens.weight"]
    tensors = load_file(output / shard)
    tensors["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"] + np.float16(0.25)
    save_file(tensors, output / shard)

    result = fidelity.verify_weights(output, source)
    assert result["status"] == "fail"
    assert result["gate"]["failures"] == ["model.embed_tokens.weight"]


def test_verify_weights_requires_a_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="conversion_report.json"):
        fidelity.verify_weights(tmp_path, tmp_path / "missing.gguf")


def test_report_summary_is_readable(converted):
    _, _, report = converted
    assert "pass" in fidelity.summarize(report)
    assert "not measured" in fidelity.summarize({"fidelity": {"weight": {"status": "not_run"}}})
