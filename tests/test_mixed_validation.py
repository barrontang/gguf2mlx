import json

import pytest

from gguf2mlx.mixed_validation import validate_mixed_artifacts


def _write_mixed_fixture(tmp_path, expert_precision=4):
    report = {
        "architecture": "qwen3moe",
        "architecture_kind": "moe",
        "mixed_precision": True,
        "tensor_decisions": [
            {
                "output_name": "mlp.gate.weight",
                "category": "router",
                "precision": "float16",
                "protected": True,
            },
            {
                "output_name": "self_attn.q_proj.weight",
                "category": "attention",
                "precision": "float16",
                "protected": True,
            },
            {
                "output_name": "switch_mlp.up_proj.weight",
                "category": "expert_mlp",
                "precision": expert_precision,
                "protected": False,
            },
        ],
    }
    report["protected_router_tensors"] = [
        item for item in report["tensor_decisions"] if item["category"] == "router"
    ]
    report["compressed_expert_tensors"] = [
        item for item in report["tensor_decisions"] if item["category"] == "expert_mlp"
    ]
    (tmp_path / "conversion_report.json").write_text(json.dumps(report))
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "gguf2mlx-mixed-v1",
                "tensor_decisions": {
                    item["output_name"]: item["precision"]
                    for item in report["tensor_decisions"]
                },
            }
        )
    )


def test_validate_mixed_artifacts_accepts_policy(tmp_path):
    _write_mixed_fixture(tmp_path)
    assert validate_mixed_artifacts(tmp_path)["architecture"] == "qwen3moe"


def test_validate_mixed_artifacts_rejects_wrong_expert_precision(tmp_path):
    _write_mixed_fixture(tmp_path, expert_precision=8)
    with pytest.raises(ValueError, match="Invalid precision decisions"):
        validate_mixed_artifacts(tmp_path)


def test_validate_mixed_artifacts_rejects_jang_config_drift(tmp_path):
    _write_mixed_fixture(tmp_path)
    jang_path = tmp_path / "jang_config.json"
    jang = json.loads(jang_path.read_text())
    jang["tensor_decisions"]["mlp.gate.weight"] = 4
    jang_path.write_text(json.dumps(jang))

    with pytest.raises(ValueError, match="JANG tensor_decisions"):
        validate_mixed_artifacts(tmp_path)


def test_validate_mixed_artifacts_rejects_inconsistent_protected_tensor_list(tmp_path):
    _write_mixed_fixture(tmp_path)
    report_path = tmp_path / "conversion_report.json"
    report = json.loads(report_path.read_text())
    report["protected_router_tensors"][0]["precision"] = 8
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="Protected router list"):
        validate_mixed_artifacts(tmp_path)


def test_validate_mixed_artifacts_rejects_non_mixed_report(tmp_path):
    _write_mixed_fixture(tmp_path)
    report_path = tmp_path / "conversion_report.json"
    report = json.loads(report_path.read_text())
    report["mixed_precision"] = False
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="not marked as mixed precision"):
        validate_mixed_artifacts(tmp_path)
