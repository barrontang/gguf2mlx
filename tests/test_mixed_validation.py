import json

import pytest

from gguf2mlx.mixed_validation import validate_mixed_artifacts


def _write_mixed_fixture(tmp_path, expert_precision=4):
    report = {
        "architecture": "qwen3moe",
        "architecture_kind": "moe",
        "protected_router_tensors": [{"output_name": "mlp.gate.weight"}],
        "compressed_expert_tensors": [{"output_name": "switch_mlp.up_proj.weight"}],
        "tensor_decisions": [
            {"category": "router", "precision": "float16"},
            {"category": "attention", "precision": "float16"},
            {"category": "expert_mlp", "precision": expert_precision},
        ],
    }
    (tmp_path / "conversion_report.json").write_text(json.dumps(report))
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"format": "gguf2mlx-mixed-v1"})
    )


def test_validate_mixed_artifacts_accepts_policy(tmp_path):
    _write_mixed_fixture(tmp_path)
    assert validate_mixed_artifacts(tmp_path)["architecture"] == "qwen3moe"


def test_validate_mixed_artifacts_rejects_wrong_expert_precision(tmp_path):
    _write_mixed_fixture(tmp_path, expert_precision=8)
    with pytest.raises(ValueError, match="Invalid precision decisions"):
        validate_mixed_artifacts(tmp_path)
