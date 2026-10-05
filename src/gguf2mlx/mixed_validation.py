"""Validate gguf2mlx mixed-precision artifacts before runtime loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_mixed_artifacts(model_dir: Path) -> dict[str, Any]:
    report_path = model_dir / "conversion_report.json"
    jang_path = model_dir / "jang_config.json"
    if not report_path.exists() or not jang_path.exists():
        raise ValueError("Mixed output requires conversion_report.json and jang_config.json")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    jang = json.loads(jang_path.read_text(encoding="utf-8"))
    if report.get("architecture_kind") not in {"moe", "hybrid"}:
        raise ValueError("Mixed output is not marked as a MoE or hybrid architecture")
    if not report.get("protected_router_tensors"):
        raise ValueError("Mixed output does not contain a protected router tensor")
    if not report.get("compressed_expert_tensors"):
        raise ValueError("Mixed output does not contain a compressed expert tensor")
    if jang.get("format") != "gguf2mlx-mixed-v1":
        raise ValueError(f"Unsupported mixed format: {jang.get('format')!r}")

    decisions = report.get("tensor_decisions", [])
    invalid_high = [
        item
        for item in decisions
        if item.get("category") in {"attention", "router"}
        and item.get("precision") not in {"float16", "float32", 8}
    ]
    invalid_experts = [
        item
        for item in decisions
        if item.get("category") == "expert_mlp" and item.get("precision") != 4
    ]
    if invalid_high or invalid_experts:
        raise ValueError(
            f"Invalid precision decisions: high={invalid_high}, experts={invalid_experts}"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_mixed_artifacts(args.model_dir)
    print(
        f"Validated {report['architecture']} ({report['architecture_kind']}): "
        f"{len(report['protected_router_tensors'])} routers, "
        f"{len(report['compressed_expert_tensors'])} expert tensors"
    )


if __name__ == "__main__":
    main()
