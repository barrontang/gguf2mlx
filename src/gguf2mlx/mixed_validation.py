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
    if not isinstance(report, dict) or not isinstance(jang, dict):
        raise TypeError("Mixed output reports must contain JSON objects")
    if report.get("architecture_kind") not in {"moe", "hybrid"}:
        raise ValueError("Mixed output is not marked as a MoE or hybrid architecture")
    if report.get("mixed_precision") is not True:
        raise ValueError("Conversion report is not marked as mixed precision")
    if not report.get("protected_router_tensors"):
        raise ValueError("Mixed output does not contain a protected router tensor")
    if not report.get("compressed_expert_tensors"):
        raise ValueError("Mixed output does not contain a compressed expert tensor")
    if jang.get("format") != "gguf2mlx-mixed-v1":
        raise ValueError(f"Unsupported mixed format: {jang.get('format')!r}")

    decisions = report.get("tensor_decisions", [])
    if not isinstance(decisions, list) or not all(
        isinstance(item, dict) for item in decisions
    ):
        raise ValueError("Conversion report tensor_decisions must be a list of objects")

    invalid_high = [
        item
        for item in decisions
        if item.get("category") in {"attention", "router"}
        and (
            item.get("precision") not in ("float16", "float32", 8)
            or item.get("protected") is not True
        )
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

    protected_routers = [
        item
        for item in decisions
        if item.get("category") == "router" and item.get("protected") is True
    ]
    compressed_experts = [
        item
        for item in decisions
        if item.get("category") == "expert_mlp"
        and item.get("precision") == 4
        and item.get("protected") is not True
    ]
    if report["protected_router_tensors"] != protected_routers:
        raise ValueError("Protected router list does not match tensor_decisions")
    if report["compressed_expert_tensors"] != compressed_experts:
        raise ValueError("Compressed expert list does not match tensor_decisions")

    decision_map = {
        item.get("output_name"): item.get("precision")
        for item in decisions
        if isinstance(item.get("output_name"), str)
    }
    if len(decision_map) != len(decisions):
        raise ValueError("Each tensor decision must have a unique output_name")
    if jang.get("tensor_decisions") != decision_map:
        raise ValueError("JANG tensor_decisions do not match conversion_report.json")

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
