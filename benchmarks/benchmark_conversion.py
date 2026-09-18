"""Run a reproducible GGUF-to-MLX conversion benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _resident_kib(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        check=False,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark conversion time, peak RSS, and output size."
    )
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-mode", default="affine")
    parser.add_argument("--direct-quant", action="store_true",
                        help="Use bounded-memory direct quantization path (--direct-quant)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both standard and --direct-quant paths and print a comparison table")
    parser.add_argument("--result-json", type=Path)
    return parser


def _run_one(args, output: Path, direct_quant: bool) -> dict:
    """Run a single conversion and return the result dict."""
    command = [
        sys.executable,
        "-m",
        "gguf2mlx",
        "convert",
        "--input",
        str(args.input),
        "--output",
        str(output),
        "--dtype",
        args.dtype,
    ]
    if args.quantize:
        command.extend(
            [
                "--quantize",
                "--q-bits",
                str(args.q_bits),
                "--q-group-size",
                str(args.q_group_size),
                "--q-mode",
                args.q_mode,
            ]
        )
    if direct_quant:
        command.append("--direct-quant")

    started = time.perf_counter()
    process = subprocess.Popen(command)
    peak_rss_kib = 0
    while process.poll() is None:
        rss_kib = _resident_kib(process.pid)
        if rss_kib is not None:
            peak_rss_kib = max(peak_rss_kib, rss_kib)
        time.sleep(0.1)
    elapsed_seconds = time.perf_counter() - started

    return {
        "success": process.returncode == 0,
        "returncode": process.returncode,
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "output": str(output.resolve()),
        "output_bytes": _directory_size(output),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_rss_mib": round(peak_rss_kib / 1024, 1),
        "dtype": args.dtype,
        "quantize": args.quantize,
        "q_bits": args.q_bits if args.quantize else None,
        "q_group_size": args.q_group_size if args.quantize else None,
        "q_mode": args.q_mode if args.quantize else None,
        "direct_quant": direct_quant,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("gguf2mlx", "gguf", "mlx", "mlx-lm", "numpy", "safetensors")
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input GGUF does not exist: {args.input}")
    if args.output.exists():
        raise SystemExit(f"Benchmark output must not already exist: {args.output}")

    if args.compare:
        # Run both paths and print a side-by-side comparison.
        std_out = args.output.parent / (args.output.name + "-standard")
        dq_out = args.output.parent / (args.output.name + "-direct-quant")
        for p in (std_out, dq_out):
            if p.exists():
                raise SystemExit(f"Comparison output must not already exist: {p}")

        print("Running standard path...")
        std = _run_one(args, std_out, direct_quant=False)
        print("Running --direct-quant path...")
        dq = _run_one(args, dq_out, direct_quant=True)

        comparison = {"standard": std, "direct_quant": dq}
        if std["success"] and dq["success"]:
            rss_delta_pct = round(
                100.0 * (dq["peak_rss_mib"] - std["peak_rss_mib"]) / max(std["peak_rss_mib"], 1),
                1,
            )
            size_delta_pct = round(
                100.0 * (dq["output_bytes"] - std["output_bytes"]) / max(std["output_bytes"], 1),
                1,
            )
            elapsed_delta_pct = round(
                100.0 * (dq["elapsed_seconds"] - std["elapsed_seconds"]) / max(std["elapsed_seconds"], 0.001),
                1,
            )
            comparison["delta"] = {
                "rss_mib": round(dq["peak_rss_mib"] - std["peak_rss_mib"], 1),
                "rss_pct": rss_delta_pct,
                "output_bytes": dq["output_bytes"] - std["output_bytes"],
                "output_size_pct": size_delta_pct,
                "elapsed_seconds": round(dq["elapsed_seconds"] - std["elapsed_seconds"], 3),
                "elapsed_pct": elapsed_delta_pct,
            }
            print(
                f"\n{'':=<60}\n"
                f"  Comparison: standard vs --direct-quant\n"
                f"{'':=<60}\n"
                f"  Peak RSS   standard={std['peak_rss_mib']:.1f} MiB   "
                f"direct-quant={dq['peak_rss_mib']:.1f} MiB   "
                f"delta={rss_delta_pct:+.1f}%\n"
                f"  Output     standard={std['output_bytes'] / 1e6:.1f} MB   "
                f"direct-quant={dq['output_bytes'] / 1e6:.1f} MB   "
                f"delta={size_delta_pct:+.1f}%\n"
                f"  Elapsed    standard={std['elapsed_seconds']:.1f}s   "
                f"direct-quant={dq['elapsed_seconds']:.1f}s   "
                f"delta={elapsed_delta_pct:+.1f}%\n"
                f"{'':=<60}"
            )
        rendered = json.dumps(comparison, indent=2)
        print(rendered)
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(rendered + "\n", encoding="utf-8")
        return 0 if (std["success"] and dq["success"]) else 1

    result = _run_one(args, args.output, direct_quant=args.direct_quant)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
