"""Audit every Q4_0 decoded value against the loaded native llama.cpp library."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize
from llama_cpp.llama_cpp import _lib


def audit(source: Path) -> dict:
    native_decode = _lib.dequantize_row_q4_0
    native_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    native_decode.restype = None
    checked = bit_differences = fp16_roundtrip_differences = tensors = 0
    for tensor in GGUFReader(str(source)).tensors:
        if tensor.tensor_type != GGMLQuantizationType.Q4_0:
            continue
        tensors += 1
        blocks = tensor.data.reshape(-1, 18)
        for start in range(0, len(blocks), 32768):
            packed = np.ascontiguousarray(blocks[start : start + 32768])
            reference = dequantize(packed, GGMLQuantizationType.Q4_0).reshape(-1)
            native = np.empty(reference.size, dtype=np.float32)
            native_decode(packed.ctypes.data, native.ctypes.data, native.size)
            bit_differences += int(np.count_nonzero(reference.view(np.uint32) != native.view(np.uint32)))
            roundtrip = reference.astype(np.float16).astype(np.float32)
            fp16_roundtrip_differences += int(
                np.count_nonzero(reference.view(np.uint32) != roundtrip.view(np.uint32)),
            )
            checked += native.size
    if not tensors:
        raise ValueError("Source contains no Q4_0 tensors")
    return {
        "input": str(source.resolve()),
        "q4_tensors": tensors,
        "decoded_values_checked": checked,
        "native_bit_differences": bit_differences,
        "fp16_roundtrip_bit_differences": fp16_roundtrip_differences,
        "scale_path": "stored float16 -> reference float32 decode -> dense output dtype",
        "success": bit_differences == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    args.result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
