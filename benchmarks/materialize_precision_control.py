"""Create labeled dense GGUF controls without changing tokenizer or tensor layout."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize


def materialize(source: Path, output: Path, dtype: str, hf_model: Path | None = None) -> None:
    if output.exists() or output.resolve() == source.resolve():
        raise ValueError("Control output must be new and must not overwrite its source")
    if dtype not in {"float16", "float32"}:
        raise ValueError("Control dtype must be float16 or float32")
    reader = GGUFReader(str(source))
    arch = reader.get_field("general.architecture").contents()
    if arch != "gemma":
        raise ValueError("Precision controls currently support first-generation Gemma only")
    index = None
    if hf_model is not None:
        from gguf2mlx.gguf2mlx import _map_tensor_name

        index = json.loads((hf_model / "model.safetensors.index.json").read_text())["weight_map"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="precision-control-", dir=output.parent) as directory:
        temporary = Path(directory) / "control.gguf"
        writer = GGUFWriter(temporary, arch, use_temp_file=True)
        try:
            for key, field in reader.fields.items():
                if key.startswith("GGUF.") or key in {
                    "general.architecture", "general.file_type", "general.quantization_version",
                }:
                    continue
                writer.add_key_value(
                    key, field.contents(), field.types[0],
                    field.types[1] if field.types[0] == GGUFValueType.ARRAY else None,
                )
            writer.add_file_type(1 if dtype == "float16" else 0)
            writer.add_string(
                "gguf2mlx.control.origin",
                "unquantized HF weights" if hf_model is not None else "dequantized GGUF weights",
            )
            shard_name = None
            shard = None
            for tensor in reader.tensors:
                if index is not None:
                    import mlx.core as mx

                    name = _map_tensor_name(tensor.name, arch)
                    next_shard = index[name]
                    if next_shard != shard_name:
                        shard = mx.load(str(hf_model / next_shard))
                        shard_name = next_shard
                    array = np.asarray(shard[name].astype(mx.float32))
                    if "norm.weight" in tensor.name:
                        array = array + np.float32(1)
                elif tensor.tensor_type in {GGMLQuantizationType.F32, GGMLQuantizationType.F16}:
                    array = np.asarray(tensor.data)
                else:
                    array = dequantize(tensor.data, tensor.tensor_type)
                shape = tuple(int(dim) for dim in reversed(tensor.shape))
                if array.shape != shape:
                    raise ValueError(f"Tensor shape mismatch for {tensor.name}: {array.shape} != {shape}")
                # Native Gemma CPU norm operations require F32 vector weights.
                tensor_dtype = "float32" if array.ndim == 1 else dtype
                writer.add_tensor(tensor.name, np.asarray(array, dtype=tensor_dtype))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        finally:
            writer.close()
        os.link(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], required=True)
    parser.add_argument(
        "--hf-model", type=Path,
        help="Optional local unquantized HF mirror; otherwise decode the source GGUF (not original HF)",
    )
    args = parser.parse_args()
    materialize(args.input, args.output, args.dtype, args.hf_model)


if __name__ == "__main__":
    main()
