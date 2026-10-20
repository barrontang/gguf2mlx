"""Disk-backed safetensors shards without retaining tensor payloads in RAM."""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


class StreamingShardWriter:
    """Spool payloads to disk, then prepend each shard's safetensors header.

    The limit applies to output payload bytes, not headers. An indivisible
    tensor larger than the limit gets its own shard.
    """

    def __init__(self, root: Path, max_shard_bytes: int):
        if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int):
            raise TypeError("max_shard_bytes must be a positive integer")
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be a positive integer")
        self.root = root
        self.max_shard_bytes = max_shard_bytes
        self.size = 0
        self.total = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self._header: dict[str, dict] = {}
        self._temp = tempfile.TemporaryDirectory(prefix=".tensor-stream-", dir=root)
        self._payload = Path(self._temp.name) / "payload"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._temp.cleanup()

    def add(self, name: str, array: np.ndarray, *, dtype: str | None = None) -> None:
        if name == "__metadata__" or name in self.weight_map:
            raise ValueError(f"Duplicate or reserved output tensor name: {name}")
        if dtype is not None:
            if dtype != "bfloat16":
                raise ValueError(f"Unsupported output dtype override: {dtype}")
            from gguf import GGMLQuantizationType
            from gguf.quants import quantize

            shape = array.shape
            array = quantize(
                np.ascontiguousarray(array, dtype=np.float32), GGMLQuantizationType.BF16,
            ).view(np.uint16).reshape(shape)
            dtype_name: str | None = "BF16"
        else:
            dtype_name = None
        arr_dtype = array.dtype
        if arr_dtype.kind not in "fiub" or arr_dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError(f"Unsupported safetensors dtype: {arr_dtype}")
        dtype_name = dtype_name or ("BOOL" if arr_dtype.kind == "b" else (
            f"{'F' if arr_dtype.kind == 'f' else 'I' if arr_dtype.kind == 'i' else 'U'}"
            f"{arr_dtype.itemsize * 8}"
        ))
        if self._header and self.size + array.nbytes > self.max_shard_bytes:
            self.flush()
        # tofile handles strided layouts without building a shard-sized bytes copy.
        little_endian = array.astype(arr_dtype.newbyteorder("<"), copy=False)
        with self._payload.open("ab") as handle:
            little_endian.tofile(handle)
        self._header[name] = {
            "dtype": dtype_name,
            "shape": list(array.shape),
            "data_offsets": [self.size, self.size + array.nbytes],
        }
        self.weight_map[name] = f"model-{len(self.files) + 1:05d}-of-NNNNN.safetensors"
        self.size += array.nbytes
        self.total += array.nbytes

    def flush(self) -> None:
        if not self._header:
            return
        header = json.dumps(
            {"__metadata__": {"format": "np"}, **self._header},
            separators=(",", ":"),
        ).encode("utf-8")
        header += b" " * (-len(header) % 8)
        path = self.root / f"model-{len(self.files) + 1:05d}-of-NNNNN.safetensors"
        temporary = Path(self._temp.name) / "shard"
        with temporary.open("wb") as output, self._payload.open("rb") as payload:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            shutil.copyfileobj(payload, output, length=1024 * 1024)
        temporary.replace(path)
        self.files.append(path)
        self._payload.unlink()
        self._header.clear()
        self.size = 0

    def finish(self) -> None:
        self.flush()
        if not self.files:
            raise RuntimeError("No weights extracted!")
        renames = {}
        for i, path in enumerate(self.files, 1):
            final = self.root / f"model-{i:05d}-of-{len(self.files):05d}.safetensors"
            path.replace(final)
            renames[path.name] = final.name
            self.files[i - 1] = final
        self.weight_map = {key: renames[value] for key, value in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total}, "weight_map": self.weight_map}
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8",
        )
        self.close()
