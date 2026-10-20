from __future__ import annotations

import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from gguf.quants import quantize
from safetensors import safe_open
from safetensors.numpy import load_file

from gguf2mlx import gguf2mlx as core
from gguf2mlx.streaming import StreamingShardWriter


def _read_output(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    arrays = {}
    for name in set(index["weight_map"].values()):
        arrays.update(load_file(root / name))
    assert set(arrays) == set(index["weight_map"])
    assert index["metadata"]["total_size"] == sum(a.nbytes for a in arrays.values())
    return index, arrays


@pytest.mark.parametrize("limit,expected", [(16, 2), (15, 3), (1, 3)])
def test_shard_boundaries_and_oversized_tensors(tmp_path, limit, expected):
    with StreamingShardWriter(tmp_path, limit) as writer:
        for i in range(3):
            writer.add(f"tensor.{i}", np.arange(4, dtype=np.float16))
        writer.finish()
    index, arrays = _read_output(tmp_path)
    assert len(set(index["weight_map"].values())) == expected
    for i in range(3):
        np.testing.assert_array_equal(arrays[f"tensor.{i}"], np.arange(4))
    assert not list(tmp_path.glob(".tensor-stream-*"))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8",
                                  "int16", "int32", "int64", "uint8", "uint16",
                                  "uint32", "uint64", "bool", ">f4", ">i8"])
def test_streaming_preserves_strided_arrays_and_endianness(tmp_path, dtype):
    array = np.arange(24).reshape(4, 6).astype(dtype).T[:, ::-1]
    with StreamingShardWriter(tmp_path, 1024) as writer:
        writer.add("strided", array)
        writer.add("scalar", np.array(2, dtype=dtype))
        writer.add("empty", np.empty((0, 3), dtype=dtype))
        writer.finish()
    _, arrays = _read_output(tmp_path)
    np.testing.assert_array_equal(arrays["strided"], array)
    assert arrays["scalar"].shape == ()
    assert arrays["empty"].shape == (0, 3)
    assert arrays["strided"].dtype == np.dtype(dtype).newbyteorder("=")


def test_writer_does_not_retain_arrays(tmp_path):
    with StreamingShardWriter(tmp_path, 1024**3) as writer:
        for i in range(100):
            array = np.ones(1024, dtype=np.float32)
            reference = weakref.ref(array)
            writer.add(str(i), array)
            del array
            assert reference() is None
        assert not writer.files  # All payloads share a shard, but none stay in RAM.
        writer.finish()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_shard_size(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        StreamingShardWriter(tmp_path, limit)


@pytest.mark.parametrize("limit", [True, 1.5])
def test_invalid_shard_size_type(tmp_path, limit):
    with pytest.raises(TypeError, match="positive integer"):
        StreamingShardWriter(tmp_path, limit)


def test_duplicate_and_reserved_names_fail_even_across_shards(tmp_path):
    with StreamingShardWriter(tmp_path, 1) as writer:
        writer.add("a", np.ones(2, dtype=np.float16))
        writer.add("b", np.ones(2, dtype=np.float16))
        for name in ("a", "__metadata__"):
            with pytest.raises(ValueError, match="Duplicate or reserved"):
                writer.add(name, np.ones(2))


def test_stream_cleanup_after_io_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("gguf2mlx.streaming.shutil.copyfileobj", fail)
    with pytest.raises(OSError, match="disk full"), StreamingShardWriter(tmp_path, 1024) as writer:
        writer.add("a", np.ones(8, dtype=np.float16))
        writer.finish()
    assert not list(tmp_path.iterdir())


def _write_gguf(path, arch, arrays, raw_dtype=None):
    writer = GGUFWriter(path, arch)
    for name, array in arrays.items():
        writer.add_tensor(name, array, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GGUFReader(path)


def test_float_conversion_shards_using_output_not_input_bytes(tmp_path):
    arrays = {f"blk.{i}.attn_q.weight": np.arange(12, dtype=np.float32).reshape(3, 4)
              for i in range(3)}
    reader = _write_gguf(tmp_path / "source.gguf", "llama", arrays)
    output = tmp_path / "out"
    output.mkdir()
    core.extract_and_convert_weights(reader, "llama", output, max_shard_bytes=48)
    index, converted = _read_output(output)
    assert len(set(index["weight_map"].values())) == 2
    assert index["metadata"]["total_size"] == 72
    for name, array in arrays.items():
        np.testing.assert_array_equal(converted[core._map_tensor_name(name, "llama")],
                                      array.astype(np.float16))
    for file in output.glob("*.safetensors"):
        with safe_open(file, framework="np") as handle:
            assert handle.metadata() == {"format": "np"}


def test_direct_quant_streams_packed_weights_scales_and_biases(tmp_path):
    array = np.linspace(-1, 1, 128, dtype=np.float32).reshape(2, 64)
    qtype = GGMLQuantizationType.Q8_0
    reader = _write_gguf(tmp_path / "source.gguf", "llama",
                         {"blk.0.ffn_gate.weight": quantize(array, qtype)}, raw_dtype=qtype)
    output = tmp_path / "out"
    output.mkdir()
    core.extract_and_convert_weights_direct_quant(
        reader, "llama", output, "float16", 4, 64, "affine", max_shard_bytes=8,
    )
    index, converted = _read_output(output)
    decoded = core._decode_tensor_to_array(reader.tensors[0], "float16", "unused")
    packed, scales, biases = core._quantize_affine_4bit(decoded, 64)
    stem = "model.layers.0.mlp.gate_proj"
    for suffix, expected in [("weight", packed), ("scales", scales), ("biases", biases)]:
        np.testing.assert_array_equal(converted[f"{stem}.{suffix}"], expected)
    assert len(set(index["weight_map"].values())) == 2
    assert index["metadata"]["total_size"] == packed.nbytes + scales.nbytes + biases.nbytes


def test_fusion_reads_headers_without_buffering_across_layers(monkeypatch):
    tensors = [
        SimpleNamespace(name=f"blk.{i}.attn_{kind}.weight", tensor_type=0)
        for kind in ("q", "k", "v") for i in range(10)
    ]
    references = []

    def decode(tensor, *args):
        if tensor.name.endswith("attn_q.weight"):
            assert all(reference() is None for reference in references)
        array = np.ones((2, 4), dtype=np.float16)
        references.append(weakref.ref(array))
        return array

    monkeypatch.setattr(core, "_decode_tensor_to_array", decode)
    reader = SimpleNamespace(tensors=tensors, get_field=lambda name: None)
    for _, _, name, array, _identity in core._iter_weight_arrays(reader, "phi3", "float16"):
        assert name.endswith("qkv_proj.weight")
        assert array.shape == (6, 4)
        del array
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("arch,name,message", [
    ("phi3", "blk.0.attn_q.weight", "Incomplete Phi-3"),
    ("deepseek2", "blk.0.attn_k_b.weight", "Incomplete split kv_b"),
])
def test_incomplete_fusion_does_not_publish_index(tmp_path, arch, name, message):
    reader = _write_gguf(tmp_path / "source.gguf", arch,
                         {name: np.ones((2, 4), dtype=np.float32)})
    with pytest.raises(RuntimeError, match=message):
        core.extract_and_convert_weights(reader, arch, tmp_path)
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not list(tmp_path.glob(".tensor-stream-*"))


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), 1e-20, 1e308])
def test_public_invalid_shard_limit_does_not_create_output(tmp_path, size):
    output = tmp_path / "output"
    assert not core.convert("unused.gguf", str(output), max_shard_gb=size)
    assert not output.exists()


@pytest.mark.parametrize("command", [[], ["convert"]])
def test_cli_forwards_shard_limit(tmp_path, monkeypatch, command):
    options = {}

    def convert(*args, **kwargs):
        options.update(kwargs)
        return True

    monkeypatch.setattr(core, "convert", convert)
    core.main([*command, "--input", "unused.gguf", "--output", str(tmp_path),
               "--max-shard-gb", "2"])
    assert options["max_shard_gb"] == 2.0


@pytest.mark.parametrize("mode", ["float", "direct", "vlm"])
def test_public_conversion_forwards_byte_limit_to_each_pipeline(tmp_path, monkeypatch, mode):
    from gguf2mlx import vlm

    options = {}

    def run(source, destination, *args, **kwargs):
        options.update(kwargs)
        destination = Path(destination)
        destination.mkdir()
        (destination / "marker").touch()
        return True

    monkeypatch.setattr(core, "_convert", run)
    monkeypatch.setattr(core, "_convert_direct_quantized", run)
    monkeypatch.setattr(vlm, "convert_vlm", run)
    assert core.convert(
        "unused.gguf", str(tmp_path / "out"), max_shard_gb=0.5,
        model_type="vlm" if mode == "vlm" else "llm",
        quantize=mode == "direct", direct_quant=mode == "direct",
    )
    assert options["max_shard_bytes"] == 512 * 1024**2
