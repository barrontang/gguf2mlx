from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter
from safetensors.numpy import save_file

from gguf2mlx import gguf2mlx as core


@pytest.mark.parametrize("source_dtype", [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("output_dtype", ["float16", "float32"])
def test_dense_gguf_decode_preserves_non_square_row_major_matrix(tmp_path, source_dtype, output_dtype):
    path = tmp_path / "layout.gguf"
    array = np.arange(15, dtype=source_dtype).reshape(3, 5)
    writer = GGUFWriter(path, "gemma")
    writer.add_tensor("blk.0.attn_q.weight", array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    tensor = GGUFReader(path).tensors[0]
    assert tuple(tensor.shape) == (5, 3)
    decoded = core._decode_tensor_to_array(tensor, output_dtype, tensor.name)
    np.testing.assert_array_equal(decoded, array.astype(output_dtype))


def test_precision_control_keeps_metadata_and_tensor_layout(tmp_path):
    path = Path(__file__).parents[1] / "benchmarks" / "materialize_precision_control.py"
    spec = importlib.util.spec_from_file_location("precision_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "source.gguf", tmp_path / "control.gguf"
    array = np.arange(15, dtype=np.float32).reshape(3, 5)
    writer = GGUFWriter(source, "gemma")
    writer.add_token_list(["<unk>", "a", "b"])
    writer.add_token_scores([0.0, -1.0, -2.0])
    writer.add_tensor("blk.0.attn_q.weight", array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    module.materialize(source, output, "float16")
    reader = GGUFReader(output)
    assert reader.get_field("tokenizer.ggml.tokens").contents() == ["<unk>", "a", "b"]
    assert reader.tensors[0].tensor_type == GGMLQuantizationType.F16
    np.testing.assert_array_equal(reader.tensors[0].data, array.astype(np.float16))
    with pytest.raises(ValueError, match="must be new"):
        module.materialize(source, output, "float16")


def test_gemma_runtime_adapter_is_written_with_config(tmp_path):
    config = {"model_type": "gemma"}
    core._write_runtime_adapter(tmp_path, "gemma", config)
    assert config["model_file"] == "gemma_model.py"
    assert (tmp_path / config["model_file"]).is_file()


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_precision_control_hf_norm_restore_and_shape(tmp_path, dtype):
    import json

    path = Path(__file__).parents[1] / "benchmarks" / "materialize_precision_control.py"
    spec = importlib.util.spec_from_file_location("precision_control_hf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "source.gguf", tmp_path / "control.gguf"
    writer = GGUFWriter(source, "gemma")
    writer.add_tensor("output_norm.weight", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    save_file({"model.norm.weight": np.array([0, 1, 2, 3], dtype=np.float32)}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.norm.weight": "weights.safetensors"},
    }))
    module.materialize(source, output, dtype, tmp_path)
    tensor = GGUFReader(output).tensors[0]
    assert tensor.tensor_type == GGMLQuantizationType.F32
    np.testing.assert_array_equal(tensor.data, [1, 2, 3, 4])
