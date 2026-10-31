"""Qwen2.5-VL: the ViT that shares Qwen2-VL's GGUF names but not its maths."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from gguf2mlx import gguf2mlx as core
from gguf2mlx.vision_qwen2 import Qwen2VisionAdapter
from gguf2mlx.vision_qwen25 import Qwen25VisionAdapter
from tests.vlm_plan_fixtures import qwen25_config, write_plan_fixture


def convert(tmp_path, **kwargs):
    adapter = Qwen25VisionAdapter(qwen25_config())
    fixture = write_plan_fixture(
        tmp_path, adapter, qwen25_config(), gguf_arch="qwen2vl", **kwargs
    )
    return adapter, fixture


def convert_direct(fixture):
    """Call the converter itself: ``core.convert`` reports rather than raises."""
    from gguf2mlx import vlm

    return vlm.convert_vlm(
        str(fixture.main), fixture.output, "float16",
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )


def test_converts_every_declared_tensor(tmp_path):
    _, fixture = convert(tmp_path)
    assert core.convert(
        str(fixture.main), str(fixture.output), "float16",
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )
    stored = load_file(str(fixture.output / "model.safetensors"))
    assert set(stored) == set(fixture.expected)
    for name, array in fixture.expected.items():
        assert np.allclose(stored[name], array.astype(np.float16), atol=2e-3), name


def test_fused_qkv_is_assembled_in_q_k_v_order(tmp_path):
    """A Q/K/V concatenation in the wrong order loads and computes nonsense."""
    adapter, fixture = convert(tmp_path)
    assert core.convert(
        str(fixture.main), str(fixture.output), "float16",
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )
    stored = load_file(str(fixture.output / "model.safetensors"))
    for layer in range(adapter.vision["depth"]):
        expected = np.concatenate([
            fixture.sources[f"v.blk.{layer}.attn_{part}.weight"]
            for part in ("q", "k", "v")
        ], axis=0)
        assert np.array_equal(
            stored[f"vision_tower.blocks.{layer}.attn.qkv.weight"],
            expected.astype(np.float16),
        )


def test_plan_is_self_consistent(tmp_path):
    adapter = Qwen25VisionAdapter(qwen25_config())
    declared = {source for _, sources, _ in adapter.vision_plan for source in sources}
    assert declared == adapter.source_names
    for target, sources, axis in adapter.vision_plan:
        assert target in adapter.expected
        if len(sources) == 1:
            continue
        shape = adapter.expected[target]
        if target.removeprefix("vision_tower.") in adapter.stacked_targets:
            # A new axis is inserted, so each source is one rank smaller.
            assert shape[axis] == len(sources)
            stacked = shape[:axis] + shape[axis + 1:]
            assert all(adapter.source_shapes[source] == stacked for source in sources)
        else:
            assert shape[axis] == sum(
                adapter.source_shapes[source][axis] for source in sources
            )


def test_rmsnorm_blocks_declare_no_bias(tmp_path):
    adapter = Qwen25VisionAdapter(qwen25_config())
    assert "vision_tower.blocks.0.norm1.weight" in adapter.expected
    assert "vision_tower.blocks.0.norm1.bias" not in adapter.expected
    assert "vision_tower.merger.ln_q.bias" not in adapter.expected


def test_rejects_the_qwen2vl_projector(tmp_path):
    """The shared GGUF naming makes this the most likely mix-up to ship."""
    _, fixture = convert(tmp_path, vision_overrides={"clip.projector_type": "qwen2vl"})
    with pytest.raises(ValueError, match="Unsupported Qwen2.5-VL projector type"):
        convert_direct(fixture)


def test_rejects_a_gelu_vision_mlp(tmp_path):
    _, fixture = convert(tmp_path, vision_overrides={"clip.use_gelu": True})
    with pytest.raises(ValueError, match="SiLU ViT MLP"):
        convert_direct(fixture)


def test_missing_source_tensor_is_reported_not_skipped(tmp_path):
    _, fixture = convert(tmp_path, drop_sources=("v.blk.1.ffn_gate.weight",))
    with pytest.raises(ValueError, match="v.blk.1.ffn_gate.weight"):
        convert_direct(fixture)


@pytest.mark.parametrize("field,value,match", [
    ("hidden_act", "gelu_pytorch_tanh", "SiLU vision activation"),
    ("out_hidden_size", 32, "out_hidden_size"),
    ("temporal_patch_size", 1, "patch/merge/channel"),
    ("spatial_merge_size", 3, "patch/merge/channel"),
    ("fullatt_block_indexes", [5], "fullatt_block_indexes"),
    ("fullatt_block_indexes", [], "fullatt_block_indexes"),
])
def test_rejects_unsupported_vision_configuration(field, value, match):
    config = qwen25_config()
    config["vision_config"][field] = value
    with pytest.raises(ValueError, match=match):
        Qwen25VisionAdapter(config)


@pytest.mark.parametrize("sections", [[4, 2, 1], [1, 1, 1], [4, 2, 2, 2]])
def test_rejects_mrope_sections_that_do_not_partition_head_dim(sections):
    config = qwen25_config()
    config["rope_scaling"] = {"type": "mrope", "mrope_section": sections}
    with pytest.raises(ValueError, match="mrope_section"):
        Qwen25VisionAdapter(config)


def test_rejects_a_biased_language_mlp():
    config = qwen25_config()
    config["mlp_bias"] = True
    with pytest.raises(ValueError, match="unbiased SiLU"):
        Qwen25VisionAdapter(config)


def test_qwen2_adapter_refuses_a_qwen25_checkpoint():
    """The two must not be aliases in either direction."""
    with pytest.raises(ValueError, match="Qwen2-VL"):
        Qwen2VisionAdapter(qwen25_config())


def test_architecture_registry_names_a_distinct_adapter():
    from gguf2mlx import vlm

    assert vlm.ARCHITECTURE_REGISTRY["qwen2_vl"] != vlm.ARCHITECTURE_REGISTRY["qwen2_5_vl"]


def test_config_keeps_text_hyperparameters_flat(tmp_path):
    """mlx-vlm's Qwen2.5-VL reader rebuilds text_config from the root."""
    _, fixture = convert(tmp_path)
    assert core.convert(
        str(fixture.main), str(fixture.output), "float16",
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )
    written = json.loads((fixture.output / "config.json").read_text())
    assert written["model_type"] == "qwen2_5_vl"
    assert written["hidden_size"] == 64
    assert written["text_config"]["hidden_size"] == 64
    assert written["vision_config"]["model_type"] == "qwen2_5_vl"
    assert written["vision_config"]["torch_dtype"] == "float16"
