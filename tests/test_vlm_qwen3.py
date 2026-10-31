"""Qwen3-VL: fused QKV, a learned position table, and deepstack mergers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from gguf2mlx import gguf2mlx as core
from gguf2mlx import vlm
from gguf2mlx.vision_qwen3 import Qwen3VisionAdapter
from tests.vlm_plan_fixtures import qwen3_config, write_plan_fixture


def convert(tmp_path, **kwargs):
    adapter = Qwen3VisionAdapter(qwen3_config())
    fixture = write_plan_fixture(
        tmp_path, adapter, qwen3_config(), gguf_arch="qwen3vl", **kwargs
    )
    return adapter, fixture


def convert_direct(fixture):
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


def test_deepstack_mergers_are_ranked_not_layer_indexed():
    """The trap: GGUF names deepstack by layer, the target by occurrence.

    A 24-layer model exporting mergers on layers 5/11/17 must produce
    ``deepstack_merger_list.0/1/2``. Mapping ``v.deepstack.11`` straight through
    would target index 11, which does not exist, and only surface at load time.
    """
    config = qwen3_config()
    config["vision_config"].update(depth=24, deepstack_visual_indexes=[5, 11, 17])
    adapter = Qwen3VisionAdapter(config)
    plan = {
        target: sources for target, sources, _ in adapter.vision_plan
        if "deepstack_merger_list" in target
    }
    assert plan["vision_tower.deepstack_merger_list.0.norm.weight"] == (
        "v.deepstack.5.norm.weight",
    )
    assert plan["vision_tower.deepstack_merger_list.1.linear_fc1.weight"] == (
        "v.deepstack.11.fc1.weight",
    )
    assert plan["vision_tower.deepstack_merger_list.2.linear_fc2.weight"] == (
        "v.deepstack.17.fc2.weight",
    )
    assert "vision_tower.deepstack_merger_list.11.norm.weight" not in plan


def test_deepstack_mergers_normalise_the_merged_width():
    """``use_postshuffle_norm=True`` makes the norm wider than the ViT width."""
    config = qwen3_config()
    config["vision_config"].update(depth=24, deepstack_visual_indexes=[5])
    adapter = Qwen3VisionAdapter(config)
    merged = adapter.vision["hidden_size"] * adapter.vision["spatial_merge_size"] ** 2
    assert merged != adapter.vision["hidden_size"]
    assert adapter.expected["vision_tower.deepstack_merger_list.0.norm.weight"] == (merged,)
    assert adapter.expected["vision_tower.merger.norm.weight"] == (
        adapter.vision["hidden_size"],
    )


def test_qk_norm_is_required_for_the_language_half():
    adapter = Qwen3VisionAdapter(qwen3_config())
    assert "language_model.model.layers.0.self_attn.q_norm.weight" in adapter.expected
    assert "language_model.model.layers.0.self_attn.k_norm.weight" in adapter.expected
    assert "language_model.model.layers.0.self_attn.q_proj.bias" not in adapter.expected


def test_plan_is_self_consistent():
    adapter = Qwen3VisionAdapter(qwen3_config())
    declared = {source for _, sources, _ in adapter.vision_plan for source in sources}
    assert declared == adapter.source_names
    for target, sources, axis in adapter.vision_plan:
        assert target in adapter.expected
        if len(sources) == 1:
            continue
        shape = adapter.expected[target]
        if target.removeprefix("vision_tower.") in adapter.stacked_targets:
            assert shape[axis] == len(sources)
            stacked = shape[:axis] + shape[axis + 1:]
            assert all(adapter.source_shapes[source] == stacked for source in sources)
        else:
            assert shape[axis] == sum(
                adapter.source_shapes[source][axis] for source in sources
            )


def test_rejects_the_qwen25_projector(tmp_path):
    _, fixture = convert(tmp_path, vision_overrides={"clip.projector_type": "qwen2.5vl_merger"})
    with pytest.raises(ValueError, match="Unsupported Qwen3-VL projector type"):
        convert_direct(fixture)


def test_requires_deepstack_layer_flags(tmp_path):
    _, fixture = convert(tmp_path, drop_vision_kv=("clip.vision.is_deepstack_layers",))
    with pytest.raises(ValueError, match="is_deepstack_layers"):
        convert_direct(fixture)


def test_rejects_a_deepstack_layer_disagreement(tmp_path):
    _, fixture = convert(
        tmp_path, vision_overrides={"clip.vision.is_deepstack_layers": [True, True]}
    )
    with pytest.raises(ValueError, match="disagree with HF deepstack_visual_indexes"):
        convert_direct(fixture)


def test_missing_source_tensor_is_reported_not_skipped(tmp_path):
    _, fixture = convert(tmp_path, drop_sources=("v.deepstack.1.fc2.weight",))
    with pytest.raises(ValueError, match="v.deepstack.1.fc2.weight"):
        convert_direct(fixture)


@pytest.mark.parametrize("field,value,match", [
    ("hidden_act", "silu", "tanh-approximate GELU"),
    ("out_hidden_size", 32, "out_hidden_size"),
    ("num_position_embeddings", 10, "perfect square"),
    ("deepstack_visual_indexes", [5], "deepstack_visual_indexes"),
    ("deepstack_visual_indexes", [1, 1], "deepstack_visual_indexes"),
])
def test_rejects_unsupported_vision_configuration(field, value, match):
    config = qwen3_config()
    config["vision_config"][field] = value
    with pytest.raises(ValueError, match=match):
        Qwen3VisionAdapter(config)


def test_rejects_non_interleaved_mrope():
    config = qwen3_config()
    config["text_config"]["rope_scaling"]["mrope_interleaved"] = False
    with pytest.raises(ValueError, match="Interleaved mRoPE"):
        Qwen3VisionAdapter(config)


def test_rejects_biased_language_attention():
    config = qwen3_config()
    config["text_config"]["attention_bias"] = True
    with pytest.raises(ValueError, match="unbiased Q/K/V"):
        Qwen3VisionAdapter(config)


def test_config_is_written_with_a_nested_text_config(tmp_path):
    """mlx-vlm reads Qwen3-VL text hyperparameters from ``text_config``."""
    adapter, fixture = convert(tmp_path)
    assert core.convert(
        str(fixture.main), str(fixture.output), "float16",
        mmproj=str(fixture.mmproj), hf_model=str(fixture.root), offline=True,
    )
    written = json.loads((fixture.output / "config.json").read_text())
    assert written["model_type"] == "qwen3_vl"
    assert written["text_config"]["hidden_size"] == adapter.text["hidden_size"]
    assert written["text_config"]["model_type"] == "qwen3_vl_text"
    assert written["text_config"]["torch_dtype"] == "float16"
    # Nested only: flattening would let mlx-vlm read a stale root value instead.
    assert "hidden_size" not in written
    assert written["vision_config"]["deepstack_visual_indexes"] == [1]
