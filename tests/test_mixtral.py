"""Mixtral conversion: expert stacking, router mapping, and what gets refused.

Three failure modes are covered here, and only the first is obvious:

1. **Unmapped tensors.** The legacy export names experts
   ``blk.0.ffn_gate.3.weight``, which matches no Llama pattern. Before this
   adapter such a file converted "successfully" with 768 tensor names copied
   through verbatim and no MLP at all.
2. **The wrong module path.** Mixtral's MoE block is ``block_sparse_moe`` in
   mlx-lm; Qwen-MoE and DeepSeek use ``mlp``. Emitting the ``mlp`` path yields a
   directory that converts and then fails to load.
3. **Expert order.** The stacked tensor has an expert axis, so reversing the
   order, skipping one, or stacking a subset all produce a valid-looking
   ``[n_experts, out, in]`` array whose rows no longer match the router's output
   columns. The fixture fills expert ``e`` with ``e + 1`` so the tests can see it.

The parameter names are checked against mlx-lm itself rather than against a copy
of its naming convention, so this file fails when mlx-lm renames a module.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gguf import GGUFReader
from safetensors import safe_open

from gguf2mlx import gguf2mlx as core
from tests.mixtral_fixtures import DEFAULT_DIMS, write_mixtral_gguf


def convert(tmp_path, **kwargs):
    """Convert a fixture and return (config, {name: array})."""
    gguf_path = write_mixtral_gguf(tmp_path / "mixtral.gguf", **kwargs)
    output_dir = tmp_path / "out"
    assert core.convert(str(gguf_path), str(output_dir)) is True

    config = json.loads((output_dir / "config.json").read_text())
    weights = {}
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(output_dir / shard), framework="np") as handle:
            for name in set(handle.keys()):
                weights[name] = handle.get_tensor(name)
    return config, weights


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    return convert(tmp_path_factory.mktemp("legacy"))


@pytest.fixture(scope="module")
def stacked(tmp_path_factory):
    return convert(tmp_path_factory.mktemp("stacked"), layout="stacked")


# --- detection ---------------------------------------------------------------


def test_expert_metadata_makes_a_llama_file_a_mixtral(tmp_path):
    path = write_mixtral_gguf(tmp_path / "model.gguf")
    reader = GGUFReader(str(path))

    assert core.get_metadata_str(reader, "general.architecture") == "llama"
    assert core.detect_architecture(reader) == "mixtral"
    assert core.classify_architecture(reader, "mixtral") == "moe"


def test_a_dense_llama_stays_dense(tmp_path):
    """The upgrade hinges on expert_count, so its absence must not trigger it."""
    path = write_mixtral_gguf(tmp_path / "model.gguf", expert_count=0, plan={
        "token_embd.weight": (16, 8),
        "output.weight": (16, 8),
        "output_norm.weight": (8,),
        "blk.0.attn_norm.weight": (8,),
        "blk.0.ffn_norm.weight": (8,),
        "blk.0.attn_q.weight": (8, 8),
        "blk.0.attn_k.weight": (4, 8),
        "blk.0.attn_v.weight": (4, 8),
        "blk.0.attn_output.weight": (8, 8),
        "blk.0.ffn_gate.weight": (12, 8),
        "blk.0.ffn_up.weight": (12, 8),
        "blk.0.ffn_down.weight": (8, 12),
    })

    assert core.detect_architecture(GGUFReader(str(path))) == "llama"


@pytest.mark.parametrize("layout", ["legacy", "stacked"])
def test_preserve_refuses_mixtral_rather_than_mismapping_it(tmp_path, layout):
    """`preserve` never covered this family, and detection is what says so.

    Resolving an 8x7B file to `llama` -- which *is* a preserve architecture --
    is how `--quant-source preserve` used to accept one and re-emit its expert
    tensors under names nothing loads. Naming the architecture correctly turns
    that silent miss-map into a refusal that points at the working mode.
    """
    gguf_path = write_mixtral_gguf(tmp_path / "model.gguf", layout=layout)
    output = tmp_path / "out"

    assert core.convert(str(gguf_path), str(output), quant_source="preserve") is False
    assert not (output / "model.safetensors.index.json").exists()


# --- config ------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["legacy", "stacked"])
def test_config_declares_the_mlx_mixtral_shape(layout, request):
    config, _ = request.getfixturevalue(layout)

    assert config["model_type"] == "mixtral"
    assert config["architectures"] == ["MixtralForCausalLM"]
    assert config["num_local_experts"] == DEFAULT_DIMS["experts"]
    assert config["num_experts_per_tok"] == DEFAULT_DIMS["used"]
    # The expert width is the only FFN width Mixtral has; mlx-lm sizes every
    # SwitchGLU from it.
    assert config["intermediate_size"] == DEFAULT_DIMS["ffn"]
    assert config["_gguf_architecture"] == "mixtral"


# --- tensor names ------------------------------------------------------------


def test_every_tensor_lands_on_an_mlx_lm_parameter(legacy):
    """Name conformance against mlx-lm itself, not against a copied convention."""
    pytest.importorskip("mlx", reason="mlx-lm's parameter names need Apple Silicon")
    from mlx.utils import tree_flatten
    from mlx_lm.models.mixtral import MixtralModel, ModelArgs

    config, weights = legacy
    args = ModelArgs(
        model_type=config["model_type"],
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        num_local_experts=config["num_local_experts"],
        num_experts_per_tok=config["num_experts_per_tok"],
    )
    expected = {
        f"model.{name}" for name, _ in tree_flatten(MixtralModel(args).parameters())
    } | {"lm_head.weight"}

    assert set(weights) == expected


def test_no_gguf_tensor_name_survives_into_the_artifact(legacy):
    """A pass-through would leave ``blk.`` names in the output, as it used to."""
    _, weights = legacy

    assert not [name for name in weights if name.startswith("blk.")]
    assert not [name for name in weights if "_exps" in name]


# --- expert stacking ---------------------------------------------------------


@pytest.mark.parametrize("layout", ["legacy", "stacked"])
def test_the_router_becomes_the_block_sparse_gate(layout, request):
    _, weights = request.getfixturevalue(layout)

    gate = weights["model.layers.0.block_sparse_moe.gate.weight"]
    assert gate.shape == (DEFAULT_DIMS["experts"], DEFAULT_DIMS["hidden"])
    # Row e must be the router row for expert e, not its transpose.
    np.testing.assert_allclose(gate[1], np.arange(8, dtype=np.float32) / 8)


@pytest.mark.parametrize("layout", ["legacy", "stacked"])
@pytest.mark.parametrize("projection", ["gate", "up", "down"])
def test_experts_are_stacked_in_router_order(layout, projection, request):
    _, weights = request.getfixturevalue(layout)

    stacked = weights[f"model.layers.0.block_sparse_moe.switch_mlp.{projection}_proj.weight"]
    assert stacked.ndim == 3
    for expert in range(DEFAULT_DIMS["experts"]):
        # Expert e is filled with e + 1 in the fixture: a reversal or an
        # off-by-one shows up as the wrong constant, not as a wrong shape.
        np.testing.assert_allclose(stacked[expert], np.full(stacked.shape[1:], expert + 1))


def test_both_layouts_produce_identical_weights(legacy, stacked):
    """The two on-disk encodings describe the same model, so they must match."""
    _, legacy_weights = legacy
    _, stacked_weights = stacked

    assert set(legacy_weights) == set(stacked_weights)
    for name, array in legacy_weights.items():
        np.testing.assert_allclose(array, stacked_weights[name], err_msg=name)


# --- refusals ----------------------------------------------------------------


def test_a_missing_expert_is_refused(tmp_path):
    """Three of four projections would still stack; the model would be wrong."""
    path = write_mixtral_gguf(tmp_path / "model.gguf", drop_expert=1)

    with pytest.raises(RuntimeError, match="Incomplete Mixtral expert groups"):
        list(core._iter_weight_arrays(GGUFReader(str(path)), "mixtral", "float32"))


def test_an_unknown_expert_count_is_refused(tmp_path):
    path = write_mixtral_gguf(tmp_path / "model.gguf", expert_count=0)

    with pytest.raises(RuntimeError, match="expert_count metadata"):
        core.validate_moe_layout(
            GGUFReader(str(path)), "llama", "moe",
        )


def test_an_expert_index_beyond_the_declared_count_is_refused(tmp_path):
    """Metadata says 1 expert, the file carries 2: the extra one cannot be placed."""
    path = write_mixtral_gguf(tmp_path / "model.gguf", expert_count=1)

    with pytest.raises(ValueError, match="outside the declared expert count"):
        list(core._iter_weight_arrays(GGUFReader(str(path)), "mixtral", "float32"))


def test_a_duplicate_expert_tensor_is_refused():
    stack = core._ExpertStack(2)
    array = np.zeros((3, 4), dtype=np.float32)

    assert stack.add("0", "gate", 0, array) is None
    with pytest.raises(ValueError, match="Duplicate tensor for expert 0"):
        stack.add("0", "gate", 0, array)


def test_a_group_is_only_emitted_once_complete():
    """The stacker must not emit a partial stack under any arrival order."""
    stack = core._ExpertStack(3)
    array = np.ones((2, 2), dtype=np.float32)

    assert stack.add("0", "up", 2, array) is None
    assert stack.add("0", "up", 0, array) is None
    assert stack.leftovers() == ["layer 0 up: 2 of 3 experts"]
    emitted = stack.add("0", "up", 1, array)
    assert emitted is not None and emitted.shape == (3, 2, 2)
    assert stack.leftovers() == []


def test_stacking_requires_a_positive_expert_count():
    with pytest.raises(ValueError, match="positive expert count"):
        core._ExpertStack(0)


def test_the_planner_refuses_per_expert_tensors_without_a_stacker():
    """A wiring mistake must not fall through to an unmapped pass-through."""
    with pytest.raises(RuntimeError, match="need an expert stacker"):
        core._plan_tensor_emit(
            "blk.0.ffn_gate.0.weight", np.zeros((4, 4), dtype=np.float32),
            "mixtral", {"num_heads": 2, "num_key_value_heads": 1}, {}, {},
        )
