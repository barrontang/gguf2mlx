"""Cross-family MoE matrix: routers, fused gates, and where the experts live.

Each per-family suite proves its own adapter's arithmetic. This file asks the
question that only makes sense across families: for every MoE layout the
converter accepts, does the router land on the module mlx-lm actually
instantiates, and does the family's fused gating tensor have a destination?

That question has to be asked separately because the answers differ in ways no
single-family test can catch. Three module paths are in use, and choosing the
wrong one produces a directory that converts cleanly and then loads without its
MoE block:

    qwen2moe, qwen3moe, deepseek2, deepseek3, glm4moe, glm-dsa -> mlp.*
    mixtral                                                    -> block_sparse_moe.*
    gemma4                                                     -> router.*, experts.switch_glu.*

The shared-expert module is spelled two ways as well -- ``mlp.shared_expert``
(singular, Qwen2-MoE) and ``mlp.shared_experts`` (plural, DeepSeek/GLM) -- and
Qwen2-MoE's shared-expert *gate* is a separate ``shared_expert_gate`` linear
layer rather than part of either. Qwen3-MoE and Mixtral have no shared expert at
all, so the matrix records ``None`` for them instead of inventing a path; a
stray shared-expert tensor in one of those families is then left under its GGUF
name, where it is visible, rather than renamed into a module the runtime never
reads.

The expected names are pinned twice: as literals below, which keeps the mapping
reviewable, and against ``mlx_lm.models.<family>`` parameter trees, which is
what stops a plausible-looking table from being wrong.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from gguf2mlx import gguf2mlx as core

#: One row per accepted MoE layout. ``module`` is the decoder attribute that
#: holds the routed experts, ``router`` the linear layer that scores experts,
#: ``shared``/``shared_gate`` the optional shared-expert pair, and
#: ``fused_gate`` the (GGUF suffix, target) of a gating tensor whose shape does
#: not match any other family's.
FAMILIES: dict[str, dict[str, Any]] = {
    # Qwen2-MoE predates the plural spelling and keeps the shared-expert gate in
    # a separate linear layer, so its gate is a rank-2 matrix, not a scalar.
    "qwen2moe": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": "mlp.shared_expert",
        "fused_gate": ("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"),
        "correction_bias": None,
    },
    # Qwen3-MoE dropped the shared expert entirely.
    "qwen3moe": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": None,
        "fused_gate": None,
        "correction_bias": None,
    },
    # DeepSeek-V2 has no `noaux_tc` gate, but the MLA mapper is shared with V3,
    # so `exp_probs_b` maps the same way for all four families below. The tensor
    # only exists in GGUFs whose router is bias-corrected.
    "deepseek2": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": "mlp.shared_experts",
        "fused_gate": None,
        "correction_bias": "mlp.gate.e_score_correction_bias",
    },
    "deepseek3": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": "mlp.shared_experts",
        "fused_gate": None,
        "correction_bias": "mlp.gate.e_score_correction_bias",
    },
    "glm4moe": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": "mlp.shared_experts",
        "fused_gate": None,
        "correction_bias": "mlp.gate.e_score_correction_bias",
    },
    "glm-dsa": {
        "module": "mlp.switch_mlp",
        "router": "mlp.gate",
        "shared": "mlp.shared_experts",
        "fused_gate": None,
        "correction_bias": "mlp.gate.e_score_correction_bias",
    },
    # Mixtral is the outlier: its MoE block is a sibling of the MLP, not the MLP.
    "mixtral": {
        "module": "block_sparse_moe.switch_mlp",
        "router": "block_sparse_moe.gate",
        "shared": None,
        "fused_gate": None,
        "correction_bias": None,
    },
    # Gemma4's fused gating is a whole router submodule plus a per-layer scale,
    # and its experts are `switch_glu`, not `switch_mlp`. Its mapper also refuses
    # anything the table does not name, which is why it has no pass-through rows.
    "gemma4": {
        "module": "experts.switch_glu",
        "router": "router.proj",
        "shared": None,
        "fused_gate": ("ffn_gate_inp_s.weight", "router.scale"),
        "correction_bias": None,
        "strict": True,
    },
}

#: Families whose mapper rejects an unlisted tensor instead of passing it through.
STRICT = frozenset(arch for arch, row in FAMILIES.items() if row.get("strict"))

#: Where the layer index is placed in an expectation, chosen so a hardcoded
#: layer 0 in the converter would fail rather than pass.
LAYER = "3"

_YARN = {
    "type": "yarn",
    "factor": 1.0,
    "original_max_position_embeddings": 64,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}

#: Tiny arguments per family: one decoder layer and one expert group are enough
#: to enumerate the parameter tree. Literal on purpose -- a signature change in
#: the pinned mlx-lm must fail this file rather than silently reduce it.
#:
#: `glm-dsa` is absent because its `glm_moe_dsa` arguments need an indexer
#: configuration this matrix has no other use for; that module's resolution is
#: covered by `tests/test_runtime_registry.py`.
RUNTIME: dict[str, tuple[str, dict[str, Any]]] = {
    "qwen2moe": ("qwen2_moe", {
        "model_type": "qwen2_moe", "vocab_size": 16, "hidden_size": 8,
        "intermediate_size": 12, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 12,
        "shared_expert_intermediate_size": 12, "rms_norm_eps": 1e-5,
    }),
    "qwen3moe": ("qwen3_moe", {
        "model_type": "qwen3_moe", "vocab_size": 16, "hidden_size": 8,
        "intermediate_size": 12, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 12,
        "head_dim": 4, "max_position_embeddings": 64, "decoder_sparse_step": 1,
        "mlp_only_layers": [], "rms_norm_eps": 1e-5, "rope_theta": 1e6,
        "tie_word_embeddings": False, "norm_topk_prob": True,
    }),
    "mixtral": ("mixtral", {
        "model_type": "mixtral", "vocab_size": 16, "hidden_size": 8,
        "intermediate_size": 12, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "num_local_experts": 2, "num_experts_per_tok": 1,
    }),
    "deepseek2": ("deepseek_v2", {
        "vocab_size": 16, "hidden_size": 8, "intermediate_size": 12,
        "moe_intermediate_size": 12, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "n_shared_experts": 2, "n_routed_experts": 4, "num_experts_per_tok": 1,
        "kv_lora_rank": 4, "q_lora_rank": 4, "qk_rope_head_dim": 2,
        "v_head_dim": 4, "qk_nope_head_dim": 4, "first_k_dense_replace": 0,
        "moe_layer_freq": 1, "rope_scaling": _YARN,
    }),
    "deepseek3": ("deepseek_v3", {
        "vocab_size": 16, "hidden_size": 8, "intermediate_size": 12,
        "moe_intermediate_size": 12, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "n_shared_experts": 1, "n_routed_experts": 4, "num_experts_per_tok": 1,
        "kv_lora_rank": 4, "q_lora_rank": 4, "qk_rope_head_dim": 2,
        "v_head_dim": 4, "qk_nope_head_dim": 4, "first_k_dense_replace": 0,
        "moe_layer_freq": 1, "rope_scaling": _YARN,
    }),
    "glm4moe": ("glm4_moe", {
        "model_type": "glm4_moe", "vocab_size": 16, "hidden_size": 8,
        "intermediate_size": 12, "max_position_embeddings": 64,
        "moe_intermediate_size": 12, "norm_topk_prob": True,
        "num_attention_heads": 2, "n_group": 1, "head_dim": 4, "topk_group": 1,
        "n_shared_experts": 1, "n_routed_experts": 4, "routed_scaling_factor": 1.0,
        "num_experts_per_tok": 1, "first_k_dense_replace": 0,
        "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
        "rope_theta": 1e6, "rope_scaling": None, "use_qk_norm": True,
        "tie_word_embeddings": False, "attention_bias": False,
        "partial_rotary_factor": 1.0,
    }),
    "gemma4": ("gemma4_text", {
        "enable_moe_block": True, "num_experts": 2, "top_k_experts": 1,
        "moe_intermediate_size": 12, "num_hidden_layers": 1, "hidden_size": 8,
        "num_attention_heads": 2, "head_dim": 4, "global_head_dim": 4,
        "vocab_size": 16, "vocab_size_per_layer_input": 16,
        "hidden_size_per_layer_input": 8, "intermediate_size": 12,
        "num_kv_shared_layers": 0, "sliding_window": 4,
        "max_position_embeddings": 64, "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
    }),
}


def _mapped(name: str, arch: str) -> str | None:
    """Map ``name`` for ``arch``, treating a strict family's refusal as ``None``."""
    if arch in STRICT:
        try:
            return core._map_tensor_name(name, arch)
        except ValueError:
            return None
    return core._map_tensor_name(name, arch)


def _runtime_parameters(arch: str) -> set[str]:
    from mlx.utils import tree_flatten

    module_name, kwargs = RUNTIME[arch]
    module = importlib.import_module(f"mlx_lm.models.{module_name}")
    model = module.Model(module.ModelArgs(**kwargs))
    return {name for name, _ in tree_flatten(model.parameters())}


# --- the table itself --------------------------------------------------------


def test_the_matrix_covers_every_moe_architecture():
    """A new MoE family must decide where its experts live, not inherit one."""
    classified = set(core.MOE_ARCHES)
    declared = set(FAMILIES)
    assert classified <= declared, (
        "these architectures are classified as MoE but have no module decision "
        f"in this matrix: {sorted(classified - declared)}"
    )
    assert declared - classified == {"gemma4"}, (
        "the matrix has rows that are neither in MOE_ARCHES nor the Gemma4 MoE "
        f"layout: {sorted(declared - classified - {'gemma4'})}"
    )


def test_the_runtime_table_explains_its_gaps():
    """Every family is checked against mlx-lm except the one named here."""
    assert set(RUNTIME) <= set(FAMILIES)
    assert set(FAMILIES) - set(RUNTIME) == {"glm-dsa"}, (
        f"undocumented runtime gaps: {sorted(set(FAMILIES) - set(RUNTIME) - {'glm-dsa'})}"
    )


# --- routers -----------------------------------------------------------------


@pytest.mark.parametrize("arch", sorted(FAMILIES))
def test_the_router_lands_on_the_declared_module(arch):
    family = FAMILIES[arch]
    expected = f"model.layers.{LAYER}.{family['router']}.weight"

    assert core._map_tensor_name(f"blk.{LAYER}.ffn_gate_inp.weight", arch) == expected


@pytest.mark.parametrize("arch", sorted(FAMILIES))
def test_no_router_is_left_under_its_gguf_name(arch):
    """A router that survives unmapped leaves the block without expert scores."""
    mapped = _mapped(f"blk.{LAYER}.ffn_gate_inp.weight", arch)

    assert mapped is not None
    assert "blk." not in mapped
    assert mapped.endswith((".gate.weight", ".router.proj.weight"))


@pytest.mark.parametrize("arch", sorted(FAMILIES))
def test_the_noaux_tc_correction_bias_is_mapped_where_the_family_carries_it(arch):
    """Bias-corrected routers need their correction tensor, or routing drifts."""
    expected = FAMILIES[arch]["correction_bias"]
    mapped = _mapped(f"blk.{LAYER}.exp_probs_b", arch)

    if expected is None:
        # No destination: the tensor keeps its GGUF name (or is refused by a
        # strict family), which is the honest outcome for a router that has no
        # such parameter.
        assert mapped is None or mapped == f"blk.{LAYER}.exp_probs_b"
    else:
        assert mapped == f"model.layers.{LAYER}.{expected}"


# --- experts -----------------------------------------------------------------


@pytest.mark.parametrize("arch", sorted(FAMILIES))
@pytest.mark.parametrize("projection", ["gate", "up", "down"])
def test_stacked_experts_land_in_the_switch_module(arch, projection):
    family = FAMILIES[arch]
    expected = f"model.layers.{LAYER}.{family['module']}.{projection}_proj.weight"

    mapped = core._map_tensor_name(f"blk.{LAYER}.ffn_{projection}_exps.weight", arch)

    assert mapped == expected


@pytest.mark.parametrize("arch", sorted(FAMILIES))
@pytest.mark.parametrize("projection", ["gate", "up", "down"])
def test_a_shared_expert_is_only_named_where_the_family_has_one(arch, projection):
    """The plural/singular split is real: one spelling is a module nobody reads."""
    family = FAMILIES[arch]
    source = f"blk.{LAYER}.ffn_{projection}_shexp.weight"
    mapped = _mapped(source, arch)

    if family["shared"] is None:
        assert mapped is None or mapped == source
    else:
        assert mapped == f"model.layers.{LAYER}.{family['shared']}.{projection}_proj.weight"


def test_the_shared_expert_spellings_are_not_interchangeable():
    """Pin the two spellings so a refactor cannot normalise them into one."""
    singular = {
        arch for arch, row in FAMILIES.items() if row["shared"] == "mlp.shared_expert"
    }
    plural = {
        arch for arch, row in FAMILIES.items() if row["shared"] == "mlp.shared_experts"
    }

    assert singular == {"qwen2moe"}
    assert plural == {"deepseek2", "deepseek3", "glm4moe", "glm-dsa"}


# --- fused gating ------------------------------------------------------------


def test_the_fused_gate_table_lists_exactly_the_families_that_have_one():
    assert {
        arch for arch, row in FAMILIES.items() if row["fused_gate"] is not None
    } == {"qwen2moe", "gemma4"}


@pytest.mark.parametrize("arch", ["qwen2moe", "gemma4"])
def test_the_fused_gate_lands_on_its_target(arch):
    """Qwen2-MoE's shared-expert gate is a matrix; Gemma4's is a per-layer scale."""
    source, target = FAMILIES[arch]["fused_gate"]

    assert core._map_tensor_name(f"blk.{LAYER}.{source}", arch) == (
        f"model.layers.{LAYER}.{target}"
    )


@pytest.mark.parametrize(
    "arch", sorted(set(FAMILIES) - {"qwen2moe", "gemma4"} - STRICT)
)
def test_a_family_without_a_fused_gate_does_not_invent_one(arch):
    """Renaming this tensor for another family would map it to the wrong width."""
    source = f"blk.{LAYER}.ffn_gate_inp_shexp.weight"

    assert core._map_tensor_name(source, arch) == source


def test_the_strict_family_refuses_an_unnamed_gating_tensor():
    """Gemma4 refuses rather than leaking a Qwen2-MoE tensor into its artifact."""
    with pytest.raises(ValueError, match="Unsupported gemma4 tensor"):
        core._map_tensor_name(f"blk.{LAYER}.ffn_gate_inp_shexp.weight", "gemma4")


# --- against the pinned runtime ---------------------------------------------


@pytest.mark.parametrize("arch", sorted(RUNTIME))
def test_the_matrix_names_exist_in_the_pinned_runtime(arch):
    """The table is only worth anything if mlx-lm has the names it predicts."""
    pytest.importorskip("mlx", reason="mlx-lm's parameter names need Apple Silicon")
    tree = _runtime_parameters(arch)
    family = FAMILIES[arch]

    expected = {f"model.layers.0.{family['router']}.weight"}
    expected |= {
        f"model.layers.0.{family['module']}.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }
    if family["shared"] is not None:
        expected |= {
            f"model.layers.0.{family['shared']}.{projection}_proj.weight"
            for projection in ("gate", "up", "down")
        }
    if family["fused_gate"] is not None:
        expected.add(f"model.layers.0.{family['fused_gate'][1]}")

    assert expected <= tree, (
        f"{arch}: the matrix predicts parameters the pinned runtime does not "
        f"have: {sorted(expected - tree)}"
    )


@pytest.mark.parametrize("arch", sorted(RUNTIME))
def test_the_runtime_exposes_exactly_the_three_switch_projections(arch):
    """A fused switch projection in the runtime would leave a GGUF tensor homeless.

    Phi-3's dense MLP is a single ``gate_up_proj``; if a MoE family ever gained
    that shape, three separate ``ffn_*_exps`` tensors would have nowhere to go and
    the conversion would have to split them. Counting the runtime's own
    projections is what makes that visible before a real checkpoint hits it.
    """
    pytest.importorskip("mlx", reason="mlx-lm's parameter names need Apple Silicon")
    tree = _runtime_parameters(arch)
    prefix = f"model.layers.0.{FAMILIES[arch]['module']}."

    projections = {name[len(prefix):] for name in tree if name.startswith(prefix)}

    assert projections == {
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    }
