"""LLaVA-NeXT conversion: the anyres parameter, the tower's block arithmetic.

Three failure modes motivate this file, and the first two are silent:

1. **A dropped ``image_newline``.** It is the only LLaVA-NeXT tensor that is
   neither ``v.*`` nor ``mm.*``, so a converter that routes by prefix maps it to
   nothing and writes a directory that loads and stitches image tiles without the
   separator row the model was trained with.
2. **A tower one block too long.** Real ``mmproj`` files store 23 CLIP blocks for
   a 24-layer tower, because HF's ``vision_feature_layer = -2`` with 24 blocks is
   block 22's output, and block 23 only feeds the pooler. Emitting 24 layers asks
   for 32 tensors no file has; emitting 23 while keeping ``-2`` selects block 21
   and produces finite, wrong features. Both are covered here.
3. **A missing ``post_layernorm``.** mlx-vlm loads strictly and declares it, so it
   has to exist; the GGUF never has it. The value written is the identity, and the
   claim that this is unobservable is tested against mlx itself rather than
   asserted in a comment.

Failures are checked through ``vlm.convert_vlm`` (which raises) rather than the
CLI wrapper (which prints and returns ``False``), so a test asserts the reason,
not just the exit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors import safe_open

from gguf2mlx import vlm
from tests.llava_next_fixtures import (
    GRID,
    TOKENS,
    MetadataReader,
    clip_metadata,
    llava_next_config,
    plan_sources,
    text_metadata,
    write_llava_next_fixture,
)


def fixture(
    tmp_path,
    *,
    dims=None,
    config=None,
    plan_stored=None,
    write_stored=None,
    clip_overrides=None,
    include_post_norm=False,
    include_newline=True,
    hf_config=None,
    gguf_tokens=None,
    ffn_convention="mtmd",
):
    """Write a release whose plan and whose metadata can disagree on purpose."""
    config = config if config is not None else llava_next_config(dims)
    adapter, sources = plan_sources(
        config, dims=dims, stored_blocks=plan_stored,
        include_post_norm=include_post_norm, include_newline=include_newline,
        ffn_convention=ffn_convention,
    )
    written = write_llava_next_fixture(
        tmp_path, config, sources, dims=dims,
        stored_blocks=write_stored, clip_overrides=clip_overrides,
        gguf_tokens=gguf_tokens,
    )
    if hf_config is not None:
        (written.root / "config.json").write_text(json.dumps(hf_config))
    return written, adapter, sources


def run(written):
    """Convert through the real entry point, which raises on a refusal."""
    vlm.convert_vlm(
        str(written.main), written.output, "float32",
        mmproj=str(written.mmproj), hf_model=str(written.root), offline=True,
    )
    config = json.loads((written.output / "config.json").read_text())
    weights: dict[str, np.ndarray] = {}
    index = json.loads((written.output / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(written.output / shard), framework="np") as handle:
            for name in set(handle.keys()):
                weights[name] = handle.get_tensor(name)
    report = json.loads((written.output / "vlm_conversion_report.json").read_text())
    return config, weights, report


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    path = tmp_path_factory.mktemp("llava_next")
    written, adapter, _sources = fixture(path)
    config, weights, report = run(written)
    return written, adapter, config, weights, report


# -- the anyres parameter -----------------------------------------------------


def test_image_newline_is_emitted_outside_the_vision_tower(converted):
    """The one LLaVA-NeXT tensor a ``v./mm.`` filter would drop."""
    _written, adapter, _config, weights, report = converted

    assert "image_newline" in weights
    assert weights["image_newline"].shape == (adapter.text["hidden_size"],)
    assert weights["image_newline"].shape == adapter.source_shapes[
        "model.image_newline"
    ]
    assert report["tensor_counts"]["vision_gguf"] > 0


def test_a_release_without_image_newline_is_refused(tmp_path):
    """A LLaVA-1.5 mmproj must not convert as LLaVA-NeXT."""
    written, _adapter, _sources = fixture(tmp_path, include_newline=False)

    with pytest.raises(ValueError, match="image_newline"):
        run(written)


def test_grid_pinpoints_reach_the_output_config(converted):
    _written, _adapter, config, _weights, _report = converted

    assert config["model_type"] == "llava_next"
    assert config["image_grid_pinpoints"] == GRID
    assert config["vision_feature_select_strategy"] == "default"


# -- block arithmetic ---------------------------------------------------------


def test_a_short_tower_moves_the_feature_layer_to_the_last_block(converted):
    """23 stored blocks of a 24-layer tower: the feature is the last one."""
    _written, adapter, config, weights, report = converted

    assert config["vision_config"]["num_hidden_layers"] == 2
    assert config["vision_feature_layer"] == -1
    assert report["vision_layout"]["hf_vision_layers"] == 3
    assert report["vision_layout"]["stored_vision_blocks"] == 2
    layers = {
        name.split("encoder.layers.")[1].split(".")[0]
        for name in weights if "encoder.layers." in name
    }
    assert layers == {"0", "1"}
    assert adapter.vision["num_hidden_layers"] == 2


def test_a_complete_tower_keeps_the_configured_feature_layer(tmp_path):
    """A full export needs ``-2``; moving it would silently drop a real block."""
    written, _adapter, _sources = fixture(tmp_path, plan_stored=3, write_stored=3)
    config, weights, _report = run(written)

    assert config["vision_config"]["num_hidden_layers"] == 3
    assert config["vision_feature_layer"] == -2
    layers = {
        name.split("encoder.layers.")[1].split(".")[0]
        for name in weights if "encoder.layers." in name
    }
    assert layers == {"0", "1", "2"}


@pytest.mark.parametrize("stored", [1, 4])
def test_a_block_count_that_is_neither_the_tower_nor_one_short_is_refused(
    tmp_path, stored,
):
    written, _adapter, _sources = fixture(tmp_path, write_stored=stored)

    with pytest.raises(ValueError, match="CLIP blocks"):
        run(written)


# -- the post norm ------------------------------------------------------------


def test_an_absent_post_norm_is_written_as_the_identity_and_recorded(converted):
    _written, adapter, _config, weights, report = converted

    width = adapter.vision["hidden_size"]
    np.testing.assert_array_equal(
        weights["vision_tower.vision_model.post_layernorm.weight"], np.ones(width)
    )
    np.testing.assert_array_equal(
        weights["vision_tower.vision_model.post_layernorm.bias"], np.zeros(width)
    )
    assert report["tensor_counts"]["vision_identity_filled"] == 2
    reason = report["vision_identity_fills"][
        "vision_tower.vision_model.post_layernorm.weight"
    ]
    assert "pooler" in reason


def test_a_present_post_norm_is_copied_rather_than_replaced(tmp_path):
    written, adapter, _sources = fixture(tmp_path, include_post_norm=True)
    _config, weights, report = run(written)

    np.testing.assert_allclose(
        weights["vision_tower.vision_model.post_layernorm.weight"],
        np.full(adapter.vision["hidden_size"], 1.5),
    )
    np.testing.assert_allclose(
        weights["vision_tower.vision_model.post_layernorm.bias"],
        np.full(adapter.vision["hidden_size"], 0.25),
    )
    assert report["tensor_counts"]["vision_identity_filled"] == 0
    assert report["vision_identity_fills"] == {}


# -- fail-closed --------------------------------------------------------------


@pytest.mark.parametrize("override,message", [
    ({"clip.projector_type": "qwen2vl_merger"}, "projector type"),
    ({"clip.use_gelu": True}, "quick_gelu"),
    ({"clip.has_siglip": True}, "vision encoder"),
    ({"clip.vision.image_aspect_ratio": "pad"}, "image_aspect_ratio"),
    ({"clip.vision.mm_patch_merge_type": "flat"}, "mm_patch_merge_type"),
    ({"clip.vision.image_grid_pinpoints": [8, 32, 32, 8]}, "grid_pinpoints"),
    ({"clip.vision.image_crop_resolution": 0}, "must be positive"),
])
def test_vision_metadata_that_is_not_this_layout_is_refused(tmp_path, override, message):
    written, _adapter, _sources = fixture(tmp_path, clip_overrides=override)

    with pytest.raises(ValueError, match=message):
        run(written)


def test_a_legacy_crop_resolution_is_recorded_rather_than_refused(tmp_path):
    """Real llava-v1.6 exports declare crop 224 against a 336 tile.

    Neither ``image_crop_resolution`` nor ``image_split_resolution`` reaches the
    runtime: tiles are cut by ``preprocessor_config.json``, whose crop size and grid
    ``validate_processor`` already pins to ``config.json``. Refusing a file that
    llama.cpp itself runs would be theatre, so the pair is recorded as evidence.
    """
    written, _adapter, _sources = fixture(
        tmp_path, clip_overrides={"clip.vision.image_crop_resolution": 6},
    )

    _config, _weights, report = run(written)

    assert report["vision_layout"]["gguf_image_crop"] == {
        "crop_resolution": 6, "split_resolution": None, "processor_tile": 8,
    }


@pytest.mark.parametrize("mutation,message", [
    ({"image_grid_pinpoints": None}, "anyres marker"),
    ({"vision_feature_select_strategy": "full"}, "feature selection"),
    ({"use_image_newline_parameter": False}, "image_newline"),
    ({"mm_projector_type": "linear"}, "mlp2x_gelu"),
    ({"image_token_index": None}, "image_token_index"),
])
def test_hf_configs_this_adapter_does_not_implement_are_refused(
    tmp_path, mutation, message,
):
    written, _adapter, _sources = fixture(
        tmp_path, hf_config={**llava_next_config(), **mutation},
    )

    with pytest.raises(ValueError, match=message):
        run(written)


def test_a_mistral_text_half_is_refused_with_its_own_reason(tmp_path):
    config = llava_next_config()
    text = {**config["text_config"], "model_type": "mistral"}
    written, _adapter, _sources = fixture(
        tmp_path, hf_config={**config, "text_config": text},
    )

    with pytest.raises(ValueError, match="sliding-window|mistral"):
        run(written)


def test_a_legacy_flat_llava_config_is_refused_with_a_next_step(tmp_path):
    written, _adapter, _sources = fixture(tmp_path, hf_config={
        "model_type": "llava_next",
        "architectures": ["LlavaNextForConditionalGeneration"],
        "image_grid_pinpoints": GRID,
        "image_token_index": 15,
        "vocab_size": 16,
    })

    with pytest.raises(ValueError, match="nested text_config and vision_config"):
        run(written)


def test_a_processor_without_anyres_is_refused(tmp_path):
    written, _adapter, _sources = fixture(tmp_path)
    preprocessor = json.loads((written.root / "preprocessor_config.json").read_text())
    preprocessor.pop("aspect_ratio_setting")
    (written.root / "preprocessor_config.json").write_text(json.dumps(preprocessor))

    with pytest.raises(ValueError, match="aspect_ratio_setting"):
        run(written)


def test_a_siglip_tower_is_named_as_the_reason(tmp_path):
    config = llava_next_config()
    vision = {**config["vision_config"], "model_type": "siglip_vision_model"}
    written, _adapter, _sources = fixture(
        tmp_path, hf_config={**config, "vision_config": vision},
    )

    with pytest.raises(ValueError, match="CLIP vision"):
        run(written)


# -- language half -----------------------------------------------------------


def test_language_dimensions_come_from_the_gguf_when_the_config_omits_them(converted):
    """A real llava_next config.json has no text hidden_size; the GGUF does."""
    written, adapter, config, weights, _report = converted

    assert "hidden_size" not in written.config["text_config"]
    assert config["text_config"]["hidden_size"] == adapter.text["hidden_size"]
    assert config["text_config"]["num_hidden_layers"] == adapter.text["num_hidden_layers"]
    assert "language_model.model.embed_tokens.weight" in weights


def test_a_gguf_that_disagrees_with_a_declared_dimension_is_refused(tmp_path):
    config = llava_next_config()
    written, _adapter, _sources = fixture(
        tmp_path,
        hf_config={**config, "text_config": {**config["text_config"], "hidden_size": 24}},
    )

    with pytest.raises(ValueError, match="incompatible with HF hidden_size"):
        run(written)


# -- runtime conformance ------------------------------------------------------


def _runtime_config(config: dict):
    """Build mlx-vlm's own config objects from a converted directory's config."""
    from mlx_vlm.models.llava_next.config import ModelConfig, TextConfig, VisionConfig

    def pick(cls, values: dict):
        fields = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in fields})

    return ModelConfig(
        model_type="llava_next",
        text_config=pick(TextConfig, config["text_config"]),
        vision_config=pick(VisionConfig, config["vision_config"]),
        image_token_index=config["image_token_index"],
        vision_feature_layer=config["vision_feature_layer"],
        vision_feature_select_strategy=config.get(
            "vision_feature_select_strategy", "default"
        ),
        vocab_size=config["vocab_size"],
    )


def test_emitted_names_are_exactly_the_runtime_parameter_tree(converted):
    """Name conformance against mlx-vlm itself, not against a copied convention."""
    pytest.importorskip("mlx_vlm", reason="needs the pinned VLM runtime")
    from mlx.utils import tree_flatten
    from mlx_vlm.models.llava_next import Model

    _written, _adapter, config, weights, _report = converted
    expected = {name for name, _ in tree_flatten(Model(_runtime_config(config)).parameters())}
    emitted = set(weights)

    assert not emitted - expected, (
        f"these tensors would be ignored by mlx-vlm: {sorted(emitted - expected)[:8]}"
    )
    assert not expected - emitted, (
        "mlx-vlm loads strictly and needs these tensors, which conversion dropped: "
        f"{sorted(expected - emitted)[:8]}"
    )


def test_the_identity_post_norm_cannot_change_the_image_features(converted):
    """The claim behind the fill: no live code path reads that parameter.

    ``llava_next`` computes ``post_layernorm`` on the CLS slot of the pooler
    output, and ``get_input_embeddings`` takes ``hidden_states[layer]`` instead.
    Perturbing the norm must therefore leave the features exactly where they were.
    """
    import mlx.core as mx
    pytest.importorskip("mlx_vlm", reason="needs the pinned VLM runtime")
    from mlx_vlm.models.llava_next import Model

    _written, adapter, config, _weights, _report = converted
    vision = adapter.vision
    tower = Model(_runtime_config(config)).vision_tower
    side = vision["image_size"]
    pixels = mx.array(
        np.random.default_rng(0).normal(0, 0.3, (1, side, side, 3)).astype(np.float32)
    )

    def features() -> np.ndarray:
        *_ignored, hidden = tower(pixels, output_hidden_states=True)
        return np.array(hidden[-1])

    before = features()
    width = vision["hidden_size"]
    tower.vision_model.post_layernorm.weight = mx.full((width,), 7.0, mx.float32)
    tower.vision_model.post_layernorm.bias = mx.full((width,), -3.0, mx.float32)
    after = features()

    np.testing.assert_array_equal(before, after)


# -- the CLIP MLP has two naming conventions ----------------------------------


@pytest.mark.parametrize("convention,expansion", [
    ("mtmd", "ffn_up"),
    ("llava-tools", "ffn_down"),
])
def test_each_clip_ffn_convention_maps_its_expansion_to_fc1(
    tmp_path, convention, expansion,
):
    """The names alone cannot decide the CLIP MLP, so the shapes do.

    llama.cpp's LLaVA-era loader reads ``ffn_down`` as ``ff_i`` -- the GELU input --
    and ``ffn_up`` as the output projection; the modern mtmd graph reads ``ffn_up``
    as the expansion. Every pre-mtmd ``mmproj``, llava-1.5 and llava-1.6 alike, is
    of the first kind, and mapping it by name produces a directory that loads and
    then scores images through a swapped MLP.
    """
    written, _adapter, sources = fixture(tmp_path, ffn_convention=convention)

    _config, weights, report = run(written)

    np.testing.assert_array_equal(
        weights["vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"],
        sources[f"v.blk.0.{expansion}.weight"],
    )
    assert report["vision_layout"]["clip_ffn_convention"] == convention


def _configure(tensors, dims=None):
    from gguf2mlx.vision_llava import LlavaNextAdapter

    adapter = LlavaNextAdapter(llava_next_config(dims))
    adapter.validate_text_metadata(MetadataReader(text_metadata(dims)))
    adapter.configure_layout([MetadataReader(clip_metadata(dims), tensors)])
    return adapter


@pytest.mark.parametrize("tensors,dims,message", [
    ((("v.blk.0.ffn_up.weight", (8, 12)),), None, "missing v.blk.0.ffn_down.weight"),
    (
        (("v.blk.0.ffn_up.weight", (12, 8)), ("v.blk.0.ffn_down.weight", (8, 8))),
        None, "match neither known convention",
    ),
    (
        (("v.blk.0.ffn_up.weight", (8, 8)), ("v.blk.0.ffn_down.weight", (8, 8))),
        {"vision_ffn": 8}, "cannot tell the CLIP ffn tensors apart",
    ),
])
def test_an_unresolvable_clip_mlp_is_refused(tensors, dims, message):
    with pytest.raises(ValueError, match=message):
        _configure(tensors, dims)


# -- the text half has to be the same checkpoint ------------------------------


def test_a_gguf_without_the_image_token_is_refused_with_both_counts(tmp_path):
    """A repack that trimmed the multimodal row cannot drive mlx-vlm.

    This is the shape of the real failure: llama.cpp's LLaVA exports carry the
    LLaVA-1.6 vocabulary (32000 tokens, no ``<image>``) while ``llava-hf``'s
    checkpoint declares 32064 rows with ``image_token_index = 32000``. mlx-vlm
    embeds the prompt before splicing image features, so ID 32000 has to exist.
    """
    written, _adapter, _sources = fixture(tmp_path, gguf_tokens=TOKENS[:-1])

    with pytest.raises(ValueError) as error:
        run(written)

    message = str(error.value)
    assert "declares 15 tokens" in message
    assert "vocab_size 16" in message
    assert "15=<image>" in message
    assert "config.image_token_index=15" in message


def test_a_gguf_with_rows_the_hf_tokenizer_never_names_is_refused(tmp_path):
    written, _adapter, _sources = fixture(tmp_path, gguf_tokens=[*TOKENS, "x", "y"])

    with pytest.raises(ValueError) as error:
        run(written)

    message = str(error.value)
    assert "declares 18 tokens" in message
    assert "vocab_size 16" in message
    assert "have no HF token" in message
