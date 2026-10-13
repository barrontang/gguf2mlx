"""Opt-in, offline Apple Silicon image-path parity against original HF-layout weights."""

from __future__ import annotations

import json
import os
import platform

import numpy as np
import pytest

from gguf2mlx import gguf2mlx as core
from tests.test_vlm_qwen2 import write_fixture

pytestmark = [
    pytest.mark.skipif(os.getenv("GGUF2MLX_RUN_VLM_E2E") != "1",
                       reason="Set GGUF2MLX_RUN_VLM_E2E=1 for native image-path tests."),
    pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                       reason="Requires Apple Silicon and tests/requirements-vlm.txt."),
]


def write_processor_assets(fixture, architecture):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(
        vocab={token: i for i, token in enumerate(fixture.tokens)}, unk_token="<unk>"
    ))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<unk>",
        additional_special_tokens=fixture.tokens[4:8],
    )
    image = "<image>" if architecture == "llava" else (
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    fast.chat_template = image + " word"
    fast.save_pretrained(fixture.root)
    processor = (
        {"processor_class": "LlavaProcessor", "patch_size": 4,
         "vision_feature_select_strategy": "default", "num_additional_image_tokens": 1}
        if architecture == "llava" else {"processor_class": "Qwen2VLProcessor"}
    )
    (fixture.root / "processor_config.json").write_text(json.dumps(processor))


@pytest.mark.parametrize("architecture", ["llava", "qwen2_vl"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("quantized_source", [False, True])
def test_native_load_image_logits_and_generation(
    tmp_path, architecture, quantized, dtype, quantized_source,
):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_model, prepare_inputs
    from PIL import Image

    fixture = write_fixture(tmp_path, architecture, quantized_source=quantized_source)
    write_processor_assets(fixture, architecture)
    assert core.convert(
        str(fixture.main), str(fixture.output), dtype, mmproj=str(fixture.mmproj),
        hf_model=str(fixture.root), offline=True, quantize=quantized, direct_quant=quantized,
    )
    # No monkeypatches, external weights, or network access: both load the same
    # randomly initialized checkpoint through the actual downstream loader.
    reference = load_model(fixture.root, strict=True)
    reference.set_dtype(mx.float16 if dtype == "float16" else mx.float32)
    converted, processor = load(str(fixture.output), strict=True)
    ref_params = dict(tree_flatten(reference.parameters()))
    out_params = dict(tree_flatten(converted.parameters()))
    for name, expected_weight in ref_params.items():
        if quantized and name.startswith("language_model.") and name.endswith("_proj.weight"):
            module = name.removesuffix(".weight")
            restored = mx.dequantize(
                out_params[name], out_params[module + ".scales"],
                out_params[module + ".biases"], group_size=64, bits=4,
            )
            np.testing.assert_allclose(np.array(restored), np.array(expected_weight), atol=0.03)
            packed = np.array(out_params[name])
            codes = ((packed[..., None] >> (np.arange(8) * 4)) & 15).reshape(expected_weight.shape)
            scales = np.array(out_params[module + ".scales"]).repeat(64, axis=-1)
            biases = np.array(out_params[module + ".biases"]).repeat(64, axis=-1)
            # Compare packed MLX matmul with independently unpacked float weights,
            # not an arbitrary quality threshold on an untrained random model.
            ref_params[name] = mx.array((codes * scales + biases).astype(dtype))
        else:
            np.testing.assert_array_equal(np.array(out_params[name]), np.array(expected_weight))
    reference.load_weights(list(ref_params.items()), strict=True)
    prompt = apply_chat_template(processor, fixture.config, "word", num_images=1)
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    logits, embeddings = [], []
    for image in images:
        inputs = prepare_inputs(processor, images=[image], prompts=prompt)
        ids, pixels = inputs["input_ids"], inputs["pixel_values"]
        kwargs = {key: value for key, value in inputs.items()
                  if key not in ("input_ids", "pixel_values", "attention_mask")}
        features = converted.get_input_embeddings(ids, pixels, **kwargs).inputs_embeds
        expected_features = reference.get_input_embeddings(ids, pixels, **kwargs).inputs_embeds
        np.testing.assert_allclose(np.array(features), np.array(expected_features), rtol=1e-5, atol=1e-6)
        output = converted(ids, pixels, mask=None, **kwargs).logits
        expected = reference(ids, pixels, mask=None, **kwargs).logits
        mx.eval(output, expected, features)
        actual = np.array(output)
        assert actual.shape[:2] == tuple(ids.shape)
        assert actual.shape[-1] == 16
        assert np.isfinite(actual).all()
        # FP16 fused dequant/matmul and materialized FP16 weights round at
        # different stages; allow eight unit-scale half-precision ULPs.
        tolerance = 8 * np.finfo(np.float16).eps if dtype == "float16" else 1e-5
        np.testing.assert_allclose(actual, np.array(expected), rtol=tolerance, atol=tolerance)
        logits.append(actual)
        embeddings.append(np.array(features))
    assert np.max(np.abs(embeddings[0] - embeddings[1])) > 1e-4
    assert np.max(np.abs(logits[0][:, -1] - logits[1][:, -1])) > 1e-4
    image_path = tmp_path / "red.png"
    images[0].save(image_path)
    result = generate(converted, processor, prompt, image=str(image_path),
                      max_tokens=3, temperature=0, verbose=False)
    assert 0 < result.generation_tokens <= 3
    assert isinstance(result.text, str)


def test_real_checkpoint_image_smoke(tmp_path):
    """Explicit opt-in for user-supplied matching GGUF/HF/image assets."""
    from pathlib import Path

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image, ImageOps

    source = os.getenv("GGUF2MLX_VLM_GGUF")
    if not source:
        pytest.skip("Set GGUF2MLX_VLM_GGUF, GGUF2MLX_VLM_HF and GGUF2MLX_VLM_IMAGE.")
    hf = os.environ["GGUF2MLX_VLM_HF"]
    image_path = os.environ["GGUF2MLX_VLM_IMAGE"]
    output = tmp_path / "real-vlm"
    assert core.convert(source, str(output), mmproj=os.getenv("GGUF2MLX_VLM_MMPROJ"),
                        hf_model=hf, offline=True)
    model, processor = load(str(output), strict=True)
    config = json.loads((output / "config.json").read_text())
    prompt = apply_chat_template(
        processor, config, "Describe the image.", num_images=1,
    )
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    features = []
    for picture in (image, ImageOps.invert(image)):
        inputs = prepare_inputs(processor, images=[picture], prompts=prompt)
        kwargs = {key: value for key, value in inputs.items()
                  if key not in ("input_ids", "pixel_values", "attention_mask")}
        embeddings = model.get_input_embeddings(
            inputs["input_ids"], inputs["pixel_values"], **kwargs,
        ).inputs_embeds
        logits = model(inputs["input_ids"], inputs["pixel_values"], mask=None, **kwargs).logits
        mx.eval(embeddings, logits)
        assert bool(mx.all(mx.isfinite(logits)).item())
        features.append(np.array(embeddings))
    assert np.max(np.abs(features[0] - features[1])) > 1e-4
    result = generate(model, processor, prompt, image=str(Path(image_path)), max_tokens=32,
                      temperature=0, verbose=False)
    assert result.generation_tokens > 0 and result.text.strip()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_native_source_preserving_q4_k(tmp_path, dtype):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_model, prepare_inputs
    from PIL import Image

    fixture = write_fixture(tmp_path, k_quantized_source=True, legacy=True)
    write_processor_assets(fixture, "qwen2_vl")
    assert core.convert(
        str(fixture.main), str(fixture.output), dtype, mmproj=str(fixture.mmproj),
        hf_model=str(fixture.root), offline=True, quantize=True, direct_quant=True,
        preserve_source_quant=True,
    )
    model, processor = load(str(fixture.output), strict=True)
    reference = load_model(fixture.root, strict=True)
    reference.set_dtype(mx.float32 if dtype == "float32" else mx.float16)
    parameters = dict(tree_flatten(model.parameters()))
    if dtype == "float16":
        # Isolate fused matmul rounding from the independently tested coefficient
        # cast: materialize the stored grids, not a newly quantized reference.
        restored = dict(tree_flatten(reference.parameters()))
        for name, weight in restored.items():
            stem = name.removesuffix(".weight")
            if stem + ".scales" not in parameters:
                continue
            codes = ((np.array(parameters[name])[..., None] >> (np.arange(8) * 4)) & 15)
            codes = codes.reshape(weight.shape)
            scales = np.array(parameters[stem + ".scales"]).repeat(32, axis=-1)
            biases = np.array(parameters[stem + ".biases"]).repeat(32, axis=-1)
            restored[name] = mx.array((codes * scales + biases).astype(np.float16))
        reference.load_weights(list(restored.items()), strict=True)
    assert parameters["language_model.model.layers.0.mlp.down_proj.weight"].dtype != mx.uint32
    prompt = apply_chat_template(processor, fixture.config, "word", num_images=1)
    for color in ("red", "blue"):
        inputs = prepare_inputs(processor, images=[Image.new("RGB", (8, 8), color)], prompts=prompt)
        kwargs = {"image_grid_thw": inputs["image_grid_thw"]}
        actual = model(inputs["input_ids"], inputs["pixel_values"], **kwargs).logits
        expected = reference(inputs["input_ids"], inputs["pixel_values"], **kwargs).logits
        mx.eval(actual, expected)
        tolerance = 8 * np.finfo(np.float16).eps if dtype == "float16" else 1e-5
        np.testing.assert_allclose(np.array(actual), np.array(expected),
                                   atol=tolerance, rtol=tolerance)
