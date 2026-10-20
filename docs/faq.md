# Frequently asked questions

> Short answers to the questions that come up most often.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

### How do I convert a GGUF model to MLX on a Mac?

Install `gguf2mlx[mlx]`, then run `gguf2mlx convert --input model.gguf
--output model-mlx`. The output directory can be passed to `mlx_lm.load()` when
the GGUF architecture and variant appear in the supported matrix.

### Does GGUF-to-MLX conversion restore FP16 model quality?

No. Dequantization expands the stored values into FP16 or FP32, but it cannot
recover information removed when the source GGUF was quantized.

### Does the 4-bit output preserve the original GGUF Q4 blocks?

Not by default. MLX-LM and the uniform `--direct-quant` path perform a second
quantization; the latter skips the FP16 staging directory. See
`docs/direct-quant-transcoding.md`. For Qwen2-VL with Q4_K projections,
`--quantize --direct-quant --preserve-source-quant` instead keeps those codes
and 32-element groups, while other projections remain floating-point.
Coefficient rounding to the output dtype still applies; see the
[real-checkpoint report](docs/qwen2-vl-real-validation.md).

### Why is the converted model larger than the GGUF file?

A quantized GGUF stores only a few bits per weight plus block metadata. Plain
conversion writes FP16 or FP32 safetensors, so a larger output is expected.

### Are Gemma and Phi supported?

Gemma, Gemma 2, Gemma 3, and text-only Gemma 4 have conversion adapters. Gemma 4
multimodal/audio tensors are not part of the text-only output. Phi-3 (including
LongRoPE) and Phi-4 text GGUFs have adapters; Phi-4 uses MLX-LM's Phi-3 model type.
Phi-2, Phi-MoE, and Phi-4 vision conversion remain unsupported.

### Is MLX always faster than llama.cpp?

No universal multiplier is claimed. Use the same model quality, prompt, context,
sampling settings, and hardware when comparing runtimes.
