# Reproducible benchmarks

> Recording conversion time, peak RSS, sizes and an optional perplexity guardrail.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

The repository does not claim that MLX is universally faster than llama.cpp.
Conversion and inference performance depend on model architecture, quantization,
prompt length, hardware, thermals, and library versions.

Run the included benchmark harness to record conversion time, peak RSS, input
size, output size, platform information, and installed package versions:

```bash
uv run benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx \
  --result-json ./benchmark-results/model.json
```

To benchmark the complete GGUF-to-4-bit-MLX path:

```bash
uv run --extra mlx benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx-4bit \
  --quantize \
  --result-json ./benchmark-results/model-4bit.json
```

Plain conversion output is dequantized and can be substantially larger than the
original GGUF quantized file. Use `--quantize --q-bits 4 --q-group-size 64` when
you want compact MLX-LM quantized output.

To attach a quality guardrail, add `--eval-ppl`: after conversion the harness
loads each output with `mlx_lm` and measures perplexity on a fixed corpus, so you
can see how much accuracy the GGUF-to-MLX round trip costs relative to a baseline.
Perplexity failures never abort the benchmark; they are recorded as
`success: false` in the JSON.

```bash
uv run --extra mlx benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx-4bit \
  --quantize --compare --eval-ppl \
  --ppl-dataset wikitext --ppl-num-samples 32 --ppl-seq-len 512 \
  --result-json ./benchmark-results/model-4bit.json
```

## Choosing the right tool

| Starting point | Goal | Recommended tool |
|---|---|---|
| GGUF model | Run the GGUF directly | llama.cpp or a GGUF application |
| Original Hugging Face model | Create an MLX model | `mlx_lm.convert` |
| GGUF-only model release | Create an MLX-LM directory | `gguf2mlx` |
| Existing MLX model directory | Create a portable archive | `gguf2mlx package` |
