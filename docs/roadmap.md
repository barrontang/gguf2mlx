# Roadmap status

> Completed work and the areas still open for contributors.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

Completed:

- MLX quantized output through bundled `mlx_lm.convert`
- MLX-LM load/integration tests on Apple Silicon
- strict adapter-based architecture validation
- fixture-backed Gemma, Phi-3 (4K and LongRoPE), and Phi-4 adapters
- GGUF tokenizer flags and embedded tokenizer JSON preservation
- reproducible conversion benchmark harness
- bounded-memory direct quantization pipeline (`--direct-quant`, affine 4-bit, dense and declared MoE adapters, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K–Q8_K/IQ4_NL/BF16 source qtypes)
- fail-closed MoE detection, default router protection, mixed-precision artifact output, and JSON conversion reports
- required tokenizer/chat-template CI plus scheduled opt-in two-family MoE validation
- StableLM conversion adapter (norm_eps, partial_rotary_factor, qk_layernorm, use_parallel_residual)
- `mlx_lm.load()` parity validation test for `--direct-quant` output (`tests/test_e2e.py`)
- opt-in real-GGUF convert + `mlx_lm.load()` finite-logit validation for gemma and phi3 fixtures (`tests/test_e2e.py`)
- strict LLaVA-1.5 and Qwen2-VL vision adapters, language-only direct quantization,
  and offline native image-path CI
- pinned Qwen2-VL-2B real-checkpoint conversion, all-tensor audit and four-path
  labeled image acceptance, including historical `qwen2vl_merger` compatibility
- `--eval-ppl` perplexity guardrail in the benchmark harness (reuses `mlx_lm.perplexity`; reports standard vs `--direct-quant` PPL delta)

Remaining areas for contributors:

- publish fixed-corpus perplexity numbers from `--eval-ppl` (standard vs `--direct-quant` vs `mlx_lm.convert`) so users can judge round-trip accuracy loss
- verify `gguf2mlx-mixed-v1` with a named JANG runtime; until then the format remains experimental
- extend the opt-in real-GGUF load test to the remaining conversion-enabled architectures (several are declared supported but have no end-to-end coverage; the `safe_open` bug showed how those paths silently rot)
- canonical chat-template fallback table for known model families when GGUF-embedded Jinja templates are missing or broken
- publish peak RSS and output-size comparison results from `benchmarks/benchmark_conversion.py --compare`
- evaluate GGUF→MLX conversion for high-demand non-text models (e.g. ASR/embedding) where competition is thin (pending verification that such models ship GGUF)
- publish additional architecture-specific tokenizer normalizer fixtures as upstream formats evolve
- add LLaVA-NeXT any-resolution/image-newline handling and distinct Qwen2.5/3.x
  vision adapters, including Qwen3.5's different language backbone, before
  claiming those families are supported
- expand beyond the named Qwen2-VL-2B diagnostic to larger, diverse image corpora
  and real LLaVA checkpoints; report quality separately from tiny-checkpoint parity

Vision format references:
[llama.cpp Qwen2-VL exporter](https://github.com/ggml-org/llama.cpp/blob/4413a192410fad1f054e9216114087a3e01527a0/conversion/qwenvl.py),
[GGUF tensor mapping](https://github.com/ggml-org/llama.cpp/blob/4413a192410fad1f054e9216114087a3e01527a0/gguf-py/gguf/tensor_mapping.py),
and the `mlx-vlm==0.7.6` loader/vision implementations exercised by the native tests.
