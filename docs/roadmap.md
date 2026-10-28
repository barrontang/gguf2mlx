# Roadmap status

> Completed work and the areas still open for contributors.

> Split out of the top-level README on 2026-10-20; heading levels unchanged, entries
> updated since to match what has shipped.

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
- three-tier fidelity gate (`exact` / `inexact` / lossless), the `verify` command and
  the documented `conversion-report.json` schema
- `--quant-source preserve|requant` re-expressing GGUF `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`
  blocks on MLX's affine grid instead of requantizing decoded floats, with a
  per-tensor audit in the conversion report (`requant` remains the default);
  supported for llama, qwen2, qwen3, mistral and dense Qwen3.5, fail-closed
  elsewhere
- `gguf2mlx publish`: split-GGUF discovery and merge, bounded-memory streaming,
  provenance record, license copy and pinned converter version, plus `--dry-run`
  and `--verify-load` so a publish can be reviewed before it becomes visible;
  `publish-mlx.yml` reduced to a thin shell
- pinned load matrix converting, weight-gating and `mlx_lm.load()`-ing 12 entries
  across llama, qwen2, qwen3, gemma2, phi3 and Qwen3.5 — green locally on Apple
  Silicon (2026-10-26); the last published CI run covered the original 11 (run
  39505897872)
- a runtime registry contract (`mlx-lm>=0.31.3`, `mlx-vlm==0.7.10`) checked by
  `tests/test_runtime_registry.py` and the `Runtime contract` workflow, so an
  emitted `model_type` cannot silently outrun the installed runtime
- CLI completion: `--force` for a non-empty output directory, `--json` for a
  machine-readable result, and a `logging` channel on stderr (`-v`/`-q`) that
  keeps stdout clean for `--json`
- the declared-versus-covered architecture gap is now explicit and test-enforced
  (`DECLARED_COVERAGE_GAPS`), instead of implied by prose
- deleted `rust/gguf2mlx-rs`: an unbuilt, unpublished pyo3 crate whose behaviour
  the Python path already implemented

Remaining areas for contributors:

- publish fixed-corpus perplexity numbers from `--eval-ppl` (standard vs `--direct-quant` vs `mlx_lm.convert`) so users can judge round-trip accuracy loss
- verify `gguf2mlx-mixed-v1` with a named JANG runtime; until then the format remains experimental
- extend the pinned load matrix beyond its five architectures (llama, qwen2, qwen3,
  gemma2, phi3): the gap is no longer silent — `DECLARED_COVERAGE_GAPS` lists every
  convertible architecture without a case and a test keeps that list exact — but
  each entry still needs a pinned GGUF and a real `mlx_lm` load
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
and the `mlx-vlm==0.7.10` loader/vision implementations exercised by the native tests.
