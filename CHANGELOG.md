# Changelog

## 2.1.1 - 2026-09-21

### Fixes

- Fix `'builtins.safe_open' object is not iterable` crash during weight extraction on safetensors 0.8.0+. The `_finalize_safetensor_shards` function iterated directly over a `safe_open` context manager, which is not iterable in safetensors 0.8.0; changed to `f.keys()`. This bug prevented any real GGUF conversion from completing — all shard-output paths failed at the index-building step. Affects all architectures including stablelm (reported in #14).

### Tests

- Add opt-in real-GGUF end-to-end load and finite-logit validation for gemma and phi3 fixtures (`GGUF2MLX_RUN_E2E=1`). These tests exercise the full `core.convert()` path without monkeypatching and would have caught the safe_open bug on first run.

### Features

- Widen `--direct-quant` architecture support: llama, gemma, mistral, qwen2, stablelm (was llama, gemma only).
- Widen `--direct-quant` source quantization types to 12 qtypes: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, BF16 (was Q4_0, Q5_1 only).
- Add StableLM architecture support (`stablelm`): norm_eps, partial_rotary_factor, qk_layernorm, use_parallel_residual.
- Add `--eval-ppl` perplexity quality guardrail to benchmark harness, reusing `mlx_lm.perplexity`; reports PPL delta between standard and `--direct-quant` paths. New flags: `--ppl-dataset`, `--ppl-num-samples`, `--ppl-seq-len`.
- Add `--compare` mode to benchmark harness: runs both standard and `--direct-quant` paths, prints peak RSS / output size / elapsed delta table, emits JSON.

### Documentation

- Update README roadmap to reflect completed work and research-backed next priorities: sensitivity-aware bit retention, per-arch e2e coverage, chat-template fallback, non-text model evaluation.

---

## 2.1.0 - (initial PyPI release)

- Initial public release.
