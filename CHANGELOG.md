# Changelog

## Unreleased

### Fixes

- Rebuild first-generation Gemma SentencePiece tokenizers as score-ranked BPE,
  preserving GGUF whitespace-prefix behavior, Unicode, token IDs and BOS/EOS flags.
- Use the public safetensors `keys()` API when finalizing shard indexes, with a
  real multi-shard regression test.
- Preserve NumPy row-major dense GGUF layouts instead of reshaping to GGML
  axis order and transposing, which scrambled dense matrices.
- Ship a local Gemma runtime adapter honoring `gelu_pytorch_tanh`; stock
  MLX-LM 0.31.3 uses erf GELU regardless of that configuration.
- Pack direct affine 4-bit weights as MLX-compatible uint32 words instead of
  uint8 byte arrays, including stacked expert matrices.
- Restore Qwen2MoE shared-expert tensor mappings, reshape its scalar gate to
  a linear matrix, and infer legacy GGUF expert dimensions from tensor shapes
  when explicit metadata is absent; reject conflicting metadata.
- Permit IQ4_NL source weights used inside mixed Q2_K MoE GGUF checkpoints.
- Preserve Qwen GGUF decomposed Unicode and omit a fabricated unknown token
  when GPT-2-style metadata has none, avoiding special-token registration of `!`.

### Validation

- Add portable Gemma tokenizer regressions and an opt-in native llama.cpp
  comparison (`GGUF2MLX_GEMMA_GGUF` with `GGUF2MLX_RUN_E2E=1`).
- Complete public Gemma Q4_0 conversion and zero-tolerance PPL evaluation on Mac.
  Tokenizer parity passes; PPL parity fails (absolute delta 0.41424293554212).
- Add native Q4_0 decode auditing and explicitly labeled unquantized controls.
  Full Q4 decode audit has zero bit differences; aligned dense FP32 delta is
  0.0007508356276275663, HF-mirror-derived FP16 delta is 0.08708103047850102.
  Original Q4_0 CPU versus repaired default FP16 MLX delta is
  0.4417528951986185. All strict PPL gates remain failed.
- Add synthetic Qwen2MoE standard/direct load regressions and extend real
  MoE validation to Qwen1.5. Measure router logits during the actual forward
  pass rather than applying the gate directly to raw embeddings. Optional
  environment variables persist converted models and JSON validation reports.
- Validate public Qwen1.5-MoE-A2.7B-Chat Q2_K on Apple Silicon: mixed policy
  validation, uniform MLX load, finite logits, actual routing entropy,
  chat-template generation and native tokenizer parity all pass.

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
