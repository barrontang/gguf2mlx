# Changelog

## Unreleased

### Quality gates that can actually fail and mean it

- Every conversion now writes `conversion_report.json` with a **weight-fidelity
  gate** (schema 2). Each written tensor is compared position-wise against an
  independent `gguf.quants.dequantize` decode of the source, with a per-dtype
  relative-L2 tolerance (fp16 6e-4, bf16 5e-3, fp32 1e-7). A tensor that moved
  or was mis-mapped fails the conversion instead of shipping silently.
- Fused, permuted and split tensors are recorded as `transformed` and reported
  in `conversion.skipped` rather than being counted as verified. A tensor marked
  `identity` whose stored shape no longer matches the source fails as a
  `layout_mismatch`.
- `gguf2mlx verify --model-dir DIR --weights --source model.gguf` replays the
  audit from the recorded `tensor_layout` without converting again.
  `gguf2mlx verify --model-dir DIR --load` is the numerical tier: it loads with
  `mlx_lm`, runs one forward pass and requires finite logits. Both exit non-zero
  on failure.
- Perplexity is downgraded to a **report-only** tier. Cross-runtime PPL mixes
  source quantization noise with conversion error, so it can no longer fail a
  build; see `docs/quality-gates.md`.
- `--weight-audit/--no-weight-audit` (default on) controls the extra pass, which
  roughly doubles conversion time.
- Quantized output carries the report forward with
  `artifact.stage = "mlx_lm_quantized"` and states explicitly that the 4-bit
  weights are not covered by the FP16 weight audit.

### Load matrix: "declared supported" is now testable

- `tests/load_matrix.json` pins real public GGUFs by repo, filename and SHA-256:
  Llama-3.2-1B, Qwen2.5-0.5B, Qwen3-0.6B, Gemma-2-2B and Phi-3-mini across
  Q4_0 / Q4_K_M / Q8_0 where published.
- `tests/test_load_matrix.py` (opt-in via `GGUF2MLX_LOAD_MATRIX=1`) downloads,
  hash-checks, converts, gates weights, then loads and runs each one.
- New `Load matrix` workflow runs the Q4_K_M subset nightly on Apple Silicon and
  the full sweep on manual dispatch.

### Stage 0: repository hygiene

- README reduced from 1,231 to 199 lines; the long-form material moved verbatim
  into `docs/` (support matrix, VLM, text encoder, publishing, benchmarks, FAQ,
  development, roadmap, validation logs).
- Repository traffic tracking (GPL-3.0-or-later workflow, dashboard, badge art
  and license text) moved to `barrontang/gguf2mlx-traffic`, removing the
  copyleft surface and the long-lived gist token from this Apache-2.0 repo.
- Publishing refuses PEP 440 pre-release versions on PyPI and restricts manual
  `workflow_dispatch` runs to `main`, so a development build cannot be uploaded.
- GitHub repository description and topics now describe the actual package.

### Trust and repository hygiene

- Conversion no longer guesses model dimensions: `build_config(..., strict=True)` raises
  with the missing `llama.*` / `<arch>.*` keys instead of silently falling back to
  4096/32/32/32000. `inspect` keeps the lenient behaviour.
- Missing `gguf` / `safetensors` now raise `ImportError` at import time instead of calling
  `sys.exit(1)`, so notebooks, CI wrappers and the publish pipeline can catch it;
  `python -m gguf2mlx` still exits 1 with a single-line message.
- GLM-5.2 `generation_config.json` is derived from the GGUF tokenizer metadata
  (`tokenizer.ggml.eos_token_id` / `eos_token_ids`) instead of hardcoded EOS ids and a
  fabricated `transformers_version`; the file is skipped when the metadata is absent.
- Affine 4-bit quantization rounds half away from zero (`std::round` semantics used by
  MLX) instead of `np.round`'s half-to-even, and rejects groups containing NaN/Inf with the
  offending tensor name instead of writing silent garbage.
- Version bumped to `2.1.5.dev0` so local builds no longer collide with the published
  2.1.4 wheel.
- Removed 1,211 tracked `rust/gguf2mlx-rs/target/` build artifacts from the repository.
- CI now exercises Python 3.10/3.12/3.13 and enforces an 80% coverage floor.

### Vision support

- Add a strict Qwen2-VL adapter for combined or split GGUFs: reconstruct fused
  vision QKV and temporal Conv3D kernels, retain the patch merger from GGUF,
  normalize HF configuration and validate M-RoPE, special tokens and processors.
- Extend direct affine 4-bit quantization to LLaVA and Qwen2-VL language
  projections while keeping vision/projector tensors floating-point. Reuse
  the existing quantizer; this is re-quantization, not lossless transpacking.
- Detect additional vision tensor/metadata prefixes and newer multimodal
  architecture IDs without enabling unsupported architecture fallbacks.
- Reject stale HF quantization configs, inconsistent image normalization and
  CLIP activation/normalization settings not honored by the tested MLX runtime.
- Add isolated pinned VLM test requirements and Apple Silicon CI covering actual
  MLX-VLM loading, image processing, feature/logit parity and generation with
  offline tiny F32/Q4_0 checkpoints; add an opt-in local-model image smoke test.
- Document the upstream LLaVA processor incompatibility between MLX-VLM 0.7.6
  and Transformers 5.19; isolate native tests rather than downgrading dev requirements.
- Support the historical `qwen2vl_merger` visual MLP naming and explicitly
  validated zero position placeholder. Merge HF `added_tokens_decoder` token
  definitions without relaxing conflict or vocabulary checks.
- Complete pinned Qwen2-VL-2B real-checkpoint acceptance: all 729 floating tensors
  audited, exact projected visual features against HF, and 12/12 labeled image
  checks for HF, GGUF-to-FP16 and llama.cpp. Additional language-only MLX 4-bit
  scores 11/12 exact matches; retain its longer photograph answer. Publish the offline
  four-path harness and scoped report, not a broad visual-quality certification.
- Correct the photograph failure diagnosis: the NASA suit is visibly orange;
  the uniform 4-bit answer was semantically correct, not a color hallucination.
  Report semantic garment/color checks separately from historical exact matches.
- Add explicit Qwen2-VL `--preserve-source-quant` with direct quantization:
  repack Q4_K's original codes/groups of 32 and keep other projections floating
  instead of lowering Q6_K to four bits. The pinned model passes all 12 original
  exact-answer checks; all-projection relative L2 error against decoded GGUF
  falls from 8.51% to 0.0614%. This is mixed precision (~2.7 GiB), not all-four-bit
  or bit-exact FP16 transpacking. Keep uniform quantization behavior unchanged.

### Fixes

- Keep dev-only CI portable: skip HF precision-control tests when optional
  MLX is absent, and run the Qwen BPE regression with `tokenizers` independently
  of the optional `transformers` loader integration.
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
