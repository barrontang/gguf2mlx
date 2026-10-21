# Changelog

## Unreleased

### `preserve` stops expanding Q6_K to FP16

`--quant-source preserve` re-expresses GGUF blocks on MLX's affine grid and
reuses the source integer codes. For Q6_K that only works when a 32-value
group's two 16-value sub-block scales agree — true for 0.7% of groups in a real
model. Every other Q6_K tensor was decoded to FP16, turning 0.82 bytes/value
into 2: **a Q6_K file came out 2.4x the size of the GGUF it came from**, which
is the opposite of the mode's purpose.

Q6_K is now re-fitted at the source width and emitted as 6-bit affine groups:

| Fallback | Q6_K artifact | Relative L2 |
|---|---|---|
| `6bit` (default) | 1.067x the source bytes | 1.7-1.8e-2 |
| `8bit` | 1.371x | 4.4-4.9e-3 |
| `float16` | 2.4x (2.44 bytes/value against 0.82) | 0 |

Measured on Llama-3.2-1B (bartowski). On the **Q4_K_M** file, whose Q6_K
tensors dominate `ffn_down` and `attn_v`, the artifact goes from 1.662x the
GGUF to **1.092x**. All four artifacts load in `mlx_lm` with finite logits.
Fitting is per group — the lower squared error of the source lattice and a
least-squares refit from two starting points — and groups whose pair scales
match still keep their codes bit for bit.

### Audit changes

- The preserve gate is split into tiers. `exact` (codes copied) is held to
  1e-3; `inexact` (re-fitted) is held to the ceiling registered for its width.
  The gate passes only when both do, so a re-fit can neither fail the fp16
  bound nor hide behind a loose one.
- `--preserve-fallback {6bit,8bit,float16}` selects the width.
- `conversion_report.json` records `preserve.tiers`, `preserve.size`
  (source/output bytes and ratio) and `preserve.fallback`; schema 3.
- `preserve` plans now record the *source* block type in `qtype` and the target
  width in `bits`, so `preserved_by_qtype` answers "what came in".

### Fixed

- Preserve conversions of any Llama-3 GGUF wrote `rope_freqs.weight` as a float
  tensor, so the artifact failed to load ("Received 1 parameters not in model").
  `build_config` already reproduces that table as a verified `rope_scaling`
  block; the preserve path now skips the tensor like the float path does.
- The Q6_K mapping is chunked, so an embedding table with a quarter of a
  billion values no longer allocates three full-size float32 temporaries.

## 2.2.0 - 2026-10-21

### Type checking is now a gate, not a configuration file

- `mypy src` runs in CI on Python 3.12 and must report **zero** errors (it
  previously reported 78 across 8 files and was never executed).
- `[tool.mypy]` targets 3.12 because numpy's bundled stubs use PEP 695 `type`
  statements that a 3.10 target rejects as a syntax error. 3.10/3.13 runtime
  compatibility stays covered by the test matrix.
- Fixing the errors removed three real hazards: `mlx_lm.load()` is now indexed
  instead of unpacked into exactly two names, the `streaming.py` writer no
  longer rebinds its `dtype` parameter to `np.dtype`, and MiniCPM-V validation
  checks for a missing preprocessor config instead of assuming one exists.

### Documentation links actually resolve

- Eight relative links in `docs/` pointed at `docs/...` or repo-root paths from
  inside `docs/`, so they 404'd on GitHub (`docs/docs/qwen2-vl-real-validation.md`,
  `docs/.github/workflows/publish-mlx.yml`, …). All internal links now resolve;
  `docs/validation/` entries use `../`.

### Scheduled MoE validation reports its real failure

- `Weekly MoE validation` had been failing for three weeks with a bare
  `JSONDecodeError`: an unset `GGUF2MLX_MOE_MODELS` repository variable arrives
  as an empty string, which is not valid JSON. It now parses `""` as "nothing
  configured" and exits with the variable name and expected format.

### What the first full load-matrix sweep caught

The matrix had never executed. The first sweep of all thirteen pinned cases
passed 4 of 11 end-to-end conversions and exposed four defects that no unit
test could see:

- **llama3 rope scaling was being dropped.** llama.cpp ships llama3 scaling as
  a `rope_freqs.weight` table with no `llama.rope.scaling.*` metadata. The
  mapper passed the tensor through unchanged, so `mlx_lm` rejected the model
  ("Received 1 parameters not in model"), and the config had no scaling at all.
  The table is now read, the equivalent `rope_scaling` block is inferred and
  **verified by regenerating the table**, and the tensor is not emitted. A
  table that cannot be reproduced fails the conversion instead of rotating
  positions differently.
- **Qwen3-0.6B got an `lm_head` it never had.** `build_config` forced
  `tie_word_embeddings=False` for qwen3 while the file ships no `output.weight`.
  Tie/untie is now derived from the tensor list for every architecture
  ("Missing 1 parameters: lm_head.weight").
- **Gemma 2/3 lost two of their four block norms.** The Llama mapper maps
  `ffn_norm` to `post_attention_layernorm`, which is the Gemma-1 layout;
  Gemma 2/3 needs `post_attention_norm` and `post_ffw_norm` as well, and its
  `ffn_norm` is the *pre*-FFN norm. Both post norms also need the same `-1`
  de-shift as the others (llama.cpp stores `1 + w`).
- **The job ran out of disk, not out of correctness.** A 4 GB GGUF becomes an
  8 GB FP16 tree, and one `macos-14` runner has ~14 GB. The workflow now fans
  out one job per (architecture, qtype) and the test drops the converted tree
  after each case while keeping the report.

`docs/quality-gates.md` records the symptom → cause → fix table, and the
workflow is a real matrix (`plan` job → `strategy.matrix`) rather than a loop.

### Version

- `2.2.0` replaces `2.1.5.dev0`. `publish-mlx.yml`'s `converter_version`
  default moves to `2.2.0`, the first published release that ships the
  `publish` subcommand.

### Preserve the GGUF quantization instead of requantizing

- New `--quant-source preserve|requant` on `convert`. `requant` remains the
  default, so existing conversions are unchanged; `preserve` re-expresses the
  source GGUF blocks as MLX affine weights on a 32-value grid instead of
  decoding to FP16 and quantizing again.
- `src/gguf2mlx/quantize/gguf_codes.py` decodes Q4_K, Q6_K and Q8_0 blocks and
  re-encodes the *same* integer codes with an fp16 scale and bias. Measured
  per-tensor relative L2 against `gguf.quants.dequantize`: Q4_K 2.0-2.4e-4,
  Q8_0 0.0 (bit-exact), Q6_K ~2.2e-4 when a group's two sub-block scales agree.
- Q6_K scales 16 values at a time, so a 32-value affine group cannot represent
  it exactly when those scales disagree (1.1-1.5e-2). Those tensors report
  `exact: false` and stay in FP16 instead of being silently requantized.
- Preserve fails closed: `llama`, `qwen2`, `qwen3` and `mistral` only. Any other
  architecture errors out naming the arch rather than falling back to
  requantization.
- `conversion_report.json` gains a `preserve` block: per-tensor relative L2 and
  max abs error, `preserved_by_qtype`/`preserved_by_bits`, the decisions to keep
  a tensor in FP16, and an aggregate gate at 1e-3 (fp16 scale rounding only).
  `docs/conversion-report.schema.json` describes it.
- Mixed 4/6/8-bit module widths are emitted as per-module `quantization`
  overrides in `config.json`; `mlx-lm` loads them as `QuantizedLinear`.
- `--preserve-source-quant` keeps working as a deprecated alias that warns and
  resolves to `--quant-source preserve`. On multimodal GGUFs the Qwen2-VL
  preserve path still requires `--quantize --direct-quant` and keeps its
  affine 4-bit / group-64 layout.

### `gguf2mlx publish` and split GGUF support

- New `publish` subcommand: `gguf2mlx publish --source <repo> --gguf <file>
  --dest <repo> [--source-revision] [--quantize] [--quant-source]`. The pipeline
  moved out of the workflow's heredoc payload into `src/gguf2mlx/publish.py`;
  `.github/workflows/publish-mlx.yml` is now a thin shell around the same code.
- Split GGUFs (`-00001-of-0000N`) are discovered from any shard, merged back
  into one file before conversion, and a partial shard set fails with an error
  naming the missing indices instead of publishing a truncated model.
- Safety properties are unchanged and now unit-tested: repository-id and path
  validation before the first network call, source ≠ destination, immutable
  source revision, provenance file, source license/notice files copied, and the
  destination repository created only after conversion produced weights.
- Float16 conversion needs only `numpy`/`gguf`/`safetensors`, so the workflow's
  default runner is now `ubuntu-latest`; only `quantize: true` requires a macOS
  Apple Silicon runner.
- New `publish` extra (`huggingface-hub`), and the provenance record now carries
  the GGUF SHA-256, byte size, shard list, quant source and converter version.

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
