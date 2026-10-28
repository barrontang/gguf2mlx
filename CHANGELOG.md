# Changelog

## Unreleased

### Emitted `model_type` values are checked against the runtime registry

Conversion could emit a `model_type` the installed runtime has no module for,
which only surfaces as a load failure on the user's machine. The minimum
runtimes are now `mlx-lm>=0.31.3` (Qwen3.5 landed in 0.31.0, `gemma4_text` in
0.31.3) and `mlx-vlm==0.7.10` (`minicpmv4_7`), and `tests/test_runtime_registry.py`
asserts every architecture ID this converter can emit resolves through
`mlx_lm.utils._get_classes()` / `mlx_vlm.utils.get_model_and_args()` — including
the `qwen2_vl_text` / `qwen3_vl_text` text hosts that the VLMs delegate to. A new
`Runtime contract` workflow pins both versions on Apple Silicon and runs that
suite; a portable inventory half fails when an adapter is added without the
registry test being updated.

### `preserve` covers dense Qwen3.5, and its hybrid layout is restored

`--quant-source preserve` now accepts `qwen35` and its aliases (`qwen3_5`,
`qwen3.5`, `qwen36`, `qwen38`). The quantized matrix projections of a
linear-attention block keep their source codes; `A_log`, `dt_bias`, the
depthwise `conv1d` and the gated norm have no affine matrix form and stay float,
recorded as `F32_has_no_affine_reexpression`.

Writing that fixture exposed two layout bugs in the float path, both of which
produced artifacts that could not load or ran with wrong decay:

- `A_log` was only restored for a tensor literally named `ssm_a`. Real GGUFs use
  `blk.N.ssm_a`, so `log(-ssm_a)` never ran and every layer kept the raw
  negative decay rate where MLX expects its logarithm.
- `ssm_norm.weight` was de-shifted by 1. llama.cpp skips its `+1` centering for
  `linear_attn.norm.weight` specifically, so subtracting 1 corrupted the norm.
- The squeezed `(channels, kernel)` kernel is now widened to
  `(channels, kernel, 1)`, which is what MLX's `Conv1d` wants and what keeps
  `mlx_lm`'s `qwen3_5` sanitize from re-applying the HF `+1` norm offset.

The weight audit learned a `reshaped` layout so the widened kernel is still
value-compared instead of being excused as `transformed`.

### Pinning the Qwen3.5 real checkpoint found four more defects

`unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M was supposed to be the easy part of the
matrix. It found:

- `Unsupported qwen35 tensor: blk.0.post_attention_norm.weight` — llama.cpp
  names Qwen3.5's post-attention norm `post_attention_norm`, not `ffn_norm`, and
  only the Llama spelling was mapped, so **every** conversion of a real Qwen3.5
  file failed.
- `Missing 1 parameters: language_model.lm_head.weight` — `build_config` forced
  `tie_word_embeddings: false` for Qwen3.5 while the file ships no
  `output.weight`. Tie/untie now follows the tensor list, as it already did for
  Qwen3.
- The first artifact was **1.446x** the source because 36 `Q5_K` tensors had no
  affine re-expression and fell back to FP16. `Q5_K` is now preserved exactly:
  its 6-bit scale/min is already per-32 values, and its 5-bit code is the `qs`
  low nibble plus one bit from the `qh` plane, so MLX stores the same codes in
  five words per group. Artifact: **1.084x**.
- `tokenizer.ggml.pre='qwen35'` was unlisted, so the tokenizer fell through to
  legacy BPE splitting **and** kept the default NFC normalizer — which no Qwen
  tokenizer has. The model still loaded with finite logits; only a warning in the
  matrix log gave it away. `QWEN_PRE_TOKENIZERS` and `QWEN_FAMILY_ARCHES` now
  route every Qwen ID to the same no-normalizer path, and
  `tests/test_tokenizer_fidelity.py` asserts `normalizer is None` for each of
  them and that an unknown `pre` value still warns.

The case now passes end to end locally on Apple Silicon (2026-10-26,
`mlx-lm` 0.32.2): convert → weight gate (`115/133`) → preserve gate (170
`exact`, 17 `inexact`) → `mlx_lm.load` → finite logits. Re-running the whole
matrix against the same tree gives 12/12 green, so every pinned entry has local
evidence; the `Load matrix` workflow has not been re-run since the Qwen3.5 entry
was added, so the published run still shows eleven.

### `publish` can be reviewed before it is visible

- `--dry-run` resolves, downloads, converts, and builds the card and provenance,
  then stops before `create_repo`/`upload_folder`.
- `--verify-load` runs the same check as the release gate
  (`fidelity.load_and_run`: load, one forward pass, finite logits) before the
  destination repository exists, so a layout defect fails the run instead of
  publishing an unloadable model. Loading alone would miss it: a tensor with the
  right name and shape but the wrong layout loads cleanly and computes garbage.
- Shard downloads and license copies stream in fixed-size chunks instead of
  reading each file into memory, and the merged GGUF is written incrementally.
- The model card now carries the source GGUF SHA-256 and byte count, the weight
  and preserve gate results, the artifact disk size, a `base_model_relation`,
  dtype/precision tags, and an exact reproduce command.

### The report explains every source tensor that produced no output

`conversion_report.json` schema 4 adds `source_tensors_without_output`, naming
each dropped tensor with `reproduced_from_config`, `mtp_head_not_consumed`, or
`unmapped`. Silence was the previous behaviour, which made MTP heads and
`rope_freqs` indistinguishable from an adapter gap.

### Three red CI jobs, none of which the converter caused

The first push of this cycle turned four workflows red. Two were mine and two
predated it:

- `tests/requirements-vlm.txt` pinned `mlx==0.32.2` while `mlx-vlm==0.7.10`
  requires `mlx>=0.32.4`, so `Runtime contract` and `Vision conversion` both
  died in `pip install` before a single test ran. Pinned to `0.32.4`, the
  version the Qwen3.5 run actually used, and the set now resolves.
- The nightly `Load matrix` failed on the `gemma2` entries in the forward pass —
  `Caused GPU Timeout Error` one night, `Caused GPU Hang Error` the next — after
  the conversion and the weight gate had passed. The same bytes load locally,
  `mlx` was the same `0.32.4` the previous green run used, and `gemma2-q4_k_m`
  passed on the same commit in the same run, so the runner's GPU is the common
  factor. `fidelity.load_and_run` now separates the two faults: a timeout is
  `retryable: true` and is retried once, while a hang sets
  `environment: gpu_hang` and is not retried at all, because a wedged Metal
  context cannot recover inside the process that wedged it — retrying it only
  spends another ten minutes failing. Non-finite logits and a missing runtime
  are classified as neither, so a real defect still fails immediately.
- `Weekly MoE validation` was failing every run because `GGUF2MLX_MOE_MODELS`
  was never set, and no step can download the 11–20 GB fixtures it expects. It
  now skips with a warning annotation when unconfigured instead of going red,
  and still fails hard when the variable is set but names an unusable file. A
  weekly red that no reader can act on is worse than an honest skip; the
  support-matrix line that promised this coverage now says it is inactive until
  configured, and `tests/test_docs_consistency.py` fails if that promise drifts
  from the workflow guard again.


### Documentation tracks the code again

`llms.txt`, `docs/development.md`, `docs/support-matrix.md`,
`docs/quality-gates.md`, `docs/publishing.md`, `docs/vlm-conversion.md` and the
README no longer describe a Rust build, call direct-quant "proposed", or claim
that source quantization is never preserved.

Fixing the instances would only invite the fourth round, so
`tests/test_docs_consistency.py` now asserts the invariants instead: every
`CONVERTIBLE_ARCHES` and `PRESERVE_ARCHES` entry is named in `llms.txt` (alias
spellings collapsed, word-boundary matched so `qwen3` is not satisfied by
`qwen3.5`), the quoted runtime versions match `pyproject.toml`, every
preservable block type appears in the gate table, no doc mentions the deleted
crate, and `direct-quant-transcoding.md` no longer titles itself as proposed.
Writing it immediately caught two live errors: `llms.txt` undercounted the
canonical architectures by two (`qwen36`/`qwen38` are canonical, not aliases),
and `--verify-load` was documented as running a forward pass while the code only
called `load()`. The check now delegates to `fidelity.load_and_run`, so a
publish-time verification and a CI gate agree on what "loadable" means — a
non-finite logits result and a missing runtime both fail the run rather than
passing silently.

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

### CLI: `--force`, `--json`, and a real logging channel

- `--force` replaces a non-empty `--output`. Without it a non-empty directory is
  still refused — and `--force` will not delete a *file* sitting at `--output`.
- `--json` prints one object (`ok`, `output`, `report`, `gates`, `weight`) read
  back from `conversion_report.json`, so a wrapper never has to parse the human
  summary. `verify --json` already worked this way.
- Diagnostics now go through a `logging` logger on stderr (`-v` for DEBUG, `-q`
  for warnings and errors only). With `--json`, the converter's own progress and
  failure prints are redirected to stderr too, so stdout carries exactly one
  parseable object. There were no `logging` calls before.

### `rust/gguf2mlx-rs` is deleted

Nine tracked files of pyo3 code (architecture detection, metadata, tensor and
tokenizer helpers) sat behind an optional import that never resolved: the cdylib
was never built, never published, and never mentioned in the docs, while the
Python path already computed everything it returned. The MDD left this as
"maturin or delete". A one-shot classification of a few thousand tensor names
does not justify a second toolchain in the release pipeline, so the crate and its
`rust_backend.py` shim are gone. The pre-rewrite backup and the changelog keep
the decision reversible.

### Declared-versus-covered architectures are enforced, not implied

`CONVERTIBLE_ARCHES` claims 30 architecture IDs; the pinned load matrix proves
five. The 17 canonical gaps (25 spellings) are now listed with a reason in
`DECLARED_COVERAGE_GAPS`, and
`test_every_convertible_architecture_is_covered_or_declared` fails when a new
adapter is neither pinned nor declared — and when a declared gap is filled but
its line is left behind. `docs/support-matrix.md` mirrors the table.

`README.md` is back inside its 200-line budget (254 → 200) by pushing the
preserve measurements to the quality-gate page and the `.mlx` walkthrough to its
own page.

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
