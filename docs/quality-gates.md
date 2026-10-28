# Quality gates

`conversion_report.json` is written next to every conversion and is the artifact
to read before trusting an output directory. This page explains what each tier
means, what actually fails a run, and how to re-check an artifact later.

## The tiers

| Tier | Question it answers | Deterministic | Fails the conversion |
|---|---|---|---|
| `weight` | Did the bytes on disk survive decode, transform and serialization? | Yes | **Yes** |
| `numerical` | Does the assembled model load and produce finite logits? | Yes, needs MLX | Only when run via `verify --load` |
| `perplexity` | How much accuracy did the source quantization cost? | No | **Never** |
| `preserve` | Were the source GGUF blocks re-expressed without requantizing? | Yes | **Yes** (preserve mode only) |

The separation exists because a single cross-runtime perplexity number cannot
distinguish "the converter is wrong" from "4-bit is lossy". A correct converter
*should* show a PPL delta against FP16; failing a build for that teaches nobody
anything. Weight fidelity, by contrast, has an exact expected answer.

## Reading the weight gate

```json
"fidelity": {
  "weight": {
    "status": "pass",
    "tolerance": 0.0006,
    "dtype": "float16",
    "tensors_compared": 290,
    "tensors_total": 291,
    "bit_exact_tensors": 154,
    "max_relative_l2": 0.00031,
    "worst_tensor": "model.layers.0.mlp.down_proj.weight",
    "failures": [],
    "failure_count": 0
  }
}
```

- **`tolerance`** is the dtype's unit roundoff with headroom: `float16` 6e-4,
  `bfloat16` 5e-3, `float32` 1e-7. Half-precision output legitimately loses a few
  parts in 10⁴ relative to an fp32 reference; anything beyond that is not
  rounding.
- **`tensors_compared` vs `tensors_total`** is the honest coverage number.
  Fused (Phi-3 QKV), permuted (Llama rotary) and split (MLA `attn_k_b`) tensors
  are counted as `transformed` in `conversion.skipped` and are **not** claimed as
  verified — they are covered by the architecture fixture tests instead.
- **`bit_exact_tensors`** is non-zero only when the source dtype and output dtype
  match and no transform ran.
- A tensor recorded as `identity` whose stored shape no longer matches the source
  is a hard failure (`layout_mismatch`), because the artifact and its stated plan
  disagree.
- A tensor recorded as `reshaped` (Qwen3.5's depthwise `conv1d`, which GGUF
  stores squeezed to 2-D) *is* still value-compared: the source is reshaped to
  the stored shape when the element counts match, so a transposed kernel fails
  the gate instead of being excused as "transformed".

The comparison is against `gguf.quants.dequantize` applied to the raw GGUF
bytes, not against the converter's own decode helper, so a bug in the converter
cannot hide behind a matching bug in the oracle.

## Preserve tier

`--quant-source preserve` re-expresses the Q4_K / Q5_K / Q6_K / Q8_0 blocks as
MLX affine groups (`w = scale * code + bias`, fp16 scale and bias, 32-value
groups). Where the block layout is expressible it copies the *same integer
codes* and only rounds the scale and bias to fp16.

`Q5_K` is preserved exactly, like `Q4_K`: it keeps a 6-bit scale and min per 32
values — exactly MLX's group — and its 5-bit code is the low nibble of `qs` plus
one high bit from the block's `qh` plane. MLX stores 5-bit groups as five words
per 32 values, so the codes land unmodified. Skipping this width is expensive:
the pinned Qwen3.5-0.8B Q4_K_M is 36 Q5_K tensors, and sending them to FP16
turned a 1.08x artifact into 1.45x.

`Q6_K` is the exception, and it is the reason this tier has two halves. Q6_K
scales 16 values at a time; MLX's narrowest group is 32, so a group would need
two scales. Only 0.7% of groups in a real model have matching pair scales, so
"copy the codes or give up" meant giving up on 99% of the tensor and writing
it as FP16 — which turns 0.82 bytes/value into 2 and makes a Q6_K model **2.4x
the size of the GGUF it came from**. Instead those tensors are re-fitted at the
source width and gated on their own ceiling:

| Tier | What happened | Ceiling |
|---|---|---|
| `exact` | codes copied; only fp16 scale/bias rounding | 1e-3 |
| `inexact` | re-fitted onto the 32-wide grid | 2.5e-2 (6-bit), 8e-3 (8-bit) |

The `preserve` gate passes only when every measured tier passes, so the two
numbers can never cover for each other. `--preserve-fallback` chooses the
width for the inexact tier; `6bit` is the default because it keeps the source
width:

| Fallback | Q6_K artifact | Relative L2 |
|---|---|---|
| `6bit` (default) | **1.067x** the source bytes | 1.7-1.8e-2 |
| `8bit` | 1.371x | 4.4-4.9e-3 |
| `float16` | 2.4x | 0 (lossless, 2.44x bytes) |

Measured on Llama-3.2-1B (bartowski Q4_K_M and Q6_K, 113 quantized tensors).
Fitting is per group: the lower squared error wins between the source lattice
and a least-squares refit from two starting points, and groups whose pair
scales do agree keep their codes untouched.

```json
"preserve": {
  "gate": "pass",
  "tolerance": 0.001,
  "fallback": "6bit",
  "tensors_compared": 9,
  "aggregate_relative_l2": 0.00774,
  "size": {"source_bytes": 1294336, "output_bytes": 1397760, "ratio": 1.08},
  "preserved_by_qtype": {"Q4_K": 7, "Q5_K": 1, "Q6_K": 1, "Q8_0": 1},
  "preserved_by_bits": {"4": 7, "5": 1, "6": 1, "8": 1},
  "group_size": 32,
  "tiers": {
    "exact":   {"status": "pass", "tolerance": 0.001, "tensors_compared": 8,
                "max_relative_l2": 0.000241},
    "inexact": {"status": "pass", "tolerance": 0.025, "tensors_compared": 1,
                "max_relative_l2": 0.0112}
  },
  "kept_float16": {"counts": {"F32_has_no_affine_reexpression": 3}},
  "tensors": [
    {"name": "model.layers.0.self_attn.q_proj.weight", "source": "Q4_K",
     "bits": 4, "tier": "exact", "group_size": 32, "relative_l2": 0.00021,
     "max_abs_error": 0.00012, "bit_exact": false}
  ]
}
```

Note that the top-level `tolerance` bounds the `exact` tier only; `inexact`
tensors are bounded by `tiers.inexact.tolerance`. A tensor is never compared
against a ceiling its format cannot meet.

Per-tensor relative L2 against `gguf.quants.dequantize`:

| Source | Relative L2 | Why |
|---|---|---|
| Q8_0 | 0 (bit-exact) | 8-bit codes map to 8-bit affine codes with `scale = d`, `bias = -128d` |
| Q4_K | 2.0e-4 - 7.7e-4 | fp16 rounding of the per-32 scale and min only |
| Q5_K | 2.1e-4 - 7.0e-4 | same, with the 5th code bit taken from the block's `qh` plane |
| Q6_K (pair scales agree) | ~2.2e-4 | same, within one 32-value group |
| Q6_K (pair scales disagree, 6-bit) | 1.6-1.8e-2 | **not representable** on a 32-wide grid; re-fitted |
| Q6_K (pair scales disagree, 8-bit) | 4.4-4.9e-3 | four times finer grid, 1.37x the bytes |

Each range spans Llama-3.2-1B (bartowski, 113 quantized tensors) and the pinned
Qwen3.5-0.8B Q4_K_M (98 `Q4_K`, 36 `Q5_K`, 17 `Q6_K`, 36 `Q8_0`). The upper ends
are per-tensor maxima, not typicals — Qwen3.5-0.8B's worst `exact` tensor is
7.69e-4, which is what the 1e-3 ceiling is sized against. The block type, not
the layer, decides which band a tensor lands in: in that same file a `Q6_K`
`ffn_down` measures 1.58e-2 while the `Q4_K` `ffn_down` one layer over measures
6.82e-4, and every `Q8_0`, `ssm_alpha` and `ssm_beta` tensor round-trips at
exactly 0.

Only block types with no affine re-expression at all (F32 norms, non-matrices,
unaligned rows) are decoded to float, and the report names the reason. The
fail-closed rule extends to architectures: only `llama`, `qwen2`, `qwen3`,
`mistral`, and the Qwen3.5 family (`qwen35` plus the `qwen3_5` / `qwen3.5` /
`qwen36` / `qwen38` aliases) are supported, and any other architecture raises an
error naming it rather than silently requantizing.

Qwen3.5 is hybrid: only the quantized matrix projections of a linear-attention
block are preservable. `A_log`, `dt_bias`, `conv1d` and the gated norm have no
affine matrix form, so they stay float and are counted in
`kept_float16.counts.F32_has_no_affine_reexpression`. Two layout rules make
that output loadable at all, and both are recorded in `tensor_layout`:

| Source tensor | On-disk MLX form | Recorded as |
|---|---|---|
| `blk.N.ssm_a` | `log(-ssm_a)` — GGUF stores the decay rate, MLX wants its log | `transformed` |
| `blk.N.ssm_conv1d.weight` | `(channels, kernel)` → `(channels, kernel, 1)` | `reshaped` |
| `blk.N.ssm_norm.weight`, `*_norm.weight` | unchanged | `identity` |

The norms are deliberately **not** de-shifted: llama.cpp applies its `+1`
centering to every norm except `linear_attn.norm.weight`, and MLX-LM's
`qwen3_5` sanitize only re-applies that offset when it sees an HF-layout
`conv1d`. Since this converter emits MLX layout, the GGUF norms are already the
values the runtime wants.

## Source tensors without an output

Not every source tensor becomes an output tensor, and an unexplained gap is a
silent correctness hole. The report names each one with a reason:

```json
"source_tensors_without_output": [
  {"name": "blk.4.nextn.attn_k.weight", "reason": "mtp_head_not_consumed"},
  {"name": "rope_freqs.weight", "reason": "reproduced_from_config"},
  {"name": "mystery.weight", "reason": "unmapped"}
]
```

| Reason | Meaning |
|---|---|
| `reproduced_from_config` | the tensor's effect is recreated from metadata (`llama3` rope table → `rope_scaling`) |
| `mtp_head_not_consumed` | a speculative-decoding head (`mtp.*` / `nextn.*`) that the text runtime never runs |
| `unmapped` | no adapter claimed the tensor — the case to investigate |

A non-empty `unmapped` list is a defect report, not a normal outcome: the
conversion still completes (the tensor list is data, not policy), but the
artifact deserves a `verify --load` before it is trusted.

## Re-checking an artifact

```bash
gguf2mlx verify --model-dir ./mlx-model --weights --source model.gguf
gguf2mlx verify --model-dir ./mlx-model --load
```

`--weights` replays the audit from the report's recorded `tensor_layout` without
converting again. `--load` is the numerical tier: it imports `mlx_lm`, loads the
directory, runs one short forward pass, and requires every logit to be finite.
Both exit non-zero on failure, so either can gate a release or a CI job.

A failure on a shared runner can come from the GPU rather than the weights, and
the two such failures need different remedies:

| Driver error | Meaning | What happens |
|---|---|---|
| `…ErrorTimeout` | the command buffer lost its scheduling slot | `retryable: true`; the matrix and `publish --verify-load` retry once, because the same bytes usually pass |
| `…ErrorHang` | the Metal context is wedged | `environment: gpu_hang`, not retryable — the remedy is to re-run the **job**, since an in-process retry cannot recover |

Non-finite logits and a missing runtime are never classified as either, so a
real defect still fails on the first attempt instead of hiding behind a second
one.

That distinction exists because the nightly went red twice on the `gemma2`
entries — once with `Caused GPU Timeout Error`, then with
`Caused GPU Hang Error` where the in-process retry hung again and cost ten
minutes for nothing. Both times the weight gate had already passed, `mlx` was
the same 0.32.4 the previous green run used, and `gemma2-q4_k_m` passed on the
same commit; nothing in the report let a reader tell any of that apart from a
broken conversion.

After `mlx_lm.convert` quantization, the report is carried into the quantized
directory with `artifact.stage = "mlx_lm_quantized"`: the weight audit describes
the FP16 staging output, and the 4-bit weights are explicitly **not** covered by
it. Use `--load` on the quantized directory.

## Nightly load matrix

`tests/load_matrix.json` pins real public GGUFs by repository, filename and
SHA-256. With `GGUF2MLX_LOAD_MATRIX=1`, each entry is downloaded, hash-checked,
converted, weight-gated, loaded with `mlx_lm` and exercised. The
`Load matrix` workflow fans the entries out as **one job per (architecture,
qtype)** on Apple Silicon: a converted FP16 tree is 3-5x its source GGUF and a
single runner has ~14 GB free, so running the whole sweep in one job ends in
`ENOSPC` that looks exactly like a conversion bug.

Run a single pinned case locally with
`GGUF2MLX_LOAD_MATRIX=1 GGUF2MLX_LOAD_MATRIX_ENTRY=llama-q4_k_m pytest
tests/test_load_matrix.py -q`.

The matrix currently covers Llama-3.2-1B, Qwen2.5-0.5B, Qwen3-0.6B, Gemma-2-2B,
Phi-3-mini and Qwen3.5-0.8B across Q4_0 / Q4_K_M / Q8_0 where published. MoE
architectures are excluded from the nightly set because no ≤2B public fixture
exists; their evidence remains the pinned real-model runs in
[validation](validation/native-validation-2026-10-18.md).

The Qwen3.5 entry is pinned (`unsloth/Qwen3.5-0.8B-GGUF`, Q4_K_M,
`bd258782…`) and runs with `quant_source: preserve`, so it simultaneously
exercises the hybrid layout and the preserve tier.

All twelve entries pass, locally on Apple Silicon (2026-10-26) and in CI
(2026-10-28): conversion, the weight gate, the preserve gate where applicable,
`mlx_lm.load` and a finite-logit forward pass. For the Qwen3.5 case
specifically: weight gate `115/133` with a maximum relative L2 of 2.85e-04,
preserve gate 170 `exact` / 17 `inexact`, a 1.084x artifact.

The CI green took three dispatches rather than one: the first full sweep came
back 11/12 because `gemma2-q8_0`'s runner GPU hung, and the entry then passed in
9m02s on a fresh job. That is the case the timeout-versus-hang split below was
written for.

### What the Qwen3.5 case found

It was worth pinning. Five defects only a real checkpoint exposed:

| Symptom | Root cause | Fix |
|---|---|---|
| `Unsupported qwen35 tensor: blk.0.post_attention_norm.weight` | llama.cpp names Qwen3.5's post-attention norm `post_attention_norm`, not `ffn_norm`; the mapper only knew the Llama spelling | dedicated mapping to `post_attention_layernorm.weight` |
| `Missing 1 parameters: language_model.lm_head.weight` | `build_config` forced `tie_word_embeddings: false` for Qwen3.5 while the file has no `output.weight` and upstream ties | tie/untie follows the tensor list, as for Qwen3 |
| artifact 1.446x the source | 36 `Q5_K` tensors had no affine re-expression and fell back to FP16 | `Q5_K` → 5-bit affine with the source codes copied; 1.084x |
| wrong decay / unloadable kernel | `A_log` was only restored for a tensor named `ssm_a` (real name `blk.N.ssm_a`), and the squeezed `conv1d` kernel was never widened | `blk.N.ssm_a` → `log(-x)`, kernel → `(channels, kernel, 1)`, gated norm left unshifted |
| `tokenizer.ggml.pre='qwen35'; using legacy BPE splitting` | llama.cpp names the hybrid family `qwen35` while Qwen2/2.5/3 report `qwen2`; the value was unlisted, so the tokenizer fell to legacy splitting **and** kept the default NFC normalizer, which no Qwen tokenizer has | `QWEN_PRE_TOKENIZERS` and `QWEN_FAMILY_ARCHES` route every Qwen ID to the same no-normalizer path |

The last one is the reason the load matrix asserts tokenization rather than just
a successful load: the model loaded, produced finite logits, and still would have
tokenized text differently from the reference. A warning in the matrix log was
the only signal.

### What the first full sweep found

The first run of all thirteen cases (2026-10-21, one job, `main` @ `63fe89c`)
passed **4 of 11** e2e cases. Everything it caught was invisible to the unit
suite, which is the point of the gate:

| Symptom at load | Root cause | Fix |
|---|---|---|
| `Received 1 parameters not in model: rope_freqs.weight` (llama ×3) | llama.cpp carries llama3 rope scaling as a precomputed table with **no** `llama.rope.scaling.*` metadata, and the mapper passed the tensor through. The config also lost the scaling, so positions would have rotated wrongly even after dropping the tensor | infer `factor` / `original_max_position_embeddings` from the table, **verify by regenerating it**, emit `rope_scaling: llama3`, drop the tensor |
| `Missing 1 parameters: lm_head.weight` (qwen3) | `build_config` forced `tie_word_embeddings=False` for qwen3 while Qwen3-0.6B ships no `output.weight` | tie/untie follows the tensor list for every architecture |
| `Unsupported gemma2 tensor: blk.0.post_attention_norm.weight` | the Llama mapper only knows two norms per block; Gemma 2/3 has four (`attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`) and `ffn_norm` is the *pre*-FFN norm | dedicated Gemma 2/3 mapper plus the `-1` de-shift for both post norms |
| `25165824 requested and 0 written` (phi3 Q8_0) | disk exhaustion: twelve downloads plus their FP16 outputs on one 14 GB runner | one job per entry, and drop the converted tree after each case |

Re-run after the fixes, all 11 entries green on
[run 39505897872](https://github.com/barrontang/gguf2mlx/actions/runs/39505897872)
(`main` @ `40ef9f5`, 2026-10-21, 11 jobs × ~5 min wall clock, ~11 min total):

| Entry | Weight gate | Tensors compared | Bit-exact | Max relative L2 | Tolerance |
|---|---|---|---|---|---|
| llama-q4_0 | pass | 114/146 | 32 | 2.04e-04 | 6e-04 |
| llama-q4_k_m | pass | 114/146 | 32 | 2.08e-04 | 6e-04 |
| llama-q8_0 | pass | 114/146 | 32 | 2.07e-04 | 6e-04 |
| qwen2-q4_0 | pass | 291/291 | 122 | 2.07e-04 | 6e-04 |
| qwen2-q4_k_m | pass | 291/291 | 121 | 2.18e-04 | 6e-04 |
| qwen2-q8_0 | pass | 291/291 | 121 | 2.15e-04 | 6e-04 |
| qwen3-q8_0 | pass | 310/310 | 113 | 2.08e-04 | 6e-04 |
| gemma2-q4_k_m | pass | 183/288 | 0 | 2.08e-04 | 6e-04 |
| gemma2-q8_0 | pass | 183/288 | 0 | 2.10e-04 | 6e-04 |
| phi3-q4_k_m | pass | 195/195 | 65 | 2.08e-04 | 6e-04 |
| phi3-q8_0 | pass | 195/195 | 65 | 2.07e-04 | 6e-04 |

Every case also loaded with `mlx_lm` and produced finite logits. The compared
fraction is the honest part of the table: `llama` excludes the 32 rotary
Q/K projections (row-permuted) and Gemma 2 excludes its 104 de-shifted norms,
because the audit only compares tensors written byte-for-byte. Both transforms
are covered by fixture tests, but they are not *value*-checked here.

## What is still not covered

- `--direct-quant` writes packed codes, scales and biases rather than plain
  tensors, so it is described by `tensor_decisions` (schema 1 fields) and the
  layout fixtures, not by the plain-tensor weight audit.
- The numerical tier is not run during conversion: it needs MLX, and conversion
  deliberately works on Linux without it.
- Preserve mode records the *mapping* error, not the error the source
  quantization already baked in: a tensor can be bit-exact and still be lossy
  relative to the original float model.
