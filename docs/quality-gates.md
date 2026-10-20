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

The comparison is against `gguf.quants.dequantize` applied to the raw GGUF
bytes, not against the converter's own decode helper, so a bug in the converter
cannot hide behind a matching bug in the oracle.

## Preserve tier

`--quant-source preserve` does not quantize: it decodes the Q4_K / Q6_K / Q8_0
blocks and re-encodes the *same integer codes* as MLX affine groups
(`w = scale * code + bias`, fp16 scale and bias, 32-value groups). The report
gains a `preserve` block with the gate and one record per tensor:

```json
"preserve": {
  "gate": "pass",
  "tolerance": 0.001,
  "tensors_compared": 8,
  "aggregate_relative_l2": 0.000176,
  "preserved_by_qtype": {"Q4_K": 7, "Q8_0": 1},
  "preserved_by_bits": {"4": 7, "8": 1},
  "group_size": 32,
  "kept_float16": {"counts": {"F32_has_no_affine_reexpression": 3,
                              "sub_block_scales_disagree": 1}},
  "tensors": [
    {"name": "model.layers.0.self_attn.q_proj.weight", "source": "Q4_K",
     "bits": 4, "group_size": 32, "relative_l2": 0.00021,
     "max_abs_error": 0.00012, "bit_exact": false}
  ]
}
```

Measured per-tensor relative L2 against `gguf.quants.dequantize`:

| Source | Relative L2 | Why |
|---|---|---|
| Q8_0 | 0 (bit-exact) | 8-bit codes map to 8-bit affine codes with `scale = d`, `bias = -128d` |
| Q4_K | 2.0-2.4e-4 | fp16 rounding of the per-32 scale and min only |
| Q6_K (sub-block scales agree) | ~2.2e-4 | same, within one 32-value group |
| Q6_K (sub-block scales disagree) | ~1.1-1.5e-2 | **not representable** on a 32-wide grid |

The last row is a structural limit, not a bug: Q6_K scales 16 values at a time,
so a 32-value affine group would need two scales. Those tensors report
`exact: false` and are written in **FP16** instead of being requantized; the
aggregate gate therefore only covers the tensors that were actually preserved.
The fail-closed rule extends to architectures: only `llama`, `qwen2`, `qwen3`
and `mistral` are supported, and any other architecture raises an error naming
it rather than silently requantizing.

## Re-checking an artifact

```bash
gguf2mlx verify --model-dir ./mlx-model --weights --source model.gguf
gguf2mlx verify --model-dir ./mlx-model --load
```

`--weights` replays the audit from the report's recorded `tensor_layout` without
converting again. `--load` is the numerical tier: it imports `mlx_lm`, loads the
directory, runs one short forward pass, and requires every logit to be finite.
Both exit non-zero on failure, so either can gate a release or a CI job.

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

The matrix currently covers Llama-3.2-1B, Qwen2.5-0.5B, Qwen3-0.6B, Gemma-2-2B
and Phi-3-mini across Q4_0 / Q4_K_M / Q8_0 where published. MoE architectures are
excluded from the nightly set because no ≤2B public fixture exists; their
evidence remains the pinned real-model runs in
[validation](validation/native-validation-2026-10-18.md).

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

Re-run after the fixes: llama-q4_0, qwen3-q8_0 and gemma2-q4_k_m convert, pass
the weight gate and load with finite logits locally.

## What is still not covered

- `--direct-quant` writes packed codes, scales and biases rather than plain
  tensors, so it is described by `tensor_decisions` (schema 1 fields) and the
  layout fixtures, not by the plain-tensor weight audit.
- The numerical tier is not run during conversion: it needs MLX, and conversion
  deliberately works on Linux without it.
- Preserve mode records the *mapping* error, not the error the source
  quantization already baked in: a tensor can be bit-exact and still be lossy
  relative to the original float model.
