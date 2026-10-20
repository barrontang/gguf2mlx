# Quality gates

`conversion_report.json` is written next to every conversion and is the artifact
to read before trusting an output directory. This page explains what each tier
means, what actually fails a run, and how to re-check an artifact later.

## The three tiers

| Tier | Question it answers | Deterministic | Fails the conversion |
|---|---|---|---|
| `weight` | Did the bytes on disk survive decode, transform and serialization? | Yes | **Yes** |
| `numerical` | Does the assembled model load and produce finite logits? | Yes, needs MLX | Only when run via `verify --load` |
| `perplexity` | How much accuracy did the source quantization cost? | No | **Never** |

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
`Load matrix` workflow runs the Q4_K_M subset nightly on Apple Silicon and the
full sweep on manual dispatch.

The matrix currently covers Llama-3.2-1B, Qwen2.5-0.5B, Qwen3-0.6B, Gemma-2-2B
and Phi-3-mini across Q4_0 / Q4_K_M / Q8_0 where published. MoE architectures are
excluded from the nightly set because no ≤2B public fixture exists; their
evidence remains the pinned real-model runs in
[validation](validation/native-validation-2026-10-18.md).

## What is still not covered

- `--direct-quant` writes packed codes, scales and biases rather than plain
  tensors, so it is described by `tensor_decisions` (schema 1 fields) and the
  layout fixtures, not by the plain-tensor weight audit.
- The numerical tier is not run during conversion: it needs MLX, and conversion
  deliberately works on Linux without it.
