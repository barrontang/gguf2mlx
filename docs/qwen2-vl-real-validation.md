# Qwen2-VL-2B: real-checkpoint conversion acceptance

## Conclusion and scope

The pinned **Qwen2-VL-2B-Instruct Q4_K_M + F16 mmproj** pair below completes
download verification, conversion, strict MLX-VLM loading, image preprocessing,
weight auditing and labeled image-conditioned inference on Apple Silicon.

**Use the floating-point output as the quality reference.** Uniform
language-only MLX 4-bit conversion passes the original diagnostic gate but
loses one of twelve exact-answer checks. Its photograph answer is **semantically
correct**: direct inspection confirms the suit is orange. The initial description
of its color as unsupported was wrong; this was an exact-format failure, not
evidence of hallucination. The new source-preserving Q4_K/float option passes all
twelve original exact-answer checks without changing the prompt or accepted
strings. It is mixed precision, not an all-four-bit model.
Converting Q4_K_M to FP16 does not recover the original pre-quantization weights.

## Four-bit follow-up: diagnosis and resolution

The original evaluator conflated a finite list of accepted short phrases with
semantic correctness. It still records that **historical exact-match score** for
comparability, but now adds a separate deterministic semantic label check:
the garment must be a space/astronaut suit, and any stated color must be orange.
Supported affirmative sentence forms are explicitly matched; negation, wrong
colors and additional unsupported clauses are rejected, not accepted by keyword
substring. This is a narrow task-specific rubric, not a general caption judge.
The exact accepted strings and original reports are not rewritten to obtain a
passing result.

Controlled local experiments on the same images, prompts and temperature:

| Experiment | Exact matches | NASA answer / observation |
|---|---:|---|
| Original uniform 4-bit | 11/12 | The person is wearing an orange space suit. |
| Independently unpack its stored codes/scales into floating layers | 11/12 | Same answer; not a bit-packing or quantized-matmul defect |
| MLX native affine quantizer, groups of 64 | 11/12 | The person is wearing an orange spacesuit. |
| MLX native affine quantizer, groups of 32 | 10/12 | Same long caption; also gets right/left wrong |
| Weight-only least-squares refinement, 8 iterations | 11/12 | Same long caption |
| Keep Q6_K down-projections at 8-bit or FP16, rest unchanged | 11/12 | space suit; introduces a left/right error elsewhere |
| Preserve original Q4_K codes/groups and retain Q6_K in floating layers | **12/12** | **space suit**, no other exact-answer regressions |

Uniform re-quantization changes the language weights substantially: Q4_K
has eight independently scaled 32-value groups per 256-value block, and this
Q4_K_M checkpoint deliberately retains its 28 MLP down-projections in Q6_K.
The old path discards those grids and precision choices to fit new affine
groups of 64, lowering every language projection to four bits. The resulting
change in wording is real, although the observed orange description is correct.
On the failing-format prompt, the old first-token logits favor `The` (16.109375)
over `space` (16.015625); the independent floating unpack has the same preference.
The source-preserving probe favors `space` (16.828125) over `The` (16.671875).
These are numerical observations, not a claim to identify a unique causal layer.

The fix is opt-in to avoid silently changing existing users' file sizes:

```bash
gguf2mlx convert \
  --input models/qwen2-vl-validation/gguf/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj models/qwen2-vl-validation/gguf/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  --hf-model models/qwen2-vl-validation/hf --offline \
  --quantize --direct-quant --preserve-source-quant \
  --output models/qwen2-vl-validation/mlx-4bit-preserved
```

It retains **168 Q4_K projections** as four-bit affine groups of 32 and the
**28 Q6_K projections** as FP16 (or the requested FP32). It does not read the
task images or answers during conversion, fit activation statistics, choose
individual layers using test outcomes, or patch the downstream runtime.
The output config describes the actual group size and the conversion report
records each projection's source and output precision.

| Metric | Original uniform 4-bit | Source-preserved Q4_K/float |
|---|---:|---:|
| Original exact matches | 11/12 | **12/12** |
| Semantic labels (including orange suit) | 12/12 | 12/12 |
| Relative L2 weight error, all 196 projections vs decoded GGUF | 0.085119602 | **0.000614190** |
| Maximum absolute weight error | 0.068847656 | **0.000801086** |
| Mean selected-answer NLL | 0.878537 | 1.088629 |
| Model weight bytes | See original report | 2,888,950,784 |
| Approximate directory size | 2.4 GiB | 2.7 GiB |

Projected visual features remain exactly equal to HF on all twelve inputs.
The source-preserving audit covers every stored tensor: Q4_K reconstructed
weights must stay within the analytically derived scale/bias rounding bound;
non-repacked weights must exactly match GGUF decoded and cast to FP16.
FP32 repacking is checked against the independent GGUF decoder exactly in
unit tests. **FP16 coefficient rounding and fused matmul rounding remain**;
this is not a claim of lossless FP16 arithmetic.

The strict follow-up run uses `--require-all-correct`: 100% original exact
matches **and** 100% semantic checks, not the earlier one-failure allowance.
Use the reproduction command below with `--direct-quant` pointing to
`mlx-4bit-preserved`, a new `--output`, and `--require-all-correct`.
The original uniform model remains available; its exact-format failure is not
hidden or relabeled as an exact match.

Follow-up artifacts are under `models/qwen2-vl-validation/`:
`acceptance-uniform-semantic/report.json` records the original model's
11 exact / 12 semantic passes, and `acceptance-preserved-v2/report.json`
records the new model's strict 12 / 12 pass with all-tensor audit.
`fourbit-numerical-diagnosis.json` records errors for all 196 projections.
The new reusable artifact is `mlx-4bit-preserved/`; the original `mlx-4bit/`
is intentionally unchanged. Each `probe-*.json` preserves the ablation answers.

This is a **12-case acceptance diagnostic**, not a visual benchmark certification:
eleven original synthetic images and one public-domain photograph. It does not
establish general caption quality, complex reasoning, high-resolution OCR,
multi-image/video behavior, or other checkpoints' quality. LLaVA-1.5 still has
tiny-checkpoint native tests, not this real-checkpoint acceptance result.
LLaVA-NeXT/1.6 and Qwen2.5/3.5 remain unsupported.

## Exact inputs

Both model repositories identify Apache-2.0 licensing. Original HF assets and
license are preserved locally; no weight modification or fabricated config was
needed to make the checkpoint load.

| Asset | Repository revision | SHA-256 |
|---|---|---|
| `bartowski/Qwen2-VL-2B-Instruct-GGUF` / `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | `2160e265f926e84e84dbb1c73e623b5324b23707` | `4ef095263343fc1237e8ca879790bb262bcf209f082e0a9bfce219b7ece55e8b` |
| Same repository / `mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | Same revision | `05cc3ae461a7b6aa4023312ccab549ecab77cf8677efee04f049fcbab55b8bc3` |
| `Qwen/Qwen2-VL-2B-Instruct` / `model-00001-of-00002.safetensors` | `895c3a49bc3fa70a340399125c650a463535e71c` | `994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9` |
| Same repository / `model-00002-of-00002.safetensors` | Same revision | `92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833` |

The photograph is NASA's Eileen Collins portrait, distributed as
[scikit-image's astronaut.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png).
Its [source documentation](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/_fetchers.py)
identifies NASA's Great Images database and public-domain status. The runner
resizes it to 224 x 224; the other eleven images are generated locally.
All case images, prompts, accepted answers and SHA-256 digests are saved before
inference. No image is sent to an external inference service.

## Actual format defects fixed

1. **Split tokenizer definitions.** The official `tokenizer.json` does not
   contain image/video token IDs 151655/151656, but
   `tokenizer_config.json.added_tokens_decoder` does. Validation now merges the
   explicit definitions and still rejects conflicting, duplicate, malformed and
   out-of-vocabulary IDs. Assets are copied unchanged.
2. **Historical `qwen2vl_merger` exporter.** This pair was exported using
   llama.cpp b4327's old LLaVA tools. They name visual fc1 `ffn_down` and fc2
   `ffn_up`, opposite the modern exporter. The adapter selects that layout only
   from explicit metadata, checks its zero feed-forward metadata and projection
   dimension, and rejects conflicting layouts. Its unused 10 x 10 zero position
   tensor is recognized explicitly; nonzero or differently shaped tensors fail.
   The exception is counted in the conversion report, not silently discarded.

These behaviors have independent GGUF roundtrip and negative regression tests.
The historical mapping is documented by
[b4327's image exporter](https://github.com/ggml-org/llama.cpp/blob/b4327/examples/llava/convert_image_encoder_to_gguf.py)
and [vision runtime](https://github.com/ggml-org/llama.cpp/blob/b4327/examples/llava/clip.cpp).

## Protocol and results

All four paths use greedy generation, at most 32 output tokens, and the same
twelve images/questions. Native llama.cpp uses its own Qwen template and image
processor; its results are **answer-level comparisons**, not cross-runtime
logit or perplexity parity. The three MLX paths have verified identical prompt
token hashes and image-processor pixel hashes, with float16 floating weights.
Every MLX pass checks finite logits and counterfactual visual-feature changes.

The gate was saved before the first inference: accuracy at least 75%, and no
more than one fewer correct answer than the HF reference. Answer scoring ignores
case, surrounding whitespace and terminal punctuation, but not extra text or
extra descriptions. Gate failure exits nonzero; raw answers remain saved.
Two complete runs gave the same answers and NLLs, including the 4-bit exact-format failure.

| Path | Exact matches | Mean selected-answer NLL | Peak MLX allocation |
|---|---:|---:|---:|
| Original HF weights loaded by MLX-VLM, cast FP16 | 12/12 | 1.080919 | 4,838,306,809 bytes |
| Q4_K_M GGUF decoded to FP16 MLX | 12/12 | 1.083658 | 4,838,307,465 bytes |
| Same GGUF re-quantized to MLX affine 4-bit language projections | 11/12 | 0.878537 | 2,958,772,556 bytes |
| Same GGUF + mmproj through llama.cpp | 12/12 | Not measured | Not measured |

NLL is teacher-forced loss of the first accepted answer string, averaged per
answer token then per case. Some generation answers differ in capitalization
from that selected string. It is **not corpus perplexity** and the 4-bit model's
lower mean NLL does **not** make it better: that score is neither semantic
accuracy nor exact-format compliance.
Peak allocation is `mlx.core.get_peak_memory()`, not process RSS or total
unified-memory consumption. Output directories occupy approximately 4.1 GiB
(FP16) and 2.4 GiB (language 4-bit).

### All raw answers

| Case | HF reference | GGUF to FP16 | GGUF to MLX 4-bit | llama.cpp |
|---|---|---|---|---|
| Red square | Red | Red | Red | Red |
| Blue square | Blue | Blue | Blue | Blue |
| Green square | Green | Green | green | Green |
| Yellow square | Yellow | Yellow | Yellow | Yellow |
| One circle | 1 | 1 | 1 | 1 |
| Two circles | 2 | 2 | 2 | 2 |
| Three circles | 3 | 3 | 3 | 3 |
| Red square left of blue circle | Left | Left | Left | Left |
| Red square right of blue circle | Right | Right | Right | right |
| OCR HELLO | HELLO | HELLO | HELLO | HELLO |
| OCR 4827 | 4827 | 4827 | 4827 | 4827 |
| NASA portrait: what is the person wearing? | space suit | space suit | The person is wearing an orange space suit. | space suit |

The last 4-bit answer **fails the historical exact-format check**, but is
semantically correct: the suit really is orange. It is not added to the exact
answer list after the fact. Wording changes only after additional language
re-quantization; the visual weights and projected features are unchanged.
This localizes the wording discrepancy to the quantized language path without
claiming a unique causal layer or a loss of visual understanding.

### Numerical evidence

The independent audit does not call the converter's tensor-name mapper:

- All **338 language tensors** in the floating output exactly match independent
  `gguf.quants.dequantize` followed by FP16 casting.
- All **391 vision/merger tensors** in **both** outputs exactly match the original
  HF `visual.*` weights cast to FP16. This includes full QKV, both temporal
  slices, all visual MLPs and the patch merger.
- Together these cover all **729 output tensors** in the floating model.
  Source tensors are 338 language plus 521 mmproj tensors, including split
  Q/K/V, split temporal kernels and one explicitly ignored zero placeholder.
- Across all twelve inputs, projected visual features have **max absolute
  difference 0** against the HF reference in both output modes.
- Full input embeddings are not identical to HF (max absolute difference
  0.00164794921875), as the language embedding table already underwent GGUF
  quantization. Language logits are likewise not claimed to be identical to HF.

## Reproduce

Use an Apple Silicon machine with adequate RAM and disk space. This run used
64 GiB RAM, Python 3.12.12, MLX 0.32.2, MLX-VLM 0.7.6, MLX-LM 0.31.3,
Transformers 5.16.1, huggingface-hub 1.30.0, gguf 0.19.0, NumPy 2.5.4,
Pillow 12.3.0, and llama.cpp **5490 (`fef693dc`)**.
Use [the isolated runtime requirements](../tests/requirements-vlm.txt);
do not downgrade the project's development environment or modify `mlxgen`.

Fetch the exact revisions above, verify the SHA-256 hashes, and retain the
official HF config, tokenizer, processor, template and both weight shards under
`models/qwen2-vl-validation/hf`. Place the two GGUF files in its sibling `gguf`
directory and the NASA photo at `images/astronaut.png`. Downloads are separate
from evaluation; the runner needs no Hub access. For example, download HF with:

```bash
uv run --isolated --no-project --with-editable . \
  --with-requirements tests/requirements-vlm.txt \
  hf download Qwen/Qwen2-VL-2B-Instruct \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --local-dir models/qwen2-vl-validation/hf \
  --include '*.json' --include '*.safetensors' --include '*.txt' --include LICENSE
```

Convert into new directories (existing outputs are never overwritten):

```bash
gguf2mlx convert \
  --input models/qwen2-vl-validation/gguf/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj models/qwen2-vl-validation/gguf/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  --hf-model models/qwen2-vl-validation/hf --offline \
  --output models/qwen2-vl-validation/mlx-fp16

gguf2mlx convert \
  --input models/qwen2-vl-validation/gguf/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj models/qwen2-vl-validation/gguf/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  --hf-model models/qwen2-vl-validation/hf --offline \
  --quantize --direct-quant --output models/qwen2-vl-validation/mlx-4bit
```

Run [the acceptance harness](../benchmarks/validate_vlm.py):

```bash
HF_HUB_OFFLINE=1 uv run --isolated --no-project --with-editable . \
  --with-requirements tests/requirements-vlm.txt \
  python benchmarks/validate_vlm.py \
  --gguf models/qwen2-vl-validation/gguf/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj models/qwen2-vl-validation/gguf/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
  --hf models/qwen2-vl-validation/hf \
  --converted models/qwen2-vl-validation/mlx-fp16 \
  --direct-quant models/qwen2-vl-validation/mlx-4bit \
  --photo models/qwen2-vl-validation/images/astronaut.png \
  --llama-server /opt/homebrew/bin/llama-server \
  --output models/qwen2-vl-validation/acceptance-new
```

The runner starts a local-only llama-server, waits for health, runs the native
cases, and stops that exact child process even on failure. Then it loads each
MLX model sequentially. It saves the case manifest, model/config/asset hashes,
complete answers, preprocessing hashes, answer NLLs, numerical differences,
allocation/timing measurements, native log and per-tensor audit. A failed check
is explicit; partial results are not presented as success.

The completed local run is under
`models/qwen2-vl-validation/acceptance-v2/`, with `report.json`,
`weight-audit.json`, `manifest.json` and per-runtime JSON files. These generated
artifacts and the reusable model directories are ignored by Git. This document
and the runner are the repository's portable record; local absolute paths in
the raw manifest intentionally identify the actual assets used.

## Regression verification

- Full repository suite after the four-bit follow-up: **492 passed, 36 skipped** (native/real-model tests are
  opt-in in the default run).
- Isolated native vision suite with `GGUF2MLX_RUN_VLM_E2E=1`:
  **312 passed, 1 deselected**; the deselected test is the separately executed
  real-checkpoint smoke test, which **passed** with the pinned assets.
- Repository Ruff checks, both vision modules' mypy checks (Python 3.12),
  whitespace checks, and wheel/source-distribution builds passed.
- CI includes the acceptance harness's scoring/audit/cleanup regressions and
  tiny native model matrix. It does not download the multi-gigabyte checkpoint
  or claim to rerun this real-model diagnostic on every pull request.
