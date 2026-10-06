# Direct GGUF-to-MLX Quantization Design

## Status

This document now tracks the direct quantization path implemented behind
`--direct-quant`. The default converter path still dequantizes GGUF tensors to
FP16/FP32 safetensors and can optionally invoke `mlx_lm.convert`.

The direct path currently targets an initial constrained scope and remains
experimental.

**This is not native transpacking.** It decodes source weights and requantizes
groups of 64. Native transpacking must preserve the source's groups of 32 and
integer codes without calling a destination quantizer. Steps 1–2 below establish
the quality gate and executable layout research before changing that pipeline.

## Implemented in this repository

- CLI flag: `gguf2mlx convert --quantize --direct-quant`
- Bounded-memory shard writing (no full FP16 model directory) for direct mode
- Direct affine 4-bit quantization (`q_bits=4`, `q_mode=affine`, `q_group_size=64`)
- Architecture gate: `llama` and `gemma`
- Source quantization gate for direct-quantized linear projections: `Q4_0`, `Q8_0`
- Non-projection tensors currently follow the existing FP16 fallback path
- Quantization metadata is added to `config.json` only after shard success

## Still pending for full acceptance criteria

- End-to-end `mlx_lm.load()` parity validation and fixed-corpus quality deltas
- Published RSS and output-size comparison against `mlx_lm.convert`
- Wider architecture and source quantization support

## Goals

- Avoid writing a complete FP16 model before producing quantized MLX output.
- Keep peak memory near one source tensor plus one destination tensor group.
- Produce weights that load through the public `mlx_lm.load()` path.
- Record conversion time, peak RSS, output size, and numerical error.
- Fail closed for unsupported source quantization types or tensor transforms.

## Non-goals

- Reinterpret arbitrary GGUF bytes as MLX weights without conversion.
- Promise identical logits after a second quantization step.
- Support every GGUF quantization type in the first release.
- Mix architecture adaptation and quantization-format logic in one mapper.

## Proposed pipeline

1. Read one GGUF tensor through the existing memory-mapped reader.
2. Apply the architecture adapter's rename, split, concatenate, or norm transform.
3. Convert the tensor in bounded row or group chunks.
4. Quantize those chunks into MLX packed weights, scales, and biases.
5. Write a completed safetensors shard and release all temporary arrays.
6. Add the MLX quantization metadata to `config.json` only after all shards succeed.

Architecture transforms happen before destination quantization. Phi-3 QKV fusion,
for example, cannot be implemented as a blind copy of three unrelated GGUF blocks.

## Initial scope

Start with a single destination format:

- MLX affine 4-bit
- group size 64
- two-dimensional linear weights only
- source `Q4_0` and `Q8_0` tensors
- `llama` and `gemma` architecture fixtures

Other tensors should follow the existing FP16 path until mixed-output models are
explicitly supported and tested.

## Adapter boundary

The quantization layer should receive already-normalized output tensors:

```text
GGUF tensor(s)
  -> architecture adapter
  -> logical MLX weight
  -> destination quantizer
  -> packed weight + scales + biases
```

The architecture adapter owns names, shapes, fusion, splitting, and reversible
model-family transforms. The destination quantizer owns group size, packing, and
quantization metadata.

## Acceptance criteria

- `mlx_lm.load()` succeeds and one forward pass returns finite logits.
- Peak RSS is at least 40% lower than the current FP16-then-quantize pipeline.
- No complete FP16 model directory is created.
- Output size is within 5% of normal `mlx_lm.convert` output for the same settings.
- Tokenizer output is byte-for-byte identical between the two conversion paths.
- Logit and perplexity deltas are reported on a fixed calibration corpus.
- Unsupported tensor types stop conversion before the final output is published.

## Benchmark protocol

Use `benchmarks/benchmark_conversion.py` for the current baseline and future
implementation. Record the exact model SHA-256, GGUF quantization, hardware,
macOS version, Python version, and MLX-LM version with every published result.

## Native transpacking prerequisites (steps 1–2)

### Step 1: strict perplexity integration gate

`benchmarks/benchmark_transpacking.py` uses the same
`mlx_lm.perplexity.eval_ppl` mechanism as the existing `--eval-ppl` benchmark,
but evaluates the **original Q4_0 GGUF through native llama.cpp**, via the optional
`llama-cpp-python` binding, against an already-converted local MLX directory.
It does not modify or implicitly convert either model.

On Apple Silicon, from the repository root:

```sh
CMAKE_ARGS="-DGGML_METAL=ON" python -m pip install -e '.[mlx,ppl]'
python benchmarks/benchmark_transpacking.py --eval-ppl \
  --input /absolute/path/model-Q4_0.gguf \
  --mlx-model /absolute/path/converted-mlx \
  --corpus /absolute/path/fixed-corpus.txt \
  --sequence-length 512 --num-samples 32 --add-bos \
  --result-json /absolute/path/ppl-result.json
GGUF2MLX_RUN_E2E=1 python -m pytest tests/test_transpacking_layout.py -q
```

Use a fixed, locally supplied UTF-8 corpus with at least the requested number
of tokens. PPL is `exp(mean(next-token negative log likelihood))`; it is
token-frequency-weighted, not a unigram word-frequency statistic. No corpus is
downloaded, shuffled, or sampled randomly. Both tokenizers must produce exactly
the same token IDs; requested BOS IDs and vocabulary sizes must also match.
Contiguous windows are shared, contexts reset between windows, and every next
token is scored with batch size 1. `--add-bos` prepends BOS to each window;
without it no special tokens are inserted. Trailing unused tokens are reported.
The report path must be new and outside the model directory; existing files
and aliases are never overwritten.
The source must contain Q4_0 tensors and may also contain F32, F16, or Q8_0;
K-quants and all other types are rejected at this MVP stage.

The llama.cpp binding retains every position's raw FP32 logits with
`logits_all=True`; those logits and MLX's logits use the **same loss and
reduction implementation**. The stock `llama-perplexity` CLI instead scores
only the latter part of each context window and prints PPL rounded to four
decimal places. Its scalar cannot be compared directly with the existing
MLX-LM evaluator, nor can its rounded output certify a 0.0 delta. Likewise,
`--save-all-logits` stores quantized log probabilities, not lossless logits.
This runner deliberately uses native llama.cpp inference with a shared scoring
protocol rather than treating that CLI's displayed number as a full-precision
oracle.

Exit status is 0 **only for exact equality of the unrounded PPL floats**;
there is no tolerance, percentage allowance, or rounding before comparison.
Mismatch, tokenizer divergence, unsupported source types, insufficient corpus,
missing runtime dependencies, and evaluation errors return nonzero. Reports
include PPL values and their hexadecimal representations, absolute delta,
standard errors, corpus/model/window hashes, package/platform/native backend
information, and raw FP32 logit hashes. CPU llama.cpp is the default;
`--n-gpu-layers -1` opts into available full GPU offload and is recorded.
Keep backend settings fixed when comparing runs.

**Scientific boundary:** equal aggregate PPL is a necessary project acceptance
gate, not proof of bit-identical weights or inference. Different logits can have
the same loss, and finite-precision reductions can hide differences.
`logits_identical` is reported separately; `bit_exact_transpacking_proven` remains
false even when this quality gate passes. Independent packed-code, scale-byte,
tensor-transform, and decoded-weight checks are required for that claim.
Conversely, exact source weights do not guarantee identical cross-runtime logits
or PPL: kernels, accumulation precision/order, normalization, and attention can
differ. Do not relax the zero-delta gate silently or claim the current
requantizing path has passed it.

### Step 2: Q4_0 and Q8_0 memory layout

For little-endian GGUF blocks (big-endian input requires explicit byte swapping):

| Format | Weights/block | Byte offsets | Decode |
| --- | --- | --- | --- |
| Q4_0 | 32 | 0–1: FP16 `d`; 2–17: 16 packed bytes | `d * (q - 8)` |
| Q8_0 | 32 | 0–1: FP16 `d`; 2–33: 32 signed int8 values | `d * s` |

Q4_0 is **not adjacent low/high nibble order**: the low nibble of byte `j`
represents weight `j`; its high nibble represents weight `j + 16`.
Scales can be negative. Q8_0 stores signed two's-complement values, including
`-128`; interpreting those bytes as unsigned without shifting changes weights.

MLX affine quantization uses unsigned uint32 packed words, with consecutive
codes in increasing bit positions (least significant first).
For a logical `[rows, columns]` matrix, packed shapes are
`[rows, columns / 8]` at 4 bits and `[rows, columns / 4]` at 8 bits.
Group-size-32 scales and quantization biases are `[rows, columns / 32]`.
The affine equation is `w = scale * unsigned_code + quantization_bias`:

- Q4_0: preserve each nibble `q`, reorder to consecutive weight order,
  pack eight codes per word, retain `scale=d`, set `bias=-8*d`.
- Q8_0: promote signed `s` before shifting to `q=s+128`, pack four codes per
  word, retain `scale=d`, set `bias=-128*d`. This is **8-bit MLX output**, not
  a lossless Q8_0-to-4-bit conversion.
- Do not merge two source blocks into a group of 64: their scales may differ.
- Preserve row boundaries, logical axes, and architecture permutations.
  A byte reinterpretation of Q4_0 payloads is not a valid MLX pack.

`mlx.core.quantize` accepts **floating-point weights** and computes a new
quantization; it is not an external-packed-array import API. Already-packed
arrays are consumed by `mlx.core.dequantize` and `mlx.core.quantized_matmul`.
`mlx.nn.QuantizedLinear` can consume externally assigned `weight` (uint32),
`scales`, and `biases` parameters with matching `bits`, `group_size=32`, and
affine mode. Its optional linear-layer `bias` is different from the
quantization `biases`. Safetensors/config loading must retain these settings.

FP16 scale **storage** does not by itself ensure exact affine arithmetic.
Multiplication by 8 or 128 can overflow an FP16 bias even for a finite FP16
scale (the fixtures include this counterexample); intermediate rounding and
cancellation also matter. FP32 scales/biases can retain FP16 scale values
exactly and avoid that bias overflow, but changing parameter precision affects
kernel arithmetic and must be validated independently. No universal lossless
FP16-affine claim is made here.

There is also a signed-zero counterexample even with ordinary exact scales:
negative `d` times signed integer zero yields `-0.0`, whereas affine cancellation
may yield `+0.0`. The portable Q4_0 test explicitly records this bit difference
while requiring exact numerical equality and identical bits for nonzero decoded
values. Do not describe numerical equality as universal decoded-byte equality.

`tests/test_transpacking_layout.py` is executable research only: handcrafted
blocks cover both signs of scale, all Q4 codes, all 256 Q8 codes, nibble ordering,
multiple rows/groups, exact source scale bytes, uint32 word order, signed zero,
and FP16 bias overflow. Portable tests compare decoded FP32 values against `gguf.quants`.
Opt-in Apple Silicon tests import external arrays into `dequantize`,
`quantized_matmul`, and `QuantizedLinear`. They do not implement or enable a
production transpacker. Architecture remapping and complete-model fidelity
remain future work.

### Upstream contracts

- [llama.cpp block definitions](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-common.h)
  and [reference decoding](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/ggml/src/ggml-quants.c).
- [MLX quantize/dequantize/quantized_matmul API](https://ml-explore.github.io/mlx/build/html/reference/quantization.html)
  and [QuantizedLinear implementation](https://github.com/ml-explore/mlx/blob/264c14fe650e1dd9bb0c3902dc06b08033d8f89c/python/mlx/nn/layers/quantized.py).
- [MLX native GGUF repacking](https://github.com/ml-explore/mlx/blob/264c14fe650e1dd9bb0c3902dc06b08033d8f89c/mlx/io/gguf_quants.cpp):
  already implements the Q4_0 nibble reorder and Q8_0 sign-bit flip; prefer
  studying/reusing this mechanism over inventing a second quantizer.
- [MLX-LM PPL scoring](https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/mlx_lm/perplexity.py)
  and [llama-perplexity scoring/storage](https://github.com/ggml-org/llama.cpp/blob/5e03bdd8700948b9c41c54dd1b00f28a2aebc03f/tools/perplexity/perplexity.cpp).
- [Pinned native binding evaluation contract](https://github.com/abetlen/llama-cpp-python/blob/v0.3.36/llama_cpp/llama.py):
  `reset()`, `eval()`, `scores`, and `logits_all`.

### Updated task status

- Step 1: strict runner and portable failure/equality tests established.
- Step 2: MVP layout fixtures and opt-in packed-array consumer tests established.
- Mac validation executed in the `conda` environment `gguf2mlx`:
  Apple Silicon and real Gemma/Qwen1.5 MoE tests enabled -> 319 passed / 2 skipped,
  `ruff check src/ tests/ benchmarks/` passes, and Rust unit tests ->
  3 passed (on macOS these tests require
  `RUSTFLAGS='-C link-arg=-undefined -C link-arg=dynamic_lookup'`).
  Remaining skips need external Qwen3-MoE and DeepSeek2 models, not Gemma or
  packed-array hardware support.
- Q4_0/Q8_0 layout fixtures and external packed-array checks are now covered by
  `tests/test_transpacking_layout.py` (portable + Apple-Silicon opt-in paths).
- Gemma tokenizer identity and unpatched local conversion now pass. The
  real-model strict PPL evaluation has executed, but fails the zero-delta
  requirement; see the recorded results below. No numerical parity is claimed.
- Security scan status depends on scope: dependency audit in this environment
  reports the known `diskcache` advisory (`PYSEC-2026-2447`), while source-level
  static scan findings are pre-existing and unrelated to transpacking research.
- The tokenizer follow-up fixed reconstruction and safetensors shard indexing.
  Subsequent numerical diagnosis also corrected dense GGUF row-major decoding
  and added a Gemma tanh-GELU runtime adapter. Q4_0/Q8_0 dequantization formulas
  and quantization algorithms remain unchanged.
- Core native transpacking implementation and K-quant support are intentionally
  not part of steps 1–2.

### Gemma Mac validation (2026-10-06)

Source: `mlabonne/gemma-2b-it-GGUF/gemma-2b-it.Q4_0.gguf` (first-generation
Gemma, not Gemma 2). SHA-256:
`1047c37fc64926600e08db71f49c86bbdbfb6d4a6af9ab05a2c3d651bf453ea5`.
The GGUF contains 127 Q4_0 and 37 F32 tensors. The standard local conversion
produces FP16 safetensors from this exact GGUF, not independently quantized
community MLX weights.

The original mismatch had two tokenizer reconstruction problems: Gemma's
SentencePiece vocabulary scores encode BPE merge priorities, not Unigram
probabilities, and the GGUF's default space-prefix behavior must be preserved.
The rebuilt tokenizer uses score-ranked BPE merges, preserves whitespace and
combining Unicode characters without NFC normalization, and uses the generic
fast tokenizer loader rather than a Llama-specific loader. Explicit BOS/EOS
flags and token IDs remain intact. The separate `safe_open` indexing crash is
fixed by reading its public `keys()` API.

The opt-in native regression checks vocabulary size, BOS, English, leading and
repeated spaces, tabs, newlines, composed/decomposed Unicode, Chinese, emoji,
and the complete README against llama.cpp. To reproduce:

```sh
GGUF2MLX_RUN_E2E=1 GGUF2MLX_GEMMA_GGUF=/absolute/path/gemma-2b-it.Q4_0.gguf \
  conda run -n gguf2mlx python -m pytest -q -rs tests/
conda run -n gguf2mlx python -m gguf2mlx \
  --input /absolute/path/gemma-2b-it.Q4_0.gguf \
  --output /absolute/path/new-gemma-mlx
conda run -n gguf2mlx python benchmarks/benchmark_transpacking.py --eval-ppl \
  --input /absolute/path/gemma-2b-it.Q4_0.gguf \
  --mlx-model /absolute/path/new-gemma-mlx \
  --corpus /absolute/path/fixed-corpus.txt \
  --sequence-length 128 --num-samples 8 --add-bos --n-gpu-layers 0 \
  --result-json /absolute/path/new-ppl-result.json
```

The evaluated corpus is a frozen README snapshot, SHA-256
`25e7e6231689abdb31a825f184ded778ea347b9f34d2698469a06dbaf441a581`;
both tokenizers produced the same 6,748 IDs. Eight contiguous windows of 128
tokens (BOS included) score 1,016 next tokens. This small repository-text corpus
is a reproducibility check, not a standard language-quality benchmark.

| Measurement | llama.cpp (CPU) | MLX (FP16 conversion) |
|---|---|---|
| Unrounded PPL | 143.58573263274607 | 143.17148969720395 |
| PPL hexadecimal | `0x1.1f2be525cbb05p+7` | `0x1.1e57cd7f622f1p+7` |
| Standard error | 34.199934274650985 | 34.08338813547865 |

Absolute delta is **0.41424293554212**, tolerance is **0.0**, exit status is
**1**, and raw logit hashes differ. Both evaluations completed; the failure is
no longer a tokenizer precondition failure. Cross-runtime numerical parity
remains unproven and the gate has not been relaxed.

Environment: macOS 27.0.1 arm64, conda `gguf2mlx`, Python 3.11.13, gguf 0.18.0,
NumPy 2.3.3, MLX 0.31.2, MLX-LM 0.31.3, llama-cpp-python 0.3.36.
`bit_exact_transpacking_proven` remains false.

### Numerical isolation and precision controls

All runs below use the same frozen corpus, identical token windows and scoring
protocol above. Strict tolerance remains **0.0**; every PPL comparison below
returns exit status **1**, with differing raw-logit hashes.

| Control after activation alignment | llama.cpp PPL | MLX PPL | Absolute delta |
|---|---|---|---|
| Original Q4_0, native CPU / dense MLX FP16 | 143.58573263274607 | 143.14397973754745 | 0.4417528951986185 |
| Original Q4_0, native CPU / dense MLX FP32 | 143.58573263274607 | 143.14732433867772 | 0.4384082940683527 |
| Same decoded Q4 weights, dense F32 GGUF CPU / MLX FP32 | 143.1465735030501 | 143.14732433867772 | 0.0007508356276275663 |
| Original Q4_0, native Metal (`n_gpu_layers=-1`) / MLX FP32 | 143.15920171702348 | 143.14732433867772 | 0.011877378345758416 |
| Unquantized HF mirror cast to FP16, native CPU / MLX FP16 | 117.02195028668703 | 117.10903131716553 | 0.08708103047850102 |

The final row uses public `alpindale/gemma-2b-it` **BF16** HF weights cast to
FP16, with source GGUF tokenizer/config metadata retained on both sides. It
is not an authenticated download of Google's official FP16 checkpoint, nor
proof that the mirror and community Q4_0 share the same original weights.
Compare each row's two engines, not the absolute PPL across different weight
sources. F32 norm vectors are retained for native Gemma CPU compatibility;
making them F16 causes a native unsupported-type abort before scoring.

Verified causes and boundaries:

1. **Q4 decode is exact.** `benchmarks/audit_q4_decode.py` compares every value
   against the actual loaded library's `dequantize_row_q4_0`. All **127**
   Q4_0 tensors, **2,506,096,640** values have **zero bit differences**.
   FP32 -> FP16 -> FP32 round trips also have zero bit differences for these
   decoded values. This does not audit every arbitrary future Q4 model.
2. **Scales are not native MLX packed arrays in the standard path.** GGUF `d`
   is stored as FP16, promoted exactly to FP32 by the reference decoder, then
   decoded dense weights are cast to the requested output dtype. Changing
   this exact FP16-to-FP32 promotion cannot explain the measured drift.
3. **Gemma norm restoration is not missing or doubled here.** All **75,776**
   norm values restore/reapply the `-1`/`+1` convention exactly in both tested
   FP16 and FP32 paths, relative to their corresponding dtype input.
4. **Native Q4 CPU and dense execution are different computations.** The
   loaded native library reports ggml `0.25.3`, commit `0c1e570-dirty`.
   Its `ggml_get_type_traits_cpu(Q4_0).vec_dot_type` is **Q8_0**: Q4 dot
   products use quantized activation operands, while dense MLX matmul does
   not. Even native CPU alone changes from **143.58573263274607** to
   **143.1465735030501** with identical decoded weights stored as F32.
   Metal offload greatly reduces but does not eliminate the cross-engine gap.
5. **A real graph discrepancy was repaired.** Installed MLX-LM 0.31.3 Gemma
   uses erf GELU while the converter config requests `gelu_pytorch_tanh` and
   native Gemma uses tanh GELU. A bundled local `gemma_model.py` honors this
   config using MLX-LM's supported `model_file` loader. With the same dense
   F32 weights, the PPL gap falls from **0.05256800605832268** to
   **0.0007508356276275663**. The smaller residual is still a strict failure;
   its individual contributing kernels have not been isolated.
6. **Dense GGUF layout was repaired independently.** Reader tensor shapes
   are in GGML axis order but data is NumPy row-major. Reshaping to GGML
   order and then transposing scrambled dense F16/F32 matrices. Regressions
   cover non-square matrices and every supported dense scalar dtype. This
   bug affected clean dense controls, not the original Q4 block decoder.

Reproduce controls without overwriting any source:

```sh
conda run --no-capture-output -n gguf2mlx python benchmarks/audit_q4_decode.py \
  --input /absolute/path/gemma-2b-it.Q4_0.gguf \
  --result-json /absolute/path/q4-decode-audit.json
conda run --no-capture-output -n gguf2mlx python benchmarks/materialize_precision_control.py \
  --input /absolute/path/gemma-2b-it.Q4_0.gguf \
  --output /absolute/path/new-decoded-f32.gguf --dtype float32
# Optional unquantized mirror control instead of decoded Q4 weights:
conda run --no-capture-output -n gguf2mlx python benchmarks/materialize_precision_control.py \
  --input /absolute/path/gemma-2b-it.Q4_0.gguf \
  --hf-model /absolute/path/local-unquantized-hf-mirror \
  --output /absolute/path/new-hf-f16.gguf --dtype float16
conda run --no-capture-output -n gguf2mlx python -m gguf2mlx \
  --input /absolute/path/new-decoded-f32.gguf \
  --output /absolute/path/new-decoded-f32-mlx --dtype float32
conda run --no-capture-output -n gguf2mlx python benchmarks/benchmark_transpacking.py \
  --eval-ppl --unquantized-control --input /absolute/path/new-decoded-f32.gguf \
  --mlx-model /absolute/path/new-decoded-f32-mlx \
  --corpus /absolute/path/fixed-corpus.txt \
  --sequence-length 128 --num-samples 8 --add-bos --n-gpu-layers 0 \
  --result-json /absolute/path/new-dense-control-ppl.json
```

Use matching control/input paths and `--dtype float16` for the HF-derived
FP16 row. `--unquantized-control` explicitly permits only F16/F32 weights; it
does not disable tokenizer checks, logit checks or the strict equality gate.
Reports hash the local executable model adapter as well as weights/config.
The adapter is included in built wheels and preserved by standard
`--quantize`/MLX-LM conversion. Direct-quant adapter publication is covered
separately; it does not certify direct-quant numerical or loader parity.
An additional probe found the direct path emitted byte-packed uint8 weights,
while MLX-LM 0.31.3 expects uint32-packed weights. The subsequent Qwen MoE
validation fixes this contract: eight 4-bit codes per uint32 word, low code in
the low nibble, and a 64-column projection has shape `(rows, 8)`. Synthetic
Gemma and Qwen2MoE standard/direct outputs now load and produce finite logits.
This is affine re-quantization, not a source-bit-preserving transpack.

Status: diagnosis and real-model evaluations completed; **zero-tolerance
acceptance remains blocked**, not passed. Exact decode bytes and corrected
formulas do not guarantee identical floating-point execution across these
backends. No corpus selection, result rounding, tolerance increase or
same-engine substitution was used.
