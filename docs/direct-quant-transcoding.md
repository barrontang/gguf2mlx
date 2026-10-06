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

- [llama.cpp block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
  and [reference decoding](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).
- [MLX quantize/dequantize/quantized_matmul API](https://ml-explore.github.io/mlx/build/html/reference/quantization.html)
  and [QuantizedLinear implementation](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py).
- [MLX-LM PPL scoring](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/perplexity.py)
  and [llama-perplexity scoring/storage](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp).
- [Pinned native binding evaluation contract](https://github.com/abetlen/llama-cpp-python/blob/v0.3.36/llama_cpp/llama.py):
  `reset()`, `eval()`, `scores`, and `logits_all`.

### Updated task status

- Step 1: strict runner and portable failure/equality tests established.
- Step 2: MVP layout fixtures and opt-in packed-array consumer tests established.
- Real-model zero-delta acceptance and Apple Silicon consumer execution must be
  run on a Mac with the original model and fixed corpus; no real-model parity
  result is claimed from Linux-only tests.
- Core native transpacking implementation and K-quant support are intentionally
  not part of steps 1–2.
