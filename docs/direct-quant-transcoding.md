# Direct GGUF-to-MLX Quantization Design

## Status

This document defines a future low-memory conversion path. The current converter
dequantizes each GGUF tensor to FP16 or FP32, writes safetensors, and optionally
asks MLX-LM to quantize that intermediate model.

The first implementation should optimize memory and temporary disk usage. It
must not claim bit-for-bit preservation because GGUF block formats and MLX
group-wise quantization layouts are not generally interchangeable.

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
