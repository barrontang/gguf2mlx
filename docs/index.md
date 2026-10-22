# gguf2mlx

**GGUF → HF-style safetensors that MLX-LM and `mlx-vlm` can load.**
Explicit architecture contracts, no silent precision downgrade, fail-closed when a
layout is not supported.

[PyPI package](https://pypi.org/project/gguf2mlx/) ·
[Source](https://github.com/barrontang/gguf2mlx) ·
[Issues](https://github.com/barrontang/gguf2mlx/issues)

---

## What it does

GGUF is a distribution format; MLX-LM and `mlx-vlm` expect a Hugging Face-style
directory. `gguf2mlx` bridges that for **explicitly supported** architectures by
reading GGUF metadata and tensors, rebuilding the MLX layout, and **failing closed**
when a layout is not actually supported — there is no silent "Llama fallback".

Use it when a model ships GGUF but no MLX checkpoint. If the original HF weights
exist, prefer `mlx_lm.convert`, which avoids inheriting the GGUF's quantization
error.

!!! note "Default output is FP16/BF16"
    `--quantize` re-quantizes from that copy. To keep the bytes already in the
    GGUF, use `--quant-source preserve` — see
    [Quality gates](quality-gates.md#preserve-tier).

## Install

```bash
pip install gguf2mlx            # installs MLX + MLX-LM on Apple Silicon
pipx install gguf2mlx           # isolated CLI; or: uv tool install gguf2mlx
# extras: [vlm] / [publish] add Hugging Face Hub, [all] everything, [dev] toolchain
```

Only `numpy`, `gguf` and `safetensors` are required, so conversion runs on Linux
and CI. MLX is needed only for `--quantize`.

## Quick start

```bash
# GGUF -> FP16 MLX-LM directory
gguf2mlx --input model.gguf --output ./mlx-model

# GGUF -> 4-bit with bounded memory (no FP16 staging directory)
gguf2mlx --input model-Q4_K_M.gguf --output ./mlx-model-4bit \
  --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine

# Reuse the GGUF blocks instead of quantizing the FP16 copy again
gguf2mlx convert --input model-Q4_K_M.gguf --output ./mlx-model-preserved \
  --quant-source preserve
```

```python
from mlx_lm import load, generate
model, tok = load("./mlx-model")
print(generate(model, tok, prompt="Hello from MLX", max_tokens=32))
```

`--force` replaces a non-empty `--output`, `--json` prints one machine-readable
result object, and `-v`/`-q` control stderr diagnostics.

## Trusting the output

Every conversion writes `conversion_report.json` and gates on **weight fidelity**:
each written tensor is compared position-wise against an independent decode of the
source GGUF, with a per-dtype relative-L2 tolerance. Fused or permuted tensors are
reported as `transformed` rather than silently counted as verified.

```bash
gguf2mlx verify --model-dir ./mlx-model --weights --source model.gguf
gguf2mlx verify --model-dir ./mlx-model --load
```

Perplexity is recorded but never gated: cross-runtime PPL mixes quantization noise
with conversion error, so it cannot prove a converter correct. Nightly, the pinned
load matrix converts real GGUFs and proves each one loads and runs.

## Limits worth knowing

- **`--quantize` is a second quantization.** Conversion cannot restore precision
  already lost in the source GGUF, and tokenizer fidelity depends on GGUF metadata.
- **Mixed-precision MoE output is experimental** and needs a JANG-compatible
  runtime; use uniform output when interoperability is unverified.
- **No universal speed claim** versus llama.cpp — measure on your hardware.

## Where to go next

| If you want to… | Read |
|---|---|
| Check whether your model converts | [Supported conversion matrix](support-matrix.md) |
| Understand the pipeline | [How conversion works](how-conversion-works.md) |
| Convert a vision model | [VLM conversion](vlm-conversion.md) |
| Push to the Hugging Face Hub | [Publishing](publishing.md) |
| Audit a conversion | [Quality gates](quality-gates.md) |
| See real-run evidence | [Validation logs](validation/README.md) |

## License

Apache-2.0. Repository traffic statistics are tracked separately in
[`barrontang/gguf2mlx-traffic`](https://github.com/barrontang/gguf2mlx-traffic)
(GPL-3.0-or-later); see [Stats](stats.md).
