# gguf2mlx — convert GGUF to MLX without guessing

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/gguf2mlx.svg)](https://pypi.org/project/gguf2mlx/)
[![CI](https://github.com/barrontang/gguf2mlx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/barrontang/gguf2mlx/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**GGUF → HF-style safetensors that MLX-LM and `mlx-vlm` can load.
Explicit architecture contracts, no silent precision downgrade, fail-closed when a layout is not supported.**

[Quick start](#quick-start) · [Supported models](#supported-models) · [Vision](#vision-language-models) · [Docs](#documentation)

</div>

---

## Why this exists

GGUF is a distribution format; MLX-LM and `mlx-vlm` expect a Hugging Face-style
directory. `gguf2mlx` bridges that for **explicitly supported** architectures by
reading GGUF metadata and tensors, rebuilding the MLX layout, and **failing
closed** when a layout is not actually supported — no silent "Llama fallback".
Use it when a model ships GGUF but no MLX checkpoint; if the original HF weights
exist, prefer `mlx_lm.convert`, which avoids inheriting the GGUF's quantization
error. Only `numpy`, `gguf` and `safetensors` are needed, so it runs on Linux and
CI; MLX is required only for `--quantize`.

> **The default output is FP16/BF16.** `--quantize` re-quantizes from that copy.
> To keep the bytes already in the GGUF, use `--quant-source preserve`
> (see [Preserve GGUF quantization](#preserve-gguf-quantization)).

## Install

```bash
pip install gguf2mlx            # installs MLX + MLX-LM on Apple Silicon
pipx install gguf2mlx           # isolated CLI; or: uv tool install gguf2mlx
# extras: [vlm] / [publish] add Hugging Face Hub, [all] everything, [dev] toolchain
```

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

# Package / verify a directory as a .mlx bundle: see docs/mlx-bundle.md
```

```python
from mlx_lm import load, generate
model, tok = load("./mlx-model")
print(generate(model, tok, prompt="Hello from MLX", max_tokens=32))
```

`--force` replaces a non-empty `--output`, `--json` prints one machine-readable
result object, and `-v`/`-q` control stderr diagnostics. Streaming, sharding and
offline tokenizer reconstruction: [How conversion works](docs/how-conversion-works.md).

## Is the output trustworthy?

Every conversion writes `conversion_report.json` and gates on **weight fidelity**:
each written tensor is compared position-wise against an independent decode of the
source GGUF, with a per-dtype relative-L2 tolerance. Fused or permuted tensors are
reported as `transformed` rather than silently counted as verified.

```bash
# Re-audit an existing artifact against the GGUF it came from
gguf2mlx verify --model-dir ./mlx-model --weights --source model.gguf

# Load it with mlx_lm and require finite logits from one forward pass
gguf2mlx verify --model-dir ./mlx-model --load
```

Perplexity is recorded but never gated: cross-runtime PPL mixes quantization noise
with conversion error, so it cannot prove a converter correct. Nightly, the pinned
[load matrix](tests/load_matrix.json) converts real GGUFs and proves each one loads
and runs (11/11 across llama, qwen2, qwen3, gemma2 and phi3).

## Preserve GGUF quantization

`--quant-source preserve` re-expresses the GGUF blocks that are already there as
MLX affine groups (`w = scale * code + bias`) on the blocks' 32-value grid instead
of decoding to FP16 and quantizing again: the codes are kept and only the fp16
rounding of scale and bias is new. Q4_K lands at 2.0-2.4e-4 relative L2 and Q8_0 is
bit-exact; Q6_K, whose 16-value sub-block scales cannot fit a 32-value group when
they disagree, is re-fitted at the source width and gated separately, keeping a
Q6_K artifact at **1.067x** the GGUF instead of the 2.4x the old FP16 fallback
produced (`--preserve-fallback 8bit|float16` trades error for size). It fails
closed for anything but `llama`, `qwen2`, `qwen3` and `mistral`.
Numbers and tolerances: [Preserve tier](docs/quality-gates.md#preserve-tier).

```bash
gguf2mlx convert --input model-Q8_0.gguf --output ./mlx-model-q8 --quant-source preserve
```

## Publish to the Hugging Face Hub

`gguf2mlx publish` downloads a hosted GGUF, converts it, and uploads the result
with a provenance record and the source license files. Split GGUFs are merged, a
partial shard set is rejected before anything is downloaded, and the destination
repository is created only once conversion produced weights.

```bash
pip install "gguf2mlx[publish]"
gguf2mlx publish --source bartowski/Model-GGUF --gguf Model-Q8_0.gguf \
  --dest my-org/Model-MLX --source-revision main
```

`publish-mlx.yml` is a thin shell around the same module. Details:
[Publishing](docs/publishing.md).

## Supported models

Conversion is adapter-based, not "any GGUF"; ⚠️ rows convert but need opt-in
real-checkpoint validation.

| Family | GGUF architecture IDs | Convert | End-to-end evidence |
|---|---|---|---|
| Llama | `llama`, `mistral` | ✅ + preserve | Llama-3.2-1B Q4_0/Q4_K_M/Q8_0 in the pinned load matrix |
| Qwen | `qwen2`, dense `qwen3` | ✅ + preserve | Qwen2.5-0.5B, Qwen3-0.6B in the pinned load matrix |
| Qwen3.5 family | `qwen35` + aliases | ✅ | portable fixtures |
| Qwen MoE | `qwen2moe`, `qwen3moe` | ✅ | real GGUF: Qwen1.5-MoE, Qwen3-30B-A3B |
| DeepSeek | `deepseek2`, `deepseek3` | ✅ | real GGUF: V2 Lite Q2_K |
| GLM | `glm4moe`, `glm-dsa` | ✅ / ⚠️ experimental | none |
| Gemma | `gemma`, `gemma2`, `gemma3`, text `gemma4` | ✅ | Gemma-2-2B in the pinned matrix; others fixtures |
| Phi | `phi3` (Phi-3, Phi-4) | ✅ | Phi-3-mini in the pinned matrix |
| StableLM | `stablelm` | ✅ | none |
| LLaVA / Qwen2-VL / MiniCPM-V | mmproj pairs | ✅ | Qwen2-VL-2B real-image acceptance |

`preserve` means `--quant-source preserve` is available (Q4_K, Q6_K, Q8_0); other
architectures convert but must requantize. Convertible architectures without a
pinned case are listed with reasons in the [support matrix](docs/support-matrix.md),
and a test keeps that list exact. **Inspection-only** families (recognized via
metadata, rejected on conversion) are listed in the same page.

## Vision-language models

Three explicit adapters emit directories for `mlx-vlm`: **LLaVA-1.5**,
**Qwen2-VL**, and **MiniCPM-V 4.7 1B**. Language weights and the projector come
from GGUF; hybrid mode takes only the vision tower from a matching HF checkpoint.
Qwen2.5-VL, Qwen3-VL full VLM, LLaVA-NeXT and SigLIP layouts are **not** supported.

```bash
pip install "gguf2mlx[vlm]"
gguf2mlx convert --type vlm \
  --input qwen2-vl.gguf --mmproj qwen2-vl-mmproj.gguf \
  --hf-model Qwen/Qwen2-VL-2B-Instruct --output ./qwen2-vl-mlx
```

Flags, hybrid/offline paths and quantization policy: [VLM conversion](docs/vlm-conversion.md).
Native acceptance: [Qwen2-VL report](docs/qwen2-vl-real-validation.md). Qwen3-VL
**text-only** extraction is a separate `--type text-encoder` contract: see
[text conditioning](docs/text-encoder.md).

## Limits worth knowing before you start

- **`--quantize` is a second quantization.** Use `--quant-source preserve` to keep
  the original blocks; Q6_K groups with disagreeing sub-block scales are re-fitted
  at the source width, not restored. **Conversion cannot restore precision** lost
  in the source GGUF, and **tokenizer fidelity depends on GGUF metadata** —
  embedded HF tokenizer JSON is kept, otherwise the tokenizer is reconstructed.
- **Mixed-precision MoE output is experimental** and needs a JANG-compatible
  runtime; use uniform output when interoperability is unverified.
- **No universal speed claim** versus llama.cpp — measure on your hardware.

## Documentation

Full documentation site: **<https://barrontang.github.io/gguf2mlx/>** — built from `docs/` with
MkDocs by `.github/workflows/docs.yml` (`mkdocs serve` locally).

[Quality gates](docs/quality-gates.md) · [Support matrix](docs/support-matrix.md) · [How conversion works](docs/how-conversion-works.md) ·
[VLM conversion](docs/vlm-conversion.md) · [Text conditioning](docs/text-encoder.md) · [Publishing](docs/publishing.md) · [`.mlx` bundles](docs/mlx-bundle.md) ·
[Capabilities](docs/capabilities.md) · [Benchmarks](docs/benchmarks.md) · [FAQ](docs/faq.md) · [Roadmap](docs/roadmap.md) · [Validation logs](docs/validation/) · [Development](docs/development.md)

## Development

```bash
git clone https://github.com/barrontang/gguf2mlx.git && cd gguf2mlx
uv sync --all-extras
pytest -q                                   # portable suite
GGUF2MLX_RUN_E2E=1 pytest tests/test_e2e.py # Apple Silicon, real GGUF
ruff check src/ tests/ benchmarks/ && mypy src
```

CI runs Python 3.10/3.12/3.13 with an 80% coverage floor and a mypy gate; a
separate macOS workflow runs the native VLM suites. New adapters must ship with a
tensor manifest and a load test, and a family request should name its GGUF
architecture, model, quantization type, tokenizer type and a public fixture.
Details: [Development](docs/development.md).

## License

Apache-2.0. The traffic-tracking dashboard that used to live here was GPL-3.0 and now lives in its own repository.
