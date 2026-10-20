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
directory with `config.json`, tokenizer assets and safetensors weights. `gguf2mlx`
bridges that gap for **explicitly supported** architectures by reading GGUF
metadata and tensors, rebuilding the MLX-compatible layout, and **failing closed**
when a model layout is not actually supported — no silent "Llama fallback".

Use it when a model ships GGUF but no MLX checkpoint, or when you need an
inspectable HF-style directory on an M1–M4 Mac. If the original Hugging Face
weights are available, prefer `mlx_lm.convert` — it avoids inheriting the source
GGUF's quantization error.

Conversion itself only needs `numpy`, `gguf` and `safetensors`, so it runs fine
on Linux servers and CI; MLX is required only for `--quantize`.

> **Important:** the default output is FP16/BF16 safetensors, and the optional
> 4-bit modes **re-quantize**; they are not bit-for-bit preservation of the
> original GGUF quantization blocks. The single exception is the Qwen2-VL
> `--preserve-source-quant` mode documented in the
> [VLM guide](docs/vlm-conversion.md).

## Install

```bash
pip install gguf2mlx            # installs MLX + MLX-LM on Apple Silicon
pipx install gguf2mlx           # isolated CLI
uv tool install gguf2mlx        # or with uv
```

Extras: `[vlm]` adds Hugging Face Hub for VLM conversion, `[all]` everything,
`[dev]` the test/lint toolchain.

## Quick start

```bash
# GGUF -> FP16 MLX-LM directory
gguf2mlx --input model.gguf --output ./mlx-model

# GGUF -> 4-bit MLX-LM output
gguf2mlx --input model-Q4_K_M.gguf --output ./mlx-model-4bit \
  --quantize --q-bits 4 --q-group-size 64

# Bounded-memory direct quantization (no FP16 staging directory)
gguf2mlx --input model-Q4_K_M.gguf --output ./mlx-model-4bit \
  --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine

# Inspect metadata without writing weights
gguf2mlx --input model.gguf --skip-weights

# Smaller shards (payload bytes, not source bytes)
gguf2mlx --input model.gguf --output ./mlx-model --max-shard-gb 2

# Package / verify a directory as a .mlx bundle
gguf2mlx package --model-dir ./mlx-model --output ./mlx-model.mlx
gguf2mlx verify --bundle ./mlx-model.mlx
```

Load the result:

```python
from mlx_lm import load, generate
model, tok = load("./mlx-model")
print(generate(model, tok, prompt="Hello from MLX", max_tokens=32))
```

See [How conversion works](docs/how-conversion-works.md) for streaming, sharding
and offline tokenizer reconstruction.

## Supported models

Conversion is adapter-based, not "any GGUF". Rows marked ⚠️ convert but still
need opt-in real-checkpoint validation.

| Family | GGUF architecture IDs | Convert | End-to-end evidence |
|---|---|---|---|
| Llama | `llama`, `mistral` | ✅ | synthetic direct-quant; SmolLM2-135M HF parity |
| Qwen | `qwen2`, dense `qwen3` | ✅ | synthetic direct-quant |
| Qwen3.5 family | `qwen35` + aliases | ✅ | portable fixtures |
| Qwen MoE | `qwen2moe`, `qwen3moe` | ✅ | real GGUF: Qwen1.5-MoE, Qwen3-30B-A3B |
| DeepSeek | `deepseek2`, `deepseek3` | ✅ | real GGUF: V2 Lite Q2_K |
| GLM | `glm4moe`, `glm-dsa` | ✅ / ⚠️ experimental | none |
| Gemma | `gemma`, `gemma2`, `gemma3`, text `gemma4` | ✅ | portable fixtures |
| Phi | `phi3` (Phi-3, Phi-4) | ✅ | synthetic GGUF |
| StableLM | `stablelm` | ✅ | none |
| LLaVA / Qwen2-VL / MiniCPM-V | mmproj pairs | ✅ | Qwen2-VL-2B real-image acceptance |

**Inspection-only** (recognized via metadata, rejected on conversion): `arctic`,
`bert`, `bitnet`, `bloom`, `dbrx`, `falcon`, `gpt2`, `gptneox`, `mpt`, `olmo`,
`phi`, `phi2`, `t5` and others — full list in the
[support matrix](docs/support-matrix.md).

## Vision-language models

Three explicit adapters emit directories for `mlx-vlm`: **LLaVA-1.5**,
**Qwen2-VL**, and **MiniCPM-V 4.7 1B**. Language weights and the projector come
from GGUF; hybrid mode takes only the vision tower from a matching HF
checkpoint. Qwen2.5-VL, Qwen3-VL full VLM, LLaVA-NeXT and SigLIP layouts are
**not** supported by these adapters.

```bash
pip install "gguf2mlx[vlm]"
gguf2mlx convert --type vlm \
  --input qwen2-vl.gguf --mmproj qwen2-vl-mmproj.gguf \
  --hf-model Qwen/Qwen2-VL-2B-Instruct --output ./qwen2-vl-mlx
```

Flags, hybrid/offline paths, quantization policy and the Q4_K
`--preserve-source-quant` mode: [VLM conversion](docs/vlm-conversion.md). Native
acceptance results: [Qwen2-VL report](docs/qwen2-vl-real-validation.md).

Qwen3-VL **text-only** extraction is a separate `--type text-encoder` contract
(BF16 by default, mRoPE preserved, vision tensors skipped): see
[text conditioning](docs/text-encoder.md).

## Limits worth knowing before you start

- **4-bit output is a second quantization** unless you use the Qwen2-VL
  preserve mode; it does not repack original GGUF blocks.
- **Conversion cannot restore precision** lost in the source GGUF.
- **Tokenizer fidelity depends on GGUF metadata**; embedded HF tokenizer JSON is
  preserved when present, otherwise reconstructed from the header.
- **Mixed-precision MoE output is experimental** and needs a JANG-compatible
  runtime; use uniform output when interoperability is unverified.
- **No universal speed claim** versus llama.cpp — measure on your hardware.

## Documentation

| Page | Contents |
|---|---|
| [Support matrix](docs/support-matrix.md) | Every architecture, its status, validation evidence |
| [How conversion works](docs/how-conversion-works.md) | Pipeline, streaming, tokenizer reconstruction |
| [VLM conversion](docs/vlm-conversion.md) | LLaVA / Qwen2-VL / MiniCPM-V flags and limits |
| [Text conditioning](docs/text-encoder.md) | Qwen-VL text encoder contract |
| [Publishing](docs/publishing.md) | Reusable `publish-mlx` workflow, PyPI releases |
| [`.mlx` bundles](docs/mlx-bundle.md) | Packaging and integrity verification |
| [Capabilities](docs/capabilities.md) | Feature status table |
| [Benchmarks](docs/benchmarks.md) | Conversion/RSS/PPL harness |
| [FAQ](docs/faq.md) | Common questions |
| [Roadmap](docs/roadmap.md) | Done, and open for contributors |
| [Validation](docs/validation/) | Native run logs and real-model reports |
| [Development](docs/development.md) | Test suites, opt-in fixtures, Rust extension |

## Development

```bash
git clone https://github.com/barrontang/gguf2mlx.git && cd gguf2mlx
uv sync --all-extras
pytest -q                                   # portable suite
GGUF2MLX_RUN_E2E=1 pytest tests/test_e2e.py # Apple Silicon, real GGUF
ruff check src/ tests/ benchmarks/
```

CI runs Python 3.10/3.12/3.13 with an 80% coverage floor; a separate macOS
workflow runs the native VLM suites. Details and opt-in fixture paths are in
[Development](docs/development.md).

## Contributing

New architecture adapters must ship with a tensor manifest and a load test.
When requesting a family, include the exact GGUF architecture, model name,
quantization type, tokenizer type, and a public fixture URL.

## License

Apache-2.0. The traffic-tracking dashboard that used to live here was GPL-3.0 and
now lives in its own repository.
