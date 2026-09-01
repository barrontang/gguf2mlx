# GGUF -> MLX

<div align="center">

**Convert supported GGUF checkpoints into MLX-LM-loadable safetensors on Apple Silicon.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://github.com/barrontang/gguf2mlx)
[![Validation](https://img.shields.io/badge/conversion-strict-purple)](https://github.com/barrontang/gguf2mlx)

</div>

---

## Why this repo exists

GGUF is great for distribution, but MLX and MLX-LM expect a Hugging Face-style
directory with `config.json`, tokenizer assets, and safetensors weights.

`gguf2mlx` bridges that gap for **supported architectures** by:

- reading GGUF metadata and tensors
- rebuilding MLX-LM-compatible model artifacts
- failing closed when a model layout is not actually supported

> **Important:** the current converter writes **HF-style safetensors for MLX-LM**.
> It does **not** yet emit MLX-native quantized weights, so quantized GGUF files
> are currently expanded to FP16 or FP32 during conversion.

---

## Quick start

### Install

```bash
# Base converter
pip install "gguf2mlx @ git+https://github.com/barrontang/gguf2mlx.git"

# Converter + MLX runtime for loading converted models
pip install "gguf2mlx[mlx] @ git+https://github.com/barrontang/gguf2mlx.git"

# Or with uv
uv add "gguf2mlx[mlx] @ git+https://github.com/barrontang/gguf2mlx.git"
```

The `mlx` extra installs both `mlx` and `mlx-lm`.

### Install from a local checkout

If you cloned the repo and want the `gguf2mlx` command in your current
environment, install it first:

```bash
python -m pip install -e .
# or with MLX / MLX-LM support
python -m pip install -e ".[mlx]"
```

### Convert

```bash
# Basic conversion
gguf2mlx --input model.gguf --output ./mlx-model

# GGUF -> 4-bit MLX in one command
gguf2mlx --input model-Q4.gguf --output ./mlx-model-4bit --quantize --q-bits 4 --q-group-size 64

# Float32 output
gguf2mlx --input model.gguf --output ./mlx-model-f32 --dtype float32

# Inspect metadata without writing weights
gguf2mlx --input model.gguf --skip-weights
```

### Load with MLX-LM

```bash
python -c "
from mlx_lm import load, generate
model, tok = load('./mlx-model')
print(generate(model, tok, prompt='Hello from MLX', max_tokens=32))
"
```

### Convert the MLX output to 4-bit

`gguf2mlx` can optionally run `mlx_lm.convert` for you. The one-command form is:

```bash
gguf2mlx \
  --input model-Q4.gguf \
  --output ./mlx-model-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

If you want the manual two-step version, it is:

```bash
# 1) GGUF -> FP16 MLX-LM-style safetensors
gguf2mlx --input model-Q4.gguf --output ./mlx-model

# 2) FP16 safetensors -> 4-bit MLX
mlx_lm.convert \
  --model ./mlx-model \
  --mlx-path ./mlx-model-4bit \
  -q \
  --q-bits 4 \
  --q-group-size 64
```

Then load the quantized output normally:

```bash
python -c "
from mlx_lm import load
model, tok = load('./mlx-model-4bit')
print('loaded')
"
```

---

## What is solid today

| Capability | Status |
|---|---|
| Strict architecture validation | Supported |
| Atomic staging and cleanup on failure | Supported |
| Correct `torch_dtype` propagation | Supported |
| Correct zero-valued special token IDs | Supported |
| GGUF context length -> `model_max_length` | Supported |
| MLX / MLX-LM optional dependency | Supported |
| Regression tests for recent fixes | Supported |

---

## Supported conversion matrix

`gguf2mlx` now distinguishes between:

1. **recognized for inspection** via GGUF metadata, and
2. **verified for weight conversion** with dedicated adapters.

### Verified weight conversion

| Family | Architectures |
|---|---|
| Llama | `llama`, `mistral` |
| Qwen | `qwen2`, `qwen2moe`, `qwen3moe` |
| DeepSeek | `deepseek2`, `deepseek3` |
| GLM | `glm4moe` |
| GLM (experimental) | `glm-dsa` |

### Inspection only (not converted)

These may still be recognized by metadata or `--skip-weights`, but they are
**rejected during conversion** until a dedicated, tested adapter exists:

`falcon`, `gpt2`, `bert`, `bloom`, `mpt`, `dbrx`, `gemma`, `phi`, `gptneox`,
`stablelm`, `olmo`, `cohere`, `granite`, `nemotron`, `exaone`, `openelm`,
`command-r`, `baichuan`, `xverse`, `orion`, `bitnet`, `plamo`, `codeshell`,
`minicpm`, `t5`, `jais`, `arctic`, `smolm`, `chameleon`, and others without a
verified adapter.

That means no more silent "Llama fallback" producing invalid outputs for
unrelated architectures.

---

## How conversion works

1. Read GGUF metadata and detect the architecture
2. Validate that the architecture has a supported adapter
3. Build `config.json` using source metadata and selected dtype
4. Export tokenizer assets from GGUF metadata
5. Dequantize tensors to FP16 or FP32
6. Remap tensor names into the target Hugging Face / MLX-LM layout
7. Write sharded safetensors atomically into the output directory

If any required step fails, conversion fails and the staged output is cleaned up.

---

## Current limitations

This project is intentionally more honest about scope now:

- **No MLX-native quantized output yet**; quantized GGUF expands to FP16/FP32
- **Tokenizer fidelity is still evolving** across SentencePiece, Unigram, and
  architecture-specific tokenizer variants
- **Architecture coverage is adapter-based**, not "all GGUF models"
- **Performance claims depend on model, prompt, hardware, and MLX-LM version**

If you need guaranteed support for a new family, open an issue with the exact
GGUF architecture and source model.

---

## Benchmarks

These numbers reflect **conversion output size**, not a universal inference-speed claim.

| Model | GGUF Size | Quant | Convert Time (M4 Max) | MLX Output Size |
|---|---:|---|---:|---:|
| Qwen2.5-7B | 4.7 GB | Q4_K_M | ~45s | 14.2 GB |
| Llama-3.2-3B | 2.0 GB | Q4_K_M | ~18s | 6.0 GB |
| Mistral-7B | 4.3 GB | Q4_K_M | ~42s | 13.8 GB |
| Phi-3-mini | 2.2 GB | Q4_K_M | ~20s | 6.6 GB |

Because output is currently dequantized, converted models are substantially
larger than the original GGUF quantized files.

---

## Development

```bash
git clone https://github.com/barrontang/gguf2mlx.git
cd gguf2mlx
uv sync --all-extras

# Tests
pytest

# Opt-in MLX-LM integration test (Apple Silicon)
GGUF2MLX_RUN_E2E=1 pytest tests/test_e2e.py

# Lint
ruff check src/ tests/
```

Recent regression coverage includes:

- strict rejection of unsupported architectures
- preservation of zero-valued token IDs
- correct dtype propagation into config
- atomic staging cleanup on failed writes

---

## Roadmap

Highest-value next steps:

1. MLX quantized output
2. MLX-LM load/integration tests on Apple Silicon
3. More verified architecture adapters
4. Higher-fidelity tokenizer preservation

Contributions in any of those areas are especially welcome.

---

## Contributing

PRs are welcome, especially for:

- new architecture adapters
- tokenizer fidelity improvements
- quantized MLX output
- Apple Silicon integration coverage

---

## License

MIT © [Barron Tang](https://github.com/barrontang)

---

<div align="center">

**If this repo saved you time, please star it.**

</div>
