# GGUF to MLX Converter for Apple Silicon

<div align="center">

**Convert supported GGUF language models into MLX-LM-compatible safetensors on Apple Silicon Macs.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://github.com/barrontang/gguf2mlx)
[![Validation](https://img.shields.io/badge/conversion-strict-purple)](https://github.com/barrontang/gguf2mlx)

**GGUF → dequantized safetensors → optional 4-bit MLX, with strict architecture validation.**

[Quick start](#quick-start) · [Supported models](#supported-conversion-matrix) · [Benchmarks](#reproducible-benchmarks) · [FAQ](#frequently-asked-questions)

</div>

---

## What is gguf2mlx?

GGUF is great for distribution, but MLX and MLX-LM expect a Hugging Face-style
directory with `config.json`, tokenizer assets, and safetensors weights.

`gguf2mlx` is a command-line converter for Mac users who have a model in GGUF
format but need an MLX-LM model directory. It bridges that gap for **supported
architectures** by:

- reading GGUF metadata and tensors
- rebuilding MLX-LM-compatible model artifacts
- failing closed when a model layout is not actually supported

Use it when a model release provides GGUF files but no MLX checkpoint, when you
want to load that model through `mlx_lm.load()`, or when you need an inspectable
Hugging Face-style model directory on an M1, M2, M3, or M4 Mac.

If the original Hugging Face weights are available, converting those directly
with MLX-LM is preferable because it avoids inheriting quantization error from an
already-quantized GGUF file.

> **Important:** `gguf2mlx` writes **HF-style safetensors for MLX-LM** and can
> optionally run `mlx_lm.convert` to emit 4-bit MLX output. This is MLX-LM
> re-quantization after conversion, not bit-for-bit preservation of the original
> GGUF quantization blocks.

## GitHub discoverability setup

If you want this repository to be indexed faster by GitHub search and external
search engines, configure these three surfaces:

### 1) Repository description (meta description)

Use a short, keyword-rich sentence in the repo **Description** field.

**Chinese (1-2 sentences):**

> 在 Apple Silicon（M1/M2/M3/M4）上将 GGUF 语言模型转换为 MLX-LM 兼容的
> safetensors。支持严格架构校验与可选 4-bit MLX 量化，面向可复现、可验证的模型转换流程。

**English (1-2 sentences):**

> Convert GGUF language models to MLX-LM-compatible safetensors on Apple Silicon
> (M1/M2/M3/M4). Includes strict architecture validation and optional 4-bit MLX
> quantization for a reproducible, verifiable conversion workflow.

### 2) Repository topics

Final prioritized topics for this project:

- `gguf`
- `mlx`
- `mlx-lm`
- `llm`
- `model-conversion`
- `apple-silicon`
- `safetensors`
- `quantization`
- `python`
- `rust`
- `macos`
- `huggingface`

### 3) GitHub Pages

Enable **GitHub Pages** in repository settings (even a simple README-rendered
site helps discoverability and external crawling).

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

# Experimental bounded-memory direct quantization path (llama/gemma only)
gguf2mlx --input model-Q4.gguf --output ./mlx-model-4bit-direct --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine

# Float32 output
gguf2mlx --input model.gguf --output ./mlx-model-f32 --dtype float32

# Inspect metadata without writing weights
gguf2mlx --input model.gguf --skip-weights

# New command form (equivalent conversion subcommand)
gguf2mlx convert --input model.gguf --output ./mlx-model
```

### Package an MLX model directory into `.mlx`

`gguf2mlx` can now package an MLX model directory into a single `.mlx` bundle,
auto-generate a manifest list, and include SHA-256 integrity metadata.

```bash
# Package model directory into model.mlx (+ model.mlx.sha256)
gguf2mlx package --model-dir ./mlx-model --output ./mlx-model.mlx

# Verify embedded manifest and SHA-256 integrity
gguf2mlx verify --bundle ./mlx-model.mlx
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

## What works today

| Capability | Status |
|---|---|
| Strict architecture validation | Supported |
| Atomic staging and cleanup on failure | Supported |
| Correct `torch_dtype` propagation | Supported |
| Correct zero-valued special token IDs | Supported |
| GGUF context length -> `model_max_length` | Supported |
| MLX / MLX-LM optional dependency | Supported |
| One-command 4-bit output via `mlx_lm.convert` | Supported |
| Package `.mlx` bundles + SHA-256 manifest integrity | Supported |
| Opt-in `mlx_lm.load()` integration test | Supported |
| Embedded `tokenizer.huggingface.json` preservation | Supported |
| GGUF BOS/EOS/UNK and space-prefix metadata | Supported |
| Executable Unigram and WordPiece tokenizer tests | Supported |
| Gemma and Phi-3 architecture fixtures | Supported |

---

## Supported conversion matrix

`gguf2mlx` distinguishes between:

1. **recognized for inspection** via GGUF metadata, and
2. **conversion enabled** through an explicit tensor adapter.

The current code recognizes 48 architecture identifiers for inspection and
enables conversion for 11 identifiers.

### Conversion-enabled architectures

| Family | GGUF architecture IDs | Status |
|---|---|---|
| Llama | `llama`, `mistral` | Conversion enabled |
| Qwen | `qwen2`, `qwen2moe`, `qwen3moe` | Conversion enabled |
| DeepSeek | `deepseek2`, `deepseek3` | Conversion enabled |
| GLM | `glm4moe` | Conversion enabled |
| Gemma | `gemma` | Fixture-validated adapter |
| Phi | `phi3` | Phi-3 4K fixture-validated adapter |
| GLM | `glm-dsa` | Experimental conversion only |

Gemma 2/3 and Phi-3 LongRoPE are different layouts and are not included in the
basic Gemma or Phi-3 4K support claim.

### Inspection only (not converted)

These may still be recognized by metadata or `--skip-weights`, but they are
**rejected during conversion** until a dedicated, tested adapter exists:

`arctic`, `baichuan`, `bert`, `bitnet`, `bloom`, `chameleon`, `chatglm`,
`codeshell`, `command-r`, `command-r-plus`, `dbrx`, `exaone`, `falcon`,
`gemma2`, `gemma3`, `gpt2`, `gptneox`, `granite`, `grok-1`, `jais`, `minicpm`,
`minicpm3`, `mpt`, `nemotron`, `olmo`, `olmo2`, `openelm`, `orion`, `phi`,
`phi2`, `plamo`, `refact`, `smolm`, `stablelm`, `starcoder`, `t5`, and `xverse`.

That means no more silent "Llama fallback" producing invalid outputs for
unrelated architectures.

---

## How conversion works

1. Read GGUF metadata and detect the architecture
2. Validate that the architecture has a supported adapter
3. Build `config.json` using source metadata and selected dtype
4. Export tokenizer assets from GGUF metadata
5. Dequantize GGUF tensors to FP16 or FP32
6. Remap tensor names into the target Hugging Face / MLX-LM layout
7. Optionally run `mlx_lm.convert --quantize` into a second staged directory
8. Write the selected output directory atomically

If any required step fails, conversion fails and the staged output is cleaned up.

---

## Current limitations

This project is intentionally more honest about scope now:

- **4-bit MLX output uses MLX-LM re-quantization**; source GGUF Q4 blocks are not
  preserved directly
- **Tokenizer fidelity still depends on available GGUF metadata**; embedded
  Hugging Face tokenizer JSON is preserved when present
- **Architecture coverage is adapter-based**, not "all GGUF models"
- **Performance claims depend on model, prompt, hardware, and MLX-LM version**
- **Phi-3 LongRoPE is rejected** until its factor tensors are represented safely
  in the generated MLX configuration

If you need guaranteed support for a new family, open an issue with the exact
GGUF architecture and source model.

---

## Reproducible benchmarks

The repository does not claim that MLX is universally faster than llama.cpp.
Conversion and inference performance depend on model architecture, quantization,
prompt length, hardware, thermals, and library versions.

Run the included benchmark harness to record conversion time, peak RSS, input
size, output size, platform information, and installed package versions:

```bash
uv run benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx \
  --result-json ./benchmark-results/model.json
```

To benchmark the complete GGUF-to-4-bit-MLX path:

```bash
uv run --extra mlx benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx-4bit \
  --quantize \
  --result-json ./benchmark-results/model-4bit.json
```

Plain conversion output is dequantized and can be substantially larger than the
original GGUF quantized file. Use `--quantize --q-bits 4 --q-group-size 64` when
you want compact MLX-LM quantized output.

## Choosing the right tool

| Starting point | Goal | Recommended tool |
|---|---|---|
| GGUF model | Run the GGUF directly | llama.cpp or a GGUF application |
| Original Hugging Face model | Create an MLX model | `mlx_lm.convert` |
| GGUF-only model release | Create an MLX-LM directory | `gguf2mlx` |
| Existing MLX model directory | Create a portable archive | `gguf2mlx package` |

## Frequently asked questions

### How do I convert a GGUF model to MLX on a Mac?

Install `gguf2mlx[mlx]`, then run `gguf2mlx convert --input model.gguf
--output model-mlx`. The output directory can be passed to `mlx_lm.load()` when
the GGUF architecture and variant appear in the supported matrix.

### Does GGUF-to-MLX conversion restore FP16 model quality?

No. Dequantization expands the stored values into FP16 or FP32, but it cannot
recover information removed when the source GGUF was quantized.

### Does the 4-bit output preserve the original GGUF Q4 blocks?

No. The current implementation dequantizes the GGUF and then uses MLX-LM to
perform a second quantization. A lower-memory direct pipeline is designed in
`docs/direct-quant-transcoding.md`.

### Why is the converted model larger than the GGUF file?

A quantized GGUF stores only a few bits per weight plus block metadata. Plain
conversion writes FP16 or FP32 safetensors, so a larger output is expected.

### Are Gemma and Phi supported?

The base `gemma` architecture and standard Phi-3 4K layout have fixture-backed
adapters. Gemma 2, Gemma 3, Phi-2, Phi-MoE, and Phi-3 LongRoPE remain unsupported.

### Is MLX always faster than llama.cpp?

No universal multiplier is claimed. Use the same model quality, prompt, context,
sampling settings, and hardware when comparing runtimes.

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
ruff check src/ tests/ benchmarks/
```

### Hybrid Rust migration (in progress)

The repository now includes an initial Rust core scaffold at:

- `rust/gguf2mlx-rs`

To build and enable the optional PyO3 extension locally:

```bash
python -m pip install maturin
maturin develop --manifest-path rust/gguf2mlx-rs/Cargo.toml
```

Current integration behavior:

- Python CLI/UX remains the primary entrypoint.
- Python can use an optional `gguf2mlx_rust` extension for architecture detection.
- If the Rust extension is unavailable, the existing Python logic is used unchanged.

Recent regression coverage includes:

- strict rejection of unsupported architectures
- Gemma tensor mapping and GGUF norm-weight restoration fixtures
- Phi-3 fused QKV and gated-MLP fixtures
- preservation of zero-valued token IDs
- preservation of embedded Hugging Face tokenizer JSON
- executable tokenizer encode/decode checks
- correct dtype propagation into config
- atomic staging cleanup on failed writes
- MLX-LM quantization error handling
- opt-in `mlx_lm.load()` validation of quantized output

---

## Roadmap status

Completed:

- MLX quantized output through bundled `mlx_lm.convert`
- MLX-LM load/integration tests on Apple Silicon
- strict adapter-based architecture validation
- fixture-backed Gemma and standard Phi-3 4K adapters
- GGUF tokenizer flags and embedded tokenizer JSON preservation
- reproducible conversion benchmark harness

Remaining areas for contributors:

- implement the bounded-memory pipeline described in
  `docs/direct-quant-transcoding.md`
- add opt-in real-GGUF load and logit validation for each fixture-backed adapter
- add Gemma 2/3 and Phi LongRoPE adapters without broad family fallbacks
- broader tokenizer fixture coverage for architecture-specific normalizers,
  byte fallback variants, and added-token edge cases

---

## Contributing

PRs are welcome, especially for:

- new architecture adapters backed by tensor manifests and load tests
- tokenizer fidelity improvements
- bounded-memory GGUF-to-MLX quantization
- Apple Silicon integration coverage

When requesting a new model family, include the exact GGUF architecture, model
name, quantization type, tokenizer type, and a public fixture or model URL. If
the project saves you conversion work, starring the repository helps other Mac
users discover it.

---

## License

Apache-2.0 © [Barron Tang](https://github.com/barrontang)

---

<div align="center">

**If this repo saved you time, please star it.**

</div>
