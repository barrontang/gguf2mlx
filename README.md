

# 🧠 GGUF → MLX

<div align="center">

**Convert any GGUF model to Apple MLX format — one command.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://github.com/barrontang/gguf2mlx)
[![Architectures](https://img.shields.io/badge/architectures-44+-purple)](https://github.com/barrontang/gguf2mlx)

</div>

---

### ✨ The Problem

You downloaded a GGUF model. You want to run it on **Apple Silicon** with MLX. But MLX expects safetensors with HuggingFace layout — not GGUF's quantized blobs with custom tensor names.

### 🎯 The Solution

```bash
gguf2mlx -i model-Q4_K_M.gguf -o ./mlx-model
```

One command. Full conversion. Ready for `mlx_lm.load()`.

---

## 🚀 Features

| | |
|---|---|
| 🔍 **Auto-Detect** | Infers architecture, vocab size, config from GGUF metadata |
| 🔓 **Full Dequant** | Q2_K through F16 — all GGUF quant types → float16 or float32 |
| 🔄 **Weight Transpose** | GGUF [out, in] ↔ HuggingFace [in, out] tensor layouts |
| 📦 **Smart Sharding** | Auto-splits large models into multi-file safetensors (<4.5 GB each) |
| 🪙 **Tokenizer** | Extracts vocab, merges, special tokens → HuggingFace-compatible format |
| 📊 **Progress Bar** | Real-time tqdm feedback during conversion |
| 🛟 **BOS/EOS Fix** | Auto-corrects broken special tokens for Qwen, DeepSeek, and others |
| 🏗️ **Strict conversion** | Converts supported Llama, Mistral, Qwen, DeepSeek, and GLM layouts; rejects unknown layouts |

---

## 📦 Quick Start

### Install

```bash
pip install git+https://github.com/barrontang/gguf2mlx.git
# or
uv add git+https://github.com/barrontang/gguf2mlx.git
```

### Convert

```bash
# Basic conversion
gguf2mlx --input model-Q4_K_M.gguf --output ./mlx-model

# Float32 precision (larger files, higher fidelity)
gguf2mlx -i model.gguf -o ./mlx-f32 --dtype float32

# Just inspect, don't convert
gguf2mlx -i model.gguf --skip-weights
```

### Run Inference

```bash
# One-step: convert + generate
uv run demo.py -i model.gguf -p "Explain quantum computing" --max-tokens 100

# Or use mlx-lm directly after conversion
python -c "
from mlx_lm import load, generate
model, tok = load('./mlx-model')
print(generate(model, tok, prompt='Hello, world!', max_tokens=50))
"
```

---

## 🔧 How It Works

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  model.gguf   │ ──▶ │  gguf2mlx v2.0.2│ ──▶ │  mlx-model/      │
│  (quantized)  │     │                 │     │  ├ config.json    │
│  Q4_K, Q8...  │     │  • dequantize   │     │  ├ tokenizer.json │
└──────────────┘     │  • remap names   │     │  ├ vocab.json     │
                     │  • transpose     │     │  ├ merges.txt     │
                     │  • shard split   │     │  └ model-*.safetensors
                     └─────────────────┘     └──────────────────┘
```

1. **Read** GGUF metadata → detect architecture (Llama? Qwen? DeepSeek?)
2. **Build** HuggingFace-compatible `config.json`
3. **Extract** tokenizer → `vocab.json`, `merges.txt`, `tokenizer.json`
4. **Dequantize** every tensor back to float16/float32
5. **Remap** tensor names (GGUF → HF convention) and transpose dimensions
6. **Shard** into multiple safetensors files if model exceeds 4.5 GB
7. **Output** ready-to-load MLX model directory

---

## 🏗️ Architecture Status

<details open>
<summary><b>GGUF architectures recognized for inspection</b></summary>

| Family | Architectures |
|--------|--------------|
| **Llama** | llama, mistral, falcon, stablelm |
| **Qwen** | qwen2, qwen2moe, qwen3moe |
| **DeepSeek** | deepseek2, deepseek3 |
| **GLM** | glm-dsa ⚠️ *(experimental)*, chatglm |
| **Gemma** | gemma, gemma2, gemma3 |
| **Phi** | phi, phi3 |
| **GPT** | gpt2, gptneox, gpt_bigcode, refact |
| **MoE** | dbrx, grok-1 |
| **OLMo** | olmo, olmo2 |
| **Command R** | command-r, command-r-plus |
| **Others** | bert, bloom, cohere, granite, nemotron, exaone, openelm, chatglm, baichuan, xverse, orion, bitnet, plamo, codeshell, minicpm, minicpm3, t5, jais, arctic, smolm, chameleon, mpt |

</details>

Weight conversion is currently enabled for `llama`, `mistral`, `qwen2`,
`qwen2moe`, `qwen3moe`, `deepseek2`, `deepseek3`, and experimental `glm-dsa`.
`glm4moe` is also supported. Other architectures are detected by `--skip-weights` but rejected during
conversion until their tensor layout has a dedicated, tested adapter.

> ⚠️ **GLM-5.2 (`glm-dsa`) is experimental.** The converter fully supports the
> tensor layout (MLA + DSA lightning indexer + IndexShare F/S layers +
> mixed dense/sparse MoE + MTP), but **end-to-end runnability is gated upstream**:
> (1) llama.cpp has not yet registered a `GlmMoeDsaForCausalLM`→`glm-dsa` GGUF
> producer, and (2) neither `mlx-lm` nor a released `transformers` ships a
> `glm_moe_dsa` model class yet. Until those land, this produces
> correctly-named, reference-matching weights but no runtime to load them. See
> [`PLAN.md`](PLAN.md) for the full reverse-engineering notes.

Missing one? [Open an issue](https://github.com/barrontang/gguf2mlx/issues) — we add new architectures fast.

## 🔝 Top GGUF Models (Compatibility Focus)

This repo is tuned to handle the most common GGUF model families used in downloads and local inference workflows:

1. Llama 3.x / 4.x  
2. Qwen 2.x / 2.5  
3. DeepSeek V3 / R1  
4. Mistral / Mixtral  
5. Gemma 2 / 3  
6. Phi 3 / 4  
7. Command-R / Command-R+  
8. Yi  
9. Falcon  
10. DBRX  

If a GGUF file is missing `general.architecture`, `gguf2mlx` now applies popular-name fallback detection for these families before falling back to the safe default (`llama`).

---

## 📊 Benchmarks

| Model | GGUF Size | Quant | Convert Time (M4 Max) | MLX Size |
|-------|-----------|-------|----------------------|----------|
| Qwen2.5-7B | 4.7 GB | Q4_K_M | ~45s | 14.2 GB |
| Llama-3.2-3B | 2.0 GB | Q4_K_M | ~18s | 6.0 GB |
| Mistral-7B | 4.3 GB | Q4_K_M | ~42s | 13.8 GB |
| Phi-3-mini | 2.2 GB | Q4_K_M | ~20s | 6.6 GB |

*MLX loads and runs 1.5–3× faster than GGUF on Apple Silicon.*

---

## 🧪 Development

```bash
git clone https://github.com/barrontang/gguf2mlx.git
cd gguf2mlx
uv sync --all-extras  # includes dev tools

# Run tests
pytest

# Lint
ruff check src/
```

The `mlx` extra installs both MLX and MLX-LM for loading converted models.

Conversion writes to a temporary staging directory and refuses to replace a
non-empty output directory, preventing failed runs from leaving mixed model files.

---

## 🤝 Contributing

PRs welcome! Especially for:
- New architecture weight mappings
- Additional quant type support
- Performance optimizations
- Test coverage

---

## 📜 License

MIT © [Barron Tang](https://github.com/barrontang)

---

<div align="center">

**⭐ Star this repo if it saved you time**

*Built with 🧠 on Apple Silicon*

</div>
