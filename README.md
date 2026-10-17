# gguf2mlx — Qwen-VL text conditioning and GGUF conversion for MLX

<div align="center">

**New: convert the Qwen3-VL-8B text-conditioning component into BF16 safetensors, preserving multimodal rotary positions and vision-token IDs.**

[![PyPI](https://img.shields.io/pypi/v/gguf2mlx.svg)](https://pypi.org/project/gguf2mlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://github.com/barrontang/gguf2mlx)
[![Validation](https://img.shields.io/badge/conversion-strict-purple)](https://github.com/barrontang/gguf2mlx)

**GGUF → HF-style safetensors, with explicit architecture contracts and no silent precision downgrade.**

[Qwen3-VL text conditioning](#qwen3-vl-text-conditioning) · [Quick start](#quick-start) · [Supported models](#supported-conversion-matrix) · [Validation](#text-encoder-validation-and-boundaries)

</div>

---

## Qwen3-VL text conditioning

This release adds a dedicated **`Qwen3VLTextEncoderConverter`** for the language
component of Qwen-VL checkpoints. The development target is the
**Qwen3-VL-8B conditioning encoder described for Qwen-Image-2.1**. Conversion
support is not a claim that an entire Qwen-Image pipeline has been implemented
or that a particular upstream image-generation release has been certified.

The converter builds on the existing Qwen2 tensor mapping, rather than treating
a multimodal encoder as an ordinary Qwen2 causal language model.

### Convert the text component

```bash
pip install gguf2mlx

# Explicit text-only extraction; BF16 is the default.
gguf2mlx convert --type text-encoder \
  --input /absolute/path/to/Qwen3-VL-8B.gguf \
  --output /absolute/path/to/qwen3-vl-text-bf16

# Keep decoded weights in FP32 instead.
gguf2mlx convert --type text-encoder \
  --input /absolute/path/to/Qwen3-VL-8B.gguf \
  --output /absolute/path/to/qwen3-vl-text-fp32 --dtype float32
```

Use a build containing this feature (install a local checkout with
`python -m pip install -e .` if your published package predates it).
No HF downloads or companion `mmproj` file are needed for **text-only**
conversion, but the GGUF must contain the matching tokenizer metadata.

**Select `--type text-encoder` deliberately.** Automatic multimodal detection
continues to use the existing full-VLM route; it does not silently discard a
vision tower. The accepted architecture strings are `qwen3vl`, `qwen3_vl`,
`qwen2vl`, and `qwen2_vl`. Qwen2-tagged exports with Qwen3-VL model identity or
Q/K normalization tensors receive the Qwen3 text configuration; actual
Qwen2-VL exports retain their Qwen2 text identity.

### What the conversion preserves

| Component | Result |
|---|---|
| Text embeddings and transformer projections | Existing Qwen2-compatible HF key layout |
| Qwen3 Q/K normalization | `self_attn.q_norm.weight` and `self_attn.k_norm.weight` |
| Multidimensional RoPE | GGUF `rope.dimension_sections` → `rope_scaling.mrope_section` |
| Qwen3 interleaved mRoPE | `rope_scaling.mrope_interleaved: true` |
| Long conditioning sequences | Source context length, head dimensions and RoPE base |
| Vision markers in the text stream | Image, video, vision-start and vision-end token IDs |
| Combined text + vision GGUF | Recognized vision/projector tensors skipped with a log message |
| Floating-point output | Real BF16 safetensors by default; FP32 decoding before BF16 rounding |

The output contains `config.json`, reconstructed or embedded tokenizer assets,
safetensors shards, and `model.safetensors.index.json`. Its
`gguf2mlx_component` is `text_encoder`, with `model_type: qwen3_vl_text`
(or `qwen2_vl_text`), not a fabricated full-VLM configuration.

### Tensor contract

| GGUF key | Output key |
|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `blk.{N}.attn_q.weight` | `model.layers.{N}.self_attn.q_proj.weight` |
| `blk.{N}.attn_k.weight` | `model.layers.{N}.self_attn.k_proj.weight` |
| `blk.{N}.attn_v.weight` | `model.layers.{N}.self_attn.v_proj.weight` |
| `blk.{N}.attn_output.weight` | `model.layers.{N}.self_attn.o_proj.weight` |
| `blk.{N}.attn_q_norm.weight` / `attn_k_norm.weight` | `model.layers.{N}.self_attn.q_norm.weight` / `k_norm.weight` |
| `blk.{N}.ffn_gate.weight` | `model.layers.{N}.mlp.gate_proj.weight` |
| `blk.{N}.ffn_up.weight` | `model.layers.{N}.mlp.up_proj.weight` |
| `blk.{N}.ffn_down.weight` | `model.layers.{N}.mlp.down_proj.weight` |
| `blk.{N}.attn_norm.weight` | `model.layers.{N}.input_layernorm.weight` |
| `blk.{N}.ffn_norm.weight` | `model.layers.{N}.post_attention_layernorm.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |

Supported Qwen2 attention Q/K/V biases are also mapped. Unknown **language**
tensors fail conversion instead of being silently omitted.

### Precision is an explicit choice

BF16 output is written as **BF16**, not FP16 or an integer array labelled BF16.
BF16-range values are decoded through FP32, avoiding FP16 overflow. Conversion
cannot recover information already lost in a quantized GGUF source.

Plain `--dtype float16` and `--quantize` without `--direct-quant` are rejected
for text conditioning. Reduced precision requires the explicit direct path:

```bash
# Optional, lossy affine 4-bit projections; BF16 floating tensors/scales/biases.
gguf2mlx convert --type text-encoder \
  --input /absolute/path/to/Qwen3-VL-8B.gguf \
  --output /absolute/path/to/qwen3-vl-text-4bit \
  --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine
```

This opt-in output needs a consumer supporting the emitted quantization layout.
Prefer BF16 for conditioning quality; no automatic quantized or FP16 fallback
occurs if conversion fails.

### Text-encoder validation and boundaries

- Portable tests cover dummy Qwen3-VL-8B topology metadata, architecture aliases,
  all text tensor mappings, long-context configuration, vision token IDs,
  mRoPE validation, BF16 payloads/range/rounding, vision skipping, explicit
  quantization, and cleanup on failure.
- Required topology, mRoPE sections, and vision token IDs must be present.
  Three positive rotary sections must sum to half the attention head dimension;
  a fourth trailing zero in GGUF is accepted and removed. Values are not guessed
  from a model name. Nondefault frequency scaling (such as YaRN or linear RoPE)
  is rejected rather than silently replaced by default mRoPE.
- Vision tower and merger weights are intentionally **not emitted**. There is
  no image processor, vision-feature generation, or complete Qwen3-VL runtime
  in this text-only artifact.
- The consuming conditioning pipeline must support Qwen-VL hidden states,
  multimodal positions, tokenizer semantics, and this standalone key layout.
  This is **not** a promise of drop-in `mlx_lm.load()` or `mlx_vlm.load()` support.
- Portable conversion tests do **not** certify real Qwen3-VL-8B inference,
  Qwen-Image-2.1 image quality, hidden-state parity, or Apple Silicon performance.
  Those require matching real checkpoints and native runtime validation.

---

## Other gguf2mlx workflows

GGUF is great for distribution, but MLX and MLX-LM expect a Hugging Face-style
directory with `config.json`, tokenizer assets, and safetensors weights.

Alongside text conditioning, `gguf2mlx` converts supported GGUF language models
to MLX-LM directories and supports separate LLaVA/Qwen2-VL full-VLM adapters.
It bridges that gap for **supported architectures** by:

- reading GGUF metadata and tensors
- rebuilding MLX-LM-compatible model artifacts
- failing closed when a model layout is not actually supported

Use it when a model release provides GGUF files but no MLX checkpoint, when you
want to load that model through `mlx_lm.load()`, or when you need an inspectable
Hugging Face-style model directory on an M1, M2, M3, or M4 Mac.

If the original Hugging Face weights are available, converting those directly
with MLX-LM is preferable because it avoids inheriting quantization error from an
already-quantized GGUF file.

> **Important:** The language-model route writes **HF-style safetensors for MLX-LM** and can
> optionally run `mlx_lm.convert` to emit 4-bit MLX output. This is MLX-LM
> re-quantization after conversion, not bit-for-bit preservation of the original
> GGUF quantization blocks.

---

## Quick start

### Install

For regular Mac users on Apple Silicon (M1, M2, M3, M4), installing is straightforward with no compilation or Xcode tools required:

```bash
# Standard install (automatically installs MLX and MLX-LM on Apple Silicon Macs)
pip install gguf2mlx

# Recommended for global CLI usage (isolated, avoids Homebrew Python PEP 668 restrictions):
pipx install gguf2mlx
# or with uv:
uv tool install gguf2mlx
```

#### Optional extras

- `pip install "gguf2mlx[all]"`: Complete bundle including MLX, MLX-LM, and Hugging Face Hub (for VLM models)
- `pip install "gguf2mlx[vlm]"`: Base converter + Hugging Face Hub for Vision-Language Models
- `pip install "gguf2mlx[dev]"`: Development toolchain (pytest, ruff, mypy, build, twine)

### Install from a local checkout

If you cloned the repo and want the `gguf2mlx` command in your current environment:

```bash
python -m pip install -e .
# or with all optional dependencies
python -m pip install -e ".[all]"
```

### Convert

```bash
# Basic conversion
gguf2mlx --input model.gguf --output ./mlx-model

# GGUF -> 4-bit MLX in one command
gguf2mlx --input model-Q4.gguf --output ./mlx-model-4bit --quantize --q-bits 4 --q-group-size 64

# Bounded-memory direct quantization
gguf2mlx --input model-Q4.gguf --output ./mlx-model-4bit-direct --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine

# Experimental MoE mixed precision: FP16 attention/router + 4-bit experts
gguf2mlx --input qwen3-moe.gguf --output ./qwen3-moe-mixed \
  --quantize --direct-quant --mixed-precision

# Validate the mixed artifact contract before loading it in a compatible runtime
python -m gguf2mlx.mixed_validation ./qwen3-moe-mixed

# Float32 output
gguf2mlx --input model.gguf --output ./mlx-model-f32 --dtype float32

# Limit streamed shard payloads to 2 GiB
gguf2mlx --input model.gguf --output ./mlx-model --max-shard-gb 2

# Inspect metadata without writing weights
gguf2mlx --input model.gguf --skip-weights

# New command form (equivalent conversion subcommand)
gguf2mlx convert --input model.gguf --output ./mlx-model
```

Conversion reads GGUF tensor data through memory-mapped views and decodes/transforms
one tensor (or one required fusion group) at a time. Output payloads are spooled to
disk immediately rather than accumulated in a shard-sized array dictionary. Shard
boundaries use **converted payload bytes**, including quantized weights, scales,
and biases; the index records their actual total size.

`--max-shard-gb` accepts a positive, finite size in GiB. Defaults remain 4.5 GB
(decimal) for floating-point LLM conversion and 256 MiB for direct quantization
and VLM conversion. A tensor larger than the limit occupies a shard by itself;
safetensors headers add a small amount beyond the payload limit. This setting
controls the converter's streaming stage, not a subsequent non-direct `mlx-lm`
quantization stage.

Memory is bounded by the largest active tensor/fusion group and transformation
or quantization scratch space, **not strictly by the shard limit**. Header/index
metadata grows with tensor count, and OS-managed memory-mapped pages may also
appear in RSS. Finalizing a shard uses a bounded copy buffer and temporarily
requires up to one extra shard of disk space. Failed conversions still discard
the staging directory instead of publishing a partial model.

### VLM conversion: architecture-specific direct or hybrid paths

VLM conversion is separate from MLX-LM conversion. Two explicit adapters produce
HF-style directories for `mlx-vlm`:

- **LLaVA-1.5** (`model_type: llava`): Llama + CLIP (QuickGELU,
  `layer_norm_eps=1e-5`) + two-linear GELU projector.
- **Qwen2-VL** (`model_type: qwen2_vl`, GGUF `qwen2vl`): Qwen2-VL language
  model with M-RoPE, ViT and 2x2 patch merger. Reassembles split Q/K/V in order
  and the two temporal Conv2D kernels into Conv3D; merger normalization and MLP
  always come from GGUF, even in hybrid mode.

Full Qwen2.5-VL, Qwen3-VL/3.5, LLaVA-NeXT/1.6, SigLIP, Pixtral and other layouts remain
**unsupported** by the full-VLM adapters. Qwen3-VL **text-only** extraction is
supported separately above. Other full layouts need separate
architecture contracts, not aliases to these adapters.
Both adapters have offline, real `mlx-vlm` loader/processor/generation tests on
Apple Silicon using tiny random checkpoints. **Large pretrained-model visual
quality and cross-runtime perplexity parity are not broadly certified.**
For the pinned **Qwen2-VL-2B-Instruct Q4_K_M + F16 mmproj** pair, real-checkpoint
acceptance is now complete: HF / GGUF-to-FP16 / llama.cpp score **12/12** on a
fixed image diagnostic; uniform language-only MLX 4-bit has **11/12 exact
matches**, but its longer orange-spacesuit answer is semantically correct.
The source-preserving Q4_K/float mode reaches **12/12 exact and semantic checks**.
It keeps Q4_K codes and retains higher-precision source layers as floats, rather
than forcing every projection to 4-bit. All 729 floating
output tensors pass the independent weight audit, with exact visual-feature
parity against HF on those inputs. See the [full report and reproduction
commands](docs/qwen2-vl-real-validation.md), including the original exact-format failure
and the limited scope. This is not broad visual-quality certification.

```bash
# Optional HF download support; local sources need only the base converter
pip install "gguf2mlx[vlm]"

# Path A: language + vision + projector weights from GGUF.
# HF supplies matching full VLM config/tokenizer/image processor, not weights.
gguf2mlx convert --type vlm \
  --input language.gguf --mmproj mmproj.gguf \
  --hf-model llava-hf/llava-1.5-7b-hf --output ./llava-direct

# Path B (recommended): language + projector from GGUF, original vision weights
# and matching config/tokenizer/processor from the full HF VLM checkpoint.
gguf2mlx convert --type vlm \
  --input language.gguf --mmproj mmproj.gguf \
  --hf-fallback-vision llava-hf/llava-1.5-7b-hf --output ./llava-hybrid

# Offline hybrid conversion from a matching local HF safetensors checkpoint
gguf2mlx convert --input language.gguf --mmproj mmproj.gguf \
  --hf-fallback-vision /absolute/path/to/hf-llava --offline \
  --output ./llava-offline

# Qwen2-VL: matching language GGUF + Qwen2-VL mmproj; HF supplies assets only
gguf2mlx convert --type vlm \
  --input qwen2-vl.gguf --mmproj qwen2-vl-mmproj.gguf \
  --hf-model Qwen/Qwen2-VL-2B-Instruct --output ./qwen2-vl-mlx

# Either adapter: affine 4-bit language projections, floating vision/projector
gguf2mlx convert --type vlm \
  --input language.gguf --mmproj mmproj.gguf \
  --hf-model /absolute/path/to/matching-hf-vlm --offline \
  --quantize --direct-quant --q-bits 4 --q-group-size 64 --q-mode affine \
  --output ./vlm-language-4bit

# Qwen2-VL Q4_K_M: avoid a second lossy quantization (recommended)
gguf2mlx convert --type vlm \
  --input qwen2-vl-Q4_K_M.gguf --mmproj qwen2-vl-mmproj.gguf \
  --hf-model /absolute/path/to/matching-hf-qwen2-vl --offline \
  --quantize --direct-quant --preserve-source-quant \
  --output ./qwen2-vl-source-preserved
```

- `--type auto` (default) detects multimodal architecture metadata and tensors.
  A language-only GGUF cannot identify a separately distributed vision tower:
  pass `--mmproj`, HF source options or `--type vlm`. `--type llm` rejects
  detected multimodal inputs instead of silently discarding vision weights.
- Omit `--mmproj` for a combined GGUF. Both paths require GGUF projector
  weights; hybrid conversion never replaces the language model or projector
  with HF weights.
- `--hf-model` must describe the **matching full VLM**, not a bare CLIP model.
  `--hf-fallback-vision` also supplies that source's assets; if both options are
  specified, they must reference the same source. Pin `--hf-revision` to a HF
  commit for reproducible downloads. `--offline` uses local/cached assets only.
- Matching `config.json`, `tokenizer.json`, `tokenizer_config.json` and
  `preprocessor_config.json` are required. The converter validates architecture,
  tensor shapes/completeness, image normalization metadata and tokenizer
  compatibility, then copies processor
  and tokenizer assets without executing remote code or loading pickle files.
  Not every GGUF/HF release pair is compatible; vocabulary/config mismatches
  fail rather than fabricating weights. All tokenizer IDs, including image and
  padding tokens, must fit the language embedding vocabulary; older unresized
  exports are not repaired by synthesizing embedding rows.
- Direct conversion requires a complete supported vision layout. Some CLIP mmproj
  releases omit unused vision layers or normalization tensors: use the hybrid
  path when direct completeness validation fails.
- Hybrid conversion reads only vision tensors from safetensors. Indexed HF
  checkpoints download only shards containing vision weights (a shared shard
  can also contain language weights); an unsharded checkpoint requires fetching
  that file. Direct conversion does not download HF weight files.
- Output includes `config.json`, tokenizer/processor assets, safetensors shards,
  an index and `vlm_conversion_report.json`. Validation failures clean staging
  files and never overwrite an existing nonempty output directory.
- `--dtype float16` (default) or `float32` controls floating-point tensors.
  `--quantize --direct-quant` together enable affine 4-bit, group-size-64
  quantization of language attention/MLP projection matrices only. Embeddings,
  LM head, norms, biases, vision tower and projector remain floating-point.
  Quantization is **decode then re-quantize**, not bit-preserving GGUF
  transpacking; misaligned projections fail instead of silently falling back.
  `--quantize` alone and MoE `--mixed-precision` are rejected for VLMs.
  The report lists quantized modules, logical/stored tensor counts and layouts.
- For Qwen2-VL with Q4_K language projections, add `--preserve-source-quant`
  to keep their original four-bit codes and **32-element** groups. This option
  overrides the default 64-element re-quantization layout; leave `--q-*` at their
  defaults. Other language projections (including Q6_K) are decoded to the
  requested floating dtype, not reduced to four bits. Vision/projector and
  embeddings remain floating. This is an explicit **mixed-precision** alternative,
  not a promise of an all-four-bit model or bit-exact FP16 arithmetic: scales and
  biases still round to `--dtype`. The named 2B artifact is about **2.7 GiB**
  versus 2.4 GiB for uniform 4-bit and 4.1 GiB for FP16.
  Missing Q4_K projections, LLaVA, or use without `--quantize --direct-quant`
  fail explicitly. Every projection's source/output policy is recorded.
- HF assets must be from the original unquantized checkpoint. The output
  rebuilds model configuration and quantization metadata; it does not carry
  stale HF quantization settings into the loader.
- `multi_modal_config.json` is **not** a universal MLX-VLM requirement and is
  not fabricated. LLaVA uses nested text/vision configuration; Qwen2-VL also
  needs top-level language fields for the tested loader. Processor/tokenizer
  semantics cannot be reliably inferred from tensor names or image mean/std
  alone, so matching HF assets remain required rather than guessed defaults.
- Install `mlx-vlm` separately on Apple Silicon to load the output. The tested
  runtime is pinned in `tests/requirements-vlm.txt` and should be installed in a
  separate environment. **`mlx-vlm==0.7.6`'s LLaVA processor is incompatible
  with `transformers==5.19.0`**: it forwards text-only `padding` to the image
  processor, which now rejects it. Native CI uses Transformers 5.16.1; the
  existing `dev` extra retains its newer Transformers requirement. Do not
  combine the two environments. The lightweight `vlm` extra provides downloads only.

GGUF vision quantization is not inherently unusable: quality depends on the
source quantization and model. Hybrid conversion avoids that vision
quantization loss but does not recover language/projector information already
lost in GGUF. Prefer direct HF-to-MLX-VLM conversion when all original weights
are available.

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
| Bounded-memory direct quantization (`--direct-quant`) | Supported |
| StableLM conversion adapter | Supported |
| LLaVA Llama + CLIP direct/hybrid VLM adapter | Native tiny-checkpoint image-path parity tested |
| Qwen2-VL direct/hybrid VLM adapter | Tiny parity; named 2B direct checkpoint passes real-image acceptance |
| Language-only 4-bit VLM conversion | Packed weights + native decoded-reference parity tested |
| Qwen3-VL / Qwen2-VL text conditioning | Portable conversion fixtures; BF16/mRoPE preserved; native conditioning parity unverified |

---

## Supported conversion matrix

`gguf2mlx` distinguishes between:

1. **recognized for inspection** via GGUF metadata, and
2. **conversion enabled** through an explicit tensor adapter.

Language-model conversion is enabled only for explicit adapters. Text conditioning
adds a separate, opt-in contract for the four Qwen-VL identifiers above.
Full-VLM detection and conversion retain their separate LLaVA Llama + CLIP and
Qwen2-VL adapters.

### Conversion-enabled architectures

| Family | GGUF architecture IDs | Conversion | End-to-end | MoE | Mixed precision | `.mlx` bundle | VLM |
|---|---|---|---|---|---|---|---|
| Llama | `llama`, `mistral` | ✅ | ⚠️ direct-quant synthetic | ❌ | — | ✅ generic package/verify | ❌ |
| Qwen | `qwen2` | ✅ | ⚠️ direct-quant synthetic | ❌ | — | ✅ generic package/verify | ❌ |
| Qwen MoE | `qwen2moe`, `qwen3moe` | ✅ | ✅ Qwen1.5 real GGUF; Qwen3 opt-in pending | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| DeepSeek | `deepseek2`, `deepseek3` | ✅ | ⚠️ scheduled opt-in | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| GLM | `glm4moe` | ✅ | ❌ | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| Gemma | `gemma` | ✅ fixture-validated | ✅ synthetic GGUF | ❌ | — | ✅ generic package/verify | ❌ |
| Phi | `phi3` | ✅ Phi-3 4K fixture-validated | ✅ synthetic GGUF | ❌ | — | ✅ generic package/verify | ❌ |
| StableLM | `stablelm` | ✅ | ❌ | ❌ | — | ✅ generic package/verify | ❌ |
| GLM | `glm-dsa` | ⚠️ experimental | ❌ | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| LLaVA | `llava`, or `llama` + CLIP/mmproj | ✅ direct/hybrid fixture-validated | Tiny image-path parity; pretrained quality unverified | ❌ | Language-only 4-bit | ✅ generic package/verify | ✅ Llama + CLIP only |
| Qwen2-VL | `qwen2vl` + Qwen2-VL/mmproj | ✅ direct/hybrid fixture-validated | Named 2B direct checkpoint: 12/12 image diagnostic; see [report](docs/qwen2-vl-real-validation.md) | ❌ | Uniform 4-bit: 11/12 exact; source-preserved Q4_K/float: 12/12 | ✅ generic package/verify | ✅ ViT + patch merger |
| Qwen-VL text encoder | `qwen3vl`, `qwen3_vl`, `qwen2vl`, `qwen2_vl` | ✅ `--type text-encoder`, BF16 default | Portable conversion only; native hidden-state/image quality unverified | ❌ | Explicit direct 4-bit only | ✅ generic package/verify | Text only; vision skipped |

“Conversion” means that a code path exists. “End-to-end” means a converted model
has been loaded and exercised; ⚠️ rows still require the opt-in model fixtures.
The mixed-precision contract validator checks agreement between
`conversion_report.json` and `jang_config.json`; this is not a substitute for
loading the result in a JANG-compatible runtime. `.mlx` bundle packaging and
integrity verification are architecture-independent operations on emitted model
directories and do not assert runtime compatibility.

Gemma 2/3 and Phi-3 LongRoPE are different layouts and are not included in the
basic Gemma or Phi-3 4K support claim.

### Inspection only (not converted)

These may still be recognized by metadata or `--skip-weights`, but they are
**rejected during conversion** until a dedicated, tested adapter exists:

`arctic`, `baichuan`, `bert`, `bitnet`, `bloom`, `chameleon`, `chatglm`,
`codeshell`, `command-r`, `command-r-plus`, `dbrx`, `exaone`, `falcon`,
`gemma2`, `gemma3`, `gpt2`, `gptneox`, `granite`, `grok-1`, `jais`, `minicpm`,
`minicpm3`, `mpt`, `nemotron`, `olmo`, `olmo2`, `openelm`, `orion`, `phi`,
`phi2`, `plamo`, `refact`, `smolm`, `starcoder`, `t5`, and `xverse`.

That means no more silent "Llama fallback" producing invalid outputs for
unrelated architectures.

---

## How conversion works

1. Read GGUF metadata and detect the architecture
2. Validate that the architecture has a supported adapter
3. Build `config.json` using source metadata and selected dtype
4. Export tokenizer assets from GGUF metadata
5. Decode tensors to the selected floating precision (text conditioning uses FP32 intermediates for BF16 output)
6. Remap tensor names into the target Hugging Face / MLX-LM layout
7. Optionally run `mlx_lm.convert --quantize` into a second staged directory
8. Write the selected output directory atomically

If any required step fails, conversion fails and the staged output is cleaned up.

### Offline tokenizer reconstruction

Language-model conversion writes `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.json`, and, when available, `merges.txt` and
`chat_template.jinja` directly from the GGUF header. No Hugging Face model or
tokenizer download is needed for this step.

- **BPE (`gpt2`, `bpe`)** preserves vocabulary IDs and merge order. Recognized
  `tokenizer.ggml.pre` profiles include GPT-2, Llama-3 and Qwen2; these differ in
  number splitting and Unicode normalization.
- **SentencePiece BPE (`llama`)** reconstructs score-ranked merges rather than
  incorrectly interpreting the vocabulary as Unigram.
- **WordPiece (`bert`, `wordpiece`)** reverses BERT's GGUF word-start/continuation
  representation and honors `normalizer.lowercase`, `normalizer.strip_accents`,
  CLS/SEP IDs and insertion flags.
- **Unigram (`t5`, `unigram`, `spm`, `sentencepiece`)** retains token scores,
  byte fallback, whitespace-prefix/removal flags and any precompiled character
  map. Legacy SentencePiece aliases on Gemma retain score-ranked BPE behavior.
- Matching embedded `tokenizer.huggingface.json` takes precedence over rebuilding,
  including its original normalization, added-token and post-processing rules.
  Explicit GGUF special-token IDs are preserved; absent roles are inferred only
  from known token names/types, not arbitrary vocabulary positions.

Token strings, scores and token types are **header metadata arrays**, not the
embedding tensor: weights alone cannot recover a vocabulary. Missing vocabulary,
invalid IDs, duplicate entries, invalid merges, or missing/nonfinite scores for
score-based models fail rather than creating a misleading minimal tokenizer.
Unknown BPE pre-tokenizers emit a parity warning and retain legacy splitting.
GGUF may omit original normalization or added-token options, so reconstruction
does not guarantee exact native/original tokenizer parity for every model.
The VLM path still requires matching processor/tokenizer assets as documented above.

Metadata semantics follow the
[llama.cpp GGUF definitions](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
and [tokenizer implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).


---

## Current limitations

This project is intentionally more honest about scope now:

- **Default 4-bit output re-quantizes** via MLX-LM or the direct affine quantizer.
  The explicit Qwen2-VL `--preserve-source-quant` exception retains Q4_K codes
  and leaves other projection types floating; it is not a general GGUF repacker.
- **Tokenizer fidelity still depends on available GGUF metadata**; embedded
  Hugging Face tokenizer JSON is preserved when present
- **Architecture coverage is adapter-based**, not "all GGUF models"
- **Performance claims depend on model, prompt, hardware, and MLX-LM version**
- **Phi-3 LongRoPE is rejected** until its factor tensors are represented safely
  in the generated MLX configuration
- **Mixed-precision MoE output is experimental.** It writes
  `jang_config.json` and requires a JANG-compatible runtime. No specific JANG
  runtime is claimed as verified yet; use uniform output when interoperability
  validation is unavailable.

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

To attach a quality guardrail, add `--eval-ppl`: after conversion the harness
loads each output with `mlx_lm` and measures perplexity on a fixed corpus, so you
can see how much accuracy the GGUF-to-MLX round trip costs relative to a baseline.
Perplexity failures never abort the benchmark; they are recorded as
`success: false` in the JSON.

```bash
uv run --extra mlx benchmarks/benchmark_conversion.py \
  --input ./model-Q4_K_M.gguf \
  --output ./benchmark-model-mlx-4bit \
  --quantize --compare --eval-ppl \
  --ppl-dataset wikitext --ppl-num-samples 32 --ppl-seq-len 512 \
  --result-json ./benchmark-results/model-4bit.json
```

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

Not by default. MLX-LM and the uniform `--direct-quant` path perform a second
quantization; the latter skips the FP16 staging directory. See
`docs/direct-quant-transcoding.md`. For Qwen2-VL with Q4_K projections,
`--quantize --direct-quant --preserve-source-quant` instead keeps those codes
and 32-element groups, while other projections remain floating-point.
Coefficient rounding to the output dtype still applies; see the
[real-checkpoint report](docs/qwen2-vl-real-validation.md).

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

# Offline native VLM tests (Apple Silicon; no model or image downloads)
GGUF2MLX_RUN_VLM_E2E=1 HF_HUB_OFFLINE=1 \
uv run --isolated --no-project --with-editable . \
  --with-requirements tests/requirements-vlm.txt \
  pytest tests/test_vlm.py tests/test_vlm_cli.py tests/test_vlm_qwen2.py tests/test_vlm_runtime.py

# Or install into a dedicated venv, not the dev environment:
python3 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -e . -r tests/requirements-vlm.txt

# Optional real-model image smoke test: all assets must already exist locally
GGUF2MLX_RUN_VLM_E2E=1 HF_HUB_OFFLINE=1 \
GGUF2MLX_VLM_GGUF=/models/language.gguf \
GGUF2MLX_VLM_MMPROJ=/models/mmproj.gguf \
GGUF2MLX_VLM_HF=/models/matching-original-hf-vlm \
GGUF2MLX_VLM_IMAGE=/images/example.png \
.venv-vlm/bin/python -m pytest tests/test_vlm_runtime.py -k real_checkpoint_image_smoke

# Real MoE validation (finite logits, actual router entropy, generation)
GGUF2MLX_RUN_E2E=1 \
GGUF2MLX_MOE_MODELS='{"qwen3moe":"/models/qwen3.gguf","deepseek2":"/models/deepseek2.gguf"}' \
pytest tests/test_e2e.py -k real_moe

# Smaller Qwen1.5 MoE fixture (Q2_K GGUF, about 5.89 GB)
# Output paths must be new; omit OUTPUT_ROOT to use pytest temporary directories.
GGUF2MLX_RUN_E2E=1 \
GGUF2MLX_MOE_MODELS='{"qwen2moe":"/models/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf"}' \
GGUF2MLX_MOE_OUTPUT_ROOT=/models/new-qwen-moe-validation \
GGUF2MLX_MOE_REPORT_DIR=/reports/qwen-moe \
pytest -q 'tests/test_e2e.py::test_real_moe_model_has_finite_logits_routing_entropy_and_coherent_output[qwen2moe]'

# Validate an embedded chat template against five conversation shapes
python -m gguf2mlx.chat_validation ./converted-model --expect-marker '<|im_start|>'

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
- fail-closed MoE router detection and per-tensor precision reports
- tokenizer fixtures for 10 architectures, byte fallback, and 20 added tokens
- chat-template rendering for single-turn, multi-turn, system, tool, and reasoning shapes
- VLM split-file conversion, ordered QKV and temporal-kernel reconstruction,
  hybrid vision-only replacement, M-RoPE and processor mismatch rejection
- native VLM image features and logits against original HF-layout weights
  (or an independently decoded quantized reference), image-change sensitivity,
  finite outputs and `mlx_vlm.generate`; F32/Q4_0 inputs and FP16/FP32/4-bit outputs

The dedicated **Vision conversion** workflow runs these offline tests on Apple
Silicon and fails if the runtime is unavailable instead of silently skipping.
The random-checkpoint test is a numerical/loader regression, not a visual
reasoning benchmark. The optional pretrained smoke test checks finite logits,
image sensitivity and nonempty generation; it does not certify caption accuracy.
The separate [real Qwen2-VL acceptance harness](benchmarks/validate_vlm.py) audits
all floating output weights and compares fixed labeled images against original HF
and the same GGUF through llama.cpp. It preserves raw answers and exits nonzero
on its predeclared gate failure; see the [results](docs/qwen2-vl-real-validation.md).
Image generators such as `mlxgen` / Qwen-Image can supply test images but are not
substitutes for a matching image-understanding checkpoint.

---

## Roadmap status

Completed:

- MLX quantized output through bundled `mlx_lm.convert`
- MLX-LM load/integration tests on Apple Silicon
- strict adapter-based architecture validation
- fixture-backed Gemma and standard Phi-3 4K adapters
- GGUF tokenizer flags and embedded tokenizer JSON preservation
- reproducible conversion benchmark harness
- bounded-memory direct quantization pipeline (`--direct-quant`, affine 4-bit, dense and declared MoE adapters, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K–Q8_K/IQ4_NL/BF16 source qtypes)
- fail-closed MoE detection, default router protection, mixed-precision artifact output, and JSON conversion reports
- required tokenizer/chat-template CI plus scheduled opt-in two-family MoE validation
- StableLM conversion adapter (norm_eps, partial_rotary_factor, qk_layernorm, use_parallel_residual)
- `mlx_lm.load()` parity validation test for `--direct-quant` output (`tests/test_e2e.py`)
- opt-in real-GGUF convert + `mlx_lm.load()` finite-logit validation for gemma and phi3 fixtures (`tests/test_e2e.py`)
- strict LLaVA-1.5 and Qwen2-VL vision adapters, language-only direct quantization,
  and offline native image-path CI
- pinned Qwen2-VL-2B real-checkpoint conversion, all-tensor audit and four-path
  labeled image acceptance, including historical `qwen2vl_merger` compatibility
- `--eval-ppl` perplexity guardrail in the benchmark harness (reuses `mlx_lm.perplexity`; reports standard vs `--direct-quant` PPL delta)

Remaining areas for contributors:

- publish fixed-corpus perplexity numbers from `--eval-ppl` (standard vs `--direct-quant` vs `mlx_lm.convert`) so users can judge round-trip accuracy loss
- verify `gguf2mlx-mixed-v1` with a named JANG runtime; until then the format remains experimental
- extend the opt-in real-GGUF load test to the remaining conversion-enabled architectures (several are declared supported but have no end-to-end coverage; the `safe_open` bug showed how those paths silently rot)
- canonical chat-template fallback table for known model families when GGUF-embedded Jinja templates are missing or broken
- publish peak RSS and output-size comparison results from `benchmarks/benchmark_conversion.py --compare`
- evaluate GGUF→MLX conversion for high-demand non-text models (e.g. ASR/embedding) where competition is thin (pending verification that such models ship GGUF)
- add Gemma 2/3 and Phi LongRoPE adapters without broad family fallbacks
- publish additional architecture-specific tokenizer normalizer fixtures as upstream formats evolve
- add LLaVA-NeXT any-resolution/image-newline handling and distinct Qwen2.5/3.x
  vision adapters, including Qwen3.5's different language backbone, before
  claiming those families are supported
- expand beyond the named Qwen2-VL-2B diagnostic to larger, diverse image corpora
  and real LLaVA checkpoints; report quality separately from tiny-checkpoint parity

Vision format references:
[llama.cpp Qwen2-VL exporter](https://github.com/ggml-org/llama.cpp/blob/4413a192410fad1f054e9216114087a3e01527a0/conversion/qwenvl.py),
[GGUF tensor mapping](https://github.com/ggml-org/llama.cpp/blob/4413a192410fad1f054e9216114087a3e01527a0/gguf-py/gguf/tensor_mapping.py),
and the `mlx-vlm==0.7.6` loader/vision implementations exercised by the native tests.

### Mac validation snapshot (2026-10-06)

- Environment: `conda` env `gguf2mlx` on Apple Silicon.
- Validation with Apple Silicon tests enabled and public Gemma/Qwen1.5 MoE GGUFs supplied:
  **319 passed, 2 skipped**; the remaining skips require external Qwen3-MoE
  and DeepSeek2 models. Rust unit tests => **3 passed** (macOS link flags required:
  `RUSTFLAGS='-C link-arg=-undefined -C link-arg=dynamic_lookup'`).
- Research guardrails now include:
  - fail-closed, zero-tolerance, unrounded llama.cpp vs MLX perplexity runner
    (`benchmarks/benchmark_transpacking.py`)
  - Q4_0/Q8_0 memory-layout and external packed-array contract tests
    (`tests/test_transpacking_layout.py`)
- Fixed Gemma SentencePiece BPE reconstruction and the safetensors shard-index
  iteration failure. The public `mlabonne/gemma-2b-it-GGUF` Q4_0 model now
  converts locally without monkeypatches, and token IDs match native llama.cpp.
- Real-model PPL evaluation completed on a fixed README snapshot (8 windows of
  128 tokens, BOS per window, 1,016 scored tokens, llama.cpp on CPU):
  llama.cpp **143.58573263274607**, converted FP16 MLX **143.17148969720395**,
  absolute delta **0.41424293554212**. The zero-tolerance gate correctly
  **fails**; tokenizer parity is fixed, but numerical parity is not established.
  See [the research protocol and results](docs/direct-quant-transcoding.md#gemma-mac-validation-2026-10-06).
- Numerical diagnosis found a dense GGUF row-major decoding bug and a stock
  MLX-LM Gemma erf/tanh GELU mismatch; both are corrected. All 2,506,096,640
  decoded Q4_0 values match native llama.cpp bit-for-bit. Same-weight dense
  FP32 PPL delta is now **0.0007508356276275663**, but the original Q4_0 CPU
  comparison still fails: native Q4_0 matmul uses Q8_0 activation operands,
  unlike dense MLX. The HF-mirror-derived FP16 control also fails
  (**0.08708103047850102**). No gate tolerance was changed.
- Important boundary: equal perplexity is a quality gate only; it does **not**
  by itself prove bit-identical tensors/logits.

### Qwen1.5 MoE real-model validation (2026-10-06)

- Source: [RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf](https://huggingface.co/RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf),
  revision `947580e7b3904cd4081810d0efec4520a75609c7`,
  `Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf`: **5,890,163,424 bytes**.
  SHA-256 `43d1dbd226e1321fd3638c8294022df2906c0ab158a7a2a8bd022b442e8b98bf`
  matches the Hugging Face LFS metadata.
- This smaller Qwen MoE family has about 14.3B total / 2.7B active parameters,
  24 layers and 60 experts, selecting 4 per token. Q2_K is the smallest
  compatible variant among the inspected sources; smaller IQ1/IQ2 variants
  are not supported by the direct-quant path. The file mixes Q2_K, Q3_K,
  IQ4_NL and Q6_K weights rather than exclusively containing Q2_K.
- Both mixed and uniform affine 4-bit conversions complete. The mixed
  artifact validator confirms **24 protected routers** and **72 compressed
  expert tensors**; a JANG mixed-runtime load is still not certified.
- Uniform MLX-LM load and forward pass succeed: logits shape
  **`[1, 8, 151936]`**, all finite. First-layer router logits are captured
  from the actual forward pass, shape **`[1, 8, 60]`**; normalized entropy
  **0.9375427467178497**, within the test's `(0.05, 0.95)` bounds.
- Chat-template generation produces 32 tokens / 26 distinct token IDs,
  beginning: “The blue color of the sky is due to a combination of factors,
  primarily the scattering of sunlight and the Earth's atmosphere.”
  The automated check certifies printable, non-degenerate text, not general
  semantic quality.
- Native tokenizer parity passes for English, whitespace, Chinese/emoji,
  decomposed Unicode and all **6,237 tokens** of the frozen evaluation corpus.
  Fixed accidental registration of ordinary `!` as a missing unknown token
  and Qwen GGUF's unintended NFC normalization.
- Real validation also exposed and repaired uint8/uint32 packed-weight
  incompatibility, missing Qwen2MoE shared-expert mappings/gate shape and
  legacy expert-dimension metadata handling. These are conversion fixes,
  not source-preserving transpacking or a zero-tolerance PPL pass.

---

## Contributing

PRs are welcome, especially for:

- new architecture adapters backed by tensor manifests and load tests
- tokenizer fidelity improvements
- end-to-end `mlx_lm.load()` parity validation and perplexity benchmarks for `--direct-quant`
- expanding `--direct-quant` to additional architectures and source quantization types
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
