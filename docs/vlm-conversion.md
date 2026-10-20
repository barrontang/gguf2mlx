# VLM conversion

> LLaVA-1.5, Qwen2-VL and MiniCPM-V 4.7 adapters: direct and hybrid paths, flags, and known limits.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

VLM conversion is separate from MLX-LM conversion. Three explicit adapters produce
HF-style directories for `mlx-vlm`:

- **LLaVA-1.5** (`model_type: llava`): Llama + CLIP (QuickGELU,
  `layer_norm_eps=1e-5`) + two-linear GELU projector.
- **Qwen2-VL** (`model_type: qwen2_vl`, GGUF `qwen2vl`): Qwen2-VL language
  model with M-RoPE, ViT and 2x2 patch merger. Reassembles split Q/K/V in order
  and the two temporal Conv2D kernels into Conv3D; merger normalization and MLP
  always come from GGUF, even in hybrid mode.
- **MiniCPM-V 4.7 1B** (`model_type: minicpmv4_7`): dense Qwen3.5 text GGUF plus
  a separate CLIP projector GGUF tagged `minicpmv4_7`; matching HF config,
  tokenizer, and processor assets are required. Direct quantization is not
  supported.

Full Qwen2.5-VL, Qwen3-VL/3.5, LLaVA-NeXT/1.6, SigLIP, Pixtral and other layouts remain
**unsupported** by the full-VLM adapters. Qwen3-VL **text-only** extraction is
supported separately above. Other full layouts need separate
architecture contracts, not aliases to these adapters.
LLaVA and Qwen2-VL have offline, real `mlx-vlm` loader/processor/generation tests
on Apple Silicon using tiny random checkpoints. MiniCPM-V 4.7 has portable
conversion fixtures, but native image inference is unverified. **Large
pretrained-model visual quality and cross-runtime perplexity parity are not
broadly certified.**
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

# MiniCPM-V 4.7 1B dense: Qwen3.5 language GGUF + matching MiniCPM-V mmproj
gguf2mlx convert --type vlm \
  --input minicpm-v-4.7-1b.gguf --mmproj minicpm-v-4.7-1b-mmproj.gguf \
  --hf-model openbmb/MiniCPM-V-4.7 --output ./minicpm-v-4.7-1b-mlx

# LLaVA/Qwen2-VL: affine 4-bit language projections, floating vision/projector
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
- MiniCPM-V currently requires its separate 4.7 1B dense `clip` projector GGUF.
  The 35B-A3B MoE and older 4.6 layouts are rejected; direct quantization is not
  enabled for MiniCPM-V.
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
  quantization of language attention/MLP projection matrices for LLaVA and
  Qwen2-VL only. Embeddings,
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
