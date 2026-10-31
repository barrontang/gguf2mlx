# VLM conversion

> LLaVA-1.5, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL, Qwen3-VL and MiniCPM-V 4.7 adapters: direct and hybrid paths, flags, and known limits.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

VLM conversion is separate from MLX-LM conversion. Six explicit adapters produce
HF-style directories for `mlx-vlm`:

- **LLaVA-1.5** (`model_type: llava`): Llama + CLIP (QuickGELU,
  `layer_norm_eps=1e-5`) + two-linear GELU projector.
- **LLaVA-NeXT / 1.6** (`model_type: llava_next`): the same CLIP-L/336 tower plus
  anyres tiling. Four things decide whether the directory computes the trained
  function, and three of them are silent when wrong — the top-level
  `model.image_newline` parameter, the block count, the feature layer that follows
  from it, and the CLIP MLP tensor roles. See
  [LLaVA-NeXT layout](#llava-next-layout-the-tower-is-shorter-than-hf-says).
- **Qwen2-VL** (`model_type: qwen2_vl`, GGUF `qwen2vl`): Qwen2-VL language
  model with M-RoPE, ViT and 2x2 patch merger. Reassembles split Q/K/V in order
  and the two temporal Conv2D kernels into Conv3D; merger normalization and MLP
  always come from GGUF, even in hybrid mode.
- **Qwen2.5-VL** (`model_type: qwen2_5_vl`, GGUF `qwen2vl`): the same GGUF tensor
  names as Qwen2-VL — llama.cpp drives both through one graph builder — but a
  different function: RMSNorm ViT blocks with no norm bias, a SiLU **gated**
  `ffn_gate`/`ffn_up`/`ffn_down` MLP, an RMSNorm `ln_q` merger norm, and
  `clip.projector_type=qwen2.5vl_merger`. Aliasing the two therefore yields a
  model that loads and emits finite logits while computing the wrong function,
  so the adapter rejects a Qwen2-VL export (and vice versa) instead of
  converting it.
- **Qwen3-VL** (`model_type: qwen3_vl`, GGUF `qwen3vl`): fused biased `attn_qkv`,
  LayerNorm ViT blocks *with* bias, a GELU-tanh `linear_fc1`/`linear_fc2` MLP, a
  Conv3d patch embedding *with* bias, a learned `pos_embed` table, and
  `deepstack_merger_list`. The deepstack mergers are numbered by **rank** in
  `clip.vision.is_deepstack_layers`, not by layer index, and are normalized at
  the 4x merged width. The text half nests its hyperparameters under
  `text_config` and carries per-head QK-norm instead of attention bias.
- **MiniCPM-V 4.7 1B** (`model_type: minicpmv4_7`): dense Qwen3.5 text GGUF plus
  a separate CLIP projector GGUF tagged `minicpmv4_7`; matching HF config,
  tokenizer, and processor assets are required. Direct quantization is not
  supported.

Qwen2.5-VL and Qwen3-VL convert **directly from a language GGUF plus `mmproj`,
with no HF vision weights**: on the pinned Qwen2.5-VL-3B and Qwen3-VL-2B pairs
the report records `vision_hf: 0`. The result was checked against the original
checkpoints, not just against the GGUF: all 824 (Qwen2.5-VL-3B) and 625
(Qwen3-VL-2B) output tensors match the HF index name-for-name and shape-for-shape
once the mlx-vlm `language_model.`/`vision_tower.` prefix convention is applied,
and element-wise comparison of the fused QKV, the stacked patch embedding and the
merger weights agrees to float16 resolution (max |diff| 3e-8; biases, norms and
`ln_q` bit-identical).

LLaVA-NeXT/1.6 converts **directly from a language GGUF plus a CLIP `mmproj`**, on
the same terms as the Qwen adapters: on the pinned llava-v1.6-vicuna-7b pair the
`mmproj` has 378 tensors and the plan consumes all 378, the 380 emitted names
(378 + the two identity post-norm parameters, below) all exist in the
`llava-hf/llava-v1.6-vicuna-7b-hf` index with the same shapes, and the 375-parameter
tower loads into `mlx_vlm.models.llava_next.VisionModel` strictly and runs. The
text half, however, must come from the **same** HF checkpoint: the LLaVA GGUFs that
llama.cpp ships (cjpais, cmp-nct, mradermacher) declare 32000 tokens and no
`<image>`, while `llava-hf`'s checkpoint declares 32064 with `image_token_index`
32000, and mlx-vlm embeds a prompt before it splices in image features. Those pairs
are refused with both counts and the missing token named. Standalone SigLIP towers
and Pixtral remain
**unsupported** by the full-VLM adapters. Qwen3-VL **text-only** extraction is
supported separately above. Other full layouts need separate
architecture contracts, not aliases to these adapters.
LLaVA and Qwen2-VL have offline, real `mlx-vlm` loader/processor/generation tests
on Apple Silicon using tiny random checkpoints. MiniCPM-V 4.7 now runs its native
image-path tests in the same `Vision` workflow job (`tests/test_vlm_minicpm.py`),
but large **pretrained** caption quality is still unverified. **Large pretrained
model visual quality and cross-runtime perplexity parity are not broadly
certified.**
For the pinned **Qwen2-VL-2B-Instruct Q4_K_M + F16 mmproj** pair, real-checkpoint
acceptance is now complete: HF / GGUF-to-FP16 / llama.cpp score **12/12** on a
fixed image diagnostic; uniform language-only MLX 4-bit has **11/12 exact
matches**, but its longer orange-spacesuit answer is semantically correct.
The source-preserving Q4_K/float mode reaches **12/12 exact and semantic checks**.
It keeps Q4_K codes and retains higher-precision source layers as floats, rather
than forcing every projection to 4-bit. All 729 floating
output tensors pass the independent weight audit, with exact visual-feature
parity against HF on those inputs. See the [full report and reproduction
commands](qwen2-vl-real-validation.md), including the original exact-format failure
and the limited scope. This is not broad visual-quality certification. The
Qwen2.5-VL and Qwen3-VL weight checks above are structural and per-tensor; they
are **not** a caption-quality or perplexity result.

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

# Qwen2.5-VL: language GGUF + qwen2.5vl_merger mmproj; HF supplies assets only
gguf2mlx convert --type vlm \
  --input qwen2.5-vl.gguf --mmproj qwen2.5-vl-mmproj.gguf \
  --hf-model Qwen/Qwen2.5-VL-3B-Instruct --output ./qwen2.5-vl-mlx

# Qwen3-VL: language GGUF + qwen3vl_merger mmproj, deepstack mergers included
gguf2mlx convert --type vlm \
  --input qwen3-vl.gguf --mmproj qwen3-vl-mmproj.gguf \
  --hf-model Qwen/Qwen3-VL-2B-Instruct --output ./qwen3-vl-mlx

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
- Direct conversion requires a complete supported vision layout. LLaVA-NeXT
  reconciles the two omissions a real CLIP `mmproj` has — a tower one block
  shorter than the HF config declares, and an absent `v.post_ln` — instead of
  sending you to the hybrid path; see [below](#llava-next-layout-the-tower-is-shorter-than-hf-says).
  A short attention or MLP tensor, or a block count off by more than the
  pooler-only block, still fails.
- Hybrid conversion reads only vision tensors from safetensors. Indexed HF
  checkpoints download only shards containing vision weights (a shared shard
  can also contain language weights); an unsharded checkpoint requires fetching
  that file. Direct conversion does not download HF weight files.
- Output includes `config.json`, tokenizer/processor assets, safetensors shards,
  an index and `vlm_conversion_report.json`. Validation failures clean staging
  files and never overwrite an existing nonempty output directory.
- `--dtype float16` (default) or `float32` controls floating-point tensors.
  `--quantize --direct-quant` together enable affine 4-bit, group-size-64
  quantization of language attention/MLP projection matrices for LLaVA-1.5,
  Qwen2-VL, Qwen2.5-VL and Qwen3-VL. MiniCPM-V 4.7 rejects the pair. Embeddings,
  LM head, norms, biases, vision tower and projector remain floating-point.
  Quantization is **decode then re-quantize**, not bit-preserving GGUF
  transpacking; misaligned projections fail instead of silently falling back.
  `--quantize` alone and MoE `--mixed-precision` are rejected for VLMs.
  The report lists quantized modules, logical/stored tensor counts and layouts.
- For a Qwen GGUF with Q4_K language projections, add `--preserve-source-quant`
  (now an alias of `--quant-source preserve`; both are accepted here) to keep
  their original four-bit codes and **32-element** groups. This option
  overrides the default 64-element re-quantization layout; leave `--q-*` at their
  defaults. Other language projections (including Q6_K) are decoded to the
  requested floating dtype, not reduced to four bits. Vision/projector and
  embeddings remain floating. This is an explicit **mixed-precision** alternative,
  not a promise of an all-four-bit model or bit-exact FP16 arithmetic: scales and
  biases still round to `--dtype`. The flag requires an architecture with a
  vision tensor plan, so it is accepted for Qwen2-VL, Qwen2.5-VL and Qwen3-VL and
  rejected for LLaVA and MiniCPM-V. The measured artifact is the Qwen2-VL-2B
  pair: about **2.7 GiB** versus 2.4 GiB for uniform 4-bit and 4.1 GiB for FP16.
  Size and parity have **not** been measured for the Qwen2.5-VL or Qwen3-VL
  artifacts.
  Missing Q4_K projections, LLaVA, or use without `--quantize --direct-quant`
  fail explicitly. Every projection's source/output policy is recorded.
- HF assets must be from the original unquantized checkpoint. The output
  rebuilds model configuration and quantization metadata; it does not carry
  stale HF quantization settings into the loader.
- `multi_modal_config.json` is **not** a universal MLX-VLM requirement and is
  not fabricated. LLaVA uses nested text/vision configuration; Qwen2-VL,
  Qwen2.5-VL and Qwen3-VL all write a nested `text_config`, and the Qwen2
  families *also* repeat the language hyperparameters at the top level because
  the tested loader reads them from there. Qwen3-VL does **not**: its text
  hyperparameters live only under `text_config`, matching the original
  checkpoint. Processor/tokenizer
  semantics cannot be reliably inferred from tensor names or image mean/std
  alone, so matching HF assets remain required rather than guessed defaults.
- Install `mlx-vlm` separately on Apple Silicon to load the output. The tested
  runtime is pinned to `mlx-vlm==0.7.10` in `tests/requirements-vlm.txt` and the
  `vlm` extra, and should be installed in a separate environment. Older
  `0.7.x` releases lack the `minicpmv4_7` model entry and 0.7.6's LLaVA
  processor is incompatible with `transformers==5.19.0`: it forwards text-only
  `padding` to the image processor, which now rejects it. Native CI uses
  Transformers 5.16.1; the existing `dev` extra retains its newer Transformers
  requirement. Do not combine the two environments. The `vlm` extra pulls the
  matching `mlx-vlm` pin on Apple Silicon, but the pinned test environment in
  `tests/requirements-vlm.txt` remains the reference for what CI exercises.

GGUF vision quantization is not inherently unusable: quality depends on the
source quantization and model. Hybrid conversion avoids that vision
quantization loss but does not recover language/projector information already
lost in GGUF. Prefer direct HF-to-MLX-VLM conversion when all original weights
are available.

## LLaVA-NeXT layout: the tower is shorter than HF says

LLaVA-NeXT is not a new vision encoder, so the interesting work is the places where
the GGUF and the HF config describe the *same* tower differently. Four of them,
three silent:

1. **`model.image_newline` is outside the tower.** It is a top-level `[hidden]`
   parameter in mlx-vlm, not a `vision_tower.*` one, and it is neither `v.*` nor
   `mm.*`. A converter that routes by prefix drops it and produces a directory that
   loads, stiches anyres tiles, and separates them with nothing.
2. **The stored tower is one block shorter.** A real llava-v1.6 `mmproj` declares
   `clip.vision.block_count = 23` for HF's 24 blocks, because HF's
   `vision_feature_layer = -2` with 24 blocks *is* block 22's output, and block 23
   only feeds the pooler LLaVA discards. Emitting 24 layers asks for 32 tensors no
   file has; emitting 23 while keeping `-2` silently selects block 21 and produces
   finite, wrong features. The stored count is authoritative and
   `vision_feature_layer` moves with it, to `-1`. Both facts land in
   `vision_layout` in the report.
3. **`v.post_ln` is absent.** mlx-vlm declares `post_layernorm` and loads strictly,
   so the parameter has to exist; it is written as the identity and recorded in
   `vision_identity_fills`. That is an explicit fill, not a silent default: the
   only consumer is `pooler_output`, which every `llava_next` code path discards,
   and `tests/test_vlm_llava_next.py` asserts that perturbing the identity leaves
   the image features bit-identical.
4. **`ffn_up` and `ffn_down` swap roles between exporters.** llama.cpp's LLaVA-era
   loader reads `ffn_down` as `ff_i` (the GELU input) and `ffn_up` as the output
   projection; the modern mtmd graph reads `ffn_up` as the expansion. Both files
   exist in the wild under both names, and nothing in the metadata says which is
   which — so the shapes decide: a CLIP MLP is `width -> feed_forward -> width`,
   and the tensor shaped `(feed_forward, width)` is the expansion. The choice is
   recorded as `clip_ffn_convention` (`llava-tools` or `mtmd`) in `vision_layout`.
   Every pre-mtmd export, llava-1.5 included, is `llava-tools`.

Two more GGUF fields look relevant and are not. `image_crop_resolution` and
`image_split_resolution` are LLaVA-1.5 leftovers: a real 1.6 export declares 224
next to a 336 tile. Tiles are cut by `preprocessor_config.json`, whose crop size and
grid are pinned to `config.json` by the processor check, so the pair is recorded in
`vision_layout` rather than enforced.

The **text** half is the one thing that cannot be reconciled. mlx-vlm embeds a
prompt before it splices in image features, so `image_token_index` has to be inside
the language embedding. `llava-hf`'s checkpoints satisfy that by padding to 32064
rows and putting `<image>` at 32000; the LLaVA GGUFs that llama.cpp ships declare
32000 tokens with no `<image>` at all, because llama.cpp injects image embeddings
positionally. Pairing them is refused, with both counts and the missing token named:

```text
GGUF tokenizer declares 32000 tokens but the HF text config declares vocab_size
32064: HF ID(s) 32000..32063 (32000=<image>) have no row in the GGUF embedding
table. config.image_token_index=32000 (<image>) is one of them: mlx-vlm embeds the
prompt before it splices in image features, so that row has to exist. Supply a GGUF
converted from the same HF checkpoint rather than a repack that trimmed the extra
rows.
```

## SigLIP ffn roles in MiniCPM-V are declared, not guessed

MiniCPM-V's vision tower is a SigLIP variant, so the `ffn_up`/`ffn_down` naming
trap above applies to it in principle. In practice it does not: MiniCPM-V is only
ever converted by `tools/mtmd`, whose `ffn_up` is the expansion, so there is no
second convention to resolve. `vision_minicpm.py` therefore keeps the roles in
`_VISION_BLOCKS` and asserts the declared shapes against the file, and the mapping
says why in a comment — the shape resolution used for LLaVA-NeXT looks like a
strict improvement here too, but applied to a tower with one convention it would
only invent ambiguity.

What a mislabelled file does produce is the shape mismatch on `mlp.fc1`, which now
carries the diagnosis rather than the raw numbers:

```text
VLM shape mismatch for vision_tower.encoder.layers.0.mlp.fc1.weight: got (4, 8),
expected (8, 4) — a CLIP/SigLIP MLP is width -> feed_forward -> width, so this
file labels the expansion and projection under the opposite ffn_up/ffn_down roles
from the one this adapter resolved
```
