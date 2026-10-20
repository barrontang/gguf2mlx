# Qwen3-VL text conditioning

> Extracting the text component of a Qwen-VL checkpoint as BF16/FP32 safetensors, including the tensor contract and its boundaries.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

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
