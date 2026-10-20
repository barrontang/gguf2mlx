# How conversion works

> Pipeline stages, streaming/sharding behaviour, and offline tokenizer reconstruction.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

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

SmolLM2-135M uses the Llama tensor layout with tied embeddings. Llama GGUF
conversion detects a missing separate output head and restores exported Q/K
rotary projection ordering. A local HF -> FP16 GGUF -> MLX conversion was
validated against the original model cast to FP16: all 272 parameter tensors,
logits on six tokenizer/Unicode/code prompts, and two greedy generations matched
exactly. Preserve the original tokenizer JSON in GGUF when exporting to avoid
losing source normalization semantics. This is conversion parity, not a broad
model-quality benchmark.

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
