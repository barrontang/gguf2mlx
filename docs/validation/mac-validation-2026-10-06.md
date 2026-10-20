# Mac validation snapshot (2026-10-06)

> Gemma PPL investigation and Qwen1.5 MoE real-model validation.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

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
  See [the research protocol and results](../direct-quant-transcoding.md#gemma-mac-validation-2026-10-06).
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
