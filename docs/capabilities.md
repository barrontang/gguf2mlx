# What works today

> Per-capability status table.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

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
| Gemma and Phi-3/Phi-4 architecture fixtures, including LongRoPE config | Supported |
| Bounded-memory direct quantization (`--direct-quant`) | Supported |
| Source-block preservation (`--quant-source preserve`, Q4_K/Q5_K/Q6_K/Q8_0) | Supported; gated per tensor |
| Qwen3.5 dense hybrid adapter, including preserve | Supported; real Qwen3.5-0.8B checkpoint loads with finite logits |
| StableLM conversion adapter | Supported |
| LLaVA Llama + CLIP direct/hybrid VLM adapter | Native tiny-checkpoint image-path parity tested |
| Qwen2-VL direct/hybrid VLM adapter | Tiny parity; named 2B direct checkpoint passes real-image acceptance |
| MiniCPM-V 4.7 1B VLM adapter | Native image-path CI on tiny checkpoints; pretrained caption quality unverified |
| Language-only 4-bit VLM conversion | Packed weights + native decoded-reference parity tested |
| Qwen3-VL / Qwen2-VL text conditioning | Portable conversion fixtures; BF16/mRoPE preserved; native conditioning parity unverified |
