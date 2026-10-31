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
| Per-format tokenizer builders with schema validation (`gguf2mlx/tokenizer.py`) | Supported; validation names the offending GGUF key |
| Gemma and Phi-3/Phi-4 architecture fixtures, including LongRoPE config | Supported |
| Bounded-memory direct quantization (`--direct-quant`) | Supported |
| Source-block preservation (`--quant-source preserve`, Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K/Q8_0) | Supported; gated per tensor. `IQ*`/`Q2_K`/`Q3_K` convert but cannot be preserved |
| Qwen3.5 dense hybrid adapter, including preserve | Supported; real Qwen3.5-0.8B checkpoint loads with finite logits |
| StableLM conversion adapter | Supported |
| Mixtral MoE adapter (stacked and per-expert on-disk layouts) | Supported; emitted names checked against `mlx_lm.models.mixtral` itself |
| MoE router / shared-expert / fused-gate matrix (`tests/test_moe_matrix.py`) | Supported; every layout's destinations checked against the pinned `mlx-lm` parameter trees |
| LLaVA Llama + CLIP direct/hybrid VLM adapter | Native tiny-checkpoint image-path parity tested |
| LLaVA-NeXT anyres adapter (CLIP-L/336 + `model.image_newline`) | Real llava-v1.6 mmproj: every tensor consumed, every name and shape matched against the HF index, tower loads and runs in mlx-vlm |
| Qwen2-VL direct/hybrid VLM adapter | Tiny parity; named 2B direct checkpoint passes real-image acceptance |
| MiniCPM-V 4.7 1B VLM adapter | Native image-path CI on tiny checkpoints; pretrained caption quality unverified |
| Language-only 4-bit VLM conversion | Packed weights + native decoded-reference parity tested |
| Qwen3-VL / Qwen2-VL text conditioning | Portable conversion fixtures; BF16/mRoPE preserved; native conditioning parity unverified |
