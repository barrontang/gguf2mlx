# Supported conversion matrix

> Which GGUF architectures convert, which are inspection-only, and what end-to-end evidence exists for each.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

`gguf2mlx` distinguishes between:

1. **recognized for inspection** via GGUF metadata, and
2. **conversion enabled** through an explicit tensor adapter.

Language-model conversion is enabled only for explicit adapters. Text conditioning
adds a separate, opt-in contract for the four Qwen-VL identifiers above.
Full-VLM detection and conversion provide LLaVA, Qwen2-VL, and a dense
MiniCPM-V 4.7 1B adapter.

### Conversion-enabled architectures

| Family | GGUF architecture IDs | Conversion | End-to-end | MoE | Mixed precision | `.mlx` bundle | VLM |
|---|---|---|---|---|---|---|---|
| Llama | `llama`, `mistral` | ✅ | ✅ Llama-3.2-1B Q4_0/Q4_K_M/Q8_0 real GGUF in CI | ❌ | — | ✅ generic package/verify | ❌ |
| Qwen | `qwen2`, dense `qwen3` | ✅ | ✅ Qwen2.5-0.5B Q4_0/Q4_K_M/Q8_0 and Qwen3-0.6B Q8_0 in CI | ❌ | — | ✅ generic package/verify | ❌ |
| Qwen3.5 family | `qwen35`; aliases `qwen3_5`, `qwen3.5`, `qwen36`, `qwen3_6`, `qwen3.6`, `qwen38`, `qwen3_8`, `qwen3.8` | ✅ dense layouts + preserve | ✅ Qwen3.5-0.8B Q4_K_M preserve: local run passes convert → weight gate → preserve gate → `mlx_lm.load` → finite logits (nightly job pending); aliases fixture-tested | ❌ | — | ✅ generic package/verify | ❌ |
| Qwen MoE | `qwen2moe`, `qwen3moe` | ✅ | ✅ Qwen1.5 and Qwen3-30B-A3B real-GGUF uniform 4-bit smoke tests | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| DeepSeek | `deepseek2`, `deepseek3` | ✅ | ✅ V2 Lite real Q2_K smoke test; V3 runtime unverified | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| GLM | `glm4moe` | ✅ | ❌ | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| Gemma | `gemma`, `gemma2`, `gemma3`, text-only `gemma4` | ✅ | ✅ Gemma-2-2B Q4_K_M/Q8_0 real GGUF in CI (`gemma`, `gemma3`, `gemma4` stay fixture-only) | Gemma4 MoE layout supported | — | ✅ generic package/verify | ❌ |
| Phi | `phi3` (Phi-3 and Phi-4; Phi-4 alias) | ✅ Phi-3 4K/LongRoPE and Phi-4 fixtures | ✅ Phi-3-mini Q4_K_M/Q8_0 real GGUF in CI plus synthetic GGUF | ❌ | — | ✅ generic package/verify | ❌ |
| StableLM | `stablelm` | ✅ | ❌ | ❌ | — | ✅ generic package/verify | ❌ |
| GLM | `glm-dsa` | ⚠️ experimental | ❌ | ✅ | ⚠️ experimental; JANG runtime unverified | ✅ generic package/verify | ❌ |
| LLaVA | `llava`, or `llama` + CLIP/mmproj | ✅ direct/hybrid fixture-validated | Tiny image-path parity; pretrained quality unverified | ❌ | Language-only 4-bit | ✅ generic package/verify | ✅ Llama + CLIP only |
| Qwen2-VL | `qwen2vl` + Qwen2-VL/mmproj | ✅ direct/hybrid fixture-validated | Named 2B direct checkpoint: 12/12 image diagnostic; see [report](qwen2-vl-real-validation.md) | ❌ | Uniform 4-bit: 11/12 exact; source-preserved Q4_K/float: 12/12 | ✅ generic package/verify | ✅ ViT + patch merger |
| MiniCPM-V 4.7 1B | dense `qwen35` + `clip.projector_type=minicpmv4_7` | ✅ portable + native image path | Native image-path CI; pretrained caption quality unverified | ❌ (35B-A3B excluded) | ❌ | ✅ generic package/verify | ✅ SigLIP + window/final mergers |
| Qwen-VL text encoder | `qwen3vl`, `qwen3_vl`, `qwen2vl`, `qwen2_vl` | ✅ `--type text-encoder`, BF16 default | Portable conversion only; native hidden-state/image quality unverified | ❌ | Explicit direct 4-bit only | ✅ generic package/verify | Text only; vision skipped |

“Conversion” means that a code path exists. “End-to-end” means a converted model
has been loaded and exercised; ⚠️ rows still require the opt-in model fixtures.
The mixed-precision contract validator checks agreement between
`conversion_report.json` and `jang_config.json`; this is not a substitute for
loading the result in a JANG-compatible runtime. `.mlx` bundle packaging and
integrity verification are architecture-independent operations on emitted model
directories and do not assert runtime compatibility.

The ✅ end-to-end cells are produced by the `Load matrix` workflow, which
downloads each pinned GGUF by SHA-256, converts it, runs the weight-fidelity
gate, loads the result with `mlx_lm` and requires finite logits. Published CI
evidence: [run 39505897872](https://github.com/barrontang/gguf2mlx/actions/runs/39505897872)
(11/11 entries green, 2026-10-21). A twelfth entry pins Qwen3.5-0.8B Q4_K_M in
preserve mode; locally the full twelve-entry matrix is green on Apple Silicon
(2026-10-26), including that entry, but the workflow has not been re-run since it
was added, so the published run still shows eleven. The entry list lives in
`tests/load_matrix.json`, and each job uploads the conversion report it produced.

### Pinned versus declared-only

`CONVERTIBLE_ARCHES` lists 30 architecture IDs — 22 canonical plus 8 alias
spellings; the pinned matrix covers six of those canonical architectures
(`llama`, `qwen2`, `qwen3`, `qwen35`, `gemma2`, `phi3`). The
difference is not implied — it is written down and enforced.
`tests/test_load_matrix.py` fails when an architecture is convertible but is
neither pinned nor listed in `DECLARED_COVERAGE_GAPS` with a reason, so a new
adapter cannot quietly widen the support claim. Alias spellings (`qwen3_5`,
`qwen3.5`, `qwen2vl`, …) share one canonical ID and one case. Separately,
`tests/test_runtime_registry.py` checks every emitted `model_type` against the
minimum pinned `mlx-lm`/`mlx-vlm` registries.

| Architecture IDs without a pinned case | Reason |
|---|---|
| `deepseek2`, `deepseek3`, `glm4moe`, `glm-dsa`, `qwen2moe`, `qwen3moe` | MoE/MLA sources; the smallest public GGUFs exceed the per-runner download budget, and router protection needs its own case. `weekly-moe.yml` can cover `qwen3moe` and `deepseek2` instead, but only on a runner that already holds those files: it is **inactive until `GGUF2MLX_MOE_MODELS` is set**, and reports `skipped` rather than passing. |
| `gemma`, `gemma3`, `gemma4` | share `gemma2`'s adapter path, which is pinned; no small pinned GGUF for the others yet |
| `mistral`, `phi4` | share `llama`'s tensor map and `phi3`'s sliding-window adapter, both pinned |
| `qwen36`, `qwen38` (and their alias spellings) | newer hybrid variants share the dense Qwen3.5 adapter and are fixture-tested; only `qwen35` has a pinned real-model case |
| `stablelm` | adapter ships; no pinned public GGUF yet |
| `qwen2_vl`, `qwen3_vl` | full VLM entries need the matching HF repo beside the GGUF; `vision.yml` owns that path |

Pin an entry by adding it to `tests/load_matrix.json` and deleting its line from
`DECLARED_COVERAGE_GAPS` — the test fails on a stale line, so the two cannot
drift apart.

#### Qwen3 MoE real-model validation

On 2026-10-18, `Qwen3-30B-A3B-Q4_K_M.gguf` from
[`unsloth/Qwen3-30B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/tree/d5b1d57bd0b504ac62ae6c725904e96ef228dc74)
was converted and exercised on Apple Silicon with 64 GiB unified memory.
The source is 18,556,686,912 bytes, with SHA-256
`9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48`.
The passing pytest rerun used `gguf` 0.19.0, `mlx` 0.32.2, `mlx-lm` 0.31.3,
and `numpy` 2.4.6.

| Check | Observed result |
|---|---|
| Mixed-precision conversion and artifact contract | Passed; 48 protected router tensors and 144 compressed expert tensors |
| Uniform direct 4-bit conversion and MLX-LM loading | Passed |
| Uniform-model logits | All finite; shape `[1, 8, 151936]` |
| First-layer router logits | Shape `[1, 8, 128]`; normalized entropy `0.895546`, within the required `(0.05, 0.95)` interval |
| Default thinking-mode generation | 32 tokens, 28 unique tokens; begins normal `<think>` reasoning before reaching the token limit |

**Historical test failure, now fixed:** the original `output.isprintable()`
assertion rejected the normal newline after Qwen3's `<think>` marker. The test
now accepts line breaks and tabs while rejecting empty/whitespace-only output
and other control characters; eight regression cases cover these boundaries.
The actual real-MoE pytest now passes without disabling thinking or changing
its entropy/token-diversity thresholds. A 32-token reasoning fragment does not
establish answer quality: this remains a smoke test, not a semantic benchmark.
Mixed-model inference and JANG runtime compatibility remain unverified.
The source and both conversions require approximately 52 GiB of disk.

Qwen3.5-family variants share the non-MoE Qwen3.5 tensor contract; Qwen3.x MoE
and Gemma4 multimodal/audio components are not included in these text-model
adapters. `qwen35` is the only variant with a real-checkpoint run (Qwen3.5-0.8B,
Q4_K_M, preserve); `qwen36` and `qwen38` remain fixture-tested only.

#### DeepSeek-V2 Lite real-model validation

The pinned `DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf` checkpoint also passes
mixed-artifact validation and uniform direct 4-bit conversion, MLX-LM loading,
finite-logit checks, router entropy, and generation. Its first routed layer
has logits of shape `[1, 7, 64]` and normalized entropy `0.883369`; generation
produced 32 tokens, 29 unique, beginning with an explanation of Rayleigh
scattering.

Enabling this previously skipped test exposed missing DeepSeek-V2 config:
conversion now preserves the source's dense/expert FFN sizes, leading dense
layers, shared/routed experts, query/KV LoRA layout, MLA head dimensions, and
YaRN scaling (including GGUF's log-multiplier encoding). Invalid required
metadata fails explicitly. The entropy probe measures raw router logits even
when the runtime gate returns top-k indices and scores.
This validates the named V2 Lite checkpoint, not DeepSeek-V3 or mixed-model
inference in a JANG runtime. The checkpoint's `deepseek-llm` tokenizer profile
still emits the existing legacy-BPE fallback warning; exact native tokenizer
parity for DeepSeek was not established by this smoke test.

### Inspection only (not converted)

These may still be recognized by metadata or `--skip-weights`, but they are
**rejected during conversion** until a dedicated, tested adapter exists:

`arctic`, `baichuan`, `bert`, `bitnet`, `bloom`, `chameleon`, `chatglm`,
`codeshell`, `command-r`, `command-r-plus`, `dbrx`, `exaone`, `falcon`,
`gpt2`, `gptneox`, `granite`, `grok-1`, `jais`, `minicpm`,
`minicpm3`, `mpt`, `nemotron`, `olmo`, `olmo2`, `openelm`, `orion`, `phi`,
`phi2`, `plamo`, `refact`, `smolm`, `starcoder`, `t5`, and `xverse`.

That means no more silent "Llama fallback" producing invalid outputs for
unrelated architectures.

MiniCPM-V support is limited to the dense 4.7 1B checkpoint, supplied as a
Qwen3.5 text GGUF plus its matching `minicpmv4_7` vision/projector GGUF and
original HF assets. The 35B-A3B MoE and older 4.6 layouts are rejected.
Plain MiniCPM3 is a separate text-only architecture and is not handled by the
MiniCPM-V adapter.
