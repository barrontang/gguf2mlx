# Native validation results (2026-10-18)

> Full native run: counts, pinned fixtures and how to reproduce them.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

All **645 current tests passed across batched runs**, with no remaining skipped
or unexecuted test cases. The initial full run with both opt-in flags enabled
passed 614 tests and skipped six for missing real-model fixtures. After fetching
those fixtures and fixing the Qwen3 text assertion and DeepSeek-V2 config, the
final coverage was:

| Coverage | Passed |
|---|---:|
| Portable and native synthetic tests, including new regressions | 639 |
| Real Qwen1.5 MoE, Qwen3 MoE, and DeepSeek-V2 Lite conversion/inference | 3 |
| Real Qwen1.5 MoE and Gemma tokenizer parity against llama.cpp | 2 |
| Real Qwen2-VL-2B image smoke test | 1 |

The real-model cases were selected separately to avoid repeating large
conversions; the 639-test run deselected those six cases, rather than skipping
them. Six existing missing-metadata warnings came from synthetic config tests;
the DeepSeek run also warned about its legacy tokenizer fallback.
Validation used Python 3.11.13 on macOS ARM64 with 64 GiB memory,
`tests/requirements-vlm.txt`, `gguf` 0.19.0, `numpy` 2.4.6, and
`llama-cpp-python` 0.3.36. The optional `ppl` extra supplies the native tokenizer
reference runtime:

```bash
.venv-vlm/bin/python -m pip install -e ".[ppl]" -r tests/requirements-vlm.txt
```

Real-model inputs were fetched from these pinned Hugging Face repositories:

| Repository | File(s) | Revision |
|---|---|---|
| `tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF` | `Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf` | `2b36b746552a7c99c814ee43120f318391e608c5` |
| `unsloth/Qwen3-30B-A3B-GGUF` | `Qwen3-30B-A3B-Q4_K_M.gguf` | `d5b1d57bd0b504ac62ae6c725904e96ef228dc74` |
| `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | `DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf` | `8f248fa2072348f77a8bc37754e470de1f61866e` |
| `mlabonne/gemma-2b-it-GGUF` | `gemma-2b-it.Q4_K_M.gguf` | `bafe4a9c05a88b3486d0d6fa35fa6dec48aa64d5` |
| `bartowski/Qwen2-VL-2B-Instruct-GGUF` | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf`, `mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | `2160e265f926e84e84dbb1c73e623b5324b23707` |

For Qwen2-VL, the matching local HF assets and astronaut image are the inputs
documented in the [real-checkpoint acceptance report](../qwen2-vl-real-validation.md).
The smoke test checks finite logits, image sensitivity, and generation, not a
new twelve-image acceptance run. MoE mixed output is checked only for artifact
consistency; runtime inference uses the uniform conversion.

After downloading the fixtures into the paths below, enable every current
opt-in test in one invocation (large conversion outputs use pytest temporary
directories):

```bash
GGUF2MLX_RUN_E2E=1 GGUF2MLX_RUN_VLM_E2E=1 HF_HUB_OFFLINE=1 \
GGUF2MLX_MOE_MODELS='{"qwen2moe":"models/native-validation/qwen2moe/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf","qwen3moe":"models/qwen3-moe-validation/gguf/Qwen3-30B-A3B-Q4_K_M.gguf","deepseek2":"models/native-validation/deepseek2/DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf"}' \
GGUF2MLX_GEMMA_GGUF=models/native-validation/gemma/gemma-2b-it.Q4_K_M.gguf \
GGUF2MLX_VLM_GGUF=models/qwen2-vl-validation/gguf/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
GGUF2MLX_VLM_MMPROJ=models/qwen2-vl-validation/gguf/mmproj-Qwen2-VL-2B-Instruct-f16.gguf \
GGUF2MLX_VLM_HF=models/qwen2-vl-validation/hf \
GGUF2MLX_VLM_IMAGE=models/qwen2-vl-validation/images/astronaut.png \
.venv-vlm/bin/python -m pytest -q -ra
```
