# Development

> Local test invocations, opt-in real-model suites, runtime-contract checks, and the regression inventory.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

```bash
git clone https://github.com/barrontang/gguf2mlx.git
cd gguf2mlx
uv sync --all-extras

# Tests
pytest

# Prose must match the code: architecture lists, runtime pins, and deleted modules
pytest tests/test_docs_consistency.py

# Opt-in MLX-LM integration test (Apple Silicon)
GGUF2MLX_RUN_E2E=1 pytest tests/test_e2e.py

# Offline native VLM tests (Apple Silicon; no model or image downloads)
GGUF2MLX_RUN_VLM_E2E=1 HF_HUB_OFFLINE=1 \
uv run --isolated --no-project --with-editable . \
  --with-requirements tests/requirements-vlm.txt \
  pytest tests/test_vlm.py tests/test_vlm_cli.py tests/test_vlm_qwen2.py \
  tests/test_vlm_minicpm.py tests/test_vlm_acceptance.py tests/test_vlm_runtime.py

# Emitted model_type must resolve in the minimum pinned runtimes
uv run --isolated --no-project --with-editable . \
  --with-requirements tests/requirements-vlm.txt \
  pytest tests/test_runtime_registry.py

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

# Qwen3-30B-A3B fixture (Q4_K_M GGUF, about 18.56 GB)
# Requires Apple Silicon and sufficient disk for the source and both conversions.
GGUF2MLX_RUN_E2E=1 \
GGUF2MLX_MOE_MODELS='{"qwen3moe":"/models/Qwen3-30B-A3B-Q4_K_M.gguf"}' \
GGUF2MLX_MOE_OUTPUT_ROOT=/models/new-qwen3-moe-validation \
GGUF2MLX_MOE_REPORT_DIR=/reports/qwen3-moe \
pytest -q 'tests/test_e2e.py::test_real_moe_model_has_finite_logits_routing_entropy_and_coherent_output[qwen3moe]'

# Validate an embedded chat template against five conversation shapes
python -m gguf2mlx.chat_validation ./converted-model --expect-marker '<|im_start|>'

# Lint
ruff check src/ tests/ benchmarks/
```

The Python converter is the only implementation. The former `gguf2mlx-rs` crate
was deleted because it duplicated the Python detection path, was never released,
and added a second toolchain to the release surface.

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
- MiniCPM-V 4.7 dense Qwen3.5 text, separate SigLIP projector, merger mappings,
  processor/tokenizer validation, and native image-path CI
- runtime-registry conformance for every emitted LLM and full-VLM model type
- Qwen3.5 preserved projection codes with SSM state kept in float
- Q5_K preserved as 5-bit affine groups, checked against `gguf.quants.dequantize`
  and `mlx.core.dequantize`
- Qwen3.5 hybrid layout restoration: `A_log = log(-ssm_a)`, the squeezed
  `conv1d` kernel widened to `(channels, kernel, 1)`, and the gated norm left
  unshifted — plus a `reshaped` audit layout so the kernel is still value-compared
- `publish --dry-run` never creating the destination, `--verify-load` running
  before repository creation, and shard downloads streaming instead of buffering
- native VLM image features and logits against original HF-layout weights
  (or an independently decoded quantized reference), image-change sensitivity,
  finite outputs and `mlx_vlm.generate`; F32/Q4_0 inputs and FP16/FP32/4-bit outputs

The dedicated **Vision conversion** workflow runs these offline tests on Apple
Silicon and fails if the runtime is unavailable instead of silently skipping.
The random-checkpoint test is a numerical/loader regression, not a visual
reasoning benchmark. The optional pretrained smoke test checks finite logits,
image sensitivity and nonempty generation; it does not certify caption accuracy.
The separate [real Qwen2-VL acceptance harness](https://github.com/barrontang/gguf2mlx/blob/main/benchmarks/validate_vlm.py) audits
all floating output weights and compares fixed labeled images against original HF
and the same GGUF through llama.cpp. It preserves raw answers and exits nonzero
on its predeclared gate failure; see the [results](qwen2-vl-real-validation.md).
Image generators such as `mlxgen` / Qwen-Image can supply test images but are not
substitutes for a matching image-understanding checkpoint.
