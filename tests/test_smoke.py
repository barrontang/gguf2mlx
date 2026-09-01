"""Minimal smoke test: verify the package imports and CLI is wired correctly.

We do NOT download a model in CI (too slow, too big). The actual end-to-end
test against a real GGUF lives in `tests/test_e2e.py` and is opt-in via env var.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gguf2mlx import gguf2mlx as core
from gguf2mlx.gguf2mlx import (
    build_config,
    detect_architecture,
    extract_tokenizer,
    get_metadata_float,
    get_metadata_int,
)


class _FakeField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _FakeReader:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_field(self, key):
        value = self._mapping.get(key)
        return None if value is None else _FakeField(value)


def test_package_imports():
    """The main module must be importable without NameError."""
    import gguf2mlx  # noqa: F401
    from gguf2mlx import gguf2mlx as core
    # Verify the previously-missing imports are present at module level
    assert hasattr(core, "gc"), "gc import was missing — see PR description"
    assert hasattr(core, "warnings"), "warnings import was missing — see PR description"


def test_cli_help():
    """`gguf2mlx --help` should exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "gguf2mlx", "--help"],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "input" in result.stdout.lower() or "gguf" in result.stdout.lower()
    assert "--quantize" in result.stdout
    assert "--q-bits" in result.stdout


def test_get_metadata_int_accepts_python_sequences():
    reader = _FakeReader({"scalar_list": [8], "empty_list": []})

    assert get_metadata_int(reader, "scalar_list") == 8
    assert get_metadata_int(reader, "empty_list") is None


def test_get_metadata_float_accepts_python_sequences():
    reader = _FakeReader({"scalar_tuple": (1.5,), "empty_tuple": ()})

    assert get_metadata_float(reader, "scalar_tuple") == 1.5
    assert get_metadata_float(reader, "empty_tuple") is None


def test_detect_architecture_model_name_fallbacks():
    reader = _FakeReader({"general.name": "Mixtral-8x7B-Instruct-v0.1"})
    assert detect_architecture(reader) == "mistral"

    reader = _FakeReader({"general.name": "DeepSeek-R1-Distill-Qwen-32B"})
    assert detect_architecture(reader) == "qwen2"

    reader = _FakeReader({"general.name": "DeepSeek-V3-0324"})
    assert detect_architecture(reader) == "deepseek3"

    reader = _FakeReader({"general.name": "Command-R+ 104B"})
    assert detect_architecture(reader) == "command-r-plus"

    reader = _FakeReader({"general.name": "Command-R 35B"})
    assert detect_architecture(reader) == "command-r"

    reader = _FakeReader({"general.name": "Yi-34B-Chat"})
    assert detect_architecture(reader) == "llama"

    reader = _FakeReader({"general.name": "Yi 1.5 9B Chat"})
    assert detect_architecture(reader) == "llama"


def test_build_config_preserves_zero_token_ids_and_dtype():
    reader = _FakeReader(
        {
            "tokenizer.ggml.tokens": ["<s>", "</s>"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
        }
    )

    config = build_config(reader, "llama", dtype="float32")

    assert config["bos_token_id"] == 0
    assert config["eos_token_id"] == 1
    assert config["torch_dtype"] == "float32"


def test_extract_tokenizer_uses_context_length_and_zero_token_ids(tmp_path: Path):
    reader = _FakeReader(
        {
            "tokenizer.ggml.model": "bpe",
            "tokenizer.ggml.tokens": np.array(["<s>", "</s>", "<pad>"]),
            "tokenizer.ggml.token_type": np.array([3, 3, 3]),
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.padding_token_id": 2,
        }
    )

    extract_tokenizer(reader, tmp_path, model_max_length=8192)

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer_config["bos_token"] == "<s>"
    assert tokenizer_config["eos_token"] == "</s>"
    assert tokenizer_config["model_max_length"] == 8192


def test_convert_refuses_nonempty_output_without_starting(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "existing.json").write_text("{}")
    started = False

    def fake_convert(*args, **kwargs):
        nonlocal started
        started = True
        return True

    monkeypatch.setattr(core, "_convert", fake_convert)

    assert core.convert("model.gguf", str(output_dir)) is False
    assert started is False
    assert (output_dir / "existing.json").exists()


def test_convert_cleans_staging_directory_after_failure(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"

    def fake_convert(_input, staging_dir, _dtype):
        (Path(staging_dir) / "partial.json").write_text("{}")
        return False

    monkeypatch.setattr(core, "_convert", fake_convert)

    assert core.convert("model.gguf", str(output_dir)) is False
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_convert_returns_false_and_cleans_up_after_exception(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"

    def fake_convert(_input, staging_dir, _dtype):
        (Path(staging_dir) / "partial.json").write_text("{}")
        raise RuntimeError("conversion error")

    monkeypatch.setattr(core, "_convert", fake_convert)

    assert core.convert("model.gguf", str(output_dir)) is False
    assert list(tmp_path.iterdir()) == []


def test_convert_quantizes_output_with_mlx_lm(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"
    quantize_calls = []

    def fake_convert(_input, staging_dir, _dtype):
        staging_path = Path(staging_dir)
        staging_path.mkdir(parents=True, exist_ok=True)
        (staging_path / "config.json").write_text("{}")
        return True

    def fake_quantize(model_path, output_path, q_bits, q_group_size, q_mode):
        quantize_calls.append((model_path, output_path, q_bits, q_group_size, q_mode))
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "quantized.safetensors").write_text("ok")

    monkeypatch.setattr(core, "_convert", fake_convert)
    monkeypatch.setattr(core, "_quantize_output_with_mlx_lm", fake_quantize)

    assert (
        core.convert(
            "model.gguf",
            str(output_dir),
            quantize=True,
            q_bits=4,
            q_group_size=128,
            q_mode="affine",
        )
        is True
    )
    assert len(quantize_calls) == 1
    model_path, quantized_path, q_bits, q_group_size, q_mode = quantize_calls[0]
    assert model_path.name == "fp"
    assert quantized_path.name == "quantized"
    assert q_bits == 4
    assert q_group_size == 128
    assert q_mode == "affine"
    assert (output_dir / "quantized.safetensors").exists()
    assert list(tmp_path.iterdir()) == [output_dir]


def test_convert_quantize_requires_mlx_lm(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"

    def fake_convert(_input, staging_dir, _dtype):
        staging_path = Path(staging_dir)
        staging_path.mkdir(parents=True, exist_ok=True)
        (staging_path / "config.json").write_text("{}")
        return True

    monkeypatch.setattr(core, "_convert", fake_convert)
    monkeypatch.setattr(core, "mlx_lm_convert", None)

    assert core.convert("model.gguf", str(output_dir), quantize=True) is False
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_convert_cleans_staging_directory_after_quantization_failure(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "model"

    def fake_convert(_input, staging_dir, _dtype):
        staging_path = Path(staging_dir)
        staging_path.mkdir(parents=True, exist_ok=True)
        (staging_path / "config.json").write_text("{}")
        return True

    def fake_quantize(_model_path, output_path, q_bits, q_group_size, q_mode):
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "partial.json").write_text("{}")
        raise RuntimeError("quantization error")

    monkeypatch.setattr(core, "_convert", fake_convert)
    monkeypatch.setattr(core, "_quantize_output_with_mlx_lm", fake_quantize)

    assert core.convert("model.gguf", str(output_dir), quantize=True) is False
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []
