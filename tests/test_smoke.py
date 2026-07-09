"""Minimal smoke test: verify the package imports and CLI is wired correctly.

We do NOT download a model in CI (too slow, too big). The actual end-to-end
test against a real GGUF lives in `tests/test_e2e.py` and is opt-in via env var.
"""
import subprocess
import sys

from gguf2mlx.gguf2mlx import detect_architecture, get_metadata_float, get_metadata_int


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
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "input" in result.stdout.lower() or "gguf" in result.stdout.lower()


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
