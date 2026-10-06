"""Portable tests for the fail-closed perplexity integration runner."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "benchmarks" / "benchmark_transpacking.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_transpacking", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_gate_never_rounds_or_uses_tolerance():
    assert benchmark._compare(5.0, 5.0)["passed"]
    close = math.nextafter(5.0, math.inf)
    result = benchmark._compare(5.0, close)
    assert not result["passed"]
    assert result["absolute_delta"] > 0
    assert result["llama_cpp_hex"] != result["mlx_hex"]
    assert round(5.0, 4) == round(close, 4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, 0.5])
def test_invalid_ppl_fails(bad):
    with pytest.raises(ValueError):
        benchmark._compare(5.0, bad)


def test_fixed_windows_and_bos():
    tokens = list(range(10, 30))
    np.testing.assert_array_equal(benchmark._windows(tokens, 4, 2, None),
                                  [[10, 11, 12, 13], [14, 15, 16, 17]])
    np.testing.assert_array_equal(benchmark._windows(tokens, 4, 2, 1),
                                  [[1, 10, 11, 12], [1, 13, 14, 15]])


@pytest.mark.parametrize("length,samples", [(1, 2), (4, 0), (4, -1), (10, 2)])
def test_bad_or_short_corpus_fails(length, samples):
    with pytest.raises(ValueError):
        benchmark._windows([1, 2, 3], length, samples, None)


def test_tokenizer_and_bos_mismatches_fail():
    reference = SimpleNamespace(tokenize=lambda *a, **kw: [2, 3], token_bos=lambda: 1)
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [2, 3], bos_token_id=1)
    assert benchmark._shared_tokens(reference, tokenizer, "fixed", True) == [2, 3]
    tokenizer.bos_token_id = 0
    with pytest.raises(ValueError, match="BOS"):
        benchmark._shared_tokens(reference, tokenizer, "fixed", True)
    tokenizer.encode = lambda *a, **kw: [2, 4]
    with pytest.raises(ValueError, match="tokenizers"):
        benchmark._shared_tokens(reference, tokenizer, "fixed", False)


def test_native_logits_reset_each_window():
    class Reference:
        def __init__(self):
            self.resets = 0
            self.scores = np.zeros((4, 8), dtype=np.float32)

        def reset(self):
            self.resets += 1

        def eval(self, tokens):
            self.scores[:len(tokens)] = self.resets

        def n_vocab(self):
            return 8

    reference = Reference()
    adapter = benchmark._LlamaLogits(reference, SimpleNamespace(array=np.asarray))
    logits = adapter(np.array([[1, 2, 3], [4, 5, 6]]))
    assert logits.shape == (2, 3, 8)
    assert reference.resets == 2
    assert (logits[0] == 1).all() and (logits[1] == 2).all()


@pytest.mark.parametrize("passed", [False, True])
def test_cli_exit_status_and_json(tmp_path, monkeypatch, passed):
    source, corpus, model = tmp_path / "input.gguf", tmp_path / "text.txt", tmp_path / "mlx"
    source.touch()
    corpus.write_text("fixed", encoding="utf-8")
    model.mkdir()
    report = tmp_path / "report.json"
    monkeypatch.setattr(benchmark, "_evaluate", lambda args: {"success": passed})
    status = benchmark.main([
        "--eval-ppl", "--input", str(source), "--corpus", str(corpus),
        "--mlx-model", str(model), "--result-json", str(report),
    ])
    assert status == (0 if passed else 1)
    assert json.loads(report.read_text())["success"] is passed


def test_missing_dependency_is_failure_not_skip(tmp_path, monkeypatch):
    source, corpus, model = tmp_path / "input.gguf", tmp_path / "text.txt", tmp_path / "mlx"
    source.touch()
    corpus.touch()
    model.mkdir()
    report = tmp_path / "report.json"

    def missing(args):
        raise ImportError("native backend missing")

    monkeypatch.setattr(benchmark, "_evaluate", missing)
    assert benchmark.main([
        "--eval-ppl", "--input", str(source), "--corpus", str(corpus),
        "--mlx-model", str(model), "--result-json", str(report),
    ]) == 1
    result = json.loads(report.read_text())
    assert not result["success"] and "ImportError" in result["error"]


def test_report_cannot_overwrite_corpus(tmp_path):
    source, corpus, model = tmp_path / "input.gguf", tmp_path / "text.txt", tmp_path / "mlx"
    source.touch()
    corpus.write_text("keep me", encoding="utf-8")
    model.mkdir()
    assert benchmark.main([
        "--eval-ppl", "--input", str(source), "--corpus", str(corpus),
        "--mlx-model", str(model), "--result-json", str(corpus),
    ]) == 1
    assert corpus.read_text() == "keep me"


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_report_cannot_overwrite_filesystem_alias(tmp_path, alias):
    source, corpus, model = tmp_path / "input.gguf", tmp_path / "text.txt", tmp_path / "mlx"
    source.touch()
    corpus.write_text("keep me", encoding="utf-8")
    model.mkdir()
    report = tmp_path / "report.json"
    if alias == "hardlink":
        os.link(corpus, report)
    else:
        report.symlink_to(corpus)
    assert benchmark.main([
        "--eval-ppl", "--input", str(source), "--corpus", str(corpus),
        "--mlx-model", str(model), "--result-json", str(report),
    ]) == 1
    assert corpus.read_text() == "keep me"


def test_evaluator_uses_native_public_contract_and_shared_scoring(tmp_path, monkeypatch):
    source, corpus, model = tmp_path / "input.gguf", tmp_path / "text.txt", tmp_path / "mlx"
    source.touch()
    corpus.write_text("fixed corpus", encoding="utf-8")
    model.mkdir()
    (model / "model.safetensors").touch()
    calls = []

    class Reference:
        def __init__(self, **kwargs):
            assert kwargs["logits_all"] is True
            self.scores = np.zeros((4, 8), dtype=np.float32)
            self.closed = False
            calls.append(self)

        def tokenize(self, text, **kwargs):
            assert kwargs == {"add_bos": False, "special": False}
            return list(range(8))

        def reset(self):
            calls.append("reset")

        def eval(self, tokens):
            calls.append(tokens)

        def n_vocab(self):
            return 8

        def close(self):
            self.closed = True

    class TokenizerWrapper:
        # Mirrors MLX-LM wrappers with no __len__ special method.
        def encode(self, text, **kwargs):
            return list(range(8))

        def get_vocab(self):
            return {str(i): i for i in range(8)}

    mx = ModuleType("mlx.core")
    mx.array, mx.float32, mx.eval = np.asarray, np.float32, lambda *a: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    native = ModuleType("llama_cpp")
    native.Llama = Reference
    native.llama_print_system_info = lambda: b"test backend"
    utils = ModuleType("mlx_lm.utils")
    utils.load = lambda path: (
        lambda batch: np.zeros((*batch.shape, 8), dtype=np.float32), TokenizerWrapper()
    )
    perplexity = ModuleType("mlx_lm.perplexity")

    def eval_ppl(evaluator, data, batch_size):
        assert batch_size == 1
        assert data.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
        for row in data:
            logits = evaluator(row[None, :-1])
            assert logits.shape == (1, 3, 8)
        return 8.0, 0.0

    perplexity.eval_ppl = eval_ppl
    for name, module in {
        "mlx": mlx, "mlx.core": mx, "llama_cpp": native,
        "mlx_lm": ModuleType("mlx_lm"), "mlx_lm.utils": utils,
        "mlx_lm.perplexity": perplexity,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark, "_validate_source", lambda path: {"Q4_0": 1})
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "test")
    result = benchmark._evaluate(SimpleNamespace(
        input=source, corpus=corpus, mlx_model=model, sequence_length=4,
        num_samples=2, n_gpu_layers=0, add_bos=False,
    ))
    assert calls[0].closed
    assert calls[1:] == ["reset", [0, 1, 2], "reset", [4, 5, 6]]
    assert result["success"] and result["comparison"]["logits_identical"]
    assert result["scored_tokens"] == 6
    assert result["unused_corpus_tokens"] == 0
    assert not result["bit_exact_transpacking_proven"]
