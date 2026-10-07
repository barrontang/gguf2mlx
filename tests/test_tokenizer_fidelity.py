from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gguf2mlx import gguf2mlx as core


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


def test_wordpiece_tokenizer_json_encodes_reference_text():
    tokenizer_json = core._build_tokenizer_json(
        ["[UNK]", "[CLS]", "[SEP]", "hello", "##s"],
        [2, 3, 3, 1, 1],
        [],
        [],
        "wordpiece",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=0,
    )

    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    assert tokenizer.encode("Hellos").ids == [3, 4]
    assert tokenizer.decode([3, 4]) == "hellos"


def test_extract_tokenizer_honors_gguf_flags_and_unknown_id(tmp_path):
    reader = _FakeReader(
        {
            "tokenizer.ggml.model": "sentencepiece",
            "tokenizer.ggml.pre": "default",
            "tokenizer.ggml.tokens": ["<pad>", "<s>", "<unk>", "</s>", "▁hello"],
            "tokenizer.ggml.token_type": [3, 3, 2, 3, 1],
            "tokenizer.ggml.scores": [0.0, 0.0, 0.0, 0.0, -1.0],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 3,
            "tokenizer.ggml.unknown_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.add_eos_token": True,
            "tokenizer.ggml.add_space_prefix": False,
        }
    )

    core.extract_tokenizer(reader, tmp_path)

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    tokenizer_json = json.loads((tmp_path / "tokenizer.json").read_text())
    assert tokenizer_config["add_bos_token"] is False
    assert tokenizer_config["add_eos_token"] is True
    assert tokenizer_config["unk_token"] == "<unk>"
    assert tokenizer_config["gguf_tokenizer_pre"] == "default"
    assert tokenizer_json["model"]["unk_id"] == 2
    assert tokenizer_json["pre_tokenizer"]["prepend_scheme"] == "never"


def test_extract_tokenizer_preserves_embedded_huggingface_json(tmp_path):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    embedded = tokenizer.to_str()
    reader = _FakeReader(
        {
            "tokenizer.ggml.model": "bpe",
            "tokenizer.ggml.tokens": ["<unk>", "hello"],
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.huggingface.json": embedded,
        }
    )

    core.extract_tokenizer(reader, tmp_path)

    written = json.loads((tmp_path / "tokenizer.json").read_text())
    assert written == json.loads(embedded)
    assert Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("hello").ids == [1]


@pytest.mark.parametrize("loader", ["tokenizers", "transformers"])
def test_qwen_bpe_without_unknown_does_not_register_normal_character(tmp_path, loader):
    if loader == "transformers":
        transformers = pytest.importorskip(
            "transformers", reason="HF loader integration requires optional transformers",
        )
    reader = _FakeReader({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["!", "a", "!a", "<s>", "</s>", "<pad>"],
        "tokenizer.ggml.token_type": [1, 1, 1, 3, 3, 3],
        "tokenizer.ggml.merges": ["! a"],
        "tokenizer.ggml.bos_token_id": 3,
        "tokenizer.ggml.eos_token_id": 4,
        "tokenizer.ggml.padding_token_id": 5,
    })
    core.extract_tokenizer(reader, tmp_path, arch="qwen2moe")
    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    tokenizer_json = json.loads((tmp_path / "tokenizer.json").read_text())
    assert "unk_token" not in config
    assert tokenizer_json["model"]["unk_token"] is None
    assert tokenizer_json["normalizer"] is None
    if loader == "transformers":
        tokenizer = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
        assert tokenizer.encode("!a", add_special_tokens=False) == [2]
        assert tokenizer.encode("!", add_special_tokens=False) == [0]
    else:
        tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
        assert tokenizer.encode("!a", add_special_tokens=False).ids == [2]
        assert tokenizer.encode("!", add_special_tokens=False).ids == [0]


@pytest.mark.skipif(
    os.getenv("GGUF2MLX_RUN_E2E") != "1", reason="Requires optional native MoE fixture",
)
def test_real_qwen_moe_tokenizer_matches_native(tmp_path):
    source = json.loads(os.getenv("GGUF2MLX_MOE_MODELS", "{}")).get("qwen2moe")
    if not source:
        pytest.skip("GGUF2MLX_MOE_MODELS does not provide qwen2moe")
    llama_cpp = pytest.importorskip("llama_cpp")
    from gguf import GGUFReader
    from transformers import AutoTokenizer

    core.extract_tokenizer(GGUFReader(source), tmp_path, arch="qwen2moe")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    native = llama_cpp.Llama(model_path=source, vocab_only=True, verbose=False)
    try:
        texts = [
            "Explain why the sky is blue.", "!a !! punctuation",
            "  repeated  spaces\tand\nlines", "中文测试 👋 café cafe\u0301",
            (Path(__file__).parents[1] / "README.md").read_text(),
        ]
        for text in texts:
            assert tokenizer.encode(text, add_special_tokens=False) == native.tokenize(
                text.encode(), add_bos=False, special=True,
            )
    finally:
        native.close()


def test_explicit_small_special_token_ids_are_not_rewritten(tmp_path):
    reader = _FakeReader(
        {
            "tokenizer.ggml.model": "bpe",
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "<|endoftext|>"],
            "tokenizer.ggml.token_type": [2, 3, 3, 3],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
        }
    )

    core.extract_tokenizer(reader, tmp_path)

    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer_config["bos_token"] == "<s>"
    assert tokenizer_config["eos_token"] == "</s>"


def _gemma_reader(add_space_prefix=None, add_bos=True, add_eos=False):
    tokens = [
        "<pad>", "<eos>", "<bos>", "<unk>", "<0x09>",
        "▁", "h", "e", "l", "o", "he", "hel", "hell", "hello", "▁hello",
        "é", "\u0301",
    ]
    metadata = {
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [3, 3, 3, 2, 6] + [1] * (len(tokens) - 5),
        # A Unigram model prefers individual characters at these scores.
        "tokenizer.ggml.scores": [0.0] * 5 + [-1.0] * 5 + [-10.0] * 5 + [-1.0] * 2,
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.add_bos_token": add_bos,
        "tokenizer.ggml.add_eos_token": add_eos,
    }
    if add_space_prefix is not None:
        metadata["tokenizer.ggml.add_space_prefix"] = add_space_prefix
    return _FakeReader(metadata)


@pytest.mark.parametrize("add_space_prefix", [None, True, False])
def test_gemma_rebuilds_score_ranked_bpe_without_unicode_normalization(tmp_path, add_space_prefix):
    core.extract_tokenizer(_gemma_reader(add_space_prefix), tmp_path, arch="gemma")
    data = json.loads((tmp_path / "tokenizer.json").read_text())
    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    prefix = [] if add_space_prefix is False else [5]
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert data["model"]["type"] == "BPE"
    assert tokenizer.encode("hello", add_special_tokens=False).ids == (
        [13] if add_space_prefix is False else [14]
    )
    assert tokenizer.encode(" hello", add_special_tokens=False).ids == (
        [14] if add_space_prefix is False else [5, 14]
    )
    assert tokenizer.encode("\t", add_special_tokens=False).ids == prefix + [4]
    assert tokenizer.encode("é", add_special_tokens=False).ids == prefix + [15]
    assert tokenizer.encode("e\u0301", add_special_tokens=False).ids == prefix + [7, 16]
    assert tokenizer.decode(tokenizer.encode(" hello").ids) == " hello"
    assert tokenizer.encode("").ids == [2]


@pytest.mark.parametrize("add_bos,add_eos", [(False, False), (True, False), (False, True), (True, True)])
def test_gemma_rebuilt_tokenizer_honors_special_token_flags(tmp_path, add_bos, add_eos):
    core.extract_tokenizer(_gemma_reader(add_bos=add_bos, add_eos=add_eos), tmp_path, arch="gemma")
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    expected = ([2] if add_bos else []) + [14] + ([1] if add_eos else [])
    assert tokenizer.encode("hello").ids == expected
    assert tokenizer.encode("hello", add_special_tokens=False).ids == [14]


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")]])
def test_gemma_bpe_rejects_missing_or_nonfinite_scores(scores):
    with pytest.raises(ValueError, match="finite score"):
        core._sentencepiece_bpe_merges(["a"], [1], scores)


def test_sentencepiece_bpe_merges_use_scores_not_vocab_order():
    assert core._sentencepiece_bpe_merges(
        ["a", "b", "c", "ab", "bc"], [1] * 5, [-1.0, -1.0, -1.0, -10.0, -2.0],
    ) == [["b", "c"], ["a", "b"]]


def test_finalize_real_safetensors_shards_indexes_all_keys(tmp_path):
    save_file({"a": np.zeros((2, 2), dtype=np.float16)}, tmp_path / "model-00001-of-NNNNN.safetensors")
    save_file({"b": np.ones((2, 2), dtype=np.float16)}, tmp_path / "model-00002-of-NNNNN.safetensors")
    assert core._finalize_safetensor_shards(tmp_path, 16) == (2, 2)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index == {
        "metadata": {"total_size": 16},
        "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
    }


@pytest.mark.skipif(
    os.getenv("GGUF2MLX_RUN_E2E") != "1"
    or not os.getenv("GGUF2MLX_GEMMA_GGUF")
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Set GGUF2MLX_RUN_E2E=1 and GGUF2MLX_GEMMA_GGUF on Apple Silicon",
)
def test_real_gemma_tokenizer_matches_native_llama_cpp(tmp_path):
    from gguf import GGUFReader
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    source = Path(os.environ["GGUF2MLX_GEMMA_GGUF"])
    reader = GGUFReader(str(source))
    assert core.get_metadata_str(reader, "general.architecture") == "gemma"
    core.extract_tokenizer(reader, tmp_path, arch="gemma")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    texts = [
        "", "hello", " hello", "  hello", "a  b", "\t", "\nhello",
        "é", "e\u0301", "你好，世界！", "🙂",
        (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8"),
    ]
    reference = Llama(model_path=str(source), n_ctx=32, n_gpu_layers=0, verbose=False)
    try:
        assert len(tokenizer.get_vocab()) == reference.n_vocab()
        assert tokenizer.bos_token_id == reference.token_bos()
        for text in texts:
            assert tokenizer.encode(text, add_special_tokens=False) == reference.tokenize(
                text.encode("utf-8"), add_bos=False, special=False,
            )
        assert tokenizer.encode("hello") == reference.tokenize(
            b"hello", add_bos=True, special=False,
        )
    finally:
        reference.close()
