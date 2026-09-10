from __future__ import annotations

import json

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
