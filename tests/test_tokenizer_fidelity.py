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


@pytest.mark.parametrize(
    "pre,expected", [("gpt-2", [6]), ("llama-bpe", [5, 3]), ("qwen2", [0, 1, 2, 3])],
)
def test_bpe_pre_metadata_controls_number_splitting_offline(tmp_path, monkeypatch, pre, expected):
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("Tokenizer reconstruction must not access the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    reader = _FakeReader({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": ["1", "2", "3", "4", "12", "123", "1234"],
        "tokenizer.ggml.merges": ["1 2", "12 3", "123 4"],
    })
    core.extract_tokenizer(reader, tmp_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tokenizer.encode("1234", add_special_tokens=False).ids == expected
    assert not any(role in config for role in ("unk_token", "pad_token", "bos_token", "eos_token"))


@pytest.mark.parametrize("lowercase,strip_accents,text", [
    (False, False, "Héllos"), (True, False, "héllos"), (True, True, "hellos"),
])
def test_real_bert_gguf_header_reverses_wordpiece_vocab(tmp_path, lowercase, strip_accents, text):
    from gguf import GGUFReader, GGUFWriter

    source = tmp_path / "bert.gguf"
    writer = GGUFWriter(str(source), "bert")
    writer.add_tokenizer_model("bert")
    writer.add_token_list(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "▁" + text[:-1], "s"])
    writer.add_token_types([2, 3, 3, 3, 1, 1])
    writer.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.seperator_token_id", 2)
    writer.add_uint32("tokenizer.ggml.padding_token_id", 3)
    writer.add_bool("tokenizer.ggml.normalizer.lowercase", lowercase)
    writer.add_bool("tokenizer.ggml.normalizer.strip_accents", strip_accents)
    writer.add_tensor("token_embd.weight", np.zeros((6, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    output = tmp_path / "tokenizer"
    output.mkdir()
    core.extract_tokenizer(GGUFReader(str(source)), output, arch="bert")
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    assert tokenizer.token_to_id(text[:-1]) == 4
    assert tokenizer.token_to_id("##s") == 5
    assert tokenizer.encode("Héllos").ids == [1, 4, 5, 2]
    pair = tokenizer.encode("Héllos", "Héllos")
    assert pair.ids == [1, 4, 5, 2, 4, 5, 2]
    assert pair.type_ids == [0, 0, 0, 0, 1, 1, 1]
    assert tokenizer.decode([4, 5]) == text


def test_t5_unigram_scores_whitespace_and_eos_are_preserved(tmp_path):
    reader = _FakeReader({
        "tokenizer.ggml.model": "t5",
        "tokenizer.ggml.tokens": ["<pad>", "</s>", "<unk>", "▁hello", "hello", "▁"],
        "tokenizer.ggml.token_type": [3, 3, 2, 1, 1, 1],
        "tokenizer.ggml.scores": [0.0, 0.0, 0.0, -0.1, -0.2, -2.0],
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": 2,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.remove_extra_whitespaces": True,
    })
    core.extract_tokenizer(reader, tmp_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    data = json.loads((tmp_path / "tokenizer.json").read_text())
    assert data["model"]["type"] == "Unigram"
    assert data["model"]["vocab"][3] == ["▁hello", -0.1]
    assert tokenizer.encode("  hello  ").ids == [3, 1]
    assert tokenizer.encode("hello", add_special_tokens=False).ids == [3]
    assert tokenizer.decode([3]) == " hello"


def test_llama_model_reconstructs_score_ranked_bpe_not_unigram(tmp_path):
    reader = _FakeReader({
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "▁", "a", "b", "ab", "▁ab"],
        "tokenizer.ggml.token_type": [2, 3, 3, 1, 1, 1, 1, 1],
        "tokenizer.ggml.scores": [0.0] * 3 + [-1.0] * 3 + [-10.0] * 2,
    })
    core.extract_tokenizer(reader, tmp_path)
    data = json.loads((tmp_path / "tokenizer.json").read_text())
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert data["model"]["type"] == "BPE"
    assert tokenizer.encode("ab").ids == [1, 7]
    assert tokenizer.encode("ab", add_special_tokens=False).ids == [7]


@pytest.mark.parametrize("prefix,remove", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("text", ["", "hello", "  hello", "hello  world ", "  ", "\thello\t"])
def test_t5_normalization_matches_native_ascii_space_rules(prefix, remove, text):
    data = core._build_tokenizer_json(
        ["<unk>", "▁", "hello", "world", "\t"], [2, 1, 1, 1, 1], [],
        [0.0, -1.0, -1.0, -1.0, -1.0], "t5", -1, -1, -1,
        add_space_prefix=prefix, remove_extra_whitespaces=remove,
    )
    tokenizer = Tokenizer.from_str(json.dumps(data))
    if remove:
        expected = "▁" + "▁".join(part for part in text.split(" ") if part) if text.strip(" ") else ""
    else:
        expected = ("▁" if prefix and text.strip(" ") else "") + text.replace(" ", "▁")
    assert tokenizer.normalizer.normalize_str(text) == expected


def test_inferred_wordpiece_separator_is_special_without_token_types(tmp_path):
    core.extract_tokenizer(_FakeReader({
        "tokenizer.ggml.model": "wordpiece",
        "tokenizer.ggml.tokens": ["[UNK]", "[CLS]", "[SEP]", "hello"],
    }), tmp_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert tokenizer.encode("hello").ids == [1, 3, 2]
    assert tokenizer.decode([1, 3, 2]) == "hello"


def test_t5_character_map_bytes_are_serialized_for_precompiled_normalizer(tmp_path):
    import base64

    charsmap = [1, 2, 3, 128, 255]
    core.extract_tokenizer(_FakeReader({
        "tokenizer.ggml.model": "t5",
        "tokenizer.ggml.tokens": ["<unk>", "hello"],
        "tokenizer.ggml.scores": [0.0, -1.0],
        "tokenizer.ggml.unknown_token_id": 0,
        "tokenizer.ggml.precompiled_charsmap": charsmap,
    }), tmp_path)
    document = json.loads((tmp_path / "tokenizer.json").read_text())
    normalizer = document["normalizer"]["normalizers"][0]
    assert normalizer["type"] == "Precompiled"
    assert base64.b64decode(normalizer["precompiled_charsmap"]) == bytes(charsmap)


def test_embedded_tokenizer_with_wrong_ids_is_rebuilt(tmp_path):
    tokenizer = Tokenizer(WordLevel({"a": 1, "b": 0}, unk_token="a"))
    reader = _FakeReader({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.huggingface.json": tokenizer.to_str(),
    })
    with pytest.warns(UserWarning, match="does not match GGUF token IDs"):
        core.extract_tokenizer(reader, tmp_path)
    rebuilt = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert rebuilt.encode("a").ids == [0]
    assert rebuilt.encode("b").ids == [1]


@pytest.mark.parametrize("metadata,message", [
    ({}, "no tokenizer.ggml.tokens"),
    ({"tokenizer.ggml.tokens": ["a", "a"]}, "duplicate"),
    ({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.bos_token_id": -1}, "outside"),
    ({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.eos_token_id": 2}, "outside"),
    ({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.token_type": [1, 1]}, "type"),
    ({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.scores": [float("nan")]}, "finite"),
    ({
        "tokenizer.ggml.model": "t5", "tokenizer.ggml.tokens": ["<unk>", "a"],
        "tokenizer.ggml.unknown_token_id": 0,
    }, "finite score"),
    ({
        "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.merges": ["a b"],
    }, "merge result"),
])
def test_invalid_metadata_fails_before_writing_assets(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        core.extract_tokenizer(_FakeReader(metadata), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unknown_bpe_pre_reports_reconstruction_limit(tmp_path):
    reader = _FakeReader({
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "future-tokenizer",
        "tokenizer.ggml.tokens": ["a"],
    })
    with pytest.warns(UserWarning, match="parity is not guaranteed"):
        core.extract_tokenizer(reader, tmp_path)


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


def _write_hf_tokenizer(root, tokens):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "WordLevel", "vocab": {token: i for i, token in enumerate(tokens)}},
        "added_tokens": [],
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "unk_token": tokens[0], "bos_token": tokens[1], "eos_token": tokens[2],
    }), encoding="utf-8")


def test_hf_tokenizer_files_replace_gguf_assets_after_id_validation(tmp_path):
    tokens = ["<unk>", "<s>", "</s>", "hello"]
    source = tmp_path / "original"
    _write_hf_tokenizer(source, tokens)
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text('{"vocab_size": 4}', encoding="utf-8")
    reader = _FakeReader({
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.unknown_token_id": 0,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    })
    core._copy_hf_tokenizer(reader, output, str(source), "main", False)
    assert (output / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert (output / "tokenizer_config.json").read_bytes() == (
        source / "tokenizer_config.json"
    ).read_bytes()


def test_hf_tokenizer_mismatch_does_not_replace_generated_assets(tmp_path):
    source = tmp_path / "original"
    _write_hf_tokenizer(source, ["<unk>", "<s>", "</s>", "wrong"])
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text('{"vocab_size": 4}', encoding="utf-8")
    old_tokenizer = output / "tokenizer.json"
    old_config = output / "tokenizer_config.json"
    old_tokenizer.write_text("generated tokenizer", encoding="utf-8")
    old_config.write_text("generated config", encoding="utf-8")
    reader = _FakeReader({"tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "hello"]})
    with pytest.raises(ValueError, match="mismatch at token ID 3"):
        core._copy_hf_tokenizer(reader, output, str(source), "main", False)
    assert old_tokenizer.read_text(encoding="utf-8") == "generated tokenizer"
    assert old_config.read_text(encoding="utf-8") == "generated config"


def test_hf_tokenizer_remote_uses_requested_revision_and_offline_cache(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    tokenizer_root = tmp_path / "remote"
    tokens = ["<unk>", "<s>", "</s>", "hello"]
    _write_hf_tokenizer(tokenizer_root, tokens)
    calls = []
    hub = ModuleType("huggingface_hub")

    def download(repo, filename, revision, local_files_only):
        calls.append((repo, filename, revision, local_files_only))
        return str(tokenizer_root / filename)

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text('{"vocab_size": 4}', encoding="utf-8")
    reader = _FakeReader({"tokenizer.ggml.tokens": tokens})
    core._copy_hf_tokenizer(reader, output, "org/model", "abc123", True)
    assert calls == [
        ("org/model", "tokenizer.json", "abc123", True),
        ("org/model", "tokenizer_config.json", "abc123", True),
    ]


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


@pytest.mark.parametrize("arch", ["gemma2", "gemma3"])
def test_gemma2_and_gemma3_keep_gemma_tokenizer_class(tmp_path, arch):
    core.extract_tokenizer(_gemma_reader(), tmp_path, arch=arch)

    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"


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
