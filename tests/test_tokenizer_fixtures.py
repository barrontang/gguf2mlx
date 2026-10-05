from __future__ import annotations

import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel

from gguf2mlx import gguf2mlx as core


FIXTURE = Path(__file__).parent / "fixtures" / "tokenizers" / "architectures.json"


def test_architecture_tokenizer_fixture_covers_supported_families():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(fixture["architectures"]) >= 8
    assert fixture["normalizer"] == "NFC"


def test_twenty_added_tokens_keep_ids_through_roundtrip():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    special_tokens = fixture["special_tokens"]
    tokenizer = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in special_tokens])

    for expected_id, token in enumerate(special_tokens, start=1):
        encoded = tokenizer.encode(token)
        assert encoded.ids == [expected_id]
        assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == token


def test_bpe_byte_fallback_variant_is_preserved():
    tokenizer_json = core._build_tokenizer_json(
        ["<unk>", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"],
        [2, 6, 6, 6, 6],
        [],
        [],
        "bpe",
        bos_id=0,
        eos_id=0,
        pad_id=0,
        unk_id=0,
    )
    assert tokenizer_json["model"]["byte_fallback"] is True
    assert {"type": "ByteFallback"} in tokenizer_json["decoder"]["decoders"]
