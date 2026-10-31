"""The tokenizer builders in isolation: one format at a time, and what they refuse.

``tests/test_tokenizer_fidelity.py`` asks whether a rebuilt tokenizer *encodes
like the original*. That question only makes sense once a more basic one is
answered: does each builder, given only its own GGUF fields, produce a complete
document -- and does the schema validator catch the defects that would otherwise
surface as a load failure or, worse, as a tokenizer that loads and disagrees?

So the tests here are structural. They build one format at a time from hand-made
``TokenizerSource`` values, check the four policy blocks each format requires,
check that a policy decision made for one format does not leak into another, and
load every result with ``tokenizers`` to confirm the document is real.
"""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer

from gguf2mlx import tokenizer as tk

#: GPT-2 style vocabulary: one merge, one control token, byte-level splitting.
GPT2 = {
    "model": "gpt2",
    "tokens": ["<unk>", "a", "b", "ab", "<|end|>"],
    "token_types": [tk.TOKEN_TYPE_UNKNOWN, 1, 1, 1, tk.TOKEN_TYPE_CONTROL],
    "merges": ["a b"],
    "bos_id": 0,
    "eos_id": 4,
    "pad_id": 0,
    "unk_id": 0,
}

#: SentencePiece Unigram: scores are the model.
UNIGRAM = {
    "model": "sentencepiece",
    "tokens": ["<unk>", "<s>", "▁hello", "hello", "▁world", "</s>"],
    "token_types": [
        tk.TOKEN_TYPE_UNKNOWN, tk.TOKEN_TYPE_CONTROL, 1, 1, 1, tk.TOKEN_TYPE_CONTROL,
    ],
    "scores": [0.0, -0.5, -1.0, -2.0, -3.0, -4.0],
    "bos_id": 1,
    "eos_id": 5,
    "pad_id": 0,
    "unk_id": 0,
}

#: BERT WordPiece: ``##`` continuation pieces.
WORDPIECE = {
    "model": "bert",
    "tokens": ["[UNK]", "[CLS]", "[SEP]", "hello", "##s"],
    "token_types": [tk.TOKEN_TYPE_UNKNOWN, tk.TOKEN_TYPE_CONTROL, tk.TOKEN_TYPE_CONTROL, 1, 1],
    "bos_id": 1,
    "eos_id": 2,
    "pad_id": 0,
    "unk_id": 0,
    "do_lower_case": True,
}


#: A vocabulary with nothing to merge, for tests that only care about policy.
MINIMAL = {"tokens": ["a"], "token_types": [1], "bos_id": 0, "eos_id": 0, "pad_id": 0, "unk_id": 0}

_BASES = {"gpt2": GPT2, "sentencepiece": UNIGRAM, "bert": WORDPIECE}


def source(model: str = "gpt2", **overrides) -> tk.TokenizerSource:
    """A source seeded from the sample vocabulary for ``model``, then overridden."""
    fields = dict(_BASES.get(model, MINIMAL))
    fields["model"] = model
    fields.update(overrides)
    return tk.TokenizerSource(**fields)


def loads(document: dict) -> Tokenizer:
    return Tokenizer.from_str(json.dumps(document))


# --- each builder produces a complete document on its own --------------------


def test_bpe_document_carries_all_four_policy_blocks():
    document = tk.build_bpe(source())

    assert document["model"]["type"] == "BPE"
    assert set(document) == {
        "version", "truncation", "padding", "added_tokens",
        "normalizer", "pre_tokenizer", "post_processor", "decoder", "model",
    }
    assert document["normalizer"] == {"type": "NFC"}
    assert document["pre_tokenizer"]["type"] == "Sequence"
    assert document["decoder"]["type"] == "ByteLevel"


def test_unigram_document_preserves_every_score_in_order():
    document = tk.build_unigram(source("sentencepiece"))

    assert document["model"]["type"] == "Unigram"
    assert document["model"]["vocab"] == [
        ["<unk>", 0.0], ["<s>", -0.5], ["▁hello", -1.0], ["hello", -2.0],
        ["▁world", -3.0], ["</s>", -4.0],
    ]
    assert document["model"]["unk_id"] == 0
    assert document["pre_tokenizer"]["type"] == "Metaspace"
    assert document["post_processor"] is None


def test_wordpiece_document_uses_bert_policy():
    document = tk.build_wordpiece(source("bert"))

    assert document["model"]["type"] == "WordPiece"
    assert document["model"]["continuing_subword_prefix"] == "##"
    assert document["normalizer"] == {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True,
    }
    assert document["pre_tokenizer"] == {"type": "BertPreTokenizer"}
    assert document["decoder"] == {"type": "WordPiece", "prefix": "##", "cleanup": True}


@pytest.mark.parametrize(("builder", "fields"), [
    (tk.build_bpe, {}),
    (tk.build_unigram, {"model": "sentencepiece"}),
    (tk.build_wordpiece, {"model": "bert"}),
])
def test_every_builder_returns_a_document_tokenizers_can_load(builder, fields):
    """Structural validity is not the goal; a loadable tokenizer is."""
    loaded = loads(builder(source(**fields)))
    assert loaded.get_vocab_size() == len(source(**fields).tokens)


def test_bpe_and_wordpiece_encode_through_their_declared_policy():
    """The pipeline is only right if it actually produces the expected pieces."""
    assert loads(tk.build_bpe(source())).encode("ab").ids == [3]
    assert loads(tk.build_wordpiece(source("bert"))).encode("hellos").ids == [3, 4]


def test_unigram_encoding_round_trips_through_the_metaspace_pipeline():
    """Which piece a word becomes is decided by the scores; decoding must invert it.

    The exact ID is not asserted: ``Metaspace`` prepends ``▁`` and then splits on
    it, so ``hello`` resolves to the ``hello`` piece rather than ``▁hello`` even
    though the latter scores higher. What must hold is that the word survives.
    """
    loaded = loads(tk.build_unigram(source("sentencepiece")))

    assert loaded.decode(loaded.encode("hello").ids) == "hello"


# --- formats are isolated from one another ----------------------------------


def test_wordpiece_ignores_the_sentencepiece_charsmap():
    """A charsmap normaliser is SentencePiece's; BERT must not inherit it.

    This is the leak the shared ladder used to allow: a ``Precompiled``
    normalizer set for one family stayed set for the next branch that forgot to
    overwrite it.
    """
    document = tk.build_wordpiece(source("bert", precompiled_charsmap="Zm9vdg=="))

    assert document["normalizer"]["type"] == "BertNormalizer"


@pytest.mark.parametrize(("arch", "expected"), [
    ("llama", {"type": "NFC"}),
    ("qwen2", None),
    ("qwen3vl", None),
])
def test_only_non_qwen_architectures_get_an_nfc_normalizer(arch, expected):
    """Qwen tokenizers were trained without NFC; adding it changes the input."""
    document = tk.build_bpe(source(arch=arch))

    assert document["normalizer"] == expected


def test_t5_replaces_the_default_normalizer_with_its_whitespace_contract():
    """T5 has its own normalizer, so the NFC default must not survive inside it."""
    document = tk.build_unigram(source(
        "t5",
        tokens=["<unk>", "▁hi", "hi"],
        token_types=[tk.TOKEN_TYPE_UNKNOWN, 1, 1],
        scores=[0.0, -1.0, -2.0],
    ))

    steps = [step["type"] for step in document["normalizer"]["normalizers"]]
    assert "NFC" not in steps
    assert steps[-1] == "Replace"


@pytest.mark.parametrize(("pre", "check"), [
    (None, lambda doc: doc["pre_tokenizer"]["type"] == "Sequence"),
    ("gpt-2", lambda doc: doc["pre_tokenizer"]["type"] == "ByteLevel"),
    ("gpt2", lambda doc: doc["pre_tokenizer"]["type"] == "ByteLevel"),
    (
        "llama-bpe",
        lambda doc: doc["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == tk._LLAMA3_SPLIT,
    ),
    ("qwen2", lambda doc: doc["normalizer"] is None),
])
def test_declared_pre_tokenizer_selects_the_split(pre, check):
    assert check(tk.build_bpe(source(pre=pre)))


def test_llama3_pre_suppresses_merges_at_encode_time():
    """Llama-3 vocabularies are already fully merged; re-merging changes output."""
    document = tk.build_bpe(source(pre="llama3"))

    assert document["model"]["ignore_merges"] is True


def test_unknown_pre_warns_instead_of_guessing():
    with pytest.warns(UserWarning, match="parity is not guaranteed"):
        document = tk.build_bpe(source(pre="tokenizer-from-the-future"))

    assert document["pre_tokenizer"]["type"] == "Sequence"


# --- schema validation names the field that is wrong ------------------------


@pytest.mark.parametrize(("fields", "field"), [
    ({"tokens": []}, "tokenizer.ggml.tokens"),
    ({"tokens": ["a", "a"]}, "tokenizer.ggml.tokens"),
    ({"token_types": [1, 1]}, "tokenizer.ggml.token_type"),
    ({"scores": [float("nan")]}, "tokenizer.ggml.scores"),
    ({"special_token_ids": (99,)}, "additional_special_tokens"),
])
def test_source_validation_names_the_offending_key(fields, field):
    with pytest.raises(tk.TokenizerSchemaError) as info:
        source(**fields).validate()

    assert info.value.field == field


def test_a_valid_source_passes_validation_silently():
    for fields in ({}, {"model": "sentencepiece"}, {"model": "bert"}):
        assert source(**fields).validate() is None


@pytest.mark.parametrize(("mutate", "detail"), [
    (lambda doc: doc.pop("decoder"), "missing required key"),
    (lambda doc: doc["model"].update(type="Mystery"), "unknown model type"),
    (lambda doc: doc["model"]["vocab"].update({"zzz": 99}), "contiguous 0-based"),
    (lambda doc: doc["model"]["merges"].append("a zzz"), "not in the vocabulary"),
    (
        lambda doc: doc["model"]["merges"].append("a-b-c"),
        "not a pair of pieces",
    ),
    (
        lambda doc: doc["added_tokens"].append(dict(doc["added_tokens"][0])),
        "repeats id",
    ),
    (
        lambda doc: doc["added_tokens"].append(
            {"id": 3, "content": "<unk>", "special": True},
        ),
        "repeats content",
    ),
    (
        lambda doc: doc["added_tokens"].append({"id": 99, "content": "<x>", "special": True}),
        "outside the vocabulary",
    ),
    (lambda doc: doc.update(decoder={"no": "type"}), "string 'type'"),
    (lambda doc: doc.update(normalizer={"type": "Sequence"}), "needs a normalizers"),
])
def test_document_validation_rejects_each_structural_defect(mutate, detail):
    """Every one of these would otherwise load late, or silently."""
    document = tk.build_bpe(source())
    mutate(document)

    with pytest.raises(tk.TokenizerSchemaError, match=detail):
        tk.validate_document(document)


def test_unigram_document_validation_checks_scores_and_unk():
    document = tk.build_unigram(source("sentencepiece"))
    document["model"]["vocab"][1] = ["▁hello"]

    with pytest.raises(tk.TokenizerSchemaError, match="pair"):
        tk.validate_document(document)


def test_unigram_without_scores_is_refused_by_the_builder():
    with pytest.raises(ValueError, match="finite score for every token"):
        tk.build_unigram(source("sentencepiece", scores=[]))


def test_sentencepiece_bpe_without_an_unknown_token_is_refused():
    with pytest.raises(ValueError, match="valid unknown_token_id"):
        tk.build_bpe(source("llama", unk_id=-1, tokens=["▁a", "▁b"], scores=[-1.0, -2.0]))


def test_wordpiece_without_an_unknown_token_is_refused():
    with pytest.raises(ValueError, match="valid unknown_token_id"):
        tk.build_wordpiece(source("bert", unk_id=-1))


def test_the_dispatcher_refuses_an_unknown_format_by_name():
    with pytest.raises(ValueError, match="Unsupported GGUF tokenizer model: mystery"):
        tk.build_tokenizer_json(source("mystery"))


# --- the SentencePiece-as-BPE path ------------------------------------------


def test_sentencepiece_bpe_orders_merges_by_score_not_vocabulary_index():
    """SentencePiece prefers the highest-scored piece; BPE applies merges in order."""
    tokens = ["<unk>", "a", "b", "c", "ab", "bc"]
    token_types = [tk.TOKEN_TYPE_UNKNOWN, 1, 1, 1, 1, 1]
    scores = [0.0, -1.0, -2.0, -3.0, -4.0, -9.0]

    merges = tk.sentencepiece_bpe_merges(tokens, token_types, scores)

    assert merges[0] == ["a", "b"]
    assert merges.index(["b", "c"]) > merges.index(["a", "b"])


def test_a_gemma_sentencepiece_vocabulary_gets_the_bpe_model_block():
    """Gemma declares ``spm`` but ships merges, so the format is BPE, not Unigram."""
    sp = source(
        "spm",
        arch="gemma",
        tokens=["<unk>", "▁a", "▁b", "▁ab"],
        token_types=[tk.TOKEN_TYPE_UNKNOWN, 1, 1, 1],
        scores=[0.0, -1.0, -2.0, -3.0],
    )

    assert sp.uses_sentencepiece_bpe is True
    document = tk.build_bpe(sp)
    assert document["model"]["type"] == "BPE"
    assert document["model"]["fuse_unk"] is True
    assert document["pre_tokenizer"] is None
    assert document["normalizer"]["type"] == "Sequence"


def test_a_plain_sentencepiece_vocabulary_stays_unigram():
    sp = source("sentencepiece", arch="llama")

    assert sp.uses_sentencepiece_bpe is False
    assert tk.build_tokenizer_json(sp)["model"]["type"] == "Unigram"


# --- byte fallback, which changes the decoder -------------------------------


@pytest.mark.parametrize(("tokens", "types", "expected"), [
    (["a", "b"], [1, 1], False),
    (["a", "<0x0A>"], [1, tk.TOKEN_TYPE_BYTE], True),
    # A vocabulary that lost its type array still advertises byte pieces by name.
    (["a", "<0x0A>"], [], True),
])
def test_byte_fallback_is_detected_from_types_or_piece_names(tokens, types, expected):
    assert tk.TokenizerSource(model="gpt2", tokens=tokens, token_types=types).has_byte_fallback is expected


def test_byte_fallback_adds_a_decoder_stage():
    document = tk.build_bpe(source(tokens=["a", "<0x0A>"], token_types=[1, tk.TOKEN_TYPE_BYTE], merges=[]))

    assert document["model"]["byte_fallback"] is True
    assert document["decoder"]["decoders"][0] == {"type": "ByteFallback"}
