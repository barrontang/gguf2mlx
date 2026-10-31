"""Rebuild ``tokenizer.json`` from GGUF tokenizer metadata, one builder per format.

GGUF's ``tokenizer.ggml.*`` keys describe a vocabulary and, depending on the
format, how pieces combine: ``merges`` for BPE, ``scores`` for Unigram. They do
*not* describe the rest of the pipeline. The normalizer, pre-tokenizer, decoder
and post-processor are policy that HuggingFace's ``tokenizers`` applies while
encoding, and that policy has to be inferred from the architecture, the
declared model, and ``tokenizer.ggml.pre``.

Inference is where this goes wrong quietly. A wrong normalizer or pre-tokenizer
still loads and still tokenizes; it simply disagrees with the model it came
from, and the disagreement surfaces much later as degraded generations. Keeping
that policy in a single ``if/elif`` ladder shared by every format made the risk
hard to see: a branch added for one family could change the meaning of a local
another family depended on, and the ordering of the branches became part of the
contract by accident.

So the formats are separated here:

* :func:`build_bpe` -- GPT-2 byte-level BPE, plus the SentencePiece BPE variant
  (``tokenizer.ggml.model = llama``) whose vocabulary is a SentencePiece corpus
  expressed as merges.
* :func:`build_unigram` -- SentencePiece Unigram, where ``scores`` *are* the
  model and must survive exactly.
* :func:`build_wordpiece` -- BERT's ``##`` continuation vocabulary.

Each builder returns a complete document and mutates nothing outside its own
locals. :func:`validate_document` then checks the result against the structural
contract the three share, so a builder that forgets a block fails at conversion
time instead of producing a file that ``tokenizers`` rejects later.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .families import QWEN_FAMILY_ARCHES, QWEN_PRE_TOKENIZERS

#: ``tokenizer.ggml.token_type`` values, as defined by llama.cpp.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4
TOKEN_TYPE_UNUSED = 5
TOKEN_TYPE_BYTE = 6

#: Token types that become ``added_tokens`` entries rather than model pieces.
_ADDED_TOKEN_TYPES = frozenset({
    TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED,
})
#: Token types that are emitted as ``special`` in ``added_tokens``.
_SPECIAL_TOKEN_TYPES = frozenset({TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL})

#: ``tokenizer.ggml.model`` values that name a format this module rebuilds.
SUPPORTED_MODELS = ("bpe", "gpt2", "llama", "spm", "sentencepiece", "unigram", "wordpiece")

#: The GPT-2 split, as llama.cpp uses it for every non-``gpt-2`` BPE.
_LEGACY_SPLIT = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
#: The Llama-3 split, which differs only in how digit runs are capped.
_LLAMA3_SPLIT = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

#: ``tokenizer.ggml.pre`` values that mean "GPT-2 byte-level split".
_GPT2_PRES = frozenset({"gpt-2", "gpt2"})
#: ``tokenizer.ggml.pre`` values that mean "Llama-3 split, merges bypassed".
_LLAMA3_PRES = frozenset({"llama-bpe", "llama3", "llama-v3"})

#: Keys every ``tokenizer.json`` document this module emits must carry.
_DOCUMENT_KEYS = (
    "version", "truncation", "padding", "added_tokens",
    "normalizer", "pre_tokenizer", "post_processor", "decoder", "model",
)
#: Pipeline slots whose value is either ``None`` or a typed block.
_PIPELINE_KEYS = ("normalizer", "pre_tokenizer", "post_processor", "decoder")
#: ``model.type`` values this module produces.
_MODEL_TYPES = ("BPE", "Unigram", "WordPiece")


class TokenizerSchemaError(ValueError):
    """A GGUF tokenizer field, or the document built from it, is inconsistent.

    The offending field is carried separately from the message so callers can
    react to *which* key is wrong -- a missing ``tokenizer.ggml.scores`` is a
    truncated download, while a bad ``tokenizer.ggml.token_type`` is a malformed
    export -- without matching on prose.
    """

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.detail = detail


@dataclass
class TokenizerSource:
    """The tokenizer fields of one GGUF, and the policy flags from its metadata.

    ``model`` is kept exactly as the file declared it because a few decisions
    (whether ``t5``'s whitespace rewriting applies, whether ``bert``'s ``##``
    conversion has already been done) depend on the original spelling, not on
    the normalised format name.

    Role IDs are stored resolved: ``-1`` means "this model has no such token".
    Which of them were *declared* versus inferred is the extractor's business,
    because that is where the raw fields are still available.
    """

    model: str
    tokens: list[str]
    token_types: list[int] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bos_id: int = -1
    eos_id: int = -1
    pad_id: int = -1
    unk_id: int = -1
    sep_id: int | None = None
    mask_id: int | None = None
    pre: str | None = None
    arch: str = "llama"
    add_space_prefix: bool = True
    do_lower_case: bool | None = None
    remove_extra_whitespaces: bool | None = None
    strip_accents: bool | None = None
    precompiled_charsmap: str | None = None
    special_token_ids: tuple[int, ...] = ()

    # -- derived views -----------------------------------------------------

    @property
    def normalized_model(self) -> str:
        """The format name, with GGUF's aliases folded onto one spelling."""
        return {"bert": "wordpiece", "t5": "unigram"}.get(
            self.model.lower(), self.model.lower(),
        )

    @property
    def vocab(self) -> dict[str, int]:
        """Token ID by piece, for the formats whose vocabulary is a mapping."""
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def has_byte_fallback(self) -> bool:
        """Whether the vocabulary carries ``<0xNN>`` byte pieces.

        Byte-fallback tokens are the only way to encode a character the
        vocabulary does not contain, so their presence changes the decoder:
        without it, unknown bytes are dropped instead of emitted.
        """
        return any(
            (index < len(self.token_types) and self.token_types[index] == TOKEN_TYPE_BYTE)
            or (token.startswith("<0x") and token.endswith(">"))
            for index, token in enumerate(self.tokens)
        )

    @property
    def uses_sentencepiece_bpe(self) -> bool:
        """Whether the vocabulary is SentencePiece expressed as BPE merges.

        ``llama`` always is. Gemma is the awkward case: it declares ``spm`` or
        ``unigram`` but ships a merge list, so it takes the Unigram *model block*
        with SentencePiece's pre-tokenizer.
        """
        normalized = self.normalized_model
        return normalized == "llama" or (
            self.arch == "gemma" and normalized in ("spm", "sentencepiece", "unigram")
        )

    def _role_ids(self) -> set[int]:
        roles = {self.bos_id, self.eos_id, self.pad_id, self.unk_id}
        return {token_id for token_id in roles | set(self.special_token_ids) if token_id >= 0}

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Reject metadata that cannot describe a tokenizer at all.

        Only vocabulary-intrinsic facts are checked here. Role wiring (which KV
        names the BOS token) is validated by the caller, which is the layer that
        can still tell a declared ``-1`` from an absent field.
        """
        if not self.tokens:
            raise TokenizerSchemaError(
                "tokenizer.ggml.tokens",
                "no tokenizer.ggml.tokens present; cannot reconstruct a tokenizer",
            )
        if len(set(self.tokens)) != len(self.tokens):
            raise TokenizerSchemaError(
                "tokenizer.ggml.tokens", "vocabulary contains duplicate tokens",
            )
        if self.token_types and len(self.token_types) != len(self.tokens):
            raise TokenizerSchemaError(
                "tokenizer.ggml.token_type", "requires a type for every token",
            )
        if self.scores and (
            len(self.scores) != len(self.tokens) or not np.isfinite(self.scores).all()
        ):
            raise TokenizerSchemaError(
                "tokenizer.ggml.scores", "requires a finite score for every token",
            )
        for token_id in self.special_token_ids:
            if not 0 <= token_id < len(self.tokens):
                raise TokenizerSchemaError(
                    "additional_special_tokens", "token ID is outside the vocabulary",
                )

    # -- construction ------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build the ``tokenizer.json`` document for this source."""
        return build_tokenizer_json(self)


def _added_tokens(source: TokenizerSource) -> list[dict[str, Any]]:
    """The ``added_tokens`` block: every piece that is not a normal model piece.

    Control and user-defined tokens have to be listed here even though they also
    appear in the model vocabulary, because that is how ``tokenizers`` knows not
    to split them during pre-tokenization. BOS/EOS/PAD/UNK are promoted too, and
    marked ``special``: they are never part of the text being encoded.
    """
    roles = source._role_ids()
    added: list[dict[str, Any]] = []
    for index, token in enumerate(source.tokens):
        token_type = (
            source.token_types[index] if index < len(source.token_types) else TOKEN_TYPE_NORMAL
        )
        is_role = index in roles
        if token_type not in _ADDED_TOKEN_TYPES and not is_role:
            continue
        added.append(
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": token_type in _SPECIAL_TOKEN_TYPES or is_role,
            }
        )
    return added


def _bytelevel(add_prefix_space: bool = False, *, use_regex: bool = False) -> dict[str, Any]:
    """A ``ByteLevel`` block, which pre-tokenizer, post-processor and decoder share."""
    return {
        "type": "ByteLevel",
        "add_prefix_space": add_prefix_space,
        "trim_offsets": False,
        "use_regex": use_regex,
    }


@dataclass
class _Pipeline:
    """The four policy blocks that surround a model block."""

    normalizer: dict[str, Any] | None
    pre_tokenizer: dict[str, Any] | None
    post_processor: dict[str, Any] | None
    decoder: dict[str, Any] | None


def _base_normalizer(source: TokenizerSource) -> dict[str, Any] | None:
    """NFC, except where normalising would change what the model expects.

    Qwen and T5 tokenizers were trained on text that was never NFC-normalised,
    so applying it would alter inputs relative to training. A precompiled
    SentencePiece charsmap, when present, *is* the normalizer and replaces this
    entirely.
    """
    if source.precompiled_charsmap:
        return {"type": "Precompiled", "precompiled_charsmap": source.precompiled_charsmap}
    if source.arch in QWEN_FAMILY_ARCHES or source.model == "t5":
        return None
    return {"type": "NFC"}


def _bytelevel_pipeline(source: TokenizerSource) -> _Pipeline:
    """Policy for GPT-2 style byte-level BPE.

    Three ``tokenizer.ggml.pre`` families matter: ``gpt-2`` drops the extra split
    and lets ``ByteLevel`` do it, ``llama-3`` caps digit runs at three, and the
    Qwen aliases keep the legacy split but must not normalise.
    """
    normalizer = _base_normalizer(source)
    pre_tokenizer: dict[str, Any] | None = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": _LEGACY_SPLIT},
                "behavior": "Isolated",
                "invert": False,
            },
            _bytelevel(),
        ],
    }
    decoder: dict[str, Any] | None = _bytelevel()
    if source.pre:
        if source.pre in _GPT2_PRES:
            normalizer = None
            pre_tokenizer = _bytelevel(use_regex=True)
        elif source.pre in _LLAMA3_PRES:
            normalizer = None
            # ``pre_tokenizer`` is a Sequence of exactly [Split, ByteLevel] here:
            # the branches above are the only ones that replace it.
            assert pre_tokenizer is not None
            pre_tokenizer["pretokenizers"][0]["pattern"]["Regex"] = _LLAMA3_SPLIT
        elif source.pre in QWEN_PRE_TOKENIZERS | {"deepseek-r1-qwen", "kormo", "f2llmv2"}:
            normalizer = None
        else:
            warnings.warn(
                f"Unsupported tokenizer.ggml.pre={source.pre!r}; "
                "using legacy BPE splitting, exact tokenization parity is not guaranteed",
            )
    if source.has_byte_fallback:
        decoder = {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, _bytelevel()]}
    return _Pipeline(
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        post_processor=_bytelevel(),
        decoder=decoder,
    )


def _sentencepiece_pipeline(
    source: TokenizerSource, *, sentencepiece_bpe: bool,
) -> _Pipeline:
    """Policy shared by Unigram and the SentencePiece-BPE variant.

    SentencePiece marks word boundaries with ``▁`` rather than with byte-level
    escapes, so the pipeline is a ``Metaspace`` pre-tokenizer and a decoder that
    reverses the marker, expands byte fallbacks, and trims the leading space the
    prefix introduced.
    """
    normalizer = _base_normalizer(source)
    decoder: dict[str, Any] = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    if not source.add_space_prefix:
        # Without a prefix there is no leading space to strip, and stripping
        # anyway would eat the first character of the first token.
        decoder["decoders"].pop()
    if source.remove_extra_whitespaces:
        normalizer = {
            "type": "Sequence",
            "normalizers": [
                *([normalizer] if normalizer else []),
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                {"type": "Strip", "strip_left": True, "strip_right": True},
            ],
        }
    pre_tokenizer: dict[str, Any] | None = {
        "type": "Metaspace",
        "replacement": "▁",
        "prepend_scheme": "always" if source.add_space_prefix else "never",
        "split": True,
    }
    if source.model == "t5":
        normalizer, pre_tokenizer = _t5_normalizer(source)
    if sentencepiece_bpe:
        # The SentencePiece corpus already contains the word-boundary markers, so
        # the normalizer is *only* the whitespace rewriting plus the marker swap.
        # Anything applied before that would change text relative to training,
        # which is why the NFC default is replaced rather than extended.
        normalizers: list[dict[str, Any]] = []
        if source.remove_extra_whitespaces:
            normalizers.extend([
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                {"type": "Strip", "strip_left": True, "strip_right": True},
            ])
        if source.add_space_prefix:
            normalizers.append({"type": "Prepend", "prepend": "▁"})
        normalizers.append({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})
        normalizer = {"type": "Sequence", "normalizers": normalizers}
        pre_tokenizer = None
    return _Pipeline(
        normalizer=normalizer, pre_tokenizer=pre_tokenizer, post_processor=None,
        decoder=decoder,
    )


def _t5_normalizer(
    source: TokenizerSource,
) -> tuple[dict[str, Any], None]:
    """T5's whitespace contract: collapse, strip, then mark a leading boundary.

    T5 was trained with a ``▁`` sentinel in front of text that does not already
    start with a space, and with runs of spaces collapsed to one. Reproducing
    that here is what keeps ``encode`` identical to ``SentencePieceProcessor``.
    """
    normalizers: list[dict[str, Any]] = []
    if source.precompiled_charsmap:
        normalizers = [{
            "type": "Precompiled", "precompiled_charsmap": source.precompiled_charsmap,
        }]
    if source.remove_extra_whitespaces:
        normalizers.extend([
            {"type": "Replace", "pattern": {"Regex": " +"}, "content": " "},
            {"type": "Replace", "pattern": {"Regex": "^ | $"}, "content": ""},
            {"type": "Replace", "pattern": {"Regex": "^(?=[\\s\\S])"}, "content": "▁"},
        ])
    elif source.add_space_prefix:
        normalizers.append({
            "type": "Replace", "pattern": {"Regex": "^(?=[\\s\\S]*[^ ])"}, "content": "▁",
        })
    normalizers.append({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})
    return {"type": "Sequence", "normalizers": normalizers}, None


def _wordpiece_pipeline(source: TokenizerSource) -> _Pipeline:
    """BERT's policy: clean text, split on punctuation, decode ``##`` joins.

    ``strip_accents`` and ``lowercase`` come from the GGUF when it declares them,
    because cased BERT variants exist and lowercasing them changes the vocabulary
    the model was trained against.
    """
    return _Pipeline(
        normalizer={
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": source.strip_accents,
            "lowercase": True if source.do_lower_case is None else source.do_lower_case,
        },
        pre_tokenizer={"type": "BertPreTokenizer"},
        post_processor=None,
        decoder={"type": "WordPiece", "prefix": "##", "cleanup": True},
    )


def sentencepiece_bpe_merges(
    tokens: list[str], token_types: list[int], scores: list[float],
) -> list[list[str]]:
    """Express a SentencePiece vocabulary as ordered BPE merge pairs.

    SentencePiece has scores, not merges, so the merge list has to be derived:
    every piece is splittable at any interior position where both halves also
    exist, and each such split is a candidate pair. Ordering is what makes the
    result correct -- BPE applies the earliest applicable merge first, so the
    candidate list is sorted by *score*, which is SentencePiece's own preference
    order. Sorting by vocabulary index instead would reproduce the piece list but
    not the segmentation.
    """
    if len(scores) != len(tokens) or not np.isfinite(scores).all():
        raise ValueError("SentencePiece BPE requires a finite score for every token")
    if token_types and len(token_types) != len(tokens):
        raise ValueError("SentencePiece BPE requires a type for every token")
    vocab = {
        token: index for index, token in enumerate(tokens)
        if not token_types or token_types[index] == TOKEN_TYPE_NORMAL
    }
    candidates: list[tuple[str, str, float]] = []
    for token, token_id in vocab.items():
        splits = [
            (token[:index], token[index:]) for index in range(1, len(token))
            if token[:index] in vocab and token[index:] in vocab
        ]
        splits.sort(key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        candidates.extend((left, right, scores[token_id]) for left, right in splits)
    candidates.sort(key=lambda item: (item[2], len(item[0]), len(item[1])), reverse=True)
    return [[left, right] for left, right, _ in candidates]


def _document(
    model_block: dict[str, Any], pipeline: _Pipeline, added_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": pipeline.normalizer,
        "pre_tokenizer": pipeline.pre_tokenizer,
        "post_processor": pipeline.post_processor,
        "decoder": pipeline.decoder,
        "model": model_block,
    }


def build_bpe(source: TokenizerSource) -> dict[str, Any]:
    """GPT-2 byte-level BPE, and the SentencePiece vocabulary exported as BPE."""
    added_tokens = _added_tokens(source)
    vocab = source.vocab
    if source.uses_sentencepiece_bpe:
        if not 0 <= source.unk_id < len(source.tokens):
            raise ValueError("SentencePiece BPE reconstruction requires a valid unknown_token_id")
        model_block: dict[str, Any] = {
            "type": "BPE",
            "vocab": vocab,
            "merges": sentencepiece_bpe_merges(
                source.tokens, source.token_types, source.scores,
            ),
            "unk_token": source.tokens[source.unk_id],
            "byte_fallback": source.has_byte_fallback,
            "fuse_unk": True,
        }
        return _document(model_block, _sentencepiece_pipeline(source, sentencepiece_bpe=True), added_tokens)

    for merge in source.merges:
        pieces = merge.split(" ")
        if len(pieces) != 2 or any(piece not in vocab for piece in pieces):
            raise ValueError(f"Invalid GGUF BPE merge: {merge!r}")
        if "".join(pieces) not in vocab:
            raise ValueError(f"GGUF BPE merge result is missing from the vocabulary: {merge!r}")
    model_block = {
        "type": "BPE",
        "dropout": None,
        "unk_token": (
            source.tokens[source.unk_id] if 0 <= source.unk_id < len(source.tokens) else None
        ),
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": source.has_byte_fallback,
        "vocab": vocab,
        "merges": list(source.merges) if source.merges else [],
    }
    if source.pre in _LLAMA3_PRES:
        # Llama-3 vocabularies already contain every merged piece, so applying
        # merges during encoding would re-merge text the model saw pre-split.
        model_block["ignore_merges"] = True
    return _document(model_block, _bytelevel_pipeline(source), added_tokens)


def build_unigram(source: TokenizerSource) -> dict[str, Any]:
    """SentencePiece Unigram, where the scores are the model."""
    if not 0 <= source.unk_id < len(source.tokens):
        raise ValueError("Unigram reconstruction requires a valid unknown_token_id")
    if len(source.scores) != len(source.tokens) or not np.isfinite(source.scores).all():
        raise ValueError("Unigram reconstruction requires a finite score for every token")
    model_block = {
        "type": "Unigram",
        "unk_id": source.unk_id,
        "vocab": [
            [token, source.scores[index]] for index, token in enumerate(source.tokens)
        ],
        "byte_fallback": source.has_byte_fallback,
    }
    return _document(
        model_block, _sentencepiece_pipeline(source, sentencepiece_bpe=False),
        _added_tokens(source),
    )


def build_wordpiece(source: TokenizerSource) -> dict[str, Any]:
    """BERT's WordPiece vocabulary, with ``##`` marking non-initial pieces."""
    if not 0 <= source.unk_id < len(source.tokens):
        raise ValueError("WordPiece reconstruction requires a valid unknown_token_id")
    model_block = {
        "type": "WordPiece",
        "unk_token": (
            source.tokens[source.unk_id]
            if 0 <= source.unk_id < len(source.tokens) else "[UNK]"
        ),
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": source.vocab,
    }
    return _document(model_block, _wordpiece_pipeline(source), _added_tokens(source))


def build_tokenizer_json(source: TokenizerSource) -> dict[str, Any]:
    """Dispatch to the builder named by the source, then validate the result."""
    normalized = source.normalized_model
    if normalized in ("bpe", "gpt2", "llama") or source.uses_sentencepiece_bpe:
        document = build_bpe(source)
    elif normalized in ("spm", "sentencepiece", "unigram"):
        document = build_unigram(source)
    elif normalized == "wordpiece":
        document = build_wordpiece(source)
    else:
        raise ValueError(f"Unsupported GGUF tokenizer model: {source.model}")
    return validate_document(document)


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Check a document against the structure ``tokenizers`` requires.

    This is deliberately about shape, not policy: it cannot tell whether the
    normalizer is the *right* one, but it does catch the failures that are
    otherwise silent until load time -- a dropped block, a vocabulary whose IDs
    are not a contiguous 0-based range, a merge naming a piece that does not
    exist, or two added tokens claiming the same ID.
    """
    for key in _DOCUMENT_KEYS:
        if key not in document:
            raise TokenizerSchemaError("tokenizer.json", f"missing required key {key!r}")

    model = document["model"]
    if not isinstance(model, dict):
        raise TokenizerSchemaError("tokenizer.json:model", "must be an object")
    kind = model.get("type")
    if kind not in _MODEL_TYPES:
        raise TokenizerSchemaError(
            "tokenizer.json:model.type", f"unknown model type {kind!r}",
        )

    vocab = model.get("vocab")
    if kind == "Unigram":
        if not isinstance(vocab, list) or not vocab:
            raise TokenizerSchemaError(
                "tokenizer.json:model.vocab", "Unigram vocabulary must be a non-empty list",
            )
        for index, entry in enumerate(vocab):
            if (
                not isinstance(entry, list) or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], (int, float))
            ):
                raise TokenizerSchemaError(
                    "tokenizer.json:model.vocab",
                    f"entry {index} is not a [piece, score] pair",
                )
        size = len(vocab)
        unk_id = model.get("unk_id")
        if not isinstance(unk_id, int) or not 0 <= unk_id < size:
            raise TokenizerSchemaError(
                "tokenizer.json:model.unk_id", "Unigram requires a vocabulary index",
            )
    else:
        if not isinstance(vocab, dict) or not vocab:
            raise TokenizerSchemaError(
                "tokenizer.json:model.vocab", "vocabulary must be a non-empty mapping",
            )
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise TokenizerSchemaError(
                "tokenizer.json:model.vocab",
                "token IDs are not a contiguous 0-based range",
            )
        size = len(vocab)

    if kind == "BPE":
        merges = model.get("merges")
        if not isinstance(merges, list):
            raise TokenizerSchemaError(
                "tokenizer.json:model.merges", "BPE requires a merge list",
            )
        assert isinstance(vocab, dict)
        for index, merge in enumerate(merges):
            # ``tokenizers`` accepts both encodings: the historical ``"left right"``
            # string and the newer two-element list. GGUF carries strings, so they
            # are passed through unchanged rather than rewritten.
            if isinstance(merge, str):
                pieces = merge.split(" ")
                if len(pieces) != 2:
                    raise TokenizerSchemaError(
                        "tokenizer.json:model.merges",
                        f"merge {index} is not a pair of pieces",
                    )
            elif (
                isinstance(merge, list) and len(merge) == 2
                and all(isinstance(piece, str) for piece in merge)
            ):
                pieces = merge
            else:
                raise TokenizerSchemaError(
                    "tokenizer.json:model.merges", f"merge {index} is not a pair of strings",
                )
            if any(piece not in vocab for piece in pieces):
                raise TokenizerSchemaError(
                    "tokenizer.json:model.merges",
                    f"merge {index} joins a piece that is not in the vocabulary",
                )

    added = document["added_tokens"]
    if not isinstance(added, list):
        raise TokenizerSchemaError("tokenizer.json:added_tokens", "must be a list")
    ids_seen: set[int] = set()
    content_seen: set[str] = set()
    for index, entry in enumerate(added):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), int)
            or not isinstance(entry.get("content"), str)
        ):
            raise TokenizerSchemaError(
                "tokenizer.json:added_tokens", f"entry {index} needs an int id and a str content",
            )
        if not 0 <= entry["id"] < size:
            raise TokenizerSchemaError(
                "tokenizer.json:added_tokens",
                f"entry {index} has id {entry['id']} outside the vocabulary",
            )
        if entry["id"] in ids_seen:
            raise TokenizerSchemaError(
                "tokenizer.json:added_tokens", f"entry {index} repeats id {entry['id']}",
            )
        if entry["content"] in content_seen:
            raise TokenizerSchemaError(
                "tokenizer.json:added_tokens",
                f"entry {index} repeats content {entry['content']!r}",
            )
        ids_seen.add(entry["id"])
        content_seen.add(entry["content"])

    for slot in _PIPELINE_KEYS:
        block = document[slot]
        if block is None:
            continue
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            raise TokenizerSchemaError(
                f"tokenizer.json:{slot}", "must be null or an object with a string 'type'",
            )
        if block["type"] == "Sequence" and not (
            isinstance(block.get("normalizers"), list)
            or isinstance(block.get("pretokenizers"), list)
            or isinstance(block.get("decoders"), list)
        ):
            raise TokenizerSchemaError(
                f"tokenizer.json:{slot}", "a Sequence needs a normalizers/pretokenizers/decoders list",
            )
    return document
