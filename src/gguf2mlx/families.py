"""Architecture families that change how a GGUF is interpreted.

These sets live in their own module because two layers need them: the converter
(which maps tensor names) and the tokenizer builders (which decide normalizer
and pre-tokenizer policy from the architecture). Importing the converter from
the tokenizer builders would be a cycle, and copying the lists into both places
would let them drift -- the failure mode being a Qwen checkpoint that silently
acquires an NFC normalizer and rewrites text before it reaches the vocabulary.

The shell of this module is deliberately empty of logic: constants only, so it
can be imported from anywhere without ordering constraints.
"""

from __future__ import annotations

from .qwen_vl_text import QWEN_VL_TEXT_ARCHES

QWEN35_ARCHES = {
    "qwen35", "qwen3_5", "qwen3.5", "qwen36", "qwen3_6", "qwen3.6",
    "qwen38", "qwen3_8", "qwen3.8",
}

# Qwen tokenizers have no normalizer: BPE runs straight on the raw text after a
# GPT-2-style regex split. Every Qwen architecture must be in this set, because
# the default for anything else is NFC -- which silently rewrites some inputs
# before they reach the vocabulary.
QWEN_FAMILY_ARCHES = {
    "qwen2",
    "qwen3",
    "qwen2moe",
    "qwen3moe",
    *QWEN35_ARCHES,
    *QWEN_VL_TEXT_ARCHES,
}

# Values of ``tokenizer.ggml.pre`` that select that same Qwen pre-tokenizer.
# llama.cpp names the hybrid family ``qwen35`` while Qwen2/2.5/3 report
# ``qwen2``; an unlisted value falls through to "legacy BPE splitting" with a
# warning, so the alias spellings are listed rather than inferred.
QWEN_PRE_TOKENIZERS = {
    "qwen2",
    "qwen2.5-coder",
    "qwen3",
    "qwen35",
    "qwen3.5",
    "qwen3_5",
    "qwen36",
    "qwen3.6",
    "qwen3_6",
    "qwen38",
    "qwen3.8",
    "qwen3_8",
}

GEMMA_ARCHES = {"gemma", "gemma2", "gemma3", "gemma4"}
