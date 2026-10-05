from pathlib import Path

import pytest

from gguf2mlx.chat_validation import CONVERSATION_SHAPES, validate_chat_template

FIXTURE = Path(__file__).parent / "fixtures" / "chat_templates" / "im_start.jinja"


def test_chat_template_renders_all_phase1_shapes():
    rendered = validate_chat_template(
        FIXTURE.read_text(encoding="utf-8"), ["<|im_start|>", "<|im_end|>"]
    )
    assert set(rendered) == set(CONVERSATION_SHAPES)
    assert all("{{" not in text and "{%" not in text for text in rendered.values())


def test_chat_template_failure_includes_rendered_comparison():
    with pytest.raises(ValueError, match="missing expected markers.*rendered output"):
        validate_chat_template("{{ messages[0].content }}", ["<|im_start|>"])
