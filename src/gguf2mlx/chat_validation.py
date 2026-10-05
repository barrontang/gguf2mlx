"""Render chat templates against the Phase 1 conversation-shape contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONVERSATION_SHAPES: dict[str, list[dict[str, Any]]] = {
    "single_turn": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ],
    "multi_turn": [
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "Two"},
        {"role": "user", "content": "Three"},
        {"role": "assistant", "content": "Four"},
    ],
    "system_prompt": [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ],
    "tool_call": [
        {"role": "user", "content": "Check weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "name": "weather", "content": "sunny"},
    ],
    "reasoning": [
        {"role": "user", "content": "Solve 1+1"},
        {"role": "assistant", "content": "<think>addition</think>2"},
    ],
}


def validate_chat_template(
    template_source: str, expected_markers: list[str] | None = None
) -> dict[str, str]:
    """Render canonical conversations and reject unresolved template syntax."""
    try:
        from jinja2 import Environment, StrictUndefined, TemplateError
    except ImportError as error:
        raise RuntimeError("Chat validation requires jinja2>=3.1.6") from error

    expected_markers = expected_markers or []
    environment = Environment(undefined=StrictUndefined, autoescape=False)
    try:
        template = environment.from_string(template_source)
        rendered = {
            name: template.render(
                messages=messages,
                bos_token="<s>",
                eos_token="</s>",
                add_generation_prompt=True,
                tools=[],
            )
            for name, messages in CONVERSATION_SHAPES.items()
        }
    except TemplateError as error:
        raise ValueError(f"Chat template rendering failed: {error}") from error

    for shape, text in rendered.items():
        if "{{" in text or "{%" in text:
            raise ValueError(f"{shape}: unresolved Jinja syntax in rendered output: {text!r}")
        missing = [marker for marker in expected_markers if marker not in text]
        if missing:
            raise ValueError(
                f"{shape}: missing expected markers {missing}; rendered output: {text!r}"
            )
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expect-marker", action="append", default=[])
    args = parser.parse_args(argv)

    template_path = args.model_dir / "chat_template.jinja"
    if template_path.exists():
        template_source = template_path.read_text(encoding="utf-8")
    else:
        config_path = args.model_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        template_source = config.get("chat_template")
        if not template_source:
            raise SystemExit("No chat template found in converted model")

    rendered = validate_chat_template(template_source, args.expect_marker)
    print(json.dumps(rendered, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
