"""
Pulls the JSON answer object out of a model's reply. Models asked for
"only JSON" still often wrap it in a code fence or a sentence, so this
takes the first parseable JSON object found anywhere in the text.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)

_decoder = json.JSONDecoder()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if fence := _FENCE_RE.match(text):
        text = fence.group(1).strip()

    # Fast path, then fall back to scanning for the first '{' that starts
    # a complete object (skips any prose before it and anything after).
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    position = text.find("{")
    while position != -1:
        try:
            value, _ = _decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position = text.find("{", position + 1)
            continue
        if isinstance(value, dict):
            return value
        position = text.find("{", position + 1)
    return None


def check_fields(parsed: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Human-readable problems with a parsed reply: fields that are
    missing or empty. `answer` being one of them makes the reply
    unusable (the caller retries); the others are only recorded as
    warnings on the stored response."""
    problems = []
    for field in required:
        value = parsed.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"missing or empty {field!r}")
    return problems
