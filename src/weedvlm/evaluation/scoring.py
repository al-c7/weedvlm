"""
Answer-matching helpers used by notebooks/scoring.ipynb.

Species-ID open-ended answers are free text, so they're matched by
normalised name (see `open_ended_correct`), with an optional alias table
for common/scientific-name variants (docs/SPECIES-ID.md).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# -- text normalisation ------------------------------------------------


def _light(text: Any) -> str:
    """Case, whitespace and surrounding quote/full-stop insensitive form."""
    return re.sub(r"\s+", " ", str(text).strip().strip("\"'`.").strip()).casefold()


def open_ended_correct(answer: str, truth: str, aliases: Mapping[str, list[str]]) -> bool:
    """Whether a free-text species answer names the ground-truth species.

    Matches if the answer, or any part of it outside/inside parentheses or
    comma-separated ("Common purslane (Portulaca oleracea)"), equals the
    truth or one of its aliases -- or contains one as a whole phrase
    ("Eclipta prostrata" for "Eclipta"). An answer that hedges between
    options ("A or B", "A/B") only matches by an exact part, never by
    containment. Less specific answers ("Amaranthus" for "Amaranthus
    tuberculatus") don't match."""
    accepted = {_light(truth), *(_light(a) for a in aliases.get(_light(truth), []))}
    text = _light(answer)
    parts = {text, *(p.strip() for p in re.split(r"[()\[\],;]", text) if p.strip())}
    if parts & accepted:
        return True
    if " or " in text or "/" in text:
        return False
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text) for a in accepted)


def load_aliases(raw: Mapping[str, list[str]] | None) -> dict[str, list[str]]:
    """{ground-truth name: [accepted alternatives]}, keys case-folded."""
    return {_light(name): list(alts) for name, alts in (raw or {}).items()}
