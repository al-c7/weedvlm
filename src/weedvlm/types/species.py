"""
Parsing of WeedCOCO category names into structured species info.

WeedCOCO categories are named "<role>: <name>[ (<qualifier>)]", e.g.
"weed: bassia scoparia" or "crop: maize (four-leaf stage)". The
qualifier is kept as free text rather than parsed further -- it's
often a growth stage, but not always (e.g. CropAndWeed's
"weed: hedge mustard (80)").
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

Role = Literal["crop", "weed"]

_CATEGORY_NAME_RE = re.compile(
    r"^(?P<role>crop|weed):\s*(?P<name>[^(]+?)\s*(?:\((?P<qualifier>[^)]+)\))?$"
)


class Species(BaseModel):
    role: Role
    name: str
    qualifier: str | None = None

    @property
    def display_name(self) -> str:
        return self.name[:1].upper() + self.name[1:]


def parse_category_name(category_name: str) -> Species:
    match = _CATEGORY_NAME_RE.match(category_name.strip())
    if not match:
        raise ValueError(
            f"Category name does not match the WeedCOCO 'role: name' convention: {category_name!r}"
        )

    return Species(
        role=match["role"],
        name=match["name"].strip(),
        qualifier=match["qualifier"],
    )
