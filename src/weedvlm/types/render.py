"""
A pending box-annotated render: everything needed to draw one image,
without having drawn it yet. Generators attach one of these to each
box-annotated question (alongside the image_path it'll be written to)
instead of rendering straight away, so only the questions that survive
selection ever get rendered -- see `weedvlm.pipeline.render.render_all`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderJob:
    source_path: Path
    out_path: Path
    line_width: int
    # (label, box) pairs. Labels are drawn (numbered boxes, in their
    # per-label colour) only when numbered is True; otherwise every box
    # is a plain, unlabelled outline and the labels are ignored.
    labelled_boxes: tuple[tuple[int, Box], ...]
    numbered: bool
