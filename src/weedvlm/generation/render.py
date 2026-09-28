"""
Renders bounding boxes onto copies of source images, for the various
box-annotated question formats.

Rendering is split in two so that only selected questions ever get
drawn: generators call a `plan_*` function, which works out the output
path and returns a `RenderJob` (attached to the question as its
`render`) without touching the image, and `render_all` then draws the
jobs behind whatever questions survived selection, in parallel.

- `plan_species_boxes` -- a single species' boxes, undrawn otherwise
  (species ID), so a question can point at "this plant" without
  needing full localisation.
- `plan_boxes` -- every given box, undrawn labels (density
  estimation's annotated condition).
- `plan_numbered_boxes` -- each box outlined and numbered in a
  colour that's unique per label (never per species name -- the label
  is the only thing shown), so a question can refer to "box 1" /
  "box 2" and a reviewer can tell same-labelled boxes apart from
  different-labelled ones at a glance (fine-grained ID, species
  localisation).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont

from weedvlm.pipeline.parallel import parallel_map
from weedvlm.types.dataset import ReviewedImage
from weedvlm.types.questions import QuestionBase
from weedvlm.types.render import Box, RenderJob

BOX_COLOUR = (255, 0, 0)

# One colour per label, cycled if there are more labels than colours.
# Chosen to stay distinguishable from soil/foliage backgrounds and from
# each other. Mirrored exactly (same order) in
# apps/question-preview/static/app.js's LABEL_PALETTE, so a box's colour
# in a rendered image always matches its choice's swatch in the
# previewer -- keep the two in sync if this changes.
LABEL_PALETTE_HEX = [
    "#e6194b",  # red
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#ffe119",  # yellow
    "#000075",  # navy
    "#fabed4",  # pink
    "#469990",  # teal
]


def _hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


LABEL_PALETTE = [_hex_to_rgb(h) for h in LABEL_PALETTE_HEX]


def _label_colours(label: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(box/background colour, contrasting text colour) for a given label."""
    box_colour = LABEL_PALETTE[(label - 1) % len(LABEL_PALETTE)]
    r, g, b = box_colour
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_colour = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    return box_colour, text_colour


def _line_width(image: ReviewedImage) -> int:
    return max(3, min(image.width, image.height) // 200)


def _out_path(image: ReviewedImage, out_dir: Path, slug: str) -> Path:
    return out_dir / f"{image.dataset_name}_{image.image_id}_{slug}{image.image_path.suffix}"


def plan_boxes(
    image: ReviewedImage,
    boxes: list[Box],
    out_dir: Path,
    *,
    slug: str,
) -> RenderJob:
    return RenderJob(
        source_path=image.image_path,
        out_path=_out_path(image, out_dir, slug),
        line_width=_line_width(image),
        labelled_boxes=tuple((0, box) for box in boxes),
        numbered=False,
    )


def plan_species_boxes(
    image: ReviewedImage,
    species_name: str,
    boxes: list[Box],
    out_dir: Path,
) -> RenderJob:
    return plan_boxes(image, boxes, out_dir, slug=species_name.replace(" ", "_"))


def plan_numbered_boxes(
    image: ReviewedImage,
    labelled_boxes: list[tuple[int, Box]],
    out_dir: Path,
    *,
    slug: str,
) -> RenderJob:
    return RenderJob(
        source_path=image.image_path,
        out_path=_out_path(image, out_dir, slug),
        line_width=_line_width(image),
        labelled_boxes=tuple(labelled_boxes),
        numbered=True,
    )


def _tag_position(
    box: Box, tag_w: float, tag_h: float, image_w: int, image_h: int
) -> tuple[float, float]:
    """Top-left corner for a box's number tag, placed outside the box so
    it never covers a small plant: sitting on the box's top edge, or on
    its bottom edge if there's no room above, or (only when the box
    spans nearly the whole image height) just inside its top edge.
    Aligned with the box's left edge, shifted in to stay on-image."""
    x, y, w, h = box
    tag_x = min(max(0, x), max(0, image_w - tag_w))
    if y - tag_h >= 0:
        return tag_x, y - tag_h
    if y + h + tag_h <= image_h:
        return tag_x, y + h
    return tag_x, max(0, y)


def render(job: RenderJob) -> Path:
    """Draws one job and writes it to job.out_path (whose directory must
    already exist)."""
    with PILImage.open(job.source_path) as source:
        rendered = source.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = job.line_width

    if not job.numbered:
        for _, (x, y, w, h) in job.labelled_boxes:
            draw.rectangle([x, y, x + w, y + h], outline=BOX_COLOUR, width=line_width)
        rendered.save(job.out_path)
        return job.out_path

    font = ImageFont.load_default(size=max(12, line_width * 6))
    # Outlines first, so no box's outline is drawn over another box's tag.
    for label, (x, y, w, h) in job.labelled_boxes:
        box_colour, _ = _label_colours(label)
        draw.rectangle([x, y, x + w, y + h], outline=box_colour, width=line_width)

    for label, box in job.labelled_boxes:
        box_colour, text_colour = _label_colours(label)
        text = str(label)
        pad = max(2, line_width // 2)
        text_x0, text_y0, text_x1, text_y1 = draw.textbbox((0, 0), text, font=font)
        tag_w = text_x1 - text_x0 + 2 * pad
        tag_h = text_y1 - text_y0 + 2 * pad
        tag_x, tag_y = _tag_position(box, tag_w, tag_h, rendered.width, rendered.height)
        draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill=box_colour)
        draw.text(
            (tag_x + pad - text_x0, tag_y + pad - text_y0),
            text,
            fill=text_colour,
            font=font,
        )

    rendered.save(job.out_path)
    return job.out_path


def render_all(questions: Iterable[QuestionBase], *, workers: int | None = None) -> int:
    """Renders every distinct pending render behind the given (already
    selected) questions, across `workers` processes (default: every
    CPU). Several questions can share one rendered image (e.g. every
    species asked about in a localisation image, or an open-ended
    ablation and its MC original), so jobs are de-duplicated by output
    path. Returns the number of images rendered."""
    jobs_by_path: dict[Path, RenderJob] = {}
    for question in questions:
        if question.render is not None:
            jobs_by_path.setdefault(question.render.out_path, question.render)
    jobs = list(jobs_by_path.values())

    for out_dir in {job.out_path.parent for job in jobs}:
        out_dir.mkdir(parents=True, exist_ok=True)

    parallel_map(render, jobs, workers)
    return len(jobs)
