"""
Mean-colour similarity between annotated crops, used alongside
bounding-box area similarity to shortlist "different species" pairs for
fine-grained ID. Per the task spec, the eventual method is "comparing
colours, as well as bounding box areas" -- this is the colour half of
that, kept deliberately simple (whole-crop mean RGB, not e.g. a proper
appearance/colour-histogram model) since it's a coarse pre-filter that
still goes through manual review, not the final word on similarity.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage, ImageStat

RGB = tuple[float, float, float]

_MAX_DISTANCE = (3 * 255**2) ** 0.5

# JPEGs are decoded at 1/_DRAFT_SCALE resolution (libjpeg's DCT scaling,
# far cheaper than a full decode + downscale). Not free: small boxes
# cover only a few pixels at 1/8, so their mean colour drifts. On a
# CropAndWeed/CottonWeedDet12 sample (median box ~33x33px), vs. a full
# decode, the colour pass got faster and flipped this share of
# cross-species pairs' colour_similarity >= 0.9 test:
#   1/2: 1.7x, 1.6%    1/4: 2.0x, 3.0%    1/8: 2.3x, 6.0%
# Acceptable for a coarse pre-filter ahead of manual review; lower this
# if the colour shortlist needs to track full-resolution colour closely.
# 1 disables the draft (full decode).
_DRAFT_SCALE = 8


def mean_colours(
    image_path: Path,
    boxes: dict[int, tuple[float, float, float, float]],
) -> dict[int, RGB]:
    """Mean RGB of each given (key -> bbox, in full-resolution pixels)
    crop, opening the image once. JPEGs are decoded at reduced size
    (see _DRAFT_SCALE); other formats ignore the draft and decode in
    full."""
    with PILImage.open(image_path) as source:
        full_w, full_h = source.size
        if _DRAFT_SCALE > 1:
            source.draft("RGB", (full_w // _DRAFT_SCALE, full_h // _DRAFT_SCALE))
        rgb = source.convert("RGB")

    scale_x, scale_y = rgb.width / full_w, rgb.height / full_h

    def scaled_crop(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        # Nearest-pixel edges (rounding in or out -- expanding every
        # edge instead pulls in background and biases small boxes
        # further), never less than 1px so a tiny box still has a colour.
        x0 = min(max(0, round(x * scale_x)), rgb.width - 1)
        y0 = min(max(0, round(y * scale_y)), rgb.height - 1)
        x1 = min(max(x0 + 1, round((x + w) * scale_x)), rgb.width)
        y1 = min(max(y0 + 1, round((y + h) * scale_y)), rgb.height)
        return x0, y0, x1, y1

    return {
        key: tuple(ImageStat.Stat(rgb.crop(scaled_crop(*box))).mean)
        for key, box in boxes.items()
    }


def colour_similarity(a: RGB, b: RGB) -> float:
    """1.0 for identical mean colour, down to 0.0 for black vs. white."""
    distance = sum((ca - cb) ** 2 for ca, cb in zip(a, b)) ** 0.5
    return 1 - distance / _MAX_DISTANCE
