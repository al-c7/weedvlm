"""
Interprets the bounding boxes a model gives for the localisation task and
scores them against the ground truth.

Models answer in their own native box format, so each reply says what it
used in a free-text `response_format` field. `infer_bbox_format` reads the
coordinate order and scale out of that text; a `bbox_format` set for the
model in the evaluation config (see `parse_bbox_format`) replaces it for
models that describe themselves badly. Boxes given as objects with named
coordinates ({"xmin": ..}) carry their own order. A scale that is never
stated is taken from the values: all within 0-1 means fractions of the
image, anything larger means pixels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from weedvlm.types.render import Box

ORDERS = ("xyxy", "yxyx", "xywh", "cxcywh")
_SCALES = {"pixel": None, "unit": 1.0, "100": 100.0, "1000": 1000.0}

_ORDER_HINTS = {
    "yxyx": ("yminxminymaxxmax", "y1x1y2x2", "y0x0y1x1", "yxyx", "topleftbottomright"),
    "xyxy": ("xminyminxmaxymax", "x1y1x2y2", "x0y0x1y1", "xyxy", "lefttoprightbottom"),
    "cxcywh": ("cxcywh", "cxcywidthheight", "centerxcentery", "xcenterycenter"),
    "xywh": ("xywh", "xywidthheight", "xminyminwidthheight", "x1y1widthheight", "lefttopwidthheight"),
}

# Keys of a box given as an object, to (order, the keys in that order).
_NAMED_LAYOUTS = (
    ("xyxy", ("xmin", "ymin", "xmax", "ymax")),
    ("xyxy", ("x1", "y1", "x2", "y2")),
    ("xyxy", ("left", "top", "right", "bottom")),
    ("cxcywh", ("cx", "cy", "width", "height")),
    ("cxcywh", ("cx", "cy", "w", "h")),
    ("xywh", ("x", "y", "width", "height")),
    ("xywh", ("x", "y", "w", "h")),
)
_LIST_KEYS = ("box2d", "bbox2d", "bbox", "box", "coordinates", "bounds")


@dataclass(frozen=True)
class BboxFormat:
    # None when not known: a box must then name its own coordinates.
    order: str | None
    # What a full image width/height is in the model's units; None for
    # pixels. `scale_known` is False when it has to come from the values.
    scale: float | None = None
    scale_known: bool = True

    def __str__(self) -> str:
        scale = next(name for name, value in _SCALES.items() if value == self.scale)
        return f"{self.order or 'named'}_{scale if self.scale_known else 'inferred'}"


def parse_bbox_format(spec: str) -> BboxFormat:
    """A configured format, "<order>_<scale>": order one of xyxy, yxyx,
    xywh, cxcywh; scale one of pixel, unit (0-1), 100, 1000. For example
    "yxyx_1000" or "xyxy_pixel"."""
    order, _, scale = spec.strip().lower().partition("_")
    if order not in ORDERS or scale not in _SCALES:
        raise ValueError(
            f"bbox_format {spec!r} isn't <order>_<scale> with order in {'/'.join(ORDERS)} and "
            f"scale in {'/'.join(_SCALES)}, e.g. yxyx_1000"
        )
    return BboxFormat(order, _SCALES[scale])


def infer_bbox_format(description: Any) -> BboxFormat:
    """What a model's `response_format` text says about its boxes."""
    text = str(description or "").lower()
    compact = re.sub(r"[^a-z0-9]", "", text).replace("centre", "center")

    order = next(
        (name for name, hints in _ORDER_HINTS.items() if any(hint in compact for hint in hints)),
        None,
    )
    gemini_style = "box2d" in compact
    if order is None and gemini_style:
        order = "yxyx"

    if "1000" in text or "thousand" in text:
        return BboxFormat(order, 1000.0)
    if "percent" in text or "%" in text:
        return BboxFormat(order, 100.0)
    if re.search(r"normali[sz]ed|relative|fraction|\b0(?:\.0)?\s*(?:-|–|to|,)\s*1(?:\.0)?\b", text):
        return BboxFormat(order, 1.0)
    if re.search(r"pixel|absolute|\bpx\b", text):
        return BboxFormat(order, None)
    if gemini_style:
        return BboxFormat(order, 1000.0)
    return BboxFormat(order, None, scale_known=False)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"box coordinate {value!r} isn't a number")
    return float(value)


def _coordinates(box: Any, order: str | None) -> tuple[str, list[float]]:
    """(order, the four numbers) for one box as the model gave it."""
    if isinstance(box, dict):
        keyed = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in box.items()}
        for named_order, keys in _NAMED_LAYOUTS:
            if all(key in keyed for key in keys):
                return named_order, [_number(keyed[key]) for key in keys]
        for key in _LIST_KEYS:
            if key in keyed:
                return _coordinates(keyed[key], order)
        raise ValueError(f"box {box!r} has no recognisable coordinates")
    if isinstance(box, list | tuple) and len(box) == 4:
        if order is None:
            raise ValueError("the coordinate order isn't stated in response_format")
        return order, [_number(v) for v in box]
    if isinstance(box, list | tuple) and len(box) == 2 and all(isinstance(p, list | tuple) for p in box):
        return _coordinates([*box[0], *box[1]], order)
    raise ValueError(f"box {box!r} isn't four coordinates")


def to_pixel_boxes(raw: Any, fmt: BboxFormat, width: int, height: int) -> tuple[list[Box], BboxFormat]:
    """The model's boxes as (x1, y1, x2, y2) in pixels, clipped to the
    image, and the format they were read with. Raises ValueError if they
    can't be interpreted."""
    if not isinstance(raw, list):
        raise ValueError("bboxes isn't a list")
    if len(raw) == 4 and all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw):
        raw = [raw]  # a single box given bare, not as a list of boxes

    boxes = [_coordinates(box, fmt.order) for box in raw]
    values = [v for _, coords in boxes for v in coords]

    scale = fmt.scale
    if not fmt.scale_known:
        scale = 1.0 if values and max(values) <= 1.0 else None
    elif scale == 1.0 and values and max(values) > 1.5:
        scale = 1000.0  # "normalised" boxes beyond 0-1 are on the 0-1000 scale
    x_factor, y_factor = (1.0, 1.0) if scale is None else (width / scale, height / scale)

    pixel_boxes: list[Box] = []
    for order, (a, b, c, d) in boxes:
        if order == "yxyx":
            a, b, c, d = b, a, d, c
        elif order == "xywh":
            c, d = a + c, b + d
        elif order == "cxcywh":
            a, b, c, d = a - c / 2, b - d / 2, a + c / 2, b + d / 2
        x1, x2 = sorted((a * x_factor, c * x_factor))
        y1, y2 = sorted((b * y_factor, d * y_factor))
        pixel_boxes.append(
            (
                min(max(x1, 0.0), width),
                min(max(y1, 0.0), height),
                min(max(x2, 0.0), width),
                min(max(y2, 0.0), height),
            )
        )
    return pixel_boxes, BboxFormat(fmt.order, scale)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(truth: list[Box], predicted: list[Box]) -> list[float]:
    """IoUs of predicted boxes paired one-to-one with ground-truth boxes,
    best overlaps first. Boxes that overlap nothing stay unpaired."""
    pairs = sorted(
        ((iou(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(predicted)),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    matched = []
    for overlap, i, j in pairs:
        if overlap <= 0:
            break
        if i in used_truth or j in used_predicted:
            continue
        used_truth.add(i)
        used_predicted.add(j)
        matched.append(overlap)
    return matched


def score_boxes(truth_xywh: list[Box], predicted: list[Box]) -> dict[str, Any]:
    """Scores one answer on both things the task asks for: the right
    number of boxes, and boxes in the right place.

    `localisation_score` is the summed IoU of the paired boxes divided by
    the larger of the two box counts, so it is 1 only for the right number
    of boxes each exactly on a plant, and every missing or extra box costs
    a full share. `matched_iou` is the mean IoU of the paired boxes alone:
    how well placed the boxes that did find a plant are."""
    truth = [(x, y, x + w, y + h) for x, y, w, h in truth_xywh]
    matched = match_boxes(truth, predicted)
    return {
        "true_box_count": len(truth),
        "pred_box_count": len(predicted),
        "box_count_error": len(predicted) - len(truth),
        "matched_iou": sum(matched) / len(matched) if matched else None,
        "localisation_score": sum(matched) / max(len(truth), len(predicted)),
    }
