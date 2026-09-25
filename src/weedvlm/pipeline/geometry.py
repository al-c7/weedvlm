"""
Bounding-box geometry helpers shared across question-generation tasks:
union area of a set of axis-aligned boxes (needed wherever overlapping
annotations must not be double-counted, e.g. weed coverage for density
estimation), and centre-of-mass / centre-offset, used by the
fine-grained ID and species localisation image-selection filters to
keep target boxes away from image corners.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def rectangle_union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Exact area covered by the union of the given (x, y, w, h) boxes.

    Uses coordinate compression: the boxes' edges partition the plane
    into a grid of cells, and each cell is either fully inside some box
    or fully outside all of them, so summing the area of covered cells
    gives the exact union area without approximation.
    """
    if not boxes:
        return 0.0

    xs = sorted({x for x, _, _, _ in boxes} | {x + w for x, _, w, _ in boxes})
    ys = sorted({y for _, y, _, _ in boxes} | {y + h for _, y, _, h in boxes})

    total = 0.0
    for i in range(len(xs) - 1):
        cell_w = xs[i + 1] - xs[i]
        if cell_w <= 0:
            continue
        cx = (xs[i] + xs[i + 1]) / 2

        for j in range(len(ys) - 1):
            cell_h = ys[j + 1] - ys[j]
            if cell_h <= 0:
                continue
            cy = (ys[j] + ys[j + 1]) / 2

            if any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in boxes):
                total += cell_w * cell_h

    return total


def centre_of_mass(boxes: list[Box]) -> tuple[float, float]:
    """Area-weighted centroid of the given boxes' centres."""
    total_area = 0.0
    cx_sum = 0.0
    cy_sum = 0.0
    for x, y, w, h in boxes:
        area = w * h
        cx_sum += (x + w / 2) * area
        cy_sum += (y + h / 2) * area
        total_area += area

    if total_area == 0:
        return 0.0, 0.0
    return cx_sum / total_area, cy_sum / total_area


def centre_offset_fraction(centre: tuple[float, float], width: int, height: int) -> float:
    """How far `centre` sits from the image centre, on a scale where 0.0 is
    the exact image centre and 1.0 is exactly at a corner -- i.e. "not in
    the corners" per the task specs is `centre_offset_fraction(...) <= t`
    for some tolerance t < 1.0."""
    if width == 0 or height == 0:
        return 0.0

    cx, cy = centre
    nx = (cx - width / 2) / (width / 2)
    ny = (cy - height / 2) / (height / 2)
    return (nx**2 + ny**2) ** 0.5 / 2**0.5
