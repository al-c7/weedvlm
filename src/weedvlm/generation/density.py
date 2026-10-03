"""
Weed density estimation: ask the VLM to select a density category and
estimate the number of weeds in an image and the fraction of the image
area they cover, under both the boxed and unannotated image conditions
for the same source image.

`classify_density` buckets the ground truth into a category using a
combined count/coverage score (see its docstring); the VLM is shown the
same fixed category
vocabulary (`DENSITY_CHOICES`) and must pick one, in addition to its two
numeric estimates, per the task spec. Coverage accounts for overlapping
boxes via `rectangle_union_area` rather than naively summing box areas.

An image with any weeds is skipped unless its largest weed box covers at
least `min_largest_area_fraction` of the image, so the VLM isn't asked
to count plants too small to make out. Weed-free images (the "none"
category) have no boxes to measure and are always kept.

`select_density_questions` then picks the final boxed and unannotated
sets separately, drawing the unannotated questions only from the
images whose boxed question was selected -- per the task spec, the same
source images are used under each condition.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from weedvlm.generation.balance import select_questions
from weedvlm.generation.geometry import rectangle_union_area
from weedvlm.generation.render import plan_boxes
from weedvlm.types.dataset import ReviewedImage
from weedvlm.types.questions import DensityEstimationQuestion, QuestionSource, QuestionType
from weedvlm.types.species import Role

# Display strings shown to the VLM, in the same order as DensityCategory.
DENSITY_CHOICES = ["None", "Low", "Medium", "High"]

QUESTION_TEXT = (
    "Select which density category best describes the weeds visible in this image, and "
    "estimate how many individual weeds are visible and what percentage of the image area "
    "they cover."
)


class DensityCategory(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _log_scale(value: float, low: float, high: float) -> float:
    """value's position between low (0.0) and high (1.0) on a log scale,
    clamped to [0, 1]. Log because both weed count and coverage are
    heavily right-skewed -- going from 1 to 2 weeds matters as much as
    from 10 to 20."""
    if value <= low:
        return 0.0
    return min(1.0, math.log(value / low) / math.log(high / low))


def density_score(
    weed_count: int,
    coverage_fraction: float,
    *,
    count_scale: tuple[float, float] = (1, 40),
    coverage_scale: tuple[float, float] = (0.01, 0.4),
) -> float:
    """0.0 (sparse) to 1.0 (dense): the mean of weed count and coverage,
    each placed on a log scale between its (low, high) anchors."""
    return (
        _log_scale(weed_count, *count_scale) + _log_scale(coverage_fraction, *coverage_scale)
    ) / 2


def classify_density(
    weed_count: int,
    coverage_fraction: float,
    *,
    count_scale: tuple[float, float] = (1, 40),
    coverage_scale: tuple[float, float] = (0.01, 0.4),
    medium_score: float = 0.3,
    high_score: float = 0.6,
    medium_min_count: int = 2,
    high_min_count: int = 5,
) -> DensityCategory:
    """Buckets a scene by `density_score`, which weighs count and
    coverage together rather than letting either decide alone: coverage
    alone mostly measures how close the camera was (one weed shot close
    up can fill the frame), and count alone ignores how much of the
    field the weeds actually take up. Low below medium_score, medium
    from medium_score, high from high_score -- except a scene needs at
    least medium_min_count / high_min_count weeds to be rated medium /
    high, so a single close-up plant is never "dense". No weeds at all
    is always NONE."""
    if weed_count == 0:
        return DensityCategory.NONE
    score = density_score(
        weed_count, coverage_fraction, count_scale=count_scale, coverage_scale=coverage_scale
    )
    if score >= high_score and weed_count >= high_min_count:
        return DensityCategory.HIGH
    if score >= medium_score and weed_count >= medium_min_count:
        return DensityCategory.MEDIUM
    return DensityCategory.LOW


def generate_density_estimation_questions(
    images: Sequence[ReviewedImage],
    rendered_images_dir: Path,
    *,
    role: Role = "weed",
    min_largest_area_fraction: float = 0.0,
    **classify_density_kwargs,
) -> list[DensityEstimationQuestion]:
    questions = []
    for image in images:
        image_area = image.width * image.height
        if image_area == 0:
            continue

        annotations = [a for a in image.species_annotations if a.species.role == role]
        boxes = [a.bbox for a in annotations]
        if boxes and max(w * h for _, _, w, h in boxes) / image_area < min_largest_area_fraction:
            continue
        coverage_fraction = rectangle_union_area(boxes) / image_area
        category = classify_density(len(annotations), coverage_fraction, **classify_density_kwargs)

        source = QuestionSource(
            dataset_name=image.dataset_name,
            image_id=image.image_id,
            annotation_ids=[a.annotation_id for a in annotations],
        )
        boxed_render = (
            plan_boxes(image, boxes, rendered_images_dir, slug="density") if boxes else None
        )
        boxed_path = boxed_render.out_path if boxed_render else image.image_path

        for annotated, image_path, render in (
            (True, boxed_path, boxed_render),
            (False, image.image_path, None),
        ):
            questions.append(
                DensityEstimationQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:{QuestionType.DENSITY_ESTIMATION}:"
                        f"{'boxed' if annotated else 'unannotated'}"
                    ),
                    question_type=QuestionType.DENSITY_ESTIMATION,
                    question_text=QUESTION_TEXT,
                    image_path=image_path,
                    render=render,
                    source=source,
                    annotated=annotated,
                    benchmark_class=category.value,
                    density_choices=DENSITY_CHOICES,
                    true_weed_count=len(annotations),
                    true_weed_coverage_fraction=coverage_fraction,
                    density_category=category.value,
                )
            )

    return questions


def select_density_questions(
    questions: Sequence[DensityEstimationQuestion],
    *,
    questions_per_class: int | None,
    unannotated_questions_per_class: int | None,
    rng: random.Random | None = None,
) -> tuple[list[DensityEstimationQuestion], list[DensityEstimationQuestion]]:
    """(boxed, unannotated) final selections. Boxed questions are
    balanced to exactly questions_per_class per category; unannotated
    ones to exactly unannotated_questions_per_class per category, taken
    only from the images selected for the boxed set (so it can't exceed
    questions_per_class). None leaves either unconstrained. Raises
    InsufficientQuestionsError if a category falls short."""
    rng = rng or random.Random()
    boxed = select_questions(
        [q for q in questions if q.annotated], questions_per_class=questions_per_class, rng=rng
    )
    selected_images = {(q.source.dataset_name, q.source.image_id) for q in boxed}
    unannotated = select_questions(
        [
            q
            for q in questions
            if not q.annotated and (q.source.dataset_name, q.source.image_id) in selected_images
        ],
        questions_per_class=unannotated_questions_per_class,
        rng=rng,
    )
    return boxed, unannotated
