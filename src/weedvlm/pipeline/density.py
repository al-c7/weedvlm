"""
Weed density estimation: ask the VLM to select a density category and
estimate the number of weeds in an image and the fraction of the image
area they cover, under both the boxed and unannotated image conditions
for the same source image.

`classify_density` buckets the ground truth into a category using count
and coverage thresholds; the VLM is shown the same fixed category
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

import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from weedvlm.pipeline.balance import select_questions
from weedvlm.pipeline.geometry import rectangle_union_area
from weedvlm.pipeline.render import plan_boxes
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


def classify_density(
    weed_count: int,
    coverage_fraction: float,
    *,
    # Category thresholds are placeholders -- pending real data to tune
    # against, a scene qualifies for a category once it clears either
    # the count or the coverage threshold for that level.
    low_count: int = 1,
    medium_count: int = 5,
    high_count: int = 15,
    low_coverage: float = 0.02,
    medium_coverage: float = 0.08,
    high_coverage: float = 0.20,
) -> DensityCategory:
    if weed_count >= high_count or coverage_fraction >= high_coverage:
        return DensityCategory.HIGH
    if weed_count >= medium_count or coverage_fraction >= medium_coverage:
        return DensityCategory.MEDIUM
    if weed_count >= low_count or coverage_fraction >= low_coverage:
        return DensityCategory.LOW
    return DensityCategory.NONE


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
