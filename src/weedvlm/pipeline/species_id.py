"""
Multiple-choice species identification, in the two image formats the
task is evaluated under:

- `generate_species_id_mc_questions` (boxed): for each species (of the
  given role) present in an image with enough total bounding-box area
  to be identifiable, render that species' boxes onto a copy of the
  image and ask which species they belong to, with distractors drawn
  from the other species observed in the same dataset. An image with
  several species present yields one question per qualifying species
  rather than being skipped -- each question only draws boxes for its
  own target species, so the other species present don't leak the
  answer or confuse the picture.

- `generate_species_id_unannotated_questions` (unannotated): the plain
  source image is shown with no boxes at all, so it only makes sense
  for images containing a single weed species (other roles, e.g. crop,
  may still be present) -- one question per qualifying image. This is
  a smaller set than the boxed one, since most images don't qualify.

Either way, a species is only asked about if it's large enough in the
image to identify (`_is_large_enough`): its boxes must cover at least
`min_area_fraction` of the image combined, and its single largest box
at least `min_largest_area_fraction` -- so a species present only as
many tiny seedlings, or one mid-size plant among specks, is dropped.

`select_species_id_questions` then picks the final boxed and
unannotated sets over one shared species selection, each with its own
per-species quota, so the number of unannotated questions is controlled
independently of the boxed ones. A subset of either set's questions can
then be turned into open-ended ablations via
`weedvlm.pipeline.open_ended`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from weedvlm.pipeline.balance import InsufficientQuestionsError, select_questions
from weedvlm.pipeline.render import plan_species_boxes
from weedvlm.types.dataset import ReviewedImage, SpeciesAnnotation
from weedvlm.types.questions import MultipleChoiceQuestion, QuestionSource, QuestionType
from weedvlm.types.species import Role


def _group_by_species(
    image: ReviewedImage, role: Role
) -> dict[str, list[SpeciesAnnotation]]:
    grouped: dict[str, list[SpeciesAnnotation]] = {}
    for annotation in image.species_annotations:
        if annotation.species.role == role:
            grouped.setdefault(annotation.species.name, []).append(annotation)
    return grouped


def _is_large_enough(
    annotations: list[SpeciesAnnotation],
    image_area: float,
    min_area_fraction: float,
    min_largest_area_fraction: float,
) -> bool:
    areas = [w * h for (_, _, w, h) in (a.bbox for a in annotations)]
    return (
        sum(areas) / image_area >= min_area_fraction
        and max(areas) / image_area >= min_largest_area_fraction
    )


def generate_species_id_mc_questions(
    images: Sequence[ReviewedImage],
    rendered_images_dir: Path,
    *,
    role: Role = "weed",
    num_choices: int = 4,
    min_area_fraction: float = 0.10,
    min_largest_area_fraction: float = 0.0,
    rng: random.Random | None = None,
) -> list[MultipleChoiceQuestion]:
    rng = rng or random.Random()

    display_names_by_dataset: dict[str, dict[str, str]] = {}
    for image in images:
        display_names = display_names_by_dataset.setdefault(image.dataset_name, {})
        for species in image.species_of_role(role):
            display_names[species.name] = species.display_name

    questions = []
    for image in images:
        image_area = image.width * image.height
        if image_area == 0:
            continue

        for species_name, annotations in _group_by_species(image, role).items():
            if not _is_large_enough(
                annotations, image_area, min_area_fraction, min_largest_area_fraction
            ):
                continue

            display_names = display_names_by_dataset[image.dataset_name]
            distractor_pool = [name for name in display_names if name != species_name]
            num_distractors = min(num_choices - 1, len(distractor_pool))
            option_names = [species_name, *rng.sample(distractor_pool, num_distractors)]
            rng.shuffle(option_names)

            render = plan_species_boxes(
                image, species_name, [a.bbox for a in annotations], rendered_images_dir
            )

            questions.append(
                MultipleChoiceQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:{species_name}:"
                        f"{QuestionType.SPECIES_ID_MULTIPLE_CHOICE}"
                    ),
                    question_type=QuestionType.SPECIES_ID_MULTIPLE_CHOICE,
                    question_text=f"What {role} species is located in the bounding boxes in this image?",
                    image_path=render.out_path,
                    render=render,
                    source=QuestionSource(
                        dataset_name=image.dataset_name,
                        image_id=image.image_id,
                        annotation_ids=[a.annotation_id for a in annotations],
                    ),
                    annotated=True,
                    benchmark_class=display_names[species_name],
                    choices=[display_names[name] for name in option_names],
                    answer_index=option_names.index(species_name),
                )
            )

    return questions


def generate_species_id_unannotated_questions(
    images: Sequence[ReviewedImage],
    *,
    role: Role = "weed",
    num_choices: int = 4,
    min_area_fraction: float = 0.10,
    min_largest_area_fraction: float = 0.0,
    rng: random.Random | None = None,
) -> list[MultipleChoiceQuestion]:
    rng = rng or random.Random()

    display_names_by_dataset: dict[str, dict[str, str]] = {}
    for image in images:
        display_names = display_names_by_dataset.setdefault(image.dataset_name, {})
        for species in image.species_of_role(role):
            display_names[species.name] = species.display_name

    questions = []
    for image in images:
        image_area = image.width * image.height
        if image_area == 0:
            continue

        species_names = image.unique_species_names(role)
        if len(species_names) != 1:
            continue
        (species_name,) = species_names

        annotations = [a for a in image.species_annotations if a.species.name == species_name]
        if not _is_large_enough(
            annotations, image_area, min_area_fraction, min_largest_area_fraction
        ):
            continue

        display_names = display_names_by_dataset[image.dataset_name]
        distractor_pool = [name for name in display_names if name != species_name]
        num_distractors = min(num_choices - 1, len(distractor_pool))
        option_names = [species_name, *rng.sample(distractor_pool, num_distractors)]
        rng.shuffle(option_names)

        questions.append(
            MultipleChoiceQuestion(
                id=(
                    f"{image.dataset_name}:{image.image_id}:{species_name}:"
                    f"{QuestionType.SPECIES_ID_MULTIPLE_CHOICE}:unannotated"
                ),
                question_type=QuestionType.SPECIES_ID_MULTIPLE_CHOICE,
                question_text=f"What {role} species is present in this image?",
                image_path=image.image_path,
                source=QuestionSource(
                    dataset_name=image.dataset_name,
                    image_id=image.image_id,
                    annotation_ids=[a.annotation_id for a in annotations],
                ),
                annotated=False,
                benchmark_class=display_names[species_name],
                choices=[display_names[name] for name in option_names],
                answer_index=option_names.index(species_name),
            )
        )

    return questions


def select_species_id_questions(
    boxed: Sequence[MultipleChoiceQuestion],
    unannotated: Sequence[MultipleChoiceQuestion],
    *,
    num_species: int | None,
    questions_per_species: int | None,
    unannotated_questions_per_species: int | None,
    rng: random.Random | None = None,
) -> tuple[list[MultipleChoiceQuestion], list[MultipleChoiceQuestion]]:
    """(boxed, unannotated) final selections over the same species.
    Species are chosen exactly as select_questions does for the boxed
    set, except that when unannotated_questions_per_species is set, only
    species that can also supply that many unannotated questions are
    eligible. Each chosen species then supplies exactly
    questions_per_species boxed and unannotated_questions_per_species
    unannotated questions (0 for none). None leaves either count
    unconstrained. Raises InsufficientQuestionsError if the quotas can't
    be met."""
    rng = rng or random.Random()

    eligible_boxed = list(boxed)
    if unannotated_questions_per_species:
        unannotated_counts: dict[str, int] = {}
        for q in unannotated:
            unannotated_counts[q.benchmark_class] = unannotated_counts.get(q.benchmark_class, 0) + 1
        eligible_boxed = [
            q
            for q in boxed
            if unannotated_counts.get(q.benchmark_class, 0) >= unannotated_questions_per_species
        ]
    try:
        selected_boxed = select_questions(
            eligible_boxed,
            num_classes=num_species,
            questions_per_class=questions_per_species,
            rng=rng,
        )
    except InsufficientQuestionsError as e:
        if not unannotated_questions_per_species:
            raise
        raise InsufficientQuestionsError(
            f"{e} (Only species with at least unannotated_questions_per_species="
            f"{unannotated_questions_per_species} unannotated questions were eligible -- "
            f"lowering that also helps.)"
        ) from e

    selected_species = {q.benchmark_class for q in selected_boxed}
    selected_unannotated = select_questions(
        [q for q in unannotated if q.benchmark_class in selected_species],
        questions_per_class=unannotated_questions_per_species,
        rng=rng,
    )
    return selected_boxed, selected_unannotated
