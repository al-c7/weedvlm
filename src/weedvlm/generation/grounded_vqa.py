"""
Grounded weed VQA: an image with every instance of every species (of
the given role) present numbered by bounding box -- each species gets
its own number, shared by all of its boxes in that image, but the VLM
is never told which number corresponds to which species. For each
species present, ask which number contains it; the VLM answers with a
label rather than a species name.

An image yields a question for each species in it that's large enough
to find (its largest box clears `min_largest_area_fraction`), so one
image is reused across every qualifying target species. Species below
that floor are still numbered -- they're real plants in the picture,
and leaving them unboxed would be misleading -- but are only ever
distractor options, never a target.

Called once with role="weed" for the main task, and once with
role="crop" for the baseline task described in the spec (a small set
of easier questions -- crop species are typically fewer and more
visually distinct -- used as a sanity check on the grounded-VQA format
itself, independent of weed-identification difficulty).

An image also has to clear the task spec's image-selection criteria to
be used at all: the union of every labelled box's area must clear
`min_total_area_fraction`, at least one species must clear
`min_largest_area_fraction`, and the boxes' area-weighted centre of mass mustn't sit too close to a corner
(`max_centre_offset` -- see
`weedvlm.generation.geometry.centre_offset_fraction` for the
0=centre/1=corner scale). Defaults are deliberately loose (tuned to
keep ~90% of qualifying images on the full reviewed set).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from weedvlm.generation.geometry import centre_of_mass, centre_offset_fraction, rectangle_union_area
from weedvlm.generation.render import plan_numbered_boxes
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


def _target_species(
    grouped: dict[str, list[SpeciesAnnotation]],
    image: ReviewedImage,
    min_largest_area_fraction: float,
) -> set[str]:
    """The species large enough to be asked about: those whose largest
    box clears min_largest_area_fraction of the image."""
    image_area = image.width * image.height
    return {
        name
        for name, annotations in grouped.items()
        if max(w * h for _, _, w, h in (a.bbox for a in annotations)) / image_area
        >= min_largest_area_fraction
    }


def _passes_selection(
    grouped: dict[str, list[SpeciesAnnotation]],
    image: ReviewedImage,
    min_total_area_fraction: float,
    max_centre_offset: float,
) -> bool:
    image_area = image.width * image.height
    all_boxes = [a.bbox for anns in grouped.values() for a in anns]
    if rectangle_union_area(all_boxes) / image_area < min_total_area_fraction:
        return False

    offset = centre_offset_fraction(centre_of_mass(all_boxes), image.width, image.height)
    return offset <= max_centre_offset


def generate_grounded_vqa_questions(
    images: Sequence[ReviewedImage],
    rendered_images_dir: Path,
    *,
    role: Role = "weed",
    min_species: int = 2,
    min_total_area_fraction: float = 0.01,
    min_largest_area_fraction: float = 0.003,
    max_centre_offset: float = 0.6,
    rng: random.Random | None = None,
) -> list[MultipleChoiceQuestion]:
    rng = rng or random.Random()

    questions = []
    for image in images:
        grouped = _group_by_species(image, role)
        if len(grouped) < min_species or image.width * image.height == 0:
            continue

        targets = _target_species(grouped, image, min_largest_area_fraction)
        if not targets or not _passes_selection(
            grouped, image, min_total_area_fraction, max_centre_offset
        ):
            continue

        species_names = list(grouped)
        rng.shuffle(species_names)
        label_by_species = {name: i + 1 for i, name in enumerate(species_names)}

        labelled_boxes = [
            (label_by_species[name], annotation.bbox)
            for name, annotations in grouped.items()
            for annotation in annotations
        ]
        render = plan_numbered_boxes(
            image, labelled_boxes, rendered_images_dir, slug=f"gvqa_{role}"
        )
        choices = [str(i + 1) for i in range(len(species_names))]

        for name, annotations in grouped.items():
            if name not in targets:
                continue
            display_name = annotations[0].species.display_name
            questions.append(
                MultipleChoiceQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:{name}:"
                        f"{QuestionType.GROUNDED_VQA}"
                    ),
                    question_type=QuestionType.GROUNDED_VQA,
                    question_text=(
                        f"Which numbered bounding box contains the {role} '{display_name}'?"
                    ),
                    image_path=render.out_path,
                    render=render,
                    source=QuestionSource(
                        dataset_name=image.dataset_name,
                        image_id=image.image_id,
                        annotation_ids=[a.annotation_id for a in annotations],
                    ),
                    annotated=True,
                    benchmark_class=display_name,
                    choices=choices,
                    answer_index=label_by_species[name] - 1,
                )
            )

    return questions
