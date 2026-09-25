"""
Species localisation: an image with every instance of every species (of
the given role) present numbered by bounding box -- each species gets
its own number, shared by all of its boxes in that image, but the VLM
is never told which number corresponds to which species. For each
species present, ask which number contains it; the VLM answers with a
label rather than a species name.

Called once with role="weed" for the main task, and once with
role="crop" for the baseline task described in the spec (a small set
of easier questions -- crop species are typically fewer and more
visually distinct -- used as a sanity check on the localisation format
itself, independent of weed-identification difficulty).

An image also has to clear the task spec's image-selection criteria to
be used at all: the union of every labelled box's area must clear
`min_total_area_fraction`, *every* species' largest box must clear
`min_largest_area_fraction` (each species gets a number the VLM has to
find, so none can be present only as specks), and the boxes'
area-weighted centre of mass mustn't sit too close to a corner
(`max_centre_offset` -- see
`weedvlm.pipeline.geometry.centre_offset_fraction` for the
0=centre/1=corner scale). Defaults are deliberately loose (tuned to
keep ~90% of qualifying images on the full reviewed set).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from weedvlm.pipeline.geometry import centre_of_mass, centre_offset_fraction, rectangle_union_area
from weedvlm.pipeline.render import plan_numbered_boxes
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


def _passes_selection(
    grouped: dict[str, list[SpeciesAnnotation]],
    image: ReviewedImage,
    min_total_area_fraction: float,
    min_largest_area_fraction: float,
    max_centre_offset: float,
) -> bool:
    image_area = image.width * image.height
    if image_area == 0:
        return False

    all_boxes = [a.bbox for anns in grouped.values() for a in anns]
    if rectangle_union_area(all_boxes) / image_area < min_total_area_fraction:
        return False

    for annotations in grouped.values():
        largest_area = max(w * h for _, _, w, h in (a.bbox for a in annotations))
        if largest_area / image_area < min_largest_area_fraction:
            return False

    offset = centre_offset_fraction(centre_of_mass(all_boxes), image.width, image.height)
    return offset <= max_centre_offset


def generate_species_localisation_questions(
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
        if len(grouped) < min_species:
            continue

        if not _passes_selection(
            grouped, image, min_total_area_fraction, min_largest_area_fraction, max_centre_offset
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
            image, labelled_boxes, rendered_images_dir, slug=f"loc_{role}"
        )
        choices = [str(i + 1) for i in range(len(species_names))]

        for name, annotations in grouped.items():
            display_name = annotations[0].species.display_name
            questions.append(
                MultipleChoiceQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:{name}:"
                        f"{QuestionType.SPECIES_LOCALISATION}"
                    ),
                    question_type=QuestionType.SPECIES_LOCALISATION,
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
