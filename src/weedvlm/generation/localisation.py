"""
Weed localisation (docs/WEED-LOCALISATION.md): the plain source image is
shown with no boxes, and the VLM is asked to draw a bounding box over
every instance of one named species.

An image yields a question for each species (of the given role) in it
that has at most `max_boxes` instances, every one of which covers at
least `min_box_area_fraction` of the image -- so the VLM is never asked
to box a crowd of plants, or ones too small to find.

How busy the rest of the scene must be is set separately, so the target
has to be picked out from among other plants rather than found alone on
bare soil: the image must hold at least `min_scene_weeds` plants of that
role in total (the target's own included), from at least
`min_other_species` species besides the target.
"""

from __future__ import annotations

from collections.abc import Sequence

from weedvlm.types.dataset import ReviewedImage, SpeciesAnnotation
from weedvlm.types.questions import LocalisationQuestion, QuestionSource, QuestionType
from weedvlm.types.species import Role


def generate_localisation_questions(
    images: Sequence[ReviewedImage],
    *,
    role: Role = "weed",
    max_boxes: int = 4,
    min_box_area_fraction: float = 0.02,
    min_scene_weeds: int = 1,
    min_other_species: int = 0,
) -> list[LocalisationQuestion]:
    questions = []
    for image in images:
        image_area = image.width * image.height
        if image_area == 0:
            continue

        grouped: dict[str, list[SpeciesAnnotation]] = {}
        for annotation in image.species_annotations:
            if annotation.species.role == role:
                grouped.setdefault(annotation.species.name, []).append(annotation)

        scene_weeds = sum(len(annotations) for annotations in grouped.values())
        if scene_weeds < min_scene_weeds or len(grouped) - 1 < min_other_species:
            continue

        for species_name, annotations in grouped.items():
            if len(annotations) > max_boxes:
                continue
            if any(
                w * h / image_area < min_box_area_fraction
                for _, _, w, h in (a.bbox for a in annotations)
            ):
                continue

            display_name = annotations[0].species.display_name
            questions.append(
                LocalisationQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:{species_name}:"
                        f"{QuestionType.LOCALISATION}"
                    ),
                    question_type=QuestionType.LOCALISATION,
                    question_text=(
                        f"There are instances of {display_name} in this image, please give "
                        "coordinates for bounding boxes of these species in your native format."
                    ),
                    image_path=image.image_path,
                    source=QuestionSource(
                        dataset_name=image.dataset_name,
                        image_id=image.image_id,
                        annotation_ids=[a.annotation_id for a in annotations],
                    ),
                    annotated=False,
                    benchmark_class=display_name,
                    target_species=display_name,
                    image_width=image.width,
                    image_height=image.height,
                    boxes=[a.bbox for a in annotations],
                )
            )

    return questions
