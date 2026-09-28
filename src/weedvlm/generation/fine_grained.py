"""
Fine-grained identification: within a single image, pick two annotated
plants and ask whether they're the same species or different species.

"Same species" pairs are drawn from species with two or more instances
in the same image. "Different species" pairs are drawn across species,
shortlisted as visually similar by *either* close bounding-box area or
close mean crop colour (see `weedvlm.pipeline.colour`) -- an OR, not an
AND, since each signal alone is a weak, noisy stand-in for true visual
similarity (per the task spec, a placeholder pending a proper
appearance-based check plus manual review) and requiring both would
only shrink an already-small candidate pool further.

Every candidate pair (same- or different-species) also has to clear the
task spec's image-selection criteria: the two boxes' combined area and
the larger box's area must each clear a floor
(`min_total_area_fraction` / `min_largest_area_fraction`), and their
area-weighted centre of mass mustn't sit too close to a corner
(`max_centre_offset` -- see `weedvlm.pipeline.geometry.centre_offset_fraction`
for the 0=centre/1=corner scale). Defaults are deliberately loose
(tuned to keep ~85% of raw candidate pairs on the full reviewed set) --
tighten them if pairs come out too small or too off-centre to judge.

On a small reviewed set, `max_pairs_per_image` -- not the similarity or
selection thresholds -- is usually the binding constraint on how many
questions come out: most images have far more raw candidate pairs than
the cap keeps. Loosen the cap before loosening the thresholds if you
need more questions.
"""

from __future__ import annotations

import functools
import itertools
import random
from collections.abc import Sequence
from pathlib import Path

from weedvlm.pipeline.colour import colour_similarity, mean_colours
from weedvlm.pipeline.geometry import centre_of_mass, centre_offset_fraction
from weedvlm.pipeline.parallel import parallel_map
from weedvlm.pipeline.render import plan_numbered_boxes
from weedvlm.types.dataset import ReviewedImage, SpeciesAnnotation
from weedvlm.types.questions import MultipleChoiceQuestion, QuestionSource, QuestionType
from weedvlm.types.species import Role

CHOICES = ["Same species", "Different species"]
SAME_SPECIES_INDEX = 0
DIFFERENT_SPECIES_INDEX = 1


def _area(annotation: SpeciesAnnotation) -> float:
    _, _, w, h = annotation.bbox
    return w * h


def _group_by_species(
    image: ReviewedImage, role: Role
) -> dict[str, list[SpeciesAnnotation]]:
    grouped: dict[str, list[SpeciesAnnotation]] = {}
    for annotation in image.species_annotations:
        if annotation.species.role == role:
            grouped.setdefault(annotation.species.name, []).append(annotation)
    return grouped


def _passes_selection(
    a: SpeciesAnnotation,
    b: SpeciesAnnotation,
    image: ReviewedImage,
    image_area: float,
    min_total_area_fraction: float,
    min_largest_area_fraction: float,
    max_centre_offset: float,
) -> bool:
    area_a, area_b = _area(a), _area(b)
    if (area_a + area_b) / image_area < min_total_area_fraction:
        return False
    if max(area_a, area_b) / image_area < min_largest_area_fraction:
        return False

    offset = centre_offset_fraction(
        centre_of_mass([a.bbox, b.bbox]), image.width, image.height
    )
    return offset <= max_centre_offset


def _candidate_pairs(
    image: ReviewedImage,
    role: Role,
    area_similarity_ratio: float,
    colour_similarity_ratio: float,
    min_total_area_fraction: float,
    min_largest_area_fraction: float,
    max_centre_offset: float,
) -> list[tuple[SpeciesAnnotation, SpeciesAnnotation, bool]]:
    image_area = image.width * image.height
    if image_area == 0:
        return []

    grouped = _group_by_species(image, role)
    if not grouped:
        return []

    def passes(a: SpeciesAnnotation, b: SpeciesAnnotation) -> bool:
        return _passes_selection(
            a, b, image, image_area,
            min_total_area_fraction, min_largest_area_fraction, max_centre_offset,
        )

    pairs: list[tuple[SpeciesAnnotation, SpeciesAnnotation, bool]] = []

    for annotations in grouped.values():
        pairs.extend(
            (a, b, True) for a, b in itertools.combinations(annotations, 2) if passes(a, b)
        )

    if len(grouped) < 2:
        return pairs

    boxes = {a.annotation_id: a.bbox for anns in grouped.values() for a in anns}
    colours = mean_colours(image.image_path, boxes)

    for name_a, name_b in itertools.combinations(grouped, 2):
        for a, b in itertools.product(grouped[name_a], grouped[name_b]):
            if not passes(a, b):
                continue
            area_a, area_b = _area(a), _area(b)
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
            colour_ratio = colour_similarity(
                colours[a.annotation_id], colours[b.annotation_id]
            )
            if area_ratio >= area_similarity_ratio or colour_ratio >= colour_similarity_ratio:
                pairs.append((a, b, False))

    return pairs


def generate_fine_grained_questions(
    images: Sequence[ReviewedImage],
    rendered_images_dir: Path,
    *,
    role: Role = "weed",
    area_similarity_ratio: float = 0.25,
    colour_similarity_ratio: float = 0.9,
    min_total_area_fraction: float = 0.10,
    min_largest_area_fraction: float = 0.05,
    max_centre_offset: float = 0.85,
    max_pairs_per_image: int = 6,
    rng: random.Random | None = None,
    workers: int | None = None,
) -> list[MultipleChoiceQuestion]:
    rng = rng or random.Random()

    # Finding candidate pairs decodes every multi-species image for its
    # mean crop colours, so it's spread across processes; the shuffle
    # below stays serial, in image order, to keep seeded runs stable.
    candidate_pairs = functools.partial(
        _candidate_pairs,
        role=role,
        area_similarity_ratio=area_similarity_ratio,
        colour_similarity_ratio=colour_similarity_ratio,
        min_total_area_fraction=min_total_area_fraction,
        min_largest_area_fraction=min_largest_area_fraction,
        max_centre_offset=max_centre_offset,
    )
    pairs_by_image = parallel_map(candidate_pairs, images, workers)

    questions = []
    for image, pairs in zip(images, pairs_by_image, strict=True):
        rng.shuffle(pairs)

        for a, b, same_species in pairs[:max_pairs_per_image]:
            render = plan_numbered_boxes(
                image,
                [(1, a.bbox), (2, b.bbox)],
                rendered_images_dir,
                slug=f"{a.annotation_id}_{b.annotation_id}",
            )

            questions.append(
                MultipleChoiceQuestion(
                    id=(
                        f"{image.dataset_name}:{image.image_id}:"
                        f"{a.annotation_id}-{b.annotation_id}:{QuestionType.FINE_GRAINED_ID}"
                    ),
                    question_type=QuestionType.FINE_GRAINED_ID,
                    question_text=(
                        "Are the plants in bounding boxes 1 and 2 the same species, "
                        "or different species?"
                    ),
                    image_path=render.out_path,
                    render=render,
                    source=QuestionSource(
                        dataset_name=image.dataset_name,
                        image_id=image.image_id,
                        annotation_ids=[a.annotation_id, b.annotation_id],
                    ),
                    annotated=True,
                    benchmark_class="same_species" if same_species else "different_species",
                    choices=CHOICES,
                    answer_index=SAME_SPECIES_INDEX if same_species else DIFFERENT_SPECIES_INDEX,
                )
            )

    return questions
