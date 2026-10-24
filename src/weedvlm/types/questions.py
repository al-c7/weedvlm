"""
Base types for benchmark questions. Every concrete task (multiple-choice
species ID, grounded weed VQA/set-of-marks, growth-stage ID,
fine-grained ID, density estimation, weed localisation, ...) builds its own question model
on top of QuestionBase and registers a QuestionType, so that all tasks
can eventually be rendered down to a common JSON shape.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SkipValidation

from weedvlm.types.render import Box, RenderJob


class QuestionType(StrEnum):
    SPECIES_ID_MULTIPLE_CHOICE = "species_id_multiple_choice"
    SPECIES_ID_OPEN_ENDED = "species_id_open_ended"
    GROUNDED_VQA = "grounded_vqa"
    LOCALISATION = "localisation"
    GROWTH_STAGE_ID = "growth_stage_id"
    FINE_GRAINED_ID = "fine_grained_id"
    DENSITY_ESTIMATION = "density_estimation"


class QuestionSource(BaseModel):
    """Traces a question back to the dataset it was generated from."""

    dataset_name: str
    image_id: int
    annotation_ids: list[int] = []


class QuestionBase(BaseModel):
    id: str
    question_type: QuestionType
    question_text: str
    image_path: Path
    source: QuestionSource
    # Whether the image shown to the VLM carries the ground-truth boxes
    # (True) or is the plain source image (False). Distinguishes the
    # "boxed" vs "unannotated" image-format variants that several tasks
    # evaluate separately.
    annotated: bool
    # The class this question counts against -- both when composing the
    # benchmark and in per-class scores -- e.g. the target species name
    # for species ID/grounded weed VQA, "same_species"/"different_species" for
    # fine-grained ID, or the density category for density estimation.
    benchmark_class: str
    # The not-yet-drawn render behind image_path, for box-annotated
    # questions (None when image_path is the plain source image).
    # Rendered only after selection, via
    # weedvlm.generation.render.render_all; never serialised.
    render: SkipValidation[RenderJob | None] = Field(default=None, exclude=True, repr=False)


class MultipleChoiceQuestion(QuestionBase):
    choices: list[str]
    answer_index: int


class OpenEndedQuestion(QuestionBase):
    """A free-response ablation of an existing MultipleChoiceQuestion,
    used to gauge how much of the MC accuracy is just guessing among
    the given options."""

    answer_text: str
    ablation_of: str


class DensityEstimationQuestion(QuestionBase):
    """Asks the VLM to pick a density category from `density_choices` and
    estimate weed count and coverage directly. `density_category` (a
    lowercase `DensityCategory` value) is the ground truth to check the
    VLM's chosen category against, case-insensitively."""

    density_choices: list[str]
    true_weed_count: int
    true_weed_coverage_fraction: float
    density_category: str


class LocalisationQuestion(QuestionBase):
    """Asks the VLM to draw a bounding box over every instance of
    `target_species` in an unannotated image. `boxes` is the ground
    truth, as [x, y, width, height] in pixels of the image, whose size is
    kept here so answers given in normalised coordinates can be scored."""

    target_species: str
    image_width: int
    image_height: int
    boxes: list[Box]


# Task name (as used in the configs' `tasks` toggles) -> the file
# generate-all-questions.py writes it to in the JSON storage format.
# species-id.json holds both the multiple-choice questions and their
# open-ended ablations.
TASK_FILES = {
    "species_id": "species-id.json",
    "fine_grained": "fine-grained.json",
    "density": "density.json",
    "grounded_vqa": "grounded-vqa.json",
    "localisation": "localisation.json",
}

_QUESTION_MODELS: dict[QuestionType, type[QuestionBase]] = {
    QuestionType.SPECIES_ID_MULTIPLE_CHOICE: MultipleChoiceQuestion,
    QuestionType.SPECIES_ID_OPEN_ENDED: OpenEndedQuestion,
    QuestionType.GROUNDED_VQA: MultipleChoiceQuestion,
    QuestionType.FINE_GRAINED_ID: MultipleChoiceQuestion,
    QuestionType.DENSITY_ESTIMATION: DensityEstimationQuestion,
    QuestionType.LOCALISATION: LocalisationQuestion,
}


def parse_question(raw: dict) -> QuestionBase:
    """A stored question (the JSON form of any of the models above) back
    as the model its question_type calls for."""
    question_type = QuestionType(raw["question_type"])
    try:
        model = _QUESTION_MODELS[question_type]
    except KeyError:
        raise ValueError(f"Question type {question_type} isn't supported for evaluation") from None
    return model.model_validate(raw)
