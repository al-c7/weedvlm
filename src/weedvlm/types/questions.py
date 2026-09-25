"""
Base types for benchmark questions. Every concrete task (multiple-choice
species ID, species localisation/set-of-marks, growth-stage ID,
fine-grained ID, density estimation, ...) builds its own question model
on top of QuestionBase and registers a QuestionType, so that all tasks
can eventually be rendered down to a common JSON shape.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SkipValidation

from weedvlm.types.render import RenderJob


class QuestionType(StrEnum):
    SPECIES_ID_MULTIPLE_CHOICE = "species_id_multiple_choice"
    SPECIES_ID_OPEN_ENDED = "species_id_open_ended"
    SPECIES_LOCALISATION = "species_localisation"
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
    # The class this question counts against when composing the final
    # benchmark (apps/benchmark-builder) -- e.g. the target species name
    # for species ID/localisation, "same_species"/"different_species" for
    # fine-grained ID, or the density category for density estimation.
    benchmark_class: str
    # The not-yet-drawn render behind image_path, for box-annotated
    # questions (None when image_path is the plain source image).
    # Rendered only after selection, via
    # weedvlm.pipeline.render.render_all; never serialised.
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
