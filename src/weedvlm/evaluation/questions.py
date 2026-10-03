"""
Loads the generated question files (the output of generate-all-questions.py)
for the tasks a run enables.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from weedvlm.evaluation.config import EvaluationConfig
from weedvlm.types.questions import (
    DensityEstimationQuestion,
    MultipleChoiceQuestion,
    OpenEndedQuestion,
    QuestionBase,
    QuestionType,
)

# Task toggle name -> the file generate-all-questions.py writes it to.
# species-id.json holds both the multiple-choice questions and their
# open-ended ablations.
TASK_FILES = {
    "species_id": "species-id.json",
    "fine_grained": "fine-grained.json",
    "density": "density.json",
    "grounded_vqa": "grounded-vqa.json",
}

_QUESTION_MODELS: dict[QuestionType, type[QuestionBase]] = {
    QuestionType.SPECIES_ID_MULTIPLE_CHOICE: MultipleChoiceQuestion,
    QuestionType.SPECIES_ID_OPEN_ENDED: OpenEndedQuestion,
    QuestionType.GROUNDED_VQA: MultipleChoiceQuestion,
    QuestionType.FINE_GRAINED_ID: MultipleChoiceQuestion,
    QuestionType.DENSITY_ESTIMATION: DensityEstimationQuestion,
}


def _parse(raw: dict) -> QuestionBase:
    question_type = QuestionType(raw["question_type"])
    try:
        model = _QUESTION_MODELS[question_type]
    except KeyError:
        raise ValueError(f"Question type {question_type} isn't supported for evaluation") from None
    return model.model_validate(raw)


def load_questions(config: EvaluationConfig) -> list[tuple[str, QuestionBase]]:
    """(task, question) for every question of every enabled task, in file
    order. Fails up front, naming the problem, if a task's file or any
    question's image is missing -- better than discovering it a few
    thousand paid requests in."""
    enabled = [task for task, on in config.tasks.model_dump().items() if on]
    rng = random.Random(config.seed)

    questions: list[tuple[str, QuestionBase]] = []
    for task in enabled:
        path = config.questions_dir / TASK_FILES[task]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found (task {task!r} is enabled). Run generate-all-questions.py "
                "first, or disable the task under `tasks`."
            )
        parsed = [_parse(raw) for raw in json.loads(path.read_text(encoding="utf-8"))]
        if config.max_questions_per_task is not None and len(parsed) > config.max_questions_per_task:
            keep = sorted(rng.sample(range(len(parsed)), config.max_questions_per_task))
            parsed = [parsed[i] for i in keep]
        questions.extend((task, question) for question in parsed)

    ids = [question.id for _, question in questions]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})[:3]
        raise ValueError(f"Question ids aren't unique across the loaded tasks, e.g. {duplicated}")

    missing = [q.image_path for _, q in questions if not Path(q.image_path).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} question image(s) don't exist, e.g. {missing[0]}. Were the "
            "questions moved or the rendered images deleted?"
        )
    return questions
