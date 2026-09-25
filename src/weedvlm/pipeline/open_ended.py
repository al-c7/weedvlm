"""
Turns a random subset of already-generated MultipleChoiceQuestions into
open-ended ablations: same image and ground truth, but with the choice
list removed, so the gap between MC and open-ended accuracy indicates
how much of the MC score is the VLM guessing among the options rather
than actually knowing the answer.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from weedvlm.types.questions import MultipleChoiceQuestion, OpenEndedQuestion, QuestionType

_OPEN_ENDED_SUFFIX = (
    " If you don't know, say so instead of guessing."
)


def _open_ended_text(mc_question_text: str) -> str:
    return mc_question_text.rstrip("?") + "?" + _OPEN_ENDED_SUFFIX


def generate_open_ended_questions(
    mc_questions: Sequence[MultipleChoiceQuestion],
    *,
    question_type: QuestionType = QuestionType.SPECIES_ID_OPEN_ENDED,
    fraction: float = 0.2,
    rng: random.Random | None = None,
) -> list[OpenEndedQuestion]:
    rng = rng or random.Random()

    sample_size = round(len(mc_questions) * fraction)
    sampled = rng.sample(list(mc_questions), min(sample_size, len(mc_questions)))

    return [
        OpenEndedQuestion(
            id=f"{mc.id}:open_ended",
            question_type=question_type,
            question_text=_open_ended_text(mc.question_text),
            image_path=mc.image_path,
            render=mc.render,
            source=mc.source,
            annotated=mc.annotated,
            benchmark_class=mc.benchmark_class,
            answer_text=mc.choices[mc.answer_index],
            ablation_of=mc.id,
        )
        for mc in sampled
    ]
