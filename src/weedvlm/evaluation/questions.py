"""
Loads the benchmark a run is evaluated against -- the questions of the
tasks it enables, and their images -- from the questions.parquet that
generate-all-questions.py writes (docs/DATABASE.md).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from weedvlm.evaluation.config import EvaluationConfig
from weedvlm.evaluation.prompts import load_density_config
from weedvlm.evaluation.store import ResponseRecord
from weedvlm.generation.config import DensityConfig
from weedvlm.storage import parquet
from weedvlm.types.questions import OpenEndedQuestion, QuestionBase


def _stratified_subset(
    task: str, questions: list[QuestionBase], fraction: float, seed: int
) -> list[QuestionBase]:
    """`fraction` of a task's questions, in their original order, taking
    the same share (rounded, at least one) of every (question type, image
    format, benchmark class) group.

    Each group is shuffled by a seed of its own and cut to a prefix, so
    the choice in one group doesn't depend on which other tasks are
    enabled, and a larger fraction keeps everything a smaller one did.
    Open-ended ablations are chosen first and the multiple-choice
    questions they ablate are then kept ahead of the rest of their group,
    so every ablation in the subset can still be compared with its
    original."""
    strata: dict[tuple[str, bool, str], list[QuestionBase]] = defaultdict(list)
    for question in questions:
        key = (question.question_type.value, question.annotated, question.benchmark_class)
        strata[key].append(question)

    keep: set[str] = set()
    ablated: set[str] = set()
    for key in sorted(strata, key=lambda k: not isinstance(strata[k][0], OpenEndedQuestion)):
        members = strata[key]
        random.Random(f"{seed}:{task}:{key}").shuffle(members)
        members.sort(key=lambda q: q.id not in ablated)  # stable: ablated ones first
        target = max(1, math.floor(len(members) * fraction + 0.5))
        for question in members[:target]:
            keep.add(question.id)
            if isinstance(question, OpenEndedQuestion):
                ablated.add(question.ablation_of)
    return [question for question in questions if question.id in keep]


class Benchmark:
    """The questions of a run's enabled tasks, as (task, question) in
    generation order, and access to their images. Opening one fails up
    front, naming the problem, if a task has no questions or any
    question's image is missing -- better than discovering it a few
    thousand paid requests in."""

    questions: list[tuple[str, QuestionBase]]

    def __init__(self, config: EvaluationConfig):
        self._config = config
        self._path = config.questions_file()
        enabled = [task for task, on in config.tasks.model_dump().items() if on]
        by_task = parquet.load_questions(self._path, enabled)
        empty = [task for task, questions in by_task.items() if not questions]
        if empty:
            raise ValueError(
                f"{self._path} has no questions for: {', '.join(empty)}. Generate them, or "
                "disable those tasks under `tasks`."
            )
        self._select(config, by_task)
        self._hashes = parquet.image_hashes(self._path)

        missing = sorted({str(q.image_path) for _, q in self.questions if not Path(q.image_path).exists()})
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} question image(s) don't exist, e.g. {missing[0]}. Were the "
                "questions moved or the rendered images deleted?"
            )

    def _select(self, config: EvaluationConfig, by_task: dict[str, list[QuestionBase]]) -> None:
        """Sets self.questions from every enabled task's full question
        list, applying the benchmark_fraction and then the
        max_questions_per_task subset."""
        rng = random.Random(config.seed)
        self.questions = []
        for task, parsed in by_task.items():
            if config.benchmark_fraction < 1:
                parsed = _stratified_subset(task, parsed, config.benchmark_fraction, config.seed)
            if config.max_questions_per_task is not None and len(parsed) > config.max_questions_per_task:
                keep = sorted(rng.sample(range(len(parsed)), config.max_questions_per_task))
                parsed = [parsed[i] for i in keep]
            self.questions.extend((task, question) for question in parsed)

    def read_image(self, question: QuestionBase) -> bytes:
        return Path(question.image_path).read_bytes()

    def image_hashes(self) -> dict[str, str]:
        """image_path -> sha256 of that image, for every question's image."""
        return {str(q.image_path): self._hashes[str(q.image_path)] for _, q in self.questions}

    def answer_is_current(self, question: QuestionBase, record: ResponseRecord) -> bool:
        """Whether a stored answer was given for the image the question
        has now (question ids are reused when the benchmark is
        regenerated, but the picture, and e.g. grounded VQA's box
        numbering, may differ)."""
        return record.image_sha256 == self._hashes[str(question.image_path)]

    def density_config(self) -> DensityConfig:
        """The density settings the questions were generated with."""
        return load_density_config(self._config.resolved_generation_config())

    def close(self) -> None:
        pass


def open_benchmark(config: EvaluationConfig) -> Benchmark:
    return Benchmark(config)
