"""
Loads the benchmark a run is evaluated against -- the questions of the
tasks it enables, and their images -- from whichever storage format the
run's config names: the JSON question files generate-all-questions.py
writes, or a SQLite database (docs/DATABASE.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

from weedvlm.evaluation.config import EvaluationConfig
from weedvlm.evaluation.prompts import load_density_config
from weedvlm.evaluation.store import ResponseRecord
from weedvlm.generation.config import DensityConfig, PipelineConfig
from weedvlm.storage import database
from weedvlm.types.questions import TASK_FILES, OpenEndedQuestion, QuestionBase, parse_question


def _sha256(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


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
    front, naming the problem, if a task's questions or any question's
    image is missing -- better than discovering it a few thousand paid
    requests in."""

    questions: list[tuple[str, QuestionBase]]

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

        ids = [question.id for _, question in self.questions]
        if len(set(ids)) != len(ids):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})[:3]
            raise ValueError(f"Question ids aren't unique across the loaded tasks, e.g. {duplicated}")

    def read_image(self, question: QuestionBase) -> bytes:
        raise NotImplementedError

    def image_hashes(self) -> dict[str, str | None]:
        """image_path -> the sha256 of that image as it is now (None if
        it can't be read), for every question's image."""
        raise NotImplementedError

    def answer_is_current(self, question: QuestionBase, record: ResponseRecord) -> bool:
        """Whether a stored answer was given for the image the question
        has now (question ids are reused when the benchmark is
        regenerated, but the picture, and e.g. grounded VQA's box
        numbering, may differ)."""
        raise NotImplementedError

    def density_config(self) -> DensityConfig:
        """The density settings the questions were generated with."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonBenchmark(Benchmark):
    def __init__(self, config: EvaluationConfig):
        self._config = config
        by_task: dict[str, list[QuestionBase]] = {}
        for task, on in config.tasks.model_dump().items():
            if not on:
                continue
            path = config.questions_dir / TASK_FILES[task]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found (task {task!r} is enabled). Run generate-all-questions.py "
                    "first, or disable the task under `tasks`."
                )
            by_task[task] = [
                parse_question(raw) for raw in json.loads(path.read_text(encoding="utf-8"))
            ]
        self._select(config, by_task)

        missing = [q.image_path for _, q in self.questions if not Path(q.image_path).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} question image(s) don't exist, e.g. {missing[0]}. Were the "
                "questions moved or the rendered images deleted?"
            )

    def read_image(self, question: QuestionBase) -> bytes:
        return Path(question.image_path).read_bytes()

    def image_hashes(self) -> dict[str, str | None]:
        paths = sorted({str(q.image_path) for _, q in self.questions})
        with ThreadPoolExecutor(8) as pool:
            return dict(zip(paths, pool.map(_sha256, paths), strict=True))

    def answer_is_current(self, question: QuestionBase, record: ResponseRecord) -> bool:
        # By modification time rather than hash: this is asked of every
        # stored answer before a run starts, and hashing every image
        # would take longer than most resumed runs have left to do.
        try:
            modified = datetime.fromtimestamp(Path(question.image_path).stat().st_mtime, UTC)
        except OSError:
            return False
        return modified <= datetime.fromisoformat(record.created_at)

    def density_config(self) -> DensityConfig:
        return load_density_config(self._config.resolved_generation_config())


class SqliteBenchmark(Benchmark):
    def __init__(self, config: EvaluationConfig):
        self._config = config
        self._path = config.resolved_database()
        self._conn = database.connect(self._path)
        # Images are read from worker threads (see runner._read_image).
        self._lock = threading.Lock()

        enabled = [task for task, on in config.tasks.model_dump().items() if on]
        by_task = database.load_questions(self._conn, enabled)
        empty = [task for task, questions in by_task.items() if not questions]
        if empty:
            raise ValueError(
                f"{self._path} has no questions for: {', '.join(empty)}. Generate them into it, "
                "or disable those tasks under `tasks`."
            )
        self._select(config, by_task)
        self._hashes = database.image_hashes(self._conn)

        used = {str(q.image_path) for _, q in self.questions}
        missing = [
            path
            for path in database.unstored_image_paths(self._conn)
            if path in used and not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} question image(s) aren't stored in {self._path} and don't exist "
                f"on disk either, e.g. {missing[0]}."
            )

    def read_image(self, question: QuestionBase) -> bytes:
        with self._lock:
            return database.read_image(self._conn, str(question.image_path))

    def image_hashes(self) -> dict[str, str | None]:
        return {str(q.image_path): self._hashes[str(q.image_path)] for _, q in self.questions}

    def answer_is_current(self, question: QuestionBase, record: ResponseRecord) -> bool:
        return record.image_sha256 == self._hashes[str(question.image_path)]

    def density_config(self) -> DensityConfig:
        if self._config.generation_config is not None:
            return load_density_config(self._config.generation_config)
        text = database.get_meta(self._conn, "generation_config")
        if text is None:
            raise ValueError(
                f"{self._path} doesn't record the config its questions were generated with. The "
                "density prompt explains the category rules using those thresholds; point "
                "`generation_config` at the config that produced them."
            )
        return PipelineConfig.model_validate(yaml.safe_load(text) or {}).density

    def close(self) -> None:
        self._conn.close()


def open_benchmark(config: EvaluationConfig) -> Benchmark:
    if config.storage.format == "sqlite":
        return SqliteBenchmark(config)
    return JsonBenchmark(config)
