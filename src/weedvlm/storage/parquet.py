"""
The benchmark's storage: one parquet file of questions, with the images
left as files on disk (see docs/DATABASE.md for the layout and columns).

    questions/questions.parquet   one row per question, in generation order
    questions/rendered/...        the images the questions use

Columns holding lists (choices, boxes, annotation_ids) are JSON text, so
they read the same from pandas, polars and duckdb. Each row carries its
image's path and sha256, so a response can tell whether the image it
answered is still the one the question uses.

Responses are written per run by weedvlm.evaluation.store.RunStore.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from weedvlm.types.questions import (
    DensityEstimationQuestion,
    LocalisationQuestion,
    MultipleChoiceQuestion,
    OpenEndedQuestion,
    QuestionBase,
    QuestionType,
    parse_question,
)

QUESTIONS_FILE = "questions.parquet"

QUESTION_SCHEMA = {
    "id": pl.String,
    "task": pl.String,
    "position": pl.Int64,
    "question_type": pl.String,
    "question_text": pl.String,
    "image_path": pl.String,
    "image_sha256": pl.String,
    "annotated": pl.Boolean,
    "benchmark_class": pl.String,
    "dataset_name": pl.String,
    "source_image_id": pl.Int64,
    "annotation_ids": pl.String,
    "expected_answer": pl.String,
    "choices": pl.String,
    "answer_index": pl.Int64,
    "ablation_of": pl.String,
    "true_weed_count": pl.Int64,
    "true_weed_coverage_fraction": pl.Float64,
    "target_species": pl.String,
    "image_width": pl.Int64,
    "image_height": pl.Int64,
    "boxes": pl.String,
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Writes `frame` to `path` so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, path)


# -- writing -----------------------------------------------------------


def _expected_answer(question: QuestionBase) -> str | None:
    match question:
        case OpenEndedQuestion():
            return question.answer_text
        case DensityEstimationQuestion():
            return question.density_category
        case MultipleChoiceQuestion():
            return question.choices[question.answer_index]
    return None  # localisation: the answer is `boxes`


def question_row(task: str, position: int, question: QuestionBase, image_sha256: str) -> dict:
    row = dict.fromkeys(QUESTION_SCHEMA) | {
        "id": question.id,
        "task": task,
        "position": position,
        "question_type": question.question_type.value,
        "question_text": question.question_text,
        "image_path": str(question.image_path),
        "image_sha256": image_sha256,
        "annotated": question.annotated,
        "benchmark_class": question.benchmark_class,
        "dataset_name": question.source.dataset_name,
        "source_image_id": question.source.image_id,
        "annotation_ids": json.dumps(question.source.annotation_ids),
        "expected_answer": _expected_answer(question),
    }
    match question:
        case OpenEndedQuestion():
            row["ablation_of"] = question.ablation_of
        case DensityEstimationQuestion():
            row["choices"] = json.dumps(question.density_choices)
            row["true_weed_count"] = question.true_weed_count
            row["true_weed_coverage_fraction"] = question.true_weed_coverage_fraction
        case LocalisationQuestion():
            row["target_species"] = question.target_species
            row["image_width"] = question.image_width
            row["image_height"] = question.image_height
            row["boxes"] = json.dumps(question.boxes)
        case MultipleChoiceQuestion():
            row["choices"] = json.dumps(question.choices)
            row["answer_index"] = question.answer_index
        case _:
            raise ValueError(f"Can't store question type {question.question_type}")
    return row


def check_rows(rows: Sequence[dict]) -> None:
    """The checks a database would make: unique ids, and every ablation
    pointing at a question that exists."""
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})[:3]
        raise ValueError(f"Question ids aren't unique across tasks, e.g. {duplicated}")
    known = set(ids)
    dangling = [row["id"] for row in rows if row["ablation_of"] and row["ablation_of"] not in known]
    if dangling:
        raise ValueError(f"{len(dangling)} ablation(s) refer to a missing question, e.g. {dangling[0]}")


def write_questions(path: Path, questions: Sequence[tuple[str, QuestionBase]]) -> int:
    """Replaces the benchmark at `path` with the given (task, question)
    pairs, in that order. Hashes every image the questions use (which must
    exist). Returns the number of distinct images."""
    paths = sorted({str(question.image_path) for _, question in questions})
    with ThreadPoolExecutor(8) as pool:
        hashes = dict(zip(paths, pool.map(sha256_file, paths), strict=True))

    positions: dict[str, int] = {}
    rows = []
    for task, question in questions:
        position = positions.get(task, 0)
        positions[task] = position + 1
        rows.append(question_row(task, position, question, hashes[str(question.image_path)]))
    check_rows(rows)
    write_atomic(pl.DataFrame(rows, schema=QUESTION_SCHEMA), path)
    return len(paths)


# -- reading -----------------------------------------------------------


def question_from_row(row: dict) -> QuestionBase:
    raw = {
        "id": row["id"],
        "question_type": row["question_type"],
        "question_text": row["question_text"],
        "image_path": row["image_path"],
        "source": {
            "dataset_name": row["dataset_name"],
            "image_id": row["source_image_id"],
            "annotation_ids": json.loads(row["annotation_ids"]),
        },
        "annotated": bool(row["annotated"]),
        "benchmark_class": row["benchmark_class"],
    }
    match QuestionType(row["question_type"]):
        case QuestionType.SPECIES_ID_OPEN_ENDED:
            raw |= {"answer_text": row["expected_answer"], "ablation_of": row["ablation_of"]}
        case QuestionType.DENSITY_ESTIMATION:
            raw |= {
                "density_choices": json.loads(row["choices"]),
                "true_weed_count": row["true_weed_count"],
                "true_weed_coverage_fraction": row["true_weed_coverage_fraction"],
                "density_category": row["expected_answer"],
            }
        case QuestionType.LOCALISATION:
            raw |= {
                "target_species": row["target_species"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "boxes": json.loads(row["boxes"]),
            }
        case _:
            raw |= {"choices": json.loads(row["choices"]), "answer_index": row["answer_index"]}
    return parse_question(raw)


def _read(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Generate the benchmark with generate-all-questions.py.")
    return pl.read_parquet(path)


def load_questions(path: Path, tasks: Iterable[str]) -> dict[str, list[QuestionBase]]:
    """Each given task's questions, in the order they were generated."""
    by_task: dict[str, list[QuestionBase]] = {task: [] for task in tasks}
    frame = _read(path).filter(pl.col("task").is_in(list(by_task))).sort("task", "position")
    for row in frame.iter_rows(named=True):
        by_task[row["task"]].append(question_from_row(row))
    return by_task


def image_hashes(path: Path) -> dict[str, str]:
    """image path -> sha256 recorded when the benchmark was written."""
    frame = _read(path).select("image_path", "image_sha256").unique()
    return dict(frame.iter_rows())


def read_image(path: str | Path) -> tuple[bytes, str]:
    """An image's bytes and media type."""
    return Path(path).read_bytes(), mimetypes.guess_type(str(path))[0] or "image/jpeg"
