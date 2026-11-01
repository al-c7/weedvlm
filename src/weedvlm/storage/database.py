"""
The SQLite storage format: one database holding a benchmark (its questions
and the images they use) and every evaluation run made against it. The
schema is documented table by table in docs/DATABASE.md -- keep the two in
step, and bump SCHEMA_VERSION on any change to it.

This module owns the schema and the benchmark half (questions, images).
The results half (runs, responses, scores) is written through
weedvlm.evaluation.store.SqliteRunStore.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from weedvlm.types.questions import (
    DensityEstimationQuestion,
    LocalisationQuestion,
    MultipleChoiceQuestion,
    OpenEndedQuestion,
    QuestionBase,
    QuestionType,
    parse_question,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE images (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    sha256     TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    data       BLOB
);

CREATE TABLE questions (
    id                TEXT PRIMARY KEY,
    task              TEXT NOT NULL,
    position          INTEGER NOT NULL,
    question_type     TEXT NOT NULL,
    question_text     TEXT NOT NULL,
    image_id          INTEGER NOT NULL REFERENCES images(id),
    annotated         INTEGER NOT NULL CHECK (annotated IN (0, 1)),
    benchmark_class   TEXT NOT NULL,
    dataset_name      TEXT NOT NULL,
    source_image_id   INTEGER NOT NULL,
    annotation_ids    TEXT NOT NULL,
    expected_answer   TEXT,
    choices           TEXT,
    answer_index      INTEGER,
    ablation_of       TEXT REFERENCES questions(id),
    true_weed_count   INTEGER,
    true_weed_coverage_fraction REAL,
    target_species    TEXT,
    image_width       INTEGER,
    image_height      INTEGER,
    boxes             TEXT,
    UNIQUE (task, position)
);
CREATE INDEX questions_image ON questions(image_id);

CREATE TABLE runs (
    name               TEXT PRIMARY KEY,
    prompt_version     INTEGER NOT NULL,
    config             TEXT NOT NULL,
    model_fingerprints TEXT NOT NULL,
    sessions           TEXT NOT NULL
);

CREATE TABLE responses (
    id                INTEGER PRIMARY KEY,
    run               TEXT NOT NULL REFERENCES runs(name) ON DELETE CASCADE,
    model             TEXT NOT NULL,
    question_id       TEXT NOT NULL REFERENCES questions(id),
    status            TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    answer            TEXT,
    reasoning         TEXT,
    parsed            TEXT,
    raw_response      TEXT,
    warnings          TEXT NOT NULL,
    error_kind        TEXT,
    error_message     TEXT,
    error_http_status INTEGER,
    attempts          INTEGER NOT NULL,
    retry_log         TEXT NOT NULL,
    finish_reason     TEXT,
    latency_seconds   REAL,
    usage             TEXT,
    total_tokens      INTEGER,
    cost_usd          REAL,
    generation_id     TEXT,
    provider          TEXT,
    image_sha256      TEXT,
    prompt_version    INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    UNIQUE (run, model, question_id)
);
CREATE INDEX responses_question ON responses(question_id);

CREATE TABLE scores (
    response_id         INTEGER PRIMARY KEY REFERENCES responses(id) ON DELETE CASCADE,
    suite               TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    predicted_label     TEXT,
    chance              REAL,
    true_count          INTEGER,
    est_count           REAL,
    count_error         REAL,
    true_coverage_pct   REAL,
    est_coverage_pct    REAL,
    coverage_error_pp   REAL,
    category_rank_error INTEGER,
    true_box_count      INTEGER,
    pred_box_count      INTEGER,
    box_count_error     INTEGER,
    matched_iou         REAL,
    localisation_score  REAL,
    bbox_format         TEXT,
    pred_boxes          TEXT
);

CREATE TABLE scoring_runs (
    run                       TEXT PRIMARY KEY REFERENCES runs(name) ON DELETE CASCADE,
    scored_at                 TEXT NOT NULL,
    aliases_file              TEXT,
    skipped_missing_questions INTEGER NOT NULL,
    skipped_changed_images    INTEGER NOT NULL,
    metrics                   TEXT NOT NULL
);

CREATE VIEW response_details AS
SELECT
    r.id AS response_id, r.run, r.model, q.id AS question_id, q.task, s.suite,
    q.question_type, q.annotated, q.dataset_name, q.benchmark_class, q.question_text,
    q.choices, q.expected_answer, r.status, r.answer, r.reasoning, s.outcome,
    s.predicted_label, r.error_kind, r.error_message, r.cost_usd, r.latency_seconds,
    q.image_id, i.path AS image_path
FROM responses r
JOIN questions q ON q.id = r.question_id
JOIN images i ON i.id = q.image_id
LEFT JOIN scores s ON s.response_id = r.id;

CREATE VIEW suite_summary AS
SELECT
    r.run, r.model, s.suite,
    COUNT(*) AS n_total,
    SUM(r.status = 'ok') AS n_answered,
    SUM(s.outcome = 'correct') AS correct,
    SUM(s.outcome = 'incorrect') AS incorrect,
    SUM(s.outcome = 'unsure') AS unsure,
    SUM(s.outcome = 'invalid') AS invalid,
    SUM(s.outcome = 'correct') * 1.0 / NULLIF(SUM(r.status = 'ok'), 0) AS accuracy,
    SUM(s.outcome = 'correct') * 1.0 / COUNT(*) AS accuracy_all,
    AVG(s.localisation_score) AS mean_localisation_score
FROM scores s
JOIN responses r ON r.id = s.response_id
GROUP BY r.run, r.model, s.suite;
"""


class BenchmarkInUseError(RuntimeError):
    """Raised instead of replacing a benchmark that runs have already been
    made against: their responses would be left pointing at questions
    that no longer exist, or at a different answer key."""


def connect(path: Path, *, create: bool = False) -> sqlite3.Connection:
    """Opens a benchmark database. With create, a missing (or empty) one
    is made with the current schema; otherwise it must already be one."""
    if not path.exists():
        if not create:
            raise FileNotFoundError(
                f"{path} not found. Generate the benchmark with storage format 'sqlite', or "
                "bring an existing one over with migrate-to-sqlite.py."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    # Shared between the event loop and the threads that read images;
    # callers serialise their own use (see SqliteBenchmark).
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Lets the database be read (e.g. for manual review) while a run is
    # writing to it.
    conn.execute("PRAGMA journal_mode = WAL")

    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone() is None:
        if not create or conn.execute("SELECT 1 FROM sqlite_master").fetchone() is not None:
            conn.close()
            raise ValueError(f"{path} isn't a weedvlm benchmark database")
        conn.executescript(_SCHEMA)
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()

    version = get_meta(conn, "schema_version")
    if version != str(SCHEMA_VERSION):
        conn.close()
        raise ValueError(
            f"{path} has schema version {version}, but this code reads version {SCHEMA_VERSION}"
        )
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# -- writing a benchmark -----------------------------------------------


def check_replaceable(conn: sqlite3.Connection) -> None:
    """Raises BenchmarkInUseError if the database's questions have been
    answered by any run."""
    if conn.execute("SELECT 1 FROM responses LIMIT 1").fetchone() is not None:
        raise BenchmarkInUseError(
            "This database already holds responses to its current questions, so the questions "
            "can't be replaced. Write the new benchmark to a different database, or delete the "
            "runs first (DELETE FROM runs)."
        )


def _expected_answer(question: QuestionBase) -> str | None:
    match question:
        case OpenEndedQuestion():
            return question.answer_text
        case DensityEstimationQuestion():
            return question.density_category
        case MultipleChoiceQuestion():
            return question.choices[question.answer_index]
    return None  # localisation: the answer is `boxes`


def _question_row(task: str, position: int, question: QuestionBase, image_id: int) -> dict:
    row = {
        "id": question.id,
        "task": task,
        "position": position,
        "question_type": question.question_type.value,
        "question_text": question.question_text,
        "image_id": image_id,
        "annotated": int(question.annotated),
        "benchmark_class": question.benchmark_class,
        "dataset_name": question.source.dataset_name,
        "source_image_id": question.source.image_id,
        "annotation_ids": json.dumps(question.source.annotation_ids),
        "expected_answer": _expected_answer(question),
        "choices": None,
        "answer_index": None,
        "ablation_of": None,
        "true_weed_count": None,
        "true_weed_coverage_fraction": None,
        "target_species": None,
        "image_width": None,
        "image_height": None,
        "boxes": None,
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


def write_benchmark(
    conn: sqlite3.Connection,
    questions: Sequence[tuple[str, QuestionBase]],
    *,
    generation_config: str | None,
    store_images: bool,
) -> int:
    """Replaces the database's benchmark with the given (task, question)
    pairs and the images they use -- only those images, so nothing that
    was rendered but not selected is ever stored. Returns the number of
    images. All or nothing: a failure leaves the database as it was."""
    check_replaceable(conn)

    ids = [question.id for _, question in questions]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})[:3]
        raise ValueError(f"Question ids aren't unique across tasks, e.g. {duplicated}")

    # Ablations refer to the question they were made from, which can sit
    # anywhere in the list, so that reference is checked at commit.
    conn.execute("PRAGMA defer_foreign_keys = ON")
    with conn:
        conn.execute("DELETE FROM questions")
        conn.execute("DELETE FROM images")

        image_ids: dict[str, int] = {}
        positions: dict[str, int] = {}
        for task, question in questions:
            path = str(question.image_path)
            if path not in image_ids:
                data = Path(path).read_bytes()
                image_ids[path] = conn.execute(
                    "INSERT INTO images (path, sha256, media_type, size_bytes, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        path,
                        hashlib.sha256(data).hexdigest(),
                        mimetypes.guess_type(path)[0] or "image/jpeg",
                        len(data),
                        data if store_images else None,
                    ),
                ).lastrowid
            position = positions.get(task, 0)
            positions[task] = position + 1
            row = _question_row(task, position, question, image_ids[path])
            conn.execute(
                f"INSERT INTO questions ({', '.join(row)}) "
                f"VALUES ({', '.join(':' + column for column in row)})",
                row,
            )

        conn.execute("DELETE FROM meta WHERE key = 'generation_config'")
        if generation_config is not None:
            set_meta(conn, "generation_config", generation_config)
        set_meta(conn, "benchmark_written_at", datetime.now(UTC).isoformat())
        set_meta(conn, "images_stored", str(int(store_images)))

    # The images go through the write-ahead log, which would otherwise
    # stay as large as they are.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return len(image_ids)


# -- reading a benchmark -----------------------------------------------


def _question_from_row(row: sqlite3.Row) -> QuestionBase:
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


def load_questions(conn: sqlite3.Connection, tasks: Iterable[str]) -> dict[str, list[QuestionBase]]:
    """Each given task's questions, in the order they were generated."""
    by_task: dict[str, list[QuestionBase]] = {task: [] for task in tasks}
    rows = conn.execute(
        "SELECT q.*, i.path AS image_path FROM questions q JOIN images i ON i.id = q.image_id "
        "ORDER BY q.task, q.position"
    )
    for row in rows:
        if row["task"] in by_task:
            by_task[row["task"]].append(_question_from_row(row))
    return by_task


def image_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """path -> sha256 of every image in the benchmark."""
    return {row["path"]: row["sha256"] for row in conn.execute("SELECT path, sha256 FROM images")}


def unstored_image_paths(conn: sqlite3.Connection) -> list[str]:
    """Paths of the images whose bytes aren't in the database, and so
    still have to be read from disk."""
    return [row["path"] for row in conn.execute("SELECT path FROM images WHERE data IS NULL")]


def read_image(conn: sqlite3.Connection, path: str) -> bytes:
    """An image's bytes: from the database if they were stored, else from
    the file at its path."""
    row = conn.execute("SELECT data FROM images WHERE path = ?", (path,)).fetchone()
    if row is None:
        raise KeyError(f"{path} isn't an image of this benchmark")
    return row["data"] if row["data"] is not None else Path(path).read_bytes()
