"""
Scores stored responses against the ground truth in the question files.

Each response is joined to its question by id and given an `outcome`:

    correct    the answer matches the ground truth
    incorrect  a valid answer, but the wrong one
    unsure     the model declined ("Unsure") rather than guessing
    invalid    the answer isn't one of the options (or, for density, not a category)
    failed     no usable response was obtained (see the run's run.log)

Species-ID open-ended answers are free text, so they're matched by
normalised name (see `open_ended_correct`), with an optional alias table
for common/scientific-name variants (docs/SPECIES-ID.md).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weedvlm.evaluation.prompts import UNSURE_ANSWER
from weedvlm.evaluation.store import ResponseRecord
from weedvlm.types.questions import (
    DensityEstimationQuestion,
    MultipleChoiceQuestion,
    OpenEndedQuestion,
    QuestionBase,
    QuestionType,
)

DENSITY_CATEGORIES = ["none", "low", "medium", "high"]

_UNSURE = {
    UNSURE_ANSWER.casefold(),
    "unknown",
    "not sure",
    "i don't know",
    "i do not know",
    "don't know",
    "cannot tell",
    "can't tell",
}


@dataclass
class ScoredResponse:
    question_id: str
    model: str
    task: str
    suite: str
    question_type: str
    annotated: bool
    dataset: str
    benchmark_class: str
    ablation_of: str | None
    status: str
    outcome: str
    expected: str
    # The model's answer as given, and (for closed-set questions) the
    # option/category it maps to, None if it maps to none of them.
    answer: str | None
    predicted_label: str | None
    # Random-guess accuracy for this question (None for open-ended).
    chance: float | None
    # Density only. Errors are estimate minus truth.
    true_count: int | None = None
    est_count: float | None = None
    count_error: float | None = None
    true_coverage_pct: float | None = None
    est_coverage_pct: float | None = None
    coverage_error_pp: float | None = None
    category_rank_error: int | None = None
    cost_usd: float | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    attempts: int | None = None


# -- text normalisation ------------------------------------------------


def _light(text: Any) -> str:
    """Case, whitespace and surrounding quote/full-stop insensitive form."""
    return re.sub(r"\s+", " ", str(text).strip().strip("\"'`.").strip()).casefold()


def _is_unsure(text: str) -> bool:
    return text in _UNSURE


def _label(answer: Any) -> str:
    """A grounded-VQA label: 2, 2.0, "2", "Box 2" and "#2" all mean "2"."""
    if isinstance(answer, int | float) and not isinstance(answer, bool):
        return str(int(answer)) if float(answer).is_integer() else str(answer)
    text = _light(answer)
    match = re.fullmatch(r"(?:bounding box|box|label|number)?\s*#?\s*(\d+)(?:\.0+)?", text)
    return match.group(1) if match else text


def _number(value: Any) -> float | None:
    """A model's numeric estimate: 12, 12.5 or "12%"; not booleans or junk."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%").strip())
        except ValueError:
            return None
    return None


def open_ended_correct(answer: str, truth: str, aliases: Mapping[str, list[str]]) -> bool:
    """Whether a free-text species answer names the ground-truth species.

    Matches if the answer, or any part of it outside/inside parentheses or
    comma-separated ("Common purslane (Portulaca oleracea)"), equals the
    truth or one of its aliases -- or contains one as a whole phrase
    ("Eclipta prostrata" for "Eclipta"). An answer that hedges between
    options ("A or B", "A/B") only matches by an exact part, never by
    containment. Less specific answers ("Amaranthus" for "Amaranthus
    tuberculatus") don't match."""
    accepted = {_light(truth), *(_light(a) for a in aliases.get(_light(truth), []))}
    text = _light(answer)
    parts = {text, *(p.strip() for p in re.split(r"[()\[\],;]", text) if p.strip())}
    if parts & accepted:
        return True
    if " or " in text or "/" in text:
        return False
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text) for a in accepted)


def load_aliases(raw: Mapping[str, list[str]] | None) -> dict[str, list[str]]:
    """{ground-truth name: [accepted alternatives]}, keys case-folded."""
    return {_light(name): list(alts) for name, alts in (raw or {}).items()}


# -- per-question scoring ----------------------------------------------


def suite_name(task: str, question: QuestionBase) -> str:
    """The group a question is reported in: each task under each image
    condition and question format is evaluated separately."""
    condition = "boxed" if question.annotated else "unannotated"
    match question:
        case OpenEndedQuestion():
            return f"species_id/open_ended/{condition}"
        case DensityEstimationQuestion():
            return f"density/{condition}"
    if question.question_type == QuestionType.GROUNDED_VQA:
        # The crop baseline is rendered to *_gvqa_crop.jpg (pipeline.grounded_vqa).
        return "grounded_vqa/crop_baseline" if "_gvqa_crop" in Path(question.image_path).name else "grounded_vqa/weed"
    if question.question_type == QuestionType.FINE_GRAINED_ID:
        return "fine_grained"
    return f"species_id/multiple_choice/{condition}"


def _base(task: str, question: QuestionBase, record: ResponseRecord) -> dict[str, Any]:
    return {
        "question_id": question.id,
        "model": record.model,
        "task": task,
        "suite": suite_name(task, question),
        "question_type": question.question_type.value,
        "annotated": question.annotated,
        "dataset": question.source.dataset_name,
        "benchmark_class": question.benchmark_class,
        "ablation_of": question.ablation_of if isinstance(question, OpenEndedQuestion) else None,
        "status": record.status,
        "cost_usd": record.cost_usd,
        "total_tokens": (record.usage or {}).get("total_tokens"),
        "latency_seconds": record.latency_seconds,
        "attempts": record.attempts,
    }


def _expected(question: QuestionBase) -> str:
    match question:
        case OpenEndedQuestion():
            return question.answer_text
        case DensityEstimationQuestion():
            return question.density_category
        case MultipleChoiceQuestion():
            return question.choices[question.answer_index]
    raise ValueError(f"Can't score {question.question_type}")


def _chance(question: QuestionBase) -> float | None:
    match question:
        case OpenEndedQuestion():
            return None
        case DensityEstimationQuestion():
            return 1 / len(question.density_choices)
        case MultipleChoiceQuestion():
            return 1 / len(question.choices)
    return None


def _score_choice(question: MultipleChoiceQuestion, answer: Any) -> tuple[str, str | None]:
    """(outcome, matched option) for a multiple-choice answer."""
    grounded_vqa = question.question_type == QuestionType.GROUNDED_VQA
    given = _label(answer) if grounded_vqa else _light(answer)
    options = {(_label(c) if grounded_vqa else _light(c)): c for c in question.choices}
    if given in options:
        matched = options[given]
        return ("correct" if matched == question.choices[question.answer_index] else "incorrect"), matched
    if not grounded_vqa and _is_unsure(given):
        return "unsure", None
    return "invalid", None


def _score_density(question: DensityEstimationQuestion, parsed: dict[str, Any]) -> dict[str, Any]:
    true_pct = question.true_weed_coverage_fraction * 100
    result: dict[str, Any] = {
        "true_count": question.true_weed_count,
        "true_coverage_pct": true_pct,
    }
    given = _light(parsed["answer"])
    if given in DENSITY_CATEGORIES:
        result["predicted_label"] = given
        result["outcome"] = "correct" if given == question.density_category else "incorrect"
        result["category_rank_error"] = abs(
            DENSITY_CATEGORIES.index(given) - DENSITY_CATEGORIES.index(question.density_category)
        )
    else:
        result["predicted_label"] = None
        result["outcome"] = "invalid"
    if (count := _number(parsed.get("estimated_weed_count"))) is not None:
        result["est_count"] = count
        result["count_error"] = count - question.true_weed_count
    if (coverage := _number(parsed.get("estimated_weed_coverage_percent"))) is not None:
        result["est_coverage_pct"] = coverage
        result["coverage_error_pp"] = coverage - true_pct
    return result


def score_response(
    task: str,
    question: QuestionBase,
    record: ResponseRecord,
    aliases: Mapping[str, list[str]],
) -> ScoredResponse:
    fields = _base(task, question, record)
    fields |= {
        "expected": _expected(question),
        "chance": _chance(question),
        "answer": None,
        "predicted_label": None,
    }
    if record.status != "ok" or record.parsed is None:
        return ScoredResponse(**fields, outcome="failed")

    answer = record.parsed["answer"]
    fields["answer"] = str(answer)

    match question:
        case OpenEndedQuestion():
            if _is_unsure(_light(answer)):
                outcome = "unsure"
            else:
                outcome = (
                    "correct" if open_ended_correct(str(answer), question.answer_text, aliases) else "incorrect"
                )
            return ScoredResponse(**fields, outcome=outcome)
        case DensityEstimationQuestion():
            return ScoredResponse(**(fields | _score_density(question, record.parsed)))
        case MultipleChoiceQuestion():
            outcome, matched = _score_choice(question, answer)
            fields["predicted_label"] = matched
            return ScoredResponse(**fields, outcome=outcome)
    raise ValueError(f"Can't score {question.question_type}")


# -- joining a run to its questions ------------------------------------


def _sha256(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass
class JoinResult:
    scored: list[ScoredResponse]
    # Responses that can't be scored against the current question files:
    # the question is gone, or its image is no longer the one the model
    # was shown (so e.g. grounded VQA's box numbering may not match the
    # answer key).
    missing_questions: list[str]
    changed_images: list[str]


def score_run(
    records: Mapping[tuple[str, str], ResponseRecord],
    questions: list[tuple[str, QuestionBase]],
    aliases: Mapping[str, list[str]],
    *,
    threads: int = 8,
) -> JoinResult:
    by_id = {q.id: (task, q) for task, q in questions}

    current_hash: dict[str, str | None] = {}
    with ThreadPoolExecutor(threads) as pool:
        paths = sorted({str(q.image_path) for _, q in questions})
        current_hash = dict(zip(paths, pool.map(_sha256, paths), strict=True))

    scored, missing, changed = [], [], []
    for (model, question_id), record in records.items():
        if question_id not in by_id:
            missing.append(question_id)
            continue
        task, question = by_id[question_id]
        if record.image_sha256 and current_hash[str(question.image_path)] != record.image_sha256:
            changed.append(question_id)
            continue
        scored.append(score_response(task, question, record, aliases))
    return JoinResult(scored, sorted(set(missing)), sorted(set(changed)))
