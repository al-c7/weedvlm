"""
Runs every (response, judge) pair against OpenRouter with a fixed number
of requests in flight, retrying what's worth retrying, recording every
outcome as it lands, and resuming cleanly when re-run.

How failures are handled:
    - transient errors (rate limits, 5xx, timeouts) are retried with
      exponential backoff; a 429 also pauses every worker
    - a reply with no usable JSON answer is retried on a separate budget
    - a question that still fails is recorded as status "failed" with the
      reason, and the run carries on
    - a model that fails `model_failure_threshold` questions in a row is
      dropped for the rest of the run (its questions stay pending)
    - a bad API key or exhausted credits stops the whole run
    - Ctrl-C stops taking new work and lets in-flight requests finish;
      a second Ctrl-C abandons them
    Anything not recorded as "ok" is asked again on the next run.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import random
import signal
import copy
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import pandas as pd
import textwrap

from weedvlm.reasoning.config import ReasoningConfig, ModelConfig, SAMPLED_TASKS, resolve_wrong_per_task
from weedvlm.evaluation.openrouter import ChatReply, OpenRouterClient, RequestFailure
from weedvlm.evaluation.parsing import check_fields, extract_json_object
from weedvlm.evaluation.progress import ModelProgress, Progress

from dataclasses import dataclass, fields

@dataclass
class JudgeItem:
    response_id: int
    run: str
    model: str
    question_id: str
    task: str
    question_type: str
    question_text: str
    image_id: int
    reasoning: str
    answer: str | None = None
    expected_answer: str | None = None
    matched_iou: float | None = None
    localisation_score: float | None = None
    est_count : float | None = None
    est_coverage_pct: float | None = None
    true_weed_count: float | None = None
    true_weed_coverage_fraction: float | None = None
    count_error: float | None = None
    abs_count_error: float | None = None
    coverage_error_pp: float | None = None
    abs_coverage_error_pp: float | None = None
    quality: str | None = None
    choices: list[str] | None = None
    boxes: list[list[float]] | None = None
    target_species: str | None = None
    annotated: int | None = None
    match: bool | None = None
    benchmark_class: str | None = None
    pred_boxes: list[list[float]] | None = None
    bbox_format: str | None = None
    image_width: float | None = None
    image_height: float | None = None

    def __str__(self) -> str:
        lines = [
            f"[{self.task} / {self.question_type}] response {self.response_id}",
            f"  run:      {self.run} ({self.model})",
            f"  question: {self.question_id}",
            f"  image:    {self.image_id}",
            f"  text:     {self.question_text}",
        ]
        if self.answer is not None or self.expected_answer is not None:
            lines.append(f"  answer:   {self.answer!r} (expected {self.expected_answer!r})")

        # Task-specific scores: only the ones this item has.
        shown = {"response_id", "run", "model", "question_id", "task", "question_type",
                 "question_text", "image_id", "reasoning", "answer", "expected_answer"}
        extras = [
            f"{f.name}={value:g}" if isinstance(value, float) else f"{f.name}={value}"
            for f in fields(self)
            if f.name not in shown and (value := getattr(self, f.name)) is not None
        ]
        if extras:
            lines.append("  scores:   " + ", ".join(extras))

        lines.append("  reasoning:")
        lines.append(textwrap.indent(textwrap.fill(self.reasoning, width=88), "    "))
        lines.append("  original question text:")
        lines.append(textwrap.indent(textwrap.fill(self.question_text, width=88), "    "))
        return "\n".join(lines)

def convert_boxes(boxes: list[list[float]]) -> list[list[float]]:
    results = []
    for box in boxes:
        results.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return results

# Write up a function to prepare the questions to sample, going to just use parquet
JSON_COLUMN = ["choices", "boxes", "pred_boxes"]

def to_items(df: pd.DataFrame) -> list[JudgeItem]:
    names = [f.name for f in fields(JudgeItem)]
    df = df.rename(columns={"id": "response_id"})
    df = df.reindex(columns=names)
    df = df.astype(object).where(df.notna(), None)
    # Extract the choices, boxes, and pred_boxes from JSON strings
    for col in JSON_COLUMN:
        df[col] = df[col].map(lambda v: json.loads(v) if v is not None else None) # type: ignore
    # Convert boxes from (x y w h) -> (x1 y1 x2 y2)
    df["boxes"] = df["boxes"].map(lambda bxs: convert_boxes(bxs) if bxs is not None else None) # type: ignore
    return [JudgeItem(**row) for row in df.to_dict("records")] # type: ignore

def get_example(task: str, items: list[JudgeItem]) -> JudgeItem | None:
    shuffled_items = copy.deepcopy(items)
    random.shuffle(shuffled_items)
    for item in shuffled_items:
        if item.task == task:
            return item
    return None
        

def _sample_correct_and_wrong(
    group: pd.DataFrame, label: str, fraction: float, wrong_per_task: int, seed: int
) -> pd.DataFrame:
    """Per run: `fraction` of the correct responses plus exactly
    `wrong_per_task` incorrect ones, so every model contributes the same
    number of incorrect responses. Fails loudly if a run has too few."""
    correct = group[group.match.astype(bool)]
    wrong = group[~group.match.astype(bool)]
    picked = [
        correct.groupby("run", group_keys=False).sample(frac=fraction, random_state=seed)
    ]
    for run, run_wrong in wrong.groupby("run"):
        if len(run_wrong) < wrong_per_task:
            raise ValueError(
                f"{label}: run {run!r} has {len(run_wrong)} incorrect responses with "
                f"reasoning, fewer than wrong_per_task={wrong_per_task}"
            )
        picked.append(run_wrong.sample(n=wrong_per_task, random_state=seed))
    missing = set(group.run) - set(wrong.run)
    if wrong_per_task > 0 and missing:
        raise ValueError(f"{label}: runs {sorted(missing)} have no incorrect responses with reasoning")
    return pd.concat(picked)


def get_question_samples(
    results_file: str,
    runs: list[str],
    fraction: float,
    wrong_per_task: int | dict[str, int],
    tasks: set[str] | None = None,
    seed: int = 69420,
):
    """Responses for the rationale judges. For each task (species ID split
    into multiple choice and open-ended) and each run: `fraction` of the
    correct responses and exactly `wrong_per_task` incorrect ones.
    `wrong_per_task` is either one number for every task, or a dict with
    a number for each of SAMPLED_TASKS.
    Localisation has no right or wrong, so it is sampled `fraction` per
    run and score tercile instead."""
    if fraction > 1.0:
        raise ValueError("fraction cannot be greater than one")
    elif fraction <= 0.0:
        raise ValueError("fraction cannot be less than zero")
    wrong_counts = resolve_wrong_per_task(wrong_per_task)
    if tasks is not None and (unknown := tasks - {"species_id", "fine_grained", "density", "grounded_vqa", "localisation"}):
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    df = pd.read_parquet(results_file)
    df = df[df.run.isin(runs)]
    # Drop responses without reasoning
    df = df[df.reasoning.fillna("").str.strip() != ""]

    groups = {
        "species_id_multiple_choice": (df.task == "species_id") & (df.question_type == "species_id_multiple_choice"),
        "species_id_open_ended": (df.task == "species_id") & (df.question_type == "species_id_open_ended"),
        "fine_grained": df.task == "fine_grained",
        "grounded_vqa": df.task == "grounded_vqa",
        "density": df.task == "density",
    }
    results = []
    for label, mask in groups.items():
        if tasks is not None and label.split("_multiple_choice")[0].split("_open_ended")[0] not in tasks:
            continue
        sampled = _sample_correct_and_wrong(df[mask], label, fraction, wrong_counts[label], seed)
        results += to_items(sampled)

    # Weed localisation, we don't have right and wrong questions here
    if tasks is not None and "localisation" not in tasks:
        return results
    localisation = df[df.task == "localisation"].copy()
    localisation["quality"] = pd.qcut( # Quantile based bins
        localisation["localisation_score"],
        q=3,
        labels=["low", "medium", "high"]
    )
    localisation_sample = (
        localisation
        .dropna(subset=["quality"])
        .groupby(["run", "quality"], observed=True, group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    results += to_items(localisation_sample)
    return results
