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

from weedvlm.reasoning.config import ReasoningConfig, ModelConfig
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
        

def get_question_samples(results_file: str, runs: list[str], fraction: float, seed: int = 69420):
    # Verify the fraction
    results = []
    if fraction > 1.0:
        raise ValueError("fraction cannot be greater than one")
    elif fraction <= 0.0:
        raise ValueError("fraction cannot be less than zero")
    # We will only get the images as we need them
    df = pd.read_parquet(results_file)
    # Drop responses without reasoning
    df = df[df.reasoning.fillna("").str.strip() != ""]
    # Get weed species ID, get MCQ, and open-ended separately as their own analysis
    mcq = (
        df[
            (df.task == "species_id") & (df.question_type == "species_id_multiple_choice")
        ]
    )
    mcq_samples = (
        mcq[mcq["run"].isin(runs)]
        .groupby("run", group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    open_ended = (
        df[
            (df.task == "species_id") & (df.question_type == "species_id_open_ended")
        ]
    )
    open_ended_samples = (
        open_ended[open_ended["run"].isin(runs)]
        .groupby("run", group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    # Get fine-grained ID
    fine_grained = df[(df.task == "fine_grained")]
    fine_grained_samples = (
        fine_grained[fine_grained["run"].isin(runs)]
        .groupby("run", group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )

    # Get grounded VQA
    gvqa = df[(df.task == "grounded_vqa")]
    gvqa_samples = (
        gvqa[gvqa["run"].isin(runs)]
        .groupby("run", group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    # Get weed localisation, we don't have right and wrong questions here
    localisation = df[df.task == "localisation"].copy()
    localisation = localisation[localisation.run.isin(runs)]
    localisation["quality"] = pd.qcut( # Quantile based bins
        localisation["localisation_score"],
        q=3,
        labels=["low", "medium", "high"]
    )
    localisation_sample = (
        localisation
        .dropna(subset=["reasoning", "quality"])
        .groupby(["run", "quality"], observed=True, group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )

    # Get weed density estimation
    wde = df[(df.task == "density")]
    wde_samples = (
        wde[wde["run"].isin(runs)]
        .groupby("run", group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    results += to_items(mcq_samples)
    results += to_items(open_ended_samples)
    results += to_items(fine_grained_samples)
    results += to_items(gvqa_samples)
    results += to_items(localisation_sample)
    results += to_items(wde_samples)
    return results
