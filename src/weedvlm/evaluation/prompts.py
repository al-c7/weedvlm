"""
Builds the text of the prompt for each question. The image itself is
attached separately at request time (see weedvlm.evaluation.runner).

Every task uses the same response contract: one JSON object whose first
field, `answer`, is the exact answer (an option verbatim for multiple
choice, a species name for open-ended) and whose second, `reasoning`, is
at most two sentences on why. Density additionally asks for its two numeric
estimates, since the question itself asks for them and the task is scored
on them (docs/WEED-DENSITY.md).

Bump PROMPT_VERSION on any change that alters what a model is sent, so a
resumed run refuses to mix answers from two different prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from weedvlm.generation.config import DensityConfig, PipelineConfig
from weedvlm.generation.density import classify_density, density_score
from weedvlm.types.questions import (
    DensityEstimationQuestion,
    MultipleChoiceQuestion,
    OpenEndedQuestion,
    QuestionBase,
    QuestionType,
)

PROMPT_VERSION = 1

# Offered on top of the generated options (docs/SPECIES-ID.md,
# docs/FINEGRAINED-ID.md: the VLM may decline rather than guess).
UNSURE_ANSWER = "Unsure"

SYSTEM_PROMPT = """\
You are an expert in weed identification and agronomy, helping to evaluate images taken in crop fields.

Reply with a single JSON object and nothing else: no markdown, no code fences, no text before or after it."""


@dataclass(frozen=True)
class Prompt:
    system: str
    user_text: str
    # Fields the reply must contain; anything else missing is recorded as
    # a warning on the response rather than failing it.
    required_fields: tuple[str, ...] = ("answer", "reasoning")


def load_density_config(generation_config_path: Path) -> DensityConfig:
    """The density settings the questions were generated with, read from
    the generation-config.yaml that generate-all-questions.py writes next
    to them."""
    if not generation_config_path.exists():
        raise FileNotFoundError(
            f"{generation_config_path} not found. The density prompt explains the category rules "
            "using the thresholds the questions were generated with; point `generation_config` at "
            "the config that produced them (generate-all-questions.py writes one into its out_dir)."
        )
    raw = yaml.safe_load(generation_config_path.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(raw).density


def _format_response_spec(answer_description: str, extra_fields: list[str] | None = None) -> str:
    fields = [
        f'  "answer": {answer_description}',
        '  "reasoning": a string of at most 2 sentences explaining why you chose that answer',
        *(f"  {field}" for field in extra_fields or []),
    ]
    return "Respond with a JSON object with exactly these fields:\n{\n" + ",\n".join(fields) + "\n}"


def _options_text(options: list[str]) -> str:
    return "\n".join(f"- {json.dumps(option)}" for option in options)


def _mc_prompt(question: MultipleChoiceQuestion, *, allow_unsure: bool) -> Prompt:
    options = [*question.choices, UNSURE_ANSWER] if allow_unsure else question.choices
    lines = [question.question_text, "", "Options:", _options_text(options)]
    lines += ["", _format_response_spec('the exact text of one option above, character for character')]
    if allow_unsure:
        lines.append(
            f'If you cannot tell, answer "{UNSURE_ANSWER}" rather than guessing.'
        )
    return Prompt(SYSTEM_PROMPT, "\n".join(lines))


def _grounded_vqa_prompt(question: MultipleChoiceQuestion) -> Prompt:
    lines = [
        "Each bounding box in this image is labelled with a number. Boxes with the same number "
        "contain the same species of weed; you are not told which species each number is.",
        "",
        question.question_text,
        "",
        "Labels: " + ", ".join(question.choices),
        "",
        _format_response_spec("the single label (number) you chose, from the labels above"),
    ]
    return Prompt(SYSTEM_PROMPT, "\n".join(lines))


def _open_ended_prompt(question: OpenEndedQuestion) -> Prompt:
    lines = [
        question.question_text,
        "",
        _format_response_spec(
            "the species name, as specific as you can (a common or scientific name)"
        ),
        f'If you do not know, answer "{UNSURE_ANSWER}" rather than guessing.',
    ]
    return Prompt(SYSTEM_PROMPT, "\n".join(lines))


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.3g}%"


def density_rules_text(rules: DensityConfig) -> str:
    """Plain-language statement of how a scene's density category is
    decided, generated from the same thresholds classify_density labelled
    the ground truth with (worked examples included, computed with it)."""
    count_low, count_high = rules.count_scale
    cov_low, cov_high = rules.coverage_scale

    def category(count: int, coverage: float) -> str:
        return classify_density(
            count,
            coverage,
            count_scale=rules.count_scale,
            coverage_scale=rules.coverage_scale,
            medium_score=rules.medium_score,
            high_score=rules.high_score,
            medium_min_count=rules.medium_min_count,
            high_min_count=rules.high_min_count,
        ).value.capitalize()

    def score(count: int, coverage: float) -> float:
        return density_score(
            count, coverage, count_scale=rules.count_scale, coverage_scale=rules.coverage_scale
        )

    examples = [(1, 0.3), (3, 0.3), (15, 0.03), (10, 0.25), (30, 0.03)]
    example_lines = [
        f"- {count} weed{'s' if count != 1 else ''} covering {_percent(coverage)} of the image: "
        f"score {score(count, coverage):.2f} -> {category(count, coverage)}"
        for count, coverage in examples
    ]

    return "\n".join(
        [
            "How the density category is decided:",
            "- Only weeds count. Crop plants and bare soil do not. Count individual weed plants, "
            "and measure coverage as the share of the whole image area the weeds take up "
            "(overlapping weeds are only counted once).",
            "- None: there are no weeds in the image.",
            "- Otherwise a density score from 0 to 1 is the average of two sub-scores, each placed "
            "on a logarithmic scale and capped at 0 and 1:",
            f"    - count sub-score: 0 at {count_low:g} weed, 1 at {count_high:g} or more weeds",
            f"    - coverage sub-score: 0 at {_percent(cov_low)} coverage, 1 at {_percent(cov_high)} "
            "or more",
            f"- High: score of at least {rules.high_score:g} and at least {rules.high_min_count} "
            "weeds.",
            f"- Medium: score of at least {rules.medium_score:g} and at least "
            f"{rules.medium_min_count} weeds (and not High).",
            "- Low: anything else with weeds present, including a single plant however large it "
            "looks.",
            "",
            "Worked examples:",
            *example_lines,
        ]
    )


def _density_prompt(question: DensityEstimationQuestion, rules_text: str) -> Prompt:
    if question.annotated:
        image_note = (
            "The weeds in this image are outlined with bounding boxes; each box is one weed to "
            "count."
        )
    else:
        image_note = (
            "The image has no annotations, so you must find the weeds yourself and tell them "
            "apart from the crop."
        )
    lines = [
        question.question_text,
        image_note,
        "",
        rules_text,
        "",
        "Categories:",
        _options_text(question.density_choices),
        "",
        _format_response_spec(
            "the exact text of one category above, character for character",
            extra_fields=[
                '"estimated_weed_count": an integer, how many individual weeds are visible',
                '"estimated_weed_coverage_percent": a number from 0 to 100, the percentage of the '
                "image area the weeds cover",
            ],
        ),
    ]
    return Prompt(
        SYSTEM_PROMPT,
        "\n".join(lines),
        required_fields=(
            "answer",
            "reasoning",
            "estimated_weed_count",
            "estimated_weed_coverage_percent",
        ),
    )


def build_prompt(question: QuestionBase, *, density_rules: str | None) -> Prompt:
    """The prompt for one question. density_rules is the text from
    density_rules_text(), required only for density questions."""
    match question:
        case DensityEstimationQuestion():
            if density_rules is None:
                raise ValueError("density questions need density_rules")
            return _density_prompt(question, density_rules)
        case OpenEndedQuestion():
            return _open_ended_prompt(question)
        case MultipleChoiceQuestion() if question.question_type == QuestionType.GROUNDED_VQA:
            return _grounded_vqa_prompt(question)
        case MultipleChoiceQuestion():
            return _mc_prompt(question, allow_unsure=True)
    raise ValueError(f"No prompt for question type {question.question_type}")
