"""
YAML configuration for the evaluation pipeline (run-evaluation.py): which
tasks to run, which OpenRouter models to run them on, how many requests to
keep in flight, and how failures are retried.

See evaluation.example.yaml at the repo root for an annotated example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from weedvlm.generation.config import TaskToggles

# Hard ceiling on concurrent requests. OpenRouter rate-limits per key and
# per upstream provider, and every request carries a multi-megabyte
# base64 image, so going wider than this mostly buys 429s and memory use.
MAX_CONCURRENCY = 32

# src/weedvlm/evaluation/config.py -> repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "evaluation.example.yaml"


class ModelConfig(BaseModel):
    """One OpenRouter model to evaluate. In the YAML a model may be given
    as a bare id string (all defaults) or as a mapping to override the
    request settings for just that model.

    Unset fields fall back to the top-level `defaults` (see
    EvaluationConfig). `extra_body` is merged into the request body as-is,
    for OpenRouter options like {"reasoning": {"effort": "low"}} or
    {"provider": {"order": ["..."]}}."""

    id: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RequestDefaults(BaseModel):
    temperature: float = 0.0
    # Reasoning models spend part of this budget on hidden thinking, and
    # an empty reply with finish_reason "length" is recorded as a failure
    # -- raise this (or set a reasoning effort in extra_body) for them.
    max_tokens: int = 1024
    timeout_seconds: float = 120.0


class RetryConfig(BaseModel):
    """Transient failures (429, 5xx, timeouts, dropped connections) back
    off exponentially with jitter, honouring any Retry-After header.
    Replies that arrive but can't be parsed into the required JSON get
    their own, separate retry budget."""

    max_retries: int = 5
    max_parse_retries: int = 2
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    # After this many consecutive failed questions for one model, stop
    # sending it further questions (they stay pending, so a re-run picks
    # them up). Catches a model id that doesn't exist, doesn't accept
    # images, or is down. 0 disables.
    model_failure_threshold: int = 25


class EvaluationConfig(BaseModel):
    # Directory holding the generated question files (the out_dir of
    # generate-all-questions.py): species-id.json, fine-grained.json,
    # density.json and localisation.json.
    questions_dir: Path = Path("questions")
    # The generation config that produced those questions. Density's
    # category rules are explained to the VLM using the thresholds from
    # here, so the prompt always matches how the ground truth was
    # labelled. Default: generation-config.yaml inside questions_dir.
    generation_config: Path | None = None

    # Results go to <output_dir>/<run_name>/. Re-running with the same
    # run_name resumes: questions already answered (status "ok") are
    # skipped, failed ones are tried again.
    output_dir: Path = Path("results")
    run_name: str = "run"

    tasks: TaskToggles = Field(default_factory=TaskToggles)
    models: list[ModelConfig]

    concurrency: int = Field(default=8, ge=1, le=MAX_CONCURRENCY)
    defaults: RequestDefaults = Field(default_factory=RequestDefaults)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    # Smoke-test aid, not a benchmark-sizing knob: evaluate only a seeded
    # random subset of this many questions from each enabled task.
    max_questions_per_task: int | None = Field(default=None, ge=1)
    seed: int = 0

    # Seconds between progress.json writes / non-interactive log lines.
    progress_interval_seconds: float = Field(default=5.0, gt=0)

    @field_validator("models", mode="before")
    @classmethod
    def _bare_model_ids(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [{"id": item} if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def _check(self) -> EvaluationConfig:
        if not self.models:
            raise ValueError("models must list at least one OpenRouter model id")
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError("models contains a duplicate model id")
        if not any(self.tasks.model_dump().values()):
            raise ValueError("tasks: every task is disabled, nothing to run")
        return self

    def resolved_generation_config(self) -> Path:
        return self.generation_config or self.questions_dir / "generation-config.yaml"

    def temperature_for(self, model: ModelConfig) -> float:
        return self.defaults.temperature if model.temperature is None else model.temperature

    def max_tokens_for(self, model: ModelConfig) -> int:
        return self.defaults.max_tokens if model.max_tokens is None else model.max_tokens


def resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(f"No --config given and no {DEFAULT_CONFIG_PATH} to fall back to")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvaluationConfig.model_validate(raw)
