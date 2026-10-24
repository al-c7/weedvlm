
"""
YAML configuration for the reasoning evaluation pipeline (score-reasoning.py):
which tasks to run, which OpenRouter models to run them on, and what set of results
to perform the reasoning on.

See reasoning.example.yaml at the repo root for an annotated example.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from weedvlm.generation.config import TaskToggles
from weedvlm.storage.config import StorageConfig

# Hard ceiling on concurrent requests. OpenRouter rate-limits per key and
# per upstream provider, and every request carries a multi-megabyte
# base64 image, so going wider than this mostly buys 429s and memory use.
MAX_CONCURRENCY = 32

# src/weedvlm/reasoning/config.py -> repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "reasoning.example.yaml"

class ModelConfig(BaseModel):
    """The judge model: an OpenRouter model id, plus optional overrides of
    the request defaults. extra_body is merged into the request as-is."""

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


class ReasoningConfig(BaseModel):
    questions_dir: Path = Path("questions")
    generation_config: Path | None = None
    output_dir: Path = Path("reasoning-results")
    run_name: str = "run"
    tasks: TaskToggles = Field(default_factory=TaskToggles)
    evaluation_model: ModelConfig
    concurrency: int = Field(default=8, ge=1, le=MAX_CONCURRENCY)
    defaults: RequestDefaults = Field(default_factory=RequestDefaults)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    storage: StorageConfig = Field(default_factory=lambda: StorageConfig(format="sqlite"))
    responses_to_evaluate: list[str]

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    reasoning_fraction: float = Field(default=1.0, gt=0, le=1)

    @field_validator("evaluation_model", mode="before")
    @classmethod
    def _bare_model_id(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value

    @field_validator("reasoning_fraction", mode="before")
    @classmethod
    def _parse_fraction(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return float(Fraction(value.replace(" ", "")))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"not a number or fraction like 1/3: {value!r}") from None
        return value

    @model_validator(mode="after")
    def _check(self) -> ReasoningConfig:
        if self.storage.format != "sqlite":
            raise ValueError("storage: format must be sqlite, the responses are read from the database")
        if not any(self.tasks.model_dump().values()):
            raise ValueError("tasks: every task is disabled, nothing to run")
        return self

    def resolved_generation_config(self) -> Path:
        return self.generation_config or self.questions_dir / "generation-config.yaml"

    def resolved_database(self) -> Path:
        return self.storage.resolved_database(self.questions_dir)

    def temperature_for(self, model: ModelConfig) -> float:
        return self.defaults.temperature if model.temperature is None else model.temperature

    def max_tokens_for(self, model: ModelConfig) -> int:
        return self.defaults.max_tokens if model.max_tokens is None else model.max_tokens

def resolve_reasoning_path(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(f"No --config given and no {DEFAULT_CONFIG_PATH} to fall back to")


def load_reasoning_config(path: Path) -> ReasoningConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ReasoningConfig.model_validate(raw)
