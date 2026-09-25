"""
YAML configuration for the question-generation pipeline (the
generate-*.py scripts). Every field here mirrors a CLI flag on those
scripts and carries the identical default, so a config file only needs
to set the values it wants to change -- anything left out falls back
exactly as if no config had been passed at all. An explicit CLI flag
always overrides the matching config value for that one run.

See config.example.yaml at the repo root for a fully-annotated example.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from weedvlm.types.species import Role


class SpeciesSelection(BaseModel):
    """Exact species selection for a task's final question list (see
    weedvlm.pipeline.balance), for tasks whose benchmark_class is a
    species name drawn from a large, open-ended pool (species-id,
    localisation) -- num_species picks an exact-size subset of that
    pool. Species are filtered to ones that can supply
    questions_per_species questions before num_species are chosen from
    among them, and every chosen species then supplies exactly that
    many questions (no cross-species backfilling). The task's total
    output size is therefore exactly num_species * questions_per_species
    when both are set. Either left as None leaves that dimension
    unconstrained; raises if the data can't meet what's asked."""

    num_species: int | None = None
    questions_per_species: int | None = None


class ClassBalance(BaseModel):
    """Exact per-class question count for a task whose benchmark_class
    is a small, fixed, known set rather than species (fine-grained's
    same_species/different_species; density's none/low/medium/high) --
    there's no subset of classes to select, so every class is always
    kept, and questions_per_class asks for exactly that many questions
    for each of them (raises naming the class if any one falls short).
    The task's total output size is exactly
    (number of classes present) * questions_per_class. None leaves it
    unconstrained."""

    questions_per_class: int | None = None


class SpeciesIdConfig(SpeciesSelection):
    """num_species/questions_per_species size the boxed set.
    unannotated_questions_per_species separately sizes the unannotated
    set, over the same selected species (see
    weedvlm.pipeline.species_id.select_species_id_questions): 0 for
    none, None for every qualifying one."""

    role: Role = "weed"
    num_choices: int = 4
    min_area_fraction: float = 0.01
    min_largest_area_fraction: float = 0.0
    unannotated_questions_per_species: int | None = None


class FineGrainedConfig(ClassBalance):
    role: Role = "weed"
    area_similarity_ratio: float = 0.25
    colour_similarity_ratio: float = 0.9
    min_total_area_fraction: float = 0.0005
    min_largest_area_fraction: float = 0.0003
    max_centre_offset: float = 0.85
    max_pairs_per_image: int = 6


class DensityConfig(ClassBalance):
    """questions_per_class sizes the boxed set.
    unannotated_questions_per_class separately sizes the unannotated
    set, drawn from the same images as the selected boxed questions
    (see weedvlm.pipeline.density.select_density_questions): 0 for
    none, None for all of them."""

    role: Role = "weed"
    min_largest_area_fraction: float = 0.0
    unannotated_questions_per_class: int | None = None
    low_count: int = 1
    medium_count: int = 5
    high_count: int = 15
    low_coverage: float = 0.02
    medium_coverage: float = 0.08
    high_coverage: float = 0.20


class LocalisationConfig(SpeciesSelection):
    """num_species/questions_per_species here apply only to the main
    weed-localisation set -- the crop baseline is sampled separately (as
    baseline_fraction of the crop-localisation questions) and isn't
    subject to either."""

    min_species: int = 2
    min_total_area_fraction: float = 0.01
    min_largest_area_fraction: float = 0.003
    max_centre_offset: float = 0.6


class TaskToggles(BaseModel):
    """Which tasks generate-all-questions.py runs. Ignored by the
    individual generate-<task>-questions.py scripts, which only ever
    generate their own task."""

    species_id: bool = True
    fine_grained: bool = True
    density: bool = True
    localisation: bool = True


class Outputs(BaseModel):
    """Per-task output paths, used as the --out fallback by the
    individual generate-<task>-questions.py scripts (generate-all-
    questions.py uses out_dir instead, and ignores this)."""

    species_id: Path | None = None
    fine_grained: Path | None = None
    density: Path | None = None
    localisation: Path | None = None


class PipelineConfig(BaseModel):
    seed: int | None = None
    review_paths: list[Path] | None = None
    # Raw-dataset (no review filtering) alternative to review_paths --
    # see weedvlm.pipeline.load.load_images. use_raw_datasets: true with
    # dataset_paths left at its default reads every dataset.json under
    # .datasets/; dataset_paths given outright implies raw mode even if
    # use_raw_datasets isn't set.
    use_raw_datasets: bool = False
    dataset_paths: list[Path] | None = None
    out_dir: Path | None = None
    rendered_images_dir: Path | None = None
    # Processes used for image decoding/rendering. None = every CPU.
    workers: int | None = None
    outputs: Outputs = Field(default_factory=Outputs)

    open_ended_fraction: float = 0.2
    baseline_fraction: float = 0.1

    tasks: TaskToggles = Field(default_factory=TaskToggles)

    species_id: SpeciesIdConfig = Field(default_factory=SpeciesIdConfig)
    fine_grained: FineGrainedConfig = Field(default_factory=FineGrainedConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    localisation: LocalisationConfig = Field(default_factory=LocalisationConfig)


# src/weedvlm/pipeline/config.py -> repo root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.example.yaml"


def resolve_config_path(path: Path | None) -> Path | None:
    """--config's value if given, else DEFAULT_CONFIG_PATH if it exists
    (so a plain `generate-*.py` invocation with no --config still picks
    up config.example.yaml at the repo root by default), else None (all
    built-in defaults)."""
    if path is not None:
        return path
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None


def load_pipeline_config(path: Path | None) -> PipelineConfig:
    """Loads a PipelineConfig from a YAML file, or returns the
    all-defaults config when path is None. Callers wanting the
    --config-not-given-but-a-default-file-exists behaviour should run
    the path through resolve_config_path() first."""
    if path is None:
        return PipelineConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(raw)
