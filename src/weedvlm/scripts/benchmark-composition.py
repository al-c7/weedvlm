"""
Prints the benchmark composition implied by a generation config: total
questions, questions per species, and the per-sub-task counts. Reads only
the config file (no datasets), so it reports the exact sizes the config
asks for. Anything the config leaves unset (None) depends on the data and
is reported as "data-dependent".

Usage:
    uv run src/weedvlm/scripts/benchmark-composition.py [--config config.example.yaml]
"""

import argparse
from pathlib import Path

from weedvlm.generation.config import load_pipeline_config, resolve_config_path

DATA = "data-dependent"


def mul(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a * b


def fmt(n: int | None) -> str:
    return DATA if n is None else str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_pipeline_config(config_path)
    print(f"config: {config_path or '(built-in defaults)'}\n")

    # task -> {sub-task: count}, and task -> {species: count} where per-species
    subtasks: dict[str, dict[str, int | None]] = {}
    per_species: dict[str, str] = {}

    if cfg.tasks.species_id:
        s = cfg.species_id
        boxed = mul(s.num_species, s.questions_per_species)
        unannotated = (
            0
            if s.unannotated_num_species == 0
            else mul(s.unannotated_num_species, s.questions_per_species)
        )
        mc = None if boxed is None or unannotated is None else boxed + unannotated
        # open_ended.py: round(len(mc) * fraction)
        open_ended = None if mc is None else round(mc * cfg.open_ended_fraction)
        subtasks["species_id"] = {
            "boxed MC": boxed,
            "unannotated MC": unannotated,
            "open-ended": open_ended,
        }
        per_species["species_id"] = (
            f"{s.questions_per_species} boxed MC ({s.num_species} species)"
            f" + {s.questions_per_species} unannotated MC"
            f" ({fmt(s.unannotated_num_species) if s.unannotated_num_species != 0 else 0} of them)"
            f", plus open-ended ablation of {cfg.open_ended_fraction:.0%} of the MC set"
        )

    if cfg.tasks.fine_grained:
        f = cfg.fine_grained
        subtasks["fine_grained"] = {
            "same_species": f.questions_per_class,
            "different_species": f.questions_per_class,
        }
        per_species["fine_grained"] = f"n/a (per class: {fmt(f.questions_per_class)})"

    if cfg.tasks.density:
        d = cfg.density
        subtasks["density"] = {
            "boxed (4 classes)": mul(4, d.questions_per_class),
            "unannotated (4 classes)": mul(4, d.unannotated_questions_per_class),
        }
        per_species["density"] = (
            f"n/a (per class: {fmt(d.questions_per_class)} boxed, "
            f"{fmt(d.unannotated_questions_per_class)} unannotated)"
        )

    if cfg.tasks.grounded_vqa:
        g = cfg.grounded_vqa
        subtasks["grounded_vqa"] = {
            "grounded_vqa": mul(g.num_species, g.questions_per_species)
        }
        per_species["grounded_vqa"] = (
            f"{fmt(g.questions_per_species)} ({fmt(g.num_species)} species)"
        )

    if cfg.tasks.localisation:
        loc = cfg.localisation
        subtasks["localisation"] = {"localisation": mul(loc.num_species, loc.questions_per_species)}
        per_species["localisation"] = (
            f"{fmt(loc.questions_per_species)} ({fmt(loc.num_species)} species)"
        )

    grand = 0
    exact = True
    for task, parts in subtasks.items():
        known = [v for v in parts.values() if v is not None]
        task_total = sum(known)
        task_exact = len(known) == len(parts)
        exact &= task_exact
        grand += task_total
        print(f"{task}: {task_total}{'' if task_exact else ' + data-dependent'}")
        for name, n in parts.items():
            print(f"    {name}: {fmt(n)}")
        print(f"    per species: {per_species[task]}")

    print(f"\nTOTAL: {grand}{'' if exact else ' + data-dependent items above (lower bound)'}")


if __name__ == "__main__":
    main()
