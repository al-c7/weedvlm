"""
Generates every question type in one pass from one or more WeedCOCO
datasets: species ID (boxed + unannotated + an open-ended ablation of
both), fine-grained ID, density estimation, and grounded weed VQA
(+ crop baseline) -- so you don't need to invoke each generate-*.py
script separately with the same --dataset list.

Each task is still written to its own file (species-id.json,
fine-grained.json, density.json, grounded-vqa.json) inside --out-dir,
plus a combined all-questions.json concatenating all of them, so the
previewer can be pointed at either the whole set or a single task. All
tasks share one rendered-images directory.

Usage:
    python src/weedvlm/generate-all-questions.py --out-dir questions/

By default this pulls from every dataset.json under .datasets/; pass
--dataset (repeatably) to restrict to specific ones:
    python src/weedvlm/generate-all-questions.py \
        --dataset .datasets/CropAndWeed/weedcoco.json \
        --dataset .datasets/ImageWeeds-zeamays/weedcoco.json \
        --out-dir questions/

Pass --config <file.yaml> (see config.example.yaml at the repo root) to
tune per-task image-selection/size parameters and choose which tasks
run. A CLI flag passed explicitly always overrides the config file's
value for that field.
"""

import argparse
import json
import random
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

from weedvlm.generation.balance import InsufficientQuestionsError, select_questions
from weedvlm.generation.config import (
    GENERATION_CONFIG_NAME,
    load_pipeline_config,
    resolve_config_path,
)
from weedvlm.generation.density import (
    generate_density_estimation_questions,
    select_density_questions,
)
from weedvlm.generation.fine_grained import generate_fine_grained_questions
from weedvlm.generation.load import load_images
from weedvlm.generation.render import render_all
from weedvlm.generation.grounded_vqa import generate_grounded_vqa_questions
from weedvlm.generation.open_ended import generate_open_ended_questions
from weedvlm.generation.species_id import (
    generate_species_id_mc_questions,
    generate_species_id_unannotated_questions,
    select_species_id_questions,
)


def _write_generation_config(config_path: Path | None, path: Path) -> None:
    """Records exactly how this output was generated -- the config
    file's text plus the command line (whose flags override it) -- so
    the benchmark's provenance survives later edits to the config.
    export-benchmark-parquet.py embeds this in the Parquet export."""
    header = f"# Generated {datetime.now(UTC).isoformat()} by: {shlex.join(sys.argv)}\n"
    if config_path is None:
        body = "# No config file -- built-in defaults plus the command line above.\n"
    else:
        body = f"# Config file: {config_path}\n" + config_path.read_text(encoding="utf-8")
    path.write_text(header + body, encoding="utf-8")


def _write(questions: list, path: Path) -> None:
    path.write_text(json.dumps([q.model_dump(mode="json") for q in questions], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: config.example.yaml at the repo root, if it exists)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        dest="dataset_paths",
        default=None,
        help="Path to a WeedCOCO dataset.json (repeatable), using every image/annotation as-is. "
        "Default: config's dataset_paths, or every dataset.json under .datasets/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write <task>.json, all-questions.json, and rendered/ into. "
        "Default: config's out_dir.",
    )
    parser.add_argument(
        "--open-ended-fraction",
        type=float,
        default=None,
        help="Fraction of the species-ID MC questions (boxed + unannotated) to also ask open-ended",
    )
    parser.add_argument(
        "--baseline-fraction",
        type=float,
        default=None,
        help="Fraction of the crop grounded-VQA questions to keep as the baseline set",
    )
    parser.add_argument(
        "--skip-species-id", action="store_true", help="Don't generate the species-id task"
    )
    parser.add_argument(
        "--skip-fine-grained", action="store_true", help="Don't generate the fine-grained task"
    )
    parser.add_argument(
        "--skip-density", action="store_true", help="Don't generate the density task"
    )
    parser.add_argument(
        "--skip-grounded-vqa", action="store_true", help="Don't generate the grounded-vqa task"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes to use for image decoding/rendering. Default: config's workers, or "
        "every CPU.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_pipeline_config(config_path)
    if config_path is not None:
        print(f"Using config: {config_path}")

    def merged(cli_value, cfg_value):
        return cli_value if cli_value is not None else cfg_value

    dataset_paths = args.dataset_paths or cfg.dataset_paths
    out_dir = args.out_dir or cfg.out_dir
    if out_dir is None:
        parser.error("--out-dir is required (or set out_dir in --config)")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Absolute, since this directory may be read by a server running from a
    # different working directory (e.g. the apps/review server).
    rendered_images_dir = (cfg.rendered_images_dir or out_dir / "rendered").resolve()

    open_ended_fraction = merged(args.open_ended_fraction, cfg.open_ended_fraction)
    baseline_fraction = merged(args.baseline_fraction, cfg.baseline_fraction)
    seed = merged(args.seed, cfg.seed)
    workers = merged(args.workers, cfg.workers)

    run_species_id = cfg.tasks.species_id and not args.skip_species_id
    run_fine_grained = cfg.tasks.fine_grained and not args.skip_fine_grained
    run_density = cfg.tasks.density and not args.skip_density
    run_grounded_vqa = cfg.tasks.grounded_vqa and not args.skip_grounded_vqa

    rng = random.Random(seed)

    source_paths, images = load_images(dataset_paths)
    print(
        f"Using {len(source_paths)} dataset(s): "
        f"{', '.join(p.parent.name for p in source_paths)}"
    )

    tasks: dict[str, list] = {}

    if run_species_id:
        species_cfg = cfg.species_id
        species_boxed = generate_species_id_mc_questions(
            images,
            rendered_images_dir,
            role=species_cfg.role,
            num_choices=species_cfg.num_choices,
            min_area_fraction=species_cfg.min_area_fraction,
            min_largest_area_fraction=species_cfg.min_largest_area_fraction,
            rng=rng,
        )
        species_unannotated = generate_species_id_unannotated_questions(
            images,
            role=species_cfg.role,
            num_choices=species_cfg.num_choices,
            min_area_fraction=species_cfg.min_area_fraction,
            min_largest_area_fraction=species_cfg.min_largest_area_fraction,
            rng=rng,
        )
        num_boxed_available = len(species_boxed)
        num_unannotated_available = len(species_unannotated)
        try:
            species_boxed, species_unannotated = select_species_id_questions(
                species_boxed,
                species_unannotated,
                num_species=species_cfg.num_species,
                questions_per_species=species_cfg.questions_per_species,
                unannotated_num_species=species_cfg.unannotated_num_species,
                rng=rng,
            )
        except InsufficientQuestionsError as e:
            sys.exit(f"error (species-id): {e}")
        species_mc = [*species_boxed, *species_unannotated]
        species_open_ended = generate_open_ended_questions(
            species_mc, fraction=open_ended_fraction, rng=rng
        )
        tasks["species-id.json"] = [*species_mc, *species_open_ended]
        print(
            f"  species-id:   {len(species_boxed)} boxed (of {num_boxed_available}), "
            f"{len(species_unannotated)} unannotated (of {num_unannotated_available}), "
            f"{len(species_open_ended)} open-ended"
        )

    if run_fine_grained:
        fg_cfg = cfg.fine_grained
        fine_grained_questions = generate_fine_grained_questions(
            images,
            rendered_images_dir,
            role=fg_cfg.role,
            area_similarity_ratio=fg_cfg.area_similarity_ratio,
            colour_similarity_ratio=fg_cfg.colour_similarity_ratio,
            min_total_area_fraction=fg_cfg.min_total_area_fraction,
            min_largest_area_fraction=fg_cfg.min_largest_area_fraction,
            max_centre_offset=fg_cfg.max_centre_offset,
            max_pairs_per_image=fg_cfg.max_pairs_per_image,
            rng=rng,
            workers=workers,
        )
        try:
            fine_grained_questions = select_questions(
                fine_grained_questions, questions_per_class=fg_cfg.questions_per_class, rng=rng
            )
        except InsufficientQuestionsError as e:
            sys.exit(f"error (fine-grained): {e}")
        tasks["fine-grained.json"] = fine_grained_questions
        print(f"  fine-grained: {len(fine_grained_questions)}")

    if run_density:
        density_cfg = cfg.density
        density_questions = generate_density_estimation_questions(
            images,
            rendered_images_dir,
            role=density_cfg.role,
            min_largest_area_fraction=density_cfg.min_largest_area_fraction,
            count_scale=density_cfg.count_scale,
            coverage_scale=density_cfg.coverage_scale,
            medium_score=density_cfg.medium_score,
            high_score=density_cfg.high_score,
            medium_min_count=density_cfg.medium_min_count,
            high_min_count=density_cfg.high_min_count,
        )
        try:
            density_boxed, density_unannotated = select_density_questions(
                density_questions,
                questions_per_class=density_cfg.questions_per_class,
                unannotated_questions_per_class=density_cfg.unannotated_questions_per_class,
                rng=rng,
            )
        except InsufficientQuestionsError as e:
            sys.exit(f"error (density): {e}")
        tasks["density.json"] = [*density_boxed, *density_unannotated]
        print(
            f"  density:      {len(tasks['density.json'])} "
            f"({len(density_boxed)} boxed, {len(density_unannotated)} unannotated)"
        )

    if run_grounded_vqa:
        gvqa_cfg = cfg.grounded_vqa
        grounded_vqa_weed = generate_grounded_vqa_questions(
            images,
            rendered_images_dir,
            role="weed",
            min_species=gvqa_cfg.min_species,
            min_total_area_fraction=gvqa_cfg.min_total_area_fraction,
            min_largest_area_fraction=gvqa_cfg.min_largest_area_fraction,
            max_centre_offset=gvqa_cfg.max_centre_offset,
            rng=rng,
        )
        try:
            grounded_vqa_weed = select_questions(
                grounded_vqa_weed,
                num_classes=gvqa_cfg.num_species,
                questions_per_class=gvqa_cfg.questions_per_species,
                rng=rng,
            )
        except InsufficientQuestionsError as e:
            sys.exit(f"error (grounded-vqa): {e}")
        grounded_vqa_crop = generate_grounded_vqa_questions(
            images,
            rendered_images_dir,
            role="crop",
            min_species=gvqa_cfg.min_species,
            min_total_area_fraction=gvqa_cfg.min_total_area_fraction,
            min_largest_area_fraction=gvqa_cfg.min_largest_area_fraction,
            max_centre_offset=gvqa_cfg.max_centre_offset,
            rng=rng,
        )
        baseline_size = round(len(grounded_vqa_crop) * baseline_fraction)
        grounded_vqa_baseline = rng.sample(
            grounded_vqa_crop, min(baseline_size, len(grounded_vqa_crop))
        )
        tasks["grounded-vqa.json"] = [*grounded_vqa_weed, *grounded_vqa_baseline]
        print(
            f"  grounded-vqa: {len(tasks['grounded-vqa.json'])} "
            f"({len(grounded_vqa_weed)} weed, {len(grounded_vqa_baseline)} crop baseline)"
        )

    all_questions = [q for questions in tasks.values() for q in questions]
    num_rendered = render_all(all_questions, workers=workers)
    print(f"Rendered {num_rendered} box-annotated images to {rendered_images_dir}")

    for filename, questions in tasks.items():
        _write(questions, out_dir / filename)

    _write(all_questions, out_dir / "all-questions.json")
    _write_generation_config(config_path, out_dir / GENERATION_CONFIG_NAME)

    print(f"Wrote {len(all_questions)} questions total to {out_dir}")
    print("  all-questions.json combines all of the above")


if __name__ == "__main__":
    main()
