"""
Generates every question type in one pass from one or more reviewed
WeedCOCO datasets: species ID (boxed + unannotated + an open-ended
ablation of both), fine-grained ID, density estimation, and species
localisation (+ crop baseline) -- so you don't need to invoke each
generate-*.py script separately with the same --review list.

Each task is still written to its own file (species-id.json,
fine-grained.json, density.json, localisation.json) inside --out-dir,
plus a combined all-questions.json concatenating all of them, so the
previewer can be pointed at either the whole set or a single task. All
tasks share one rendered-images directory.

Usage:
    python src/weedvlm/generate-all-questions.py --out-dir questions/

By default this pulls from every reviewed dataset (every review-*.json
under apps/weedcoco-review); pass --review (repeatably) to restrict to
specific ones:
    python src/weedvlm/generate-all-questions.py \
        --review apps/weedcoco-review/review-cropandweed.json \
        --review apps/weedcoco-review/review-zeamays.json \
        --out-dir questions/

Pass --config <file.yaml> (see config.example.yaml at the repo root) to
tune per-task image-selection/size parameters and choose which tasks
run -- everything the individual generate-*.py scripts expose, in one
file. A CLI flag passed explicitly always overrides the config file's
value for that field.

Species-id and localisation additionally support an exact species
selection (species_id.num_species/questions_per_species,
localisation.num_species/questions_per_species): num_species picks an
exact-size subset of the species pool, and each picked species then
supplies exactly questions_per_species questions, so the task's total
output size is exactly num_species * questions_per_species.
Fine-grained and density instead have a small, fixed, known class set
(same_species/different_species; none/low/medium/high) -- every class
is always kept, and fine_grained.questions_per_class /
density.questions_per_class asks for exactly that many questions per
class, so their total output size is (2 or 4) * questions_per_class.
Species-id and density's unannotated questions are sized separately
(species_id.unannotated_questions_per_species,
density.unannotated_questions_per_class), on top of those totals.
There's no single CLI flag for any of these here since the right
numbers differ per task -- use --config.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from weedvlm.pipeline.balance import InsufficientQuestionsError, select_questions
from weedvlm.pipeline.config import load_pipeline_config, resolve_config_path
from weedvlm.pipeline.density import (
    generate_density_estimation_questions,
    select_density_questions,
)
from weedvlm.pipeline.fine_grained import generate_fine_grained_questions
from weedvlm.pipeline.load import load_images
from weedvlm.pipeline.render import render_all
from weedvlm.pipeline.localisation import generate_species_localisation_questions
from weedvlm.pipeline.open_ended import generate_open_ended_questions
from weedvlm.pipeline.species_id import (
    generate_species_id_mc_questions,
    generate_species_id_unannotated_questions,
    select_species_id_questions,
)


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
        "--review",
        action="append",
        type=Path,
        dest="review_paths",
        default=None,
        help="Path to a review-*.json file (repeatable). Default: config's review_paths, or "
        "every review-*.json under apps/weedcoco-review.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        dest="dataset_paths",
        default=None,
        help="Path to a raw WeedCOCO dataset.json (repeatable) -- bypasses "
        "apps/weedcoco-review filtering entirely, using every image/annotation as-is. Implies "
        "--raw-datasets. Default: config's dataset_paths.",
    )
    parser.add_argument(
        "--raw-datasets",
        action="store_true",
        help="Use raw WeedCOCO datasets with no review filtering (auto-discovers every "
        "dataset.json under .datasets/ unless --dataset is given). Default: config's "
        "use_raw_datasets.",
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
        help="Fraction of the crop-localisation questions to keep as the baseline set",
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
        "--skip-localisation", action="store_true", help="Don't generate the localisation task"
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

    review_paths = args.review_paths or cfg.review_paths
    dataset_paths = args.dataset_paths or cfg.dataset_paths
    use_raw_datasets = args.raw_datasets or cfg.use_raw_datasets
    out_dir = args.out_dir or cfg.out_dir
    if out_dir is None:
        parser.error("--out-dir is required (or set out_dir in --config)")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Absolute, since this directory may be read by a server running from a
    # different working directory (e.g. the question-preview app).
    rendered_images_dir = (cfg.rendered_images_dir or out_dir / "rendered").resolve()

    open_ended_fraction = merged(args.open_ended_fraction, cfg.open_ended_fraction)
    baseline_fraction = merged(args.baseline_fraction, cfg.baseline_fraction)
    seed = merged(args.seed, cfg.seed)
    workers = merged(args.workers, cfg.workers)

    run_species_id = cfg.tasks.species_id and not args.skip_species_id
    run_fine_grained = cfg.tasks.fine_grained and not args.skip_fine_grained
    run_density = cfg.tasks.density and not args.skip_density
    run_localisation = cfg.tasks.localisation and not args.skip_localisation

    rng = random.Random(seed)

    source_paths, images, using_raw = load_images(
        review_paths=review_paths, dataset_paths=dataset_paths, use_raw_datasets=use_raw_datasets
    )
    if using_raw:
        print(
            f"Using {len(source_paths)} raw dataset(s) (no review filtering): "
            f"{', '.join(p.parent.name for p in source_paths)}"
        )
    else:
        print(
            f"Using {len(source_paths)} review file(s): "
            f"{', '.join(p.name for p in source_paths)}"
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
                unannotated_questions_per_species=species_cfg.unannotated_questions_per_species,
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
            low_count=density_cfg.low_count,
            medium_count=density_cfg.medium_count,
            high_count=density_cfg.high_count,
            low_coverage=density_cfg.low_coverage,
            medium_coverage=density_cfg.medium_coverage,
            high_coverage=density_cfg.high_coverage,
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

    if run_localisation:
        loc_cfg = cfg.localisation
        localisation_weed = generate_species_localisation_questions(
            images,
            rendered_images_dir,
            role="weed",
            min_species=loc_cfg.min_species,
            min_total_area_fraction=loc_cfg.min_total_area_fraction,
            min_largest_area_fraction=loc_cfg.min_largest_area_fraction,
            max_centre_offset=loc_cfg.max_centre_offset,
            rng=rng,
        )
        try:
            localisation_weed = select_questions(
                localisation_weed,
                num_classes=loc_cfg.num_species,
                questions_per_class=loc_cfg.questions_per_species,
                rng=rng,
            )
        except InsufficientQuestionsError as e:
            sys.exit(f"error (localisation): {e}")
        localisation_crop = generate_species_localisation_questions(
            images,
            rendered_images_dir,
            role="crop",
            min_species=loc_cfg.min_species,
            min_total_area_fraction=loc_cfg.min_total_area_fraction,
            min_largest_area_fraction=loc_cfg.min_largest_area_fraction,
            max_centre_offset=loc_cfg.max_centre_offset,
            rng=rng,
        )
        baseline_size = round(len(localisation_crop) * baseline_fraction)
        localisation_baseline = rng.sample(
            localisation_crop, min(baseline_size, len(localisation_crop))
        )
        tasks["localisation.json"] = [*localisation_weed, *localisation_baseline]
        print(
            f"  localisation: {len(tasks['localisation.json'])} "
            f"({len(localisation_weed)} weed, {len(localisation_baseline)} crop baseline)"
        )

    all_questions = [q for questions in tasks.values() for q in questions]
    num_rendered = render_all(all_questions, workers=workers)
    print(f"Rendered {num_rendered} box-annotated images to {rendered_images_dir}")

    for filename, questions in tasks.items():
        _write(questions, out_dir / filename)

    _write(all_questions, out_dir / "all-questions.json")

    print(f"Wrote {len(all_questions)} questions total to {out_dir}")
    print("  all-questions.json combines all of the above")


if __name__ == "__main__":
    main()
