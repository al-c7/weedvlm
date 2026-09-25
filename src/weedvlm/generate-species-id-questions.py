"""
Generates species ID questions from one or more reviewed WeedCOCO
datasets: the boxed multiple-choice set, the smaller unannotated
multiple-choice set (single-weed-species images only), and an
open-ended ablation sampled from both.

Usage:
    python src/weedvlm/generate-species-id-questions.py --out questions/species-id.json

By default this pulls from every reviewed dataset (every review-*.json
under apps/weedcoco-review); pass --review (repeatably) to restrict to
specific ones:
    python src/weedvlm/generate-species-id-questions.py \
        --review apps/weedcoco-review/review-cropandweed.json \
        --review apps/weedcoco-review/review-zeamays.json \
        --out questions/species-id.json

Every tunable below can also be set via --config <file.yaml> (see
config.example.yaml at the repo root); a CLI flag passed explicitly
always overrides the config file's value for that field.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from weedvlm.pipeline.balance import InsufficientQuestionsError
from weedvlm.pipeline.config import load_pipeline_config, resolve_config_path
from weedvlm.pipeline.load import load_images
from weedvlm.pipeline.render import render_all
from weedvlm.pipeline.open_ended import generate_open_ended_questions
from weedvlm.pipeline.species_id import (
    generate_species_id_mc_questions,
    generate_species_id_unannotated_questions,
    select_species_id_questions,
)


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
        "--out",
        type=Path,
        default=None,
        help="Path to write the questions JSON. Default: config's outputs.species_id.",
    )
    parser.add_argument(
        "--rendered-images-dir",
        type=Path,
        default=None,
        help="Directory to write box-annotated images to (default: config's "
        "rendered_images_dir, or alongside --out in 'rendered/')",
    )
    parser.add_argument("--num-choices", type=int, default=None)
    parser.add_argument(
        "--min-area-fraction",
        type=float,
        default=None,
        help="Minimum fraction of the image a species' boxes must cover to be asked about",
    )
    parser.add_argument(
        "--min-largest-area-fraction",
        type=float,
        default=None,
        help="Minimum fraction of the image a species' largest single box must cover to be "
        "asked about",
    )
    parser.add_argument(
        "--open-ended-fraction",
        type=float,
        default=None,
        help="Fraction of the MC questions (boxed + unannotated) to also ask open-ended",
    )
    parser.add_argument(
        "--num-species",
        type=int,
        default=None,
        help="Select exactly this many species for this task's output (error if fewer qualify)",
    )
    parser.add_argument(
        "--questions-per-species",
        type=int,
        default=None,
        help="Exactly this many boxed MC questions targeting each selected species (error if "
        "a species can't supply this many). Total boxed size = num_species x this.",
    )
    parser.add_argument(
        "--unannotated-questions-per-species",
        type=int,
        default=None,
        help="Exactly this many unannotated MC questions for each selected species (0 for "
        "none; only species that can supply this many are selected). Total unannotated size "
        "= num_species x this. Default: config's species_id.unannotated_questions_per_species, "
        "or every qualifying one.",
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

    review_paths = args.review_paths or cfg.review_paths
    dataset_paths = args.dataset_paths or cfg.dataset_paths
    use_raw_datasets = args.raw_datasets or cfg.use_raw_datasets
    out_path = args.out or cfg.outputs.species_id
    if out_path is None:
        parser.error("--out is required (or set outputs.species_id in --config)")
    out_path = out_path.resolve()

    rendered_images_dir = (
        args.rendered_images_dir or cfg.rendered_images_dir or out_path.parent / "rendered"
    ).resolve()

    role = cfg.species_id.role
    num_choices = args.num_choices if args.num_choices is not None else cfg.species_id.num_choices
    min_area_fraction = (
        args.min_area_fraction
        if args.min_area_fraction is not None
        else cfg.species_id.min_area_fraction
    )
    open_ended_fraction = (
        args.open_ended_fraction
        if args.open_ended_fraction is not None
        else cfg.open_ended_fraction
    )
    min_largest_area_fraction = (
        args.min_largest_area_fraction
        if args.min_largest_area_fraction is not None
        else cfg.species_id.min_largest_area_fraction
    )
    num_species = (
        args.num_species if args.num_species is not None else cfg.species_id.num_species
    )
    questions_per_species = (
        args.questions_per_species
        if args.questions_per_species is not None
        else cfg.species_id.questions_per_species
    )
    unannotated_questions_per_species = (
        args.unannotated_questions_per_species
        if args.unannotated_questions_per_species is not None
        else cfg.species_id.unannotated_questions_per_species
    )
    seed = args.seed if args.seed is not None else cfg.seed
    workers = args.workers if args.workers is not None else cfg.workers

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

    boxed_questions = generate_species_id_mc_questions(
        images,
        rendered_images_dir,
        role=role,
        num_choices=num_choices,
        min_area_fraction=min_area_fraction,
        min_largest_area_fraction=min_largest_area_fraction,
        rng=rng,
    )
    unannotated_questions = generate_species_id_unannotated_questions(
        images,
        role=role,
        num_choices=num_choices,
        min_area_fraction=min_area_fraction,
        min_largest_area_fraction=min_largest_area_fraction,
        rng=rng,
    )

    try:
        boxed_questions, unannotated_questions = select_species_id_questions(
            boxed_questions,
            unannotated_questions,
            num_species=num_species,
            questions_per_species=questions_per_species,
            unannotated_questions_per_species=unannotated_questions_per_species,
            rng=rng,
        )
    except InsufficientQuestionsError as e:
        sys.exit(f"error: {e}")
    mc_questions = [*boxed_questions, *unannotated_questions]
    open_ended_questions = generate_open_ended_questions(
        mc_questions,
        fraction=open_ended_fraction,
        rng=rng,
    )

    questions = [*mc_questions, *open_ended_questions]
    num_rendered = render_all(questions, workers=workers)
    print(f"Rendered {num_rendered} box-annotated images to {rendered_images_dir}")
    out_path.write_text(json.dumps([q.model_dump(mode="json") for q in questions], indent=2))
    print(
        f"Wrote {len(questions)} questions to {out_path} "
        f"({len(boxed_questions)} boxed MC, {len(unannotated_questions)} unannotated MC, "
        f"{len(open_ended_questions)} open-ended)"
    )


if __name__ == "__main__":
    main()
