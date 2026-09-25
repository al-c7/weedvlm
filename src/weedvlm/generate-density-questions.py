"""
Generates weed density estimation questions (boxed + unannotated
conditions, same source images) from one or more reviewed WeedCOCO
datasets.

Usage:
    python src/weedvlm/generate-density-questions.py --out questions/density.json

By default this pulls from every reviewed dataset (every review-*.json
under apps/weedcoco-review); pass --review (repeatably) to restrict to
specific ones:
    python src/weedvlm/generate-density-questions.py \
        --review apps/weedcoco-review/review-cropandweed.json \
        --out questions/density.json

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
from weedvlm.pipeline.density import (
    generate_density_estimation_questions,
    select_density_questions,
)
from weedvlm.pipeline.load import load_images
from weedvlm.pipeline.render import render_all


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
        help="Path to write the questions JSON. Default: config's outputs.density.",
    )
    parser.add_argument(
        "--rendered-images-dir",
        type=Path,
        default=None,
        help="Directory to write box-annotated images to (default: config's "
        "rendered_images_dir, or alongside --out in 'rendered/')",
    )
    parser.add_argument(
        "--min-largest-area-fraction",
        type=float,
        default=None,
        help="Minimum fraction of the image the largest weed box must cover (weed-free images "
        "are always kept)",
    )
    parser.add_argument("--low-count", type=int, default=None)
    parser.add_argument("--medium-count", type=int, default=None)
    parser.add_argument("--high-count", type=int, default=None)
    parser.add_argument("--low-coverage", type=float, default=None)
    parser.add_argument("--medium-coverage", type=float, default=None)
    parser.add_argument("--high-coverage", type=float, default=None)
    parser.add_argument(
        "--questions-per-class",
        type=int,
        default=None,
        help="Exactly this many boxed questions for each of the 4 density categories (none/"
        "low/medium/high) -- all 4 are always kept, balanced (error if any can't supply this "
        "many). Total boxed size = 4 x this.",
    )
    parser.add_argument(
        "--unannotated-questions-per-class",
        type=int,
        default=None,
        help="Exactly this many unannotated questions per density category, drawn from the "
        "selected boxed questions' images (0 for none; at most --questions-per-class). "
        "Default: config's density.unannotated_questions_per_class, or all of them.",
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
    task_cfg = cfg.density

    review_paths = args.review_paths or cfg.review_paths
    dataset_paths = args.dataset_paths or cfg.dataset_paths
    use_raw_datasets = args.raw_datasets or cfg.use_raw_datasets
    out_path = args.out or cfg.outputs.density
    if out_path is None:
        parser.error("--out is required (or set outputs.density in --config)")
    out_path = out_path.resolve()

    rendered_images_dir = (
        args.rendered_images_dir or cfg.rendered_images_dir or out_path.parent / "rendered"
    ).resolve()

    def merged(cli_value, cfg_value):
        return cli_value if cli_value is not None else cfg_value

    questions_per_class = merged(args.questions_per_class, task_cfg.questions_per_class)
    unannotated_questions_per_class = merged(
        args.unannotated_questions_per_class, task_cfg.unannotated_questions_per_class
    )
    seed = merged(args.seed, cfg.seed)
    workers = merged(args.workers, cfg.workers)

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

    questions = generate_density_estimation_questions(
        images,
        rendered_images_dir,
        role=task_cfg.role,
        min_largest_area_fraction=merged(
            args.min_largest_area_fraction, task_cfg.min_largest_area_fraction
        ),
        low_count=merged(args.low_count, task_cfg.low_count),
        medium_count=merged(args.medium_count, task_cfg.medium_count),
        high_count=merged(args.high_count, task_cfg.high_count),
        low_coverage=merged(args.low_coverage, task_cfg.low_coverage),
        medium_coverage=merged(args.medium_coverage, task_cfg.medium_coverage),
        high_coverage=merged(args.high_coverage, task_cfg.high_coverage),
    )
    try:
        boxed, unannotated = select_density_questions(
            questions,
            questions_per_class=questions_per_class,
            unannotated_questions_per_class=unannotated_questions_per_class,
            rng=rng,
        )
    except InsufficientQuestionsError as e:
        sys.exit(f"error: {e}")
    questions = [*boxed, *unannotated]

    num_rendered = render_all(questions, workers=workers)
    print(f"Rendered {num_rendered} box-annotated images to {rendered_images_dir}")
    out_path.write_text(json.dumps([q.model_dump(mode="json") for q in questions], indent=2))
    print(
        f"Wrote {len(questions)} questions to {out_path} "
        f"({len(boxed)} boxed, {len(unannotated)} unannotated)"
    )


if __name__ == "__main__":
    main()
