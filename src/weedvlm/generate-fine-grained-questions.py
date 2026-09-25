"""
Generates fine-grained (same species / different species) questions
from one or more reviewed WeedCOCO datasets.

Usage:
    python src/weedvlm/generate-fine-grained-questions.py --out questions/fine-grained.json

By default this pulls from every reviewed dataset (every review-*.json
under apps/weedcoco-review); pass --review (repeatably) to restrict to
specific ones:
    python src/weedvlm/generate-fine-grained-questions.py \
        --review apps/weedcoco-review/review-cropandweed.json \
        --out questions/fine-grained.json

Every tunable below can also be set via --config <file.yaml> (see
config.example.yaml at the repo root); a CLI flag passed explicitly
always overrides the config file's value for that field.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from weedvlm.pipeline.balance import InsufficientQuestionsError, select_questions
from weedvlm.pipeline.config import load_pipeline_config, resolve_config_path
from weedvlm.pipeline.fine_grained import generate_fine_grained_questions
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
        help="Path to write the questions JSON. Default: config's outputs.fine_grained.",
    )
    parser.add_argument(
        "--rendered-images-dir",
        type=Path,
        default=None,
        help="Directory to write box-annotated images to (default: config's "
        "rendered_images_dir, or alongside --out in 'rendered/')",
    )
    parser.add_argument(
        "--area-similarity-ratio",
        type=float,
        default=None,
        help="Min ratio (smaller/larger bbox area) for a different-species pair to be a candidate "
        "(pair qualifies on area OR colour, not both)",
    )
    parser.add_argument(
        "--colour-similarity-ratio",
        type=float,
        default=None,
        help="Min mean-crop-colour similarity (1.0 = identical) for a different-species pair to be "
        "a candidate (pair qualifies on area OR colour, not both)",
    )
    parser.add_argument(
        "--min-total-area-fraction",
        type=float,
        default=None,
        help="Minimum fraction of the image the pair's two boxes must cover combined -- kept low "
        "by default (keeps ~85%% of candidates on the full reviewed set); raise it if pairs come "
        "out too small to judge",
    )
    parser.add_argument(
        "--min-largest-area-fraction",
        type=float,
        default=None,
        help="Minimum fraction of the image the pair's larger box alone must cover",
    )
    parser.add_argument(
        "--max-centre-offset",
        type=float,
        default=None,
        help="Max area-weighted centre-of-mass offset from the image centre, on a 0 (centre) to "
        "1 (corner) scale -- keeps pairs out of the corners; kept lenient by default",
    )
    parser.add_argument(
        "--max-pairs-per-image",
        type=int,
        default=None,
        help="Usually the binding constraint on question count -- raise this before loosening "
        "--area-similarity-ratio if you need more questions",
    )
    parser.add_argument(
        "--questions-per-class",
        type=int,
        default=None,
        help="Exactly this many questions for each of the two classes (same_species/"
        "different_species) -- both are always kept, balanced (error if either can't supply "
        "this many). Total output size = 2 x this.",
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
    task_cfg = cfg.fine_grained

    review_paths = args.review_paths or cfg.review_paths
    dataset_paths = args.dataset_paths or cfg.dataset_paths
    use_raw_datasets = args.raw_datasets or cfg.use_raw_datasets
    out_path = args.out or cfg.outputs.fine_grained
    if out_path is None:
        parser.error("--out is required (or set outputs.fine_grained in --config)")
    out_path = out_path.resolve()

    rendered_images_dir = (
        args.rendered_images_dir or cfg.rendered_images_dir or out_path.parent / "rendered"
    ).resolve()

    def merged(cli_value, cfg_value):
        return cli_value if cli_value is not None else cfg_value

    area_similarity_ratio = merged(args.area_similarity_ratio, task_cfg.area_similarity_ratio)
    colour_similarity_ratio = merged(args.colour_similarity_ratio, task_cfg.colour_similarity_ratio)
    min_total_area_fraction = merged(args.min_total_area_fraction, task_cfg.min_total_area_fraction)
    min_largest_area_fraction = merged(
        args.min_largest_area_fraction, task_cfg.min_largest_area_fraction
    )
    max_centre_offset = merged(args.max_centre_offset, task_cfg.max_centre_offset)
    max_pairs_per_image = merged(args.max_pairs_per_image, task_cfg.max_pairs_per_image)
    questions_per_class = merged(args.questions_per_class, task_cfg.questions_per_class)
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

    questions = generate_fine_grained_questions(
        images,
        rendered_images_dir,
        role=task_cfg.role,
        area_similarity_ratio=area_similarity_ratio,
        colour_similarity_ratio=colour_similarity_ratio,
        min_total_area_fraction=min_total_area_fraction,
        min_largest_area_fraction=min_largest_area_fraction,
        max_centre_offset=max_centre_offset,
        max_pairs_per_image=max_pairs_per_image,
        rng=rng,
        workers=workers,
    )
    try:
        questions = select_questions(questions, questions_per_class=questions_per_class, rng=rng)
    except InsufficientQuestionsError as e:
        sys.exit(f"error: {e}")

    num_rendered = render_all(questions, workers=workers)
    print(f"Rendered {num_rendered} box-annotated images to {rendered_images_dir}")
    out_path.write_text(json.dumps([q.model_dump(mode="json") for q in questions], indent=2))
    print(f"Wrote {len(questions)} questions to {out_path}")


if __name__ == "__main__":
    main()
