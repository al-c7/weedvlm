"""
Removes annotations (and their category entries) for named categories from
a WeedCOCO-format dataset -- e.g. CropAndWeed's catch-all "none" (Vegetation)
category, which isn't an identified crop or weed species.

Usage:
    python src/weedvlm/filter-weedcoco-categories.py \
        --dataset .datasets/CropAndWeed/weedcoco.json \
        --out .datasets/CropAndWeed/weedcoco.filtered.json \
        --category none
"""

import argparse
import json
from pathlib import Path


def filter_categories(dataset: dict, excluded_names: set[str]) -> dict:
    excluded_ids = {
        c["id"] for c in dataset["categories"] if c["name"].lower() in excluded_names
    }

    kept_categories = [c for c in dataset["categories"] if c["id"] not in excluded_ids]
    kept_annotations = [
        a for a in dataset["annotations"] if a["category_id"] not in excluded_ids
    ]

    removed_count = len(dataset["annotations"]) - len(kept_annotations)
    print(
        f"Excluded categories: {sorted(c['name'] for c in dataset['categories'] if c['id'] in excluded_ids)}"
    )
    print(f"Removed {removed_count} of {len(dataset['annotations'])} annotations")

    remaining_image_ids = {a["image_id"] for a in kept_annotations}
    empty_images = sum(1 for img in dataset["images"] if img["id"] not in remaining_image_ids)
    print(f"Images left with zero annotations: {empty_images} of {len(dataset['images'])}")

    return {
        **dataset,
        "categories": kept_categories,
        "annotations": kept_annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to weedcoco.json")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the filtered dataset")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        dest="categories",
        help="Category name to exclude (case-insensitive, repeatable). Defaults to 'none'.",
    )
    args = parser.parse_args()

    excluded_names = {c.lower() for c in (args.categories or ["none"])}

    with args.dataset.open(encoding="utf-8") as f:
        dataset = json.load(f)

    filtered = filter_categories(dataset, excluded_names)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"Wrote filtered dataset to {args.out}")


if __name__ == "__main__":
    main()
