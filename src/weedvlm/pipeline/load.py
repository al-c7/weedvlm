"""
Loads a WeedCOCO dataset into a list of ReviewedImage, the entry point
every question-generation task starts from. Two sources:

- reviewed (apps/weedcoco-review): the review-*.json's decisions gate
  which images/annotations survive, dropping anything marked "bad".
- raw: every image/annotation in the dataset.json is used as-is, no
  filtering -- for once a dataset's quality has already been screened
  some other way (e.g. automatic screening upstream) and
  apps/weedcoco-review's manual pass is no longer needed.

load_images() is what the generate-*.py scripts call; it picks between
the two based on what's asked for, so a script doesn't have to
duplicate that branch itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from weedvlm.types.dataset import ReviewedImage, SpeciesAnnotation
from weedvlm.types.review import ReviewFile
from weedvlm.types.species import parse_category_name
from weedvlm.types.weedcoco import WeedCocoDataset

# src/weedvlm/pipeline/load.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REVIEW_DIR = _REPO_ROOT / "apps" / "weedcoco-review"
_DEFAULT_DATASETS_DIR = _REPO_ROOT / ".datasets"


def discover_review_paths(review_dir: Path | None = None) -> list[Path]:
    """Every review-*.json under review_dir (default: apps/weedcoco-review),
    sorted for deterministic ordering. Used by the generate-*.py scripts'
    --review default, so a generation run covers every reviewed dataset
    without having to list each one by hand."""
    review_dir = review_dir or _DEFAULT_REVIEW_DIR
    return sorted(review_dir.glob("review-*.json"))


def discover_dataset_paths(datasets_dir: Path | None = None) -> list[Path]:
    """Every weedcoco.json under datasets_dir (default: .datasets),
    sorted for deterministic ordering. Mirrors discover_review_paths,
    for the raw (no review file) loading path -- used by the
    generate-*.py scripts' --dataset default when raw mode is on."""
    datasets_dir = datasets_dir or _DEFAULT_DATASETS_DIR
    return sorted(datasets_dir.glob("*/weedcoco.json"))


def _load_images(
    dataset_path: Path,
    *,
    images_dir: Path | None,
    is_image_kept: Callable[[int], bool],
    is_annotation_kept: Callable[[int, int], bool],
) -> list[ReviewedImage]:
    dataset = WeedCocoDataset.model_validate(json.loads(dataset_path.read_text(encoding="utf-8")))

    dataset_name = dataset_path.parent.name
    images_dir = images_dir or dataset_path.parent / "images"
    species_by_category_id = {c.id: parse_category_name(c.name) for c in dataset.categories}

    annotations_by_image_id: dict[int, list] = {}
    for annotation in dataset.annotations:
        annotations_by_image_id.setdefault(annotation.image_id, []).append(annotation)

    images = []
    for image in dataset.images:
        if not is_image_kept(image.id):
            continue

        kept_annotations = [
            SpeciesAnnotation(
                annotation_id=annotation.id,
                category_id=annotation.category_id,
                species=species_by_category_id[annotation.category_id],
                bbox=annotation.bbox,
            )
            for annotation in annotations_by_image_id.get(image.id, [])
            if is_annotation_kept(image.id, annotation.id)
        ]

        images.append(
            ReviewedImage(
                dataset_name=dataset_name,
                image_id=image.id,
                image_path=images_dir / image.file_name,
                width=image.width,
                height=image.height,
                species_annotations=kept_annotations,
            )
        )

    return images


def load_reviewed_images(
    review_path: Path,
    *,
    dataset_path: Path | None = None,
    images_dir: Path | None = None,
) -> list[ReviewedImage]:
    review = ReviewFile.model_validate(json.loads(review_path.read_text(encoding="utf-8")))
    dataset_path = dataset_path or Path(review.dataset)

    return _load_images(
        dataset_path,
        images_dir=images_dir,
        is_image_kept=review.is_image_kept,
        is_annotation_kept=review.is_annotation_kept,
    )


def load_dataset_images(dataset_path: Path, *, images_dir: Path | None = None) -> list[ReviewedImage]:
    """Loads every image/annotation straight from a raw WeedCOCO
    dataset.json, with no review filtering at all."""
    return _load_images(
        dataset_path,
        images_dir=images_dir,
        is_image_kept=lambda _image_id: True,
        is_annotation_kept=lambda _image_id, _annotation_id: True,
    )


def load_reviewed_images_from(
    review_paths: list[Path] | None,
) -> tuple[list[Path], list[ReviewedImage]]:
    """Resolves review_paths (falling back to every discovered review file
    when None/empty) and loads all of them, keyed to their dataset via each
    ReviewedImage's dataset_name. Returns the resolved paths alongside the
    combined images, so a caller can report which datasets were actually
    used."""
    paths = review_paths or discover_review_paths()
    images = [image for path in paths for image in load_reviewed_images(path)]
    return paths, images


def load_images_from(dataset_paths: list[Path] | None) -> tuple[list[Path], list[ReviewedImage]]:
    """Resolves dataset_paths (falling back to every discovered raw
    dataset.json when None/empty) and loads all of them directly, with
    no review filtering. Mirrors load_reviewed_images_from."""
    paths = dataset_paths or discover_dataset_paths()
    images = [image for path in paths for image in load_dataset_images(path)]
    return paths, images


def load_images(
    *,
    review_paths: list[Path] | None,
    dataset_paths: list[Path] | None,
    use_raw_datasets: bool,
) -> tuple[list[Path], list[ReviewedImage], bool]:
    """The single entry point the generate-*.py scripts use to load
    images, switching between the reviewed and raw sources. Raw is used
    when explicitly asked for (use_raw_datasets, or dataset_paths given
    outright) -- otherwise reviewed is the default, unchanged from
    before raw-dataset support existed. Returns the resolved source
    paths, the combined images, and whether raw mode was used (so the
    caller can log which source it read from)."""
    if use_raw_datasets or dataset_paths:
        paths, images = load_images_from(dataset_paths)
        return paths, images, True
    paths, images = load_reviewed_images_from(review_paths)
    return paths, images, False
