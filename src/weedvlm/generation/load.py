"""
Loads a WeedCOCO dataset into a list of ReviewedImage, the entry point
every question-generation task starts from. Every image/annotation in
the dataset.json is used as-is, no filtering.

load_images() is what the generate-*.py scripts call.
"""

from __future__ import annotations

import json
from pathlib import Path

from weedvlm.types.dataset import ReviewedImage, SpeciesAnnotation
from weedvlm.types.species import parse_category_name
from weedvlm.types.weedcoco import WeedCocoDataset

# src/weedvlm/pipeline/load.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS_DIR = _REPO_ROOT / ".datasets"


def discover_dataset_paths(datasets_dir: Path | None = None) -> list[Path]:
    """Every weedcoco.json under datasets_dir (default: .datasets),
    sorted for deterministic ordering. Used by the generate-*.py
    scripts' --dataset default."""
    datasets_dir = datasets_dir or DEFAULT_DATASETS_DIR
    return sorted(datasets_dir.glob("*/weedcoco.json"))


def resolve_image_path(file_name: str, dataset_dir: Path, images_dir: Path | None) -> Path | None:
    """Where an image's file actually is. An explicit images_dir is
    taken as-is. Otherwise WeedCOCO exports disagree on what file_name
    is relative to -- most give it relative to <dataset>/images/, some
    (e.g. BroadleafWeedsInCommonCouch) relative to the dataset folder
    itself, already prefixed with images/ -- so both are tried. None if
    the file exists under neither."""
    if images_dir is not None:
        return images_dir / file_name
    for candidate in (dataset_dir / "images" / file_name, dataset_dir / file_name):
        if candidate.is_file():
            return candidate
    return None


def load_dataset_images(dataset_path: Path, *, images_dir: Path | None = None) -> list[ReviewedImage]:
    """Loads every image/annotation straight from a WeedCOCO dataset.json,
    with no filtering at all."""
    dataset = WeedCocoDataset.model_validate(json.loads(dataset_path.read_text(encoding="utf-8")))

    dataset_name = dataset_path.parent.name
    species_by_category_id = {c.id: parse_category_name(c.name) for c in dataset.categories}

    annotations_by_image_id: dict[int, list] = {}
    for annotation in dataset.annotations:
        annotations_by_image_id.setdefault(annotation.image_id, []).append(annotation)

    images = []
    missing: list[str] = []
    for image in dataset.images:
        image_path = resolve_image_path(image.file_name, dataset_path.parent, images_dir)
        if image_path is None:
            missing.append(image.file_name)
            continue

        kept_annotations = [
            SpeciesAnnotation(
                annotation_id=annotation.id,
                category_id=annotation.category_id,
                species=species_by_category_id[annotation.category_id],
                bbox=annotation.bbox,
            )
            for annotation in annotations_by_image_id.get(image.id, [])
        ]

        images.append(
            ReviewedImage(
                dataset_name=dataset_name,
                image_id=image.id,
                image_path=image_path,
                width=image.width,
                height=image.height,
                species_annotations=kept_annotations,
            )
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) in {dataset_path} not found under "
            f"{dataset_path.parent / 'images'} or {dataset_path.parent} (e.g. {missing[0]!r}). "
            f"Check the dataset's images were fully extracted."
        )
    return images


def load_images(dataset_paths: list[Path] | None) -> tuple[list[Path], list[ReviewedImage]]:
    """The single entry point the generate-*.py scripts use to load
    images. Resolves dataset_paths (falling back to every discovered
    dataset.json under .datasets/ when None/empty) and loads all of
    them directly, with no filtering. Returns the resolved paths
    alongside the combined images, so a caller can report which
    datasets were actually used."""
    paths = dataset_paths or discover_dataset_paths()
    images = [image for path in paths for image in load_dataset_images(path)]
    return paths, images
