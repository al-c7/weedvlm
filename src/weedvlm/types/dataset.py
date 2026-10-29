"""
The reviewed, ready-to-use view of a WeedCOCO dataset: one entry per
image that passed review, carrying only the annotations that also
passed review, each resolved to a Species. This is the base unit every
question-generation task builds from.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from weedvlm.types.species import Role, Species


class SpeciesAnnotation(BaseModel):
    annotation_id: int
    category_id: int
    species: Species
    bbox: tuple[float, float, float, float]


class ReviewedImage(BaseModel):
    dataset_name: str
    image_id: int
    image_path: Path
    width: int
    height: int
    species_annotations: list[SpeciesAnnotation]

    def species_of_role(self, role: Role) -> list[Species]:
        return [a.species for a in self.species_annotations if a.species.role == role]

    def unique_species_names(self, role: Role) -> set[str]:
        return {species.name for species in self.species_of_role(role)}
