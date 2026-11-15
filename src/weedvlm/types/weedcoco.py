"""
Pydantic models mirroring the raw WeedCOCO JSON schema (COCO extended
with AgContexts). These are intentionally permissive (`extra="allow"`)
beyond the fields we actually consume, since the exact schema varies
slightly between source datasets.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Category(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    name: str


class Image(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    file_name: str
    width: int
    height: int
    agcontext_id: int
    license: int | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    segmentation: list | None = None


class AgContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    crop_type: str | None = None


class License(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    url: str | None = None


class WeedCocoDataset(BaseModel):
    model_config = ConfigDict(extra="allow")

    images: list[Image]
    annotations: list[Annotation]
    categories: list[Category]
    agcontexts: list[AgContext] = []
    licenses: list[License] = []
    info: dict = {}
