"""
Pydantic models for the review decisions produced by the
weedcoco-review web app (apps/weedcoco-review), used to filter out bad
images/annotations before generating questions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Decision = Literal["good", "bad"]


class ImageReviewDecision(BaseModel):
    file_name: str
    decision: Decision
    annotations: dict[str, Decision] = {}
    reviewed_at: str


class ReviewFile(BaseModel):
    dataset: str
    updated_at: str
    decisions: dict[str, ImageReviewDecision]

    def is_image_kept(self, image_id: int) -> bool:
        image_decision = self.decisions.get(str(image_id))
        return image_decision is not None and image_decision.decision == "good"

    def is_annotation_kept(self, image_id: int, annotation_id: int) -> bool:
        image_decision = self.decisions.get(str(image_id))
        if image_decision is None or image_decision.decision != "good":
            return False
        # Annotations are only ever recorded when marked bad -- anything
        # else on a good image is implicitly kept.
        return image_decision.annotations.get(str(annotation_id)) != "bad"
