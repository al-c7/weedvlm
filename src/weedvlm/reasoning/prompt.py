"""
Builds the prompt to send to OpenRouter for the reasoning evaluation.
The image is pulled from the database and the prompt provides a render
with all of the ground truth provided.

The overall template for each judge prompt is the same, the judge context
is provided, followed by the questions, the response, and the output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from weedvlm.reasoning.runner import JudgeItem
from weedvlm.storage.database import read_image
from PIL import Image as PILImage, ImageDraw, ImageFont
from weedvlm.evaluation.prompts import density_rules_text, load_density_config

REASONING_SYSTEM_PROMPT = """\
You are an expert judge whose purpose is to evaluates the quality of responses
to various weed identification questions. The response that you have been
provided is considered to be correct. Your job is to evaluate the post-hoc
rationale provided with the response. You will assign scores to the rationale
based on the following criteria:

- Observation (0-10): Rationale correctly identifies evidence present in the image.
- Interpretation (0-10): Correctly infers the species or condition  from the observations.
- Grounding (0-10): Reasoning only relies on given information; does not hallucinate details.
- Consistency (0-10): The explanation logically supports the final answer without contradictions.
"""

ERROR_SYSTEM_PROMPT = """\
You are an expert judge whose purpose is to evaluate the quality of responses
to 
"""

@dataclass
class ReasoningPrompt:
    system: str
    user_text: str
    required_fields: tuple[str, ...] = ("answer", "reasoning")

class JudgeBuilder:
    conn: sqlite3.Connection
    density_rules: str

    def __init__(self, conn: sqlite3.Connection, generation_config_path: str):
        self.conn = conn
        self.generation_config_path = Path(generation_config_path)
        self.density_rules = density_rules_text(load_density_config(self.generation_config_path))

    def __load_image_from_id(self, id: int):
        row = self.conn.execute("SELECT data FROM images WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise KeyError(f"{id} doesn't correspond to an image in this benchmark")
        return row["data"] if row["data"] is not None else None

    def __load_weed_id_item(self, item: JudgeItem) -> str:
        """The judge's user text for a species ID question (multiple choice
        or open-ended, boxed or unannotated). The image is attached
        separately by the caller."""
        is_mcq = item.question_type == "species_id_multiple_choice"
        parts = [
            "Original task: identify the weed species in the image"
            + (" by choosing from the given options." if is_mcq else " (open-ended).")
        ]

        parts.append(f"The base question:\n\n{item.question_text}")
        if is_mcq and item.choices:
            options = "\n".join(f"- {choice}" for choice in item.choices)
            parts.append(f"The options were:\n{options}")

        # Ground truth
        truth = [f"Correct species: {item.expected_answer}"]
        if item.annotated == 1:
            truth.append("The target plant is outlined by a bounding box in the image.")
        else:
            truth.append("The image carries no bounding box annotations.")
        parts.append("Ground truth:\n" + "\n".join(f"- {line}" for line in truth))

        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_fine_grained_item(self, item: JudgeItem) -> str:
        """The judge's user text for a fine-grained ID question. The image needs to
        be attached separately by the caller"""
        parts = [
            "Original task: are weed species in bounding boxes 1 and 2 the same species or different species?"
        ]
        parts.append(f"The base question:\n\n{item.question_text}")
        if item.choices:
            choices = item.choices.copy()
            choices.append("Unsure")
            options = "\n".join(f"- {choice}" for choice in choices)
            parts.append(f"The options were:\n{options}")
        if item.expected_answer:
            parts.append(f"Ground truth: {item.expected_answer}")
        else:
            raise ValueError("No ground truth attached")
        parts.append("The two species are differentiated by different colour bounding boxes and numerical labels")
        parts.append("The response in this case is shown below")
        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_grounded_vqa_item(self, item: JudgeItem) -> str:
        """The judge's user text for a grounded weed visual question answering
        question. The image needs to be attached separately by the caller."""
        parts = [
            "Original task: What labelled bounding box is Species X located in?"
        ]
        parts.append(f"The base question:\n\n{item.question_text}")
        if item.choices:
            options = "\n".join(f"- {choice}" for choice in item.choices)
            parts.append(options)
        if item.expected_answer:
            parts.append(f"Ground truth: {item.expected_answer}")
        else:
            raise ValueError("No ground truth attached")
        parts.append("The options are differentiated by different bounding box colours and labels")
        parts.append("The response in this case is shown below")
        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_localisation_item(self, item: JudgeItem) -> str:
        """The judge's user text for a weed localisation question. The image
        needs to be attached by the caller."""
        parts = [
            "Original task: find instances of Species X and draw bounding boxes around them in your native format",
            "You are provided with the ground truth boxes (green) and the fitted boxes (orange-dotted)"
        ]
        parts.append(f"The base question:\n\n {item.question_text}")
        # No options this time, just provide localisation score and matched Iou, explain what these are
        if item.localisation_score is not None:
            parts.append(f"The localisation score for this question was: {item.localisation_score}")
            parts.append("Localisation score adds up the IoUs for all the bounding box pairs, then divides this by either the number of ground truth boxes or number of fitted boxes, which ever is larger")
        else:
            raise ValueError("No localisation score provided")
        if item.matched_iou is not None:
            parts.append(f"The matched intersection over union for this question was: {item.matched_iou}")
            parts.append("Matched intersection over union adds up the matched IoUs for all bounding box pairs, then divides this by the number of matched pairs")
        else:
            raise ValueError("No matched IoU provided")
        parts.append(f"Rationale: {item.reasoning}")        
        return "\n\n".join(parts)

    def __load_density_item(self, item: JudgeItem, rules_text: str) -> str:
        """The judge's user text for a weed density estimation question.
        rules_text is the category rules the model was given
        (evaluation.prompts.density_rules_text), so the judge sees the same
        definition. The image needs to be attached separately by the caller."""
        if (
            item.expected_answer is None
            or item.true_weed_count is None
            or item.true_weed_coverage_fraction is None
        ):
            raise ValueError("No ground truth attached")
        parts = [
            "Original task: determine the number of weeds and percentage weeds cover in the field image. From this determine the weed category."
        ]
        if item.annotated == 1:
            parts.append("The weeds in this image are outlined with bounding boxes; each box is one weed.")
        else:
            parts.append("The image has no annotations, so the weeds had to be found and told apart from the crop.")
        parts.append(f"The base question:\n\n{item.question_text}")
        parts.append(rules_text)
        if item.choices:
            options = "\n".join(f"- {choice}" for choice in item.choices)
            parts.append(f"The categories were:\n{options}")
        parts.append(
            "Ground truth:\n"
            f"- Correct category: {item.expected_answer}\n"
            f"- Weed count: {item.true_weed_count:g}\n"
            f"- Weed coverage: {item.true_weed_coverage_fraction * 100:.3g}% of the image"
        )
        response = f"Response: {item.answer}"
        if item.est_count is not None:
            response += f"\nEstimated weed count: {item.est_count:g}"
        if item.est_coverage_pct is not None:
            response += f"\nEstimated weed coverage: {item.est_coverage_pct:.3g}% of the image"
        parts.append(f"{response}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def load_judge_item(self, item: JudgeItem) -> str:
        # Work out the type of question
        judge_text = None
        match item.task:
            case "species_id":
                judge_text = self.__load_weed_id_item(item)
            case "fine_grained":
                judge_text = self.__load_fine_grained_item(item)
            case "grounded_vqa":
                judge_text = self.__load_grounded_vqa_item(item)
            case "localisation":
                judge_text = self.__load_localisation_item(item)
            case "density":
                judge_text = self.__load_density_item(item, self.density_rules)
            case _:
                raise ValueError("Invalid question_type")
        return judge_text


