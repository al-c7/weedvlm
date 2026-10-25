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
from typing import Literal
import sqlite3

from weedvlm.reasoning.runner import JudgeItem
from weedvlm.storage.database import read_image
from PIL import Image as PILImage, ImageDraw, ImageFont
from weedvlm.evaluation.prompts import density_rules_text, load_density_config

_RATIONALE_PREAMBLE = """\
You are an expert judge whose purpose is to evaluate the quality of responses made by a VLM
to various weed identification questions. The response you have been provided may be
correct or incorrect; you are given the ground truth, but you are not told which. For weed
localisation there is no right or wrong answer, and you are given localisation scores instead.
Your job is to evaluate the post-hoc rationale provided with the response, judging the quality
of the reasoning rather than whether the final answer is right. You will assign integer scores
from 0 (worst) to 10 (best) to the rationale based on the following criteria:
"""

TEXTUAL_RATIONALE_SYSTEM_PROMPT = _RATIONALE_PREAMBLE + """
- Interpretation: The rationale correctly infers the species or condition from the observations
  it states. Take the stated observations as given; do not penalise them here. 0 means the
  inference does not follow from the observations, 5 means it is plausible but incomplete or
  loosely argued, 10 means the observations clearly and correctly lead to the conclusion drawn
  (a rationale that reasons soundly to a wrong answer from wrong observations is not penalised here).
- Consistency: The explanation logically supports the final answer without contradictions.
  0 means it contradicts itself or the answer, 5 means it has minor gaps or tension,
  10 means every step supports the answer.

You have been given the question, the ground truth, and the text the VLM produced as its
answer. You have explicitly not been given the source image, as you are only the textual
judge, so do not assess whether the stated observations are true of the image. Reply with
only a JSON object of the following form, where each value is an integer from 0 to 10:

```json
{"interpretation": <int>, "consistency": <int>}
```
"""

VISUAL_RATIONALE_SYSTEM_PROMPT = _RATIONALE_PREAMBLE + """
- Observation: The rationale correctly identifies evidence that is present in the image.
  0 means the evidence cited is wrong or absent from the image, 5 means some of it is
  right, 10 means all of it is accurate and relevant.
- Grounding: The reasoning relies only on what is visible in the image and the information
  given in the question; it does not hallucinate details. 0 means it relies mainly on invented
  details, 5 means it includes some unsupported claims, 10 means every claim is supported.

You have been given the question, the ground truth, the source image, and the text the VLM
produced as its answer. Reply with only a JSON object of the following form, where each value
is an integer from 0 to 10:

```json
{"observation": <int>, "grounding": <int>}
```
"""

TEXTUAL_FIELDS = ("interpretation", "consistency")
VISUAL_FIELDS = ("observation", "grounding")

JudgeKind = Literal["textual", "visual"]

@dataclass
class ReasoningPrompt:
    system: str
    user_text: str
    required_fields: tuple[str, ...]
    attach_image: bool

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

    def __load_weed_id_item(self, item: JudgeItem, visual: bool) -> str:
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
        if visual:
            if item.annotated == 1:
                truth.append("The target plant is outlined by a bounding box in the image.")
            else:
                truth.append("The image carries no bounding box annotations.")
        parts.append("Ground truth:\n" + "\n".join(f"- {line}" for line in truth))

        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_fine_grained_item(self, item: JudgeItem, visual: bool) -> str:
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
        if visual:
            parts.append("The two species are differentiated by different colour bounding boxes and numerical labels")
        else:
            parts.append("The two plants are referred to by their numerical labels, 1 and 2")
        parts.append("The response in this case is shown below")
        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_grounded_vqa_item(self, item: JudgeItem, visual: bool) -> str:
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
        if visual:
            parts.append("The options are differentiated by different bounding box colours and labels")
        parts.append("The response in this case is shown below")
        parts.append(f"Response: {item.answer}\n\nRationale: {item.reasoning}")
        return "\n\n".join(parts)

    def __load_localisation_item(self, item: JudgeItem, visual: bool) -> str:
        """The judge's user text for a weed localisation question. The image
        needs to be attached by the caller."""
        parts = [
            "Original task: find instances of Species X and draw bounding boxes around them in your native format"
        ]
        if visual:
            parts.append("You are provided with the ground truth boxes (green) and the fitted boxes (orange-dotted)")
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

    def __load_density_item(self, item: JudgeItem, rules_text: str, visual: bool) -> str:
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
        if visual:
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

    def __load_user_text(self, item: JudgeItem, visual: bool) -> str:
        match item.task:
            case "species_id":
                return self.__load_weed_id_item(item, visual)
            case "fine_grained":
                return self.__load_fine_grained_item(item, visual)
            case "grounded_vqa":
                return self.__load_grounded_vqa_item(item, visual)
            case "localisation":
                return self.__load_localisation_item(item, visual)
            case "density":
                return self.__load_density_item(item, self.density_rules, visual)
            case _:
                raise ValueError(f"Invalid task: {item.task}")

    def build(self, item: JudgeItem, kind: JudgeKind) -> ReasoningPrompt:
        """The full prompt for one judge of one response. The textual judge
        never sees the image or any text that refers to it."""
        match kind:
            case "textual":
                return ReasoningPrompt(
                    TEXTUAL_RATIONALE_SYSTEM_PROMPT,
                    self.__load_user_text(item, visual=False),
                    TEXTUAL_FIELDS,
                    attach_image=False,
                )
            case "visual":
                return ReasoningPrompt(
                    VISUAL_RATIONALE_SYSTEM_PROMPT,
                    self.__load_user_text(item, visual=True),
                    VISUAL_FIELDS,
                    attach_image=True,
                )
            case _:
                raise ValueError(f"Invalid judge kind: {kind}")
