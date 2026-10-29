"""
Builds the prompt to send to OpenRouter for the reasoning evaluation.
The image is pulled from the database and the prompt provides a render
with all of the ground truth provided.

The overall template for each judge prompt is the same, the judge context
is provided, followed by the questions, the response, and the output.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
import sqlite3
import threading

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

_SCORING_GUIDANCE = """\
Use the full 0-10 range and score each criterion independently, anchoring on the specific claims
the rationale makes. A 10 requires every claim to be correct and specific enough to rule out
look-alikes in this case. Claims that are correct but generic, so that they would fit many plants,
cap a criterion at 8. A mix of correct and incorrect claims is around 5, and mostly incorrect
claims score 2 or below.

Write your "notes" first, then the scores. The notes are a brief assessment (one to three
sentences) of the rationale's strengths and problems that justifies your scores, naming the
specific claim behind any score below 10, and noting anything unusual such as a problem with
the question or ground truth. Write "none" only if every score is 10.
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
- Knowledge: The features the rationale states for the plant agree with the reference
  description of the target species given in the question. Only features that contradict the
  description, or that clearly belong to a different plant, count against it; a feature that is
  merely missing from the description does not. 0 means the stated features contradict the
  description, 5 means they are mixed, or only generic features that fit many species, 10 means
  the stated features are all consistent with the description and distinctive of the species.
  Score knowledge only when a reference description is provided; otherwise set it to null.

You have been given the question, the ground truth, and the text the VLM produced as its
answer. You have explicitly not been given the source image, as you are only the textual
judge, so do not assess whether the stated observations are true of the image.

""" + _SCORING_GUIDANCE + """
Reply with only a JSON object of the following form, where each score is an integer from 0
to 10 (knowledge may be null, as described above):

```json
{"notes": "<string>", "interpretation": <int>, "consistency": <int>, "knowledge": <int or null>}
```
"""

VISUAL_RATIONALE_SYSTEM_PROMPT = _RATIONALE_PREAMBLE + """
- Observation: The rationale correctly identifies evidence that is present in the image.
  0 means the evidence cited is wrong or absent from the image, 5 means some of it is
  right, 8 means all of it is accurate but generic, 10 means every feature cited is present in the
  image and specific enough to tell the plant from look-alikes.
- Grounding: The reasoning relies only on what is visible in the image and the information
  given in the question; it does not hallucinate details. 0 means it relies mainly on invented
  details, 5 means it includes some unsupported claims, 8 means every claim is supported but
  generic, 10 means every claim is supported and specific to what this image shows.

You have been given the question, the ground truth, the source image, and the text the VLM
produced as its answer.

""" + _SCORING_GUIDANCE + """
Reply with only a JSON object of the following form, where each score is an integer from 0 to 10:

```json
{"notes": "<string>", "observation": <int>, "grounding": <int>}
```
"""

TEXTUAL_FIELDS = ("interpretation", "consistency")
# Scored only for questions that name a species, which have a reference description.
TEXTUAL_KNOWLEDGE_FIELD = "knowledge"
VISUAL_FIELDS = ("observation", "grounding")

JudgeKind = Literal["textual", "visual"]

def load_species_descriptions(path: Path) -> dict[str, str]:
    """species name -> description, from the YAML file (see
    species-descriptions.yaml at the repo root). Each species has a
    `description` (shown to the judges) and `sources` (links, kept for
    provenance and never shown to them)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must map each species name to a description and sources")
    descriptions: dict[str, str] = {}
    for name, entry in raw.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"description", "sources"}
            or not isinstance(entry["description"], str)
            or not isinstance(entry["sources"], list)
            or not all(isinstance(link, str) for link in entry["sources"])
        ):
            raise ValueError(
                f"{path}: {name!r} must have exactly a `description` string and a `sources` "
                "list of links"
            )
        descriptions[str(name)] = entry["description"]
    return descriptions


GROUND_TRUTH_COLOUR = (0, 200, 0)
FITTED_COLOUR = (255, 140, 0)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: list[float], colour, width: int) -> None:
    """A rectangle drawn as dashes, each 5 line widths long with a gap of 3,
    so it reads as dashed however large the image is."""
    x1, y1, x2, y2 = box
    dash, gap = width * 5, width * 3
    for (ax, ay), (bx, by) in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            draw.line(
                [(ax + dx * pos, ay + dy * pos), (ax + dx * end, ay + dy * end)],
                fill=colour,
                width=width,
            )
            pos += dash + gap


def render_localisation_image(
    image: PILImage.Image, truth_xyxy: list[list[float]], pred_xyxy: list[list[float]]
) -> bytes:
    """Draws the ground truth boxes (solid green) and the model's boxes
    (dashed orange), both (x1, y1, x2, y2) in pixels, under a legend strip.
    Returns JPEG bytes."""
    image = image.convert("RGB")
    width, height = image.size
    line = max(2, round(max(width, height) / 400))
    draw = ImageDraw.Draw(image)
    for box in truth_xyxy:
        draw.rectangle(box, outline=GROUND_TRUTH_COLOUR, width=line)
    for box in pred_xyxy:
        _dashed_rectangle(draw, box, FITTED_COLOUR, line)

    font_size = max(14, round(height / 40))
    font = ImageFont.load_default(size=font_size)
    strip = round(font_size * 2)
    canvas = PILImage.new("RGB", (width, height + strip), "white")
    canvas.paste(image, (0, strip))
    draw = ImageDraw.Draw(canvas)
    mid = strip / 2
    swatch = font_size * 2
    x = font_size / 2
    draw.line([(x, mid), (x + swatch, mid)], fill=GROUND_TRUTH_COLOUR, width=line)
    draw.text((x + swatch + font_size / 2, mid), "Ground truth", fill="black", font=font, anchor="lm")
    x += swatch + font_size / 2 + draw.textlength("Ground truth", font=font) + font_size * 1.5
    _dashed_rectangle(draw, [x, mid, x + swatch, mid], FITTED_COLOUR, line)
    draw.text((x + swatch + font_size / 2, mid), "Fitted (model)", fill="black", font=font, anchor="lm")

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=92)
    return out.getvalue()


@dataclass
class ReasoningPrompt:
    system: str
    user_text: str
    required_fields: tuple[str, ...]
    attach_image: bool

class JudgeBuilder:
    conn: sqlite3.Connection
    density_rules: str

    def __init__(
        self,
        conn: sqlite3.Connection,
        generation_config_path: str,
        species_descriptions: dict[str, str],
    ):
        self.conn = conn
        self.species_descriptions = {
            name.strip().lower(): text.strip() for name, text in species_descriptions.items()
        }
        # The connection is shared with worker threads that read images.
        self._db_lock = threading.Lock()
        self.generation_config_path = Path(generation_config_path)
        self.density_rules = density_rules_text(load_density_config(self.generation_config_path))

    @staticmethod
    def target_species(item: JudgeItem) -> str | None:
        """The species a question is about, or None for tasks that name
        none (fine-grained, density)."""
        match item.task:
            case "species_id":
                return item.expected_answer
            case "grounded_vqa":
                return item.benchmark_class
            case "localisation":
                return item.target_species
            case _:
                return None

    def check_descriptions(self, items: list[JudgeItem]) -> None:
        """Fails loudly if any item's target species has no description."""
        missing = sorted(
            {
                species
                for item in items
                if (species := self.target_species(item))
                and not self.species_descriptions.get(species.strip().lower())
            }
        )
        if missing:
            raise ValueError(
                f"No description for {len(missing)} target species (add them to the species "
                f"descriptions file): {', '.join(missing)}"
            )

    def __species_part(self, item: JudgeItem) -> str | None:
        species = self.target_species(item)
        if species is None:
            return None
        description = self.species_descriptions.get(species.strip().lower())
        if not description:
            raise ValueError(f"No description for target species {species!r}")
        return (
            f"Reference description of the target species, {species}:\n{description}\n\n"
            "This is for your reference only; the VLM was not given it, and not every feature "
            "described need be visible in this particular image."
        )

    def load_image(self, image_id: int) -> tuple[bytes, str]:
        """An image's (bytes, media type): from the database if they were
        stored, else from the file at its path."""
        with self._db_lock:
            row = self.conn.execute(
                "SELECT path, data, media_type FROM images WHERE id = ?", (image_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"{image_id} doesn't correspond to an image in this benchmark")
        data = row["data"] if row["data"] is not None else Path(row["path"]).read_bytes()
        return data, row["media_type"]

    def image_for(self, item: JudgeItem) -> tuple[bytes, str]:
        """The (bytes, media type) of the image to attach for this item:
        the localisation render, else the stored image."""
        if item.task == "localisation":
            return self.render_localisation(item), "image/jpeg"
        return self.load_image(item.image_id)

    def render_localisation(self, item: JudgeItem) -> bytes:
        """The image for a localisation question as JPEG bytes: ground truth
        boxes solid green, the model's boxes dashed orange, with a legend
        strip above the image."""
        if item.boxes is None:
            raise ValueError("No ground truth boxes attached")
        image = PILImage.open(io.BytesIO(self.load_image(item.image_id)[0]))
        return render_localisation_image(image, item.boxes, item.pred_boxes or [])

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

        parts.append(self.__species_part(item))
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
        parts.append(self.__species_part(item))
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
        n_pred = len(item.pred_boxes or [])
        n_truth = len(item.boxes or [])
        if visual:
            if n_pred:
                parts.append("You are provided with the ground truth boxes (green) and the fitted boxes (orange-dashed)")
            else:
                parts.append("You are provided with the ground truth boxes (green); the model's boxes could not be drawn")
        parts.append(f"The base question:\n\n {item.question_text}")
        parts.append(self.__species_part(item))
        if n_pred:
            parts.append(f"The model drew {n_pred} box(es); there are {n_truth} ground truth box(es).")
        else:
            parts.append(
                f"The model's boxes could not be read, so no boxes were drawn; there are {n_truth} ground truth box(es)."
            )
        # No options this time, just provide localisation score and matched Iou, explain what these are
        if item.localisation_score is None:
            raise ValueError("No localisation score provided")
        parts.append(f"The localisation score for this question was: {item.localisation_score}")
        parts.append("Localisation score adds up the IoUs for all the bounding box pairs, then divides this by either the number of ground truth boxes or number of fitted boxes, which ever is larger")
        if item.matched_iou is not None:
            parts.append(f"The matched intersection over union for this question was: {item.matched_iou}")
            parts.append("Matched intersection over union adds up the matched IoUs for all bounding box pairs, then divides this by the number of matched pairs")
        else:
            parts.append("No fitted box overlapped any ground truth box, so there is no matched intersection over union for this question.")
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
                user_text = self.__load_user_text(item, visual=False)
                fields = TEXTUAL_FIELDS
                if self.target_species(item):
                    fields += (TEXTUAL_KNOWLEDGE_FIELD,)
                else:
                    user_text += (
                        "\n\nNo reference description is provided for this question, so set "
                        "knowledge to null."
                    )
                return ReasoningPrompt(
                    TEXTUAL_RATIONALE_SYSTEM_PROMPT, user_text, fields, attach_image=False
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
