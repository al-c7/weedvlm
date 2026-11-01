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
from dataclasses import dataclass, field
from functools import cache
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
from 0 (worst) to 4 (best) to the rationale, using the rubric for each criterion below:
"""

_SCORING_GUIDANCE = """\
Score each criterion independently, anchoring on the specific claims the rationale makes, and
give the level whose descriptor best fits the rationale.

Write your "notes" first, then the scores. The notes are a brief assessment (one to three
sentences) of the rationale's strengths and problems that justifies your scores, naming the
specific claim behind any score below 4, and noting anything unusual such as a problem with
the question or ground truth. Write "none" only if every score is 4.
"""

SCORE_LEVELS = 5  # scores run 0 .. SCORE_LEVELS - 1


@dataclass(frozen=True)
class Criterion:
    """One rubric criterion: an optional preamble, then one descriptor per
    score, where levels[score] describes that score."""

    preamble: str | None
    levels: tuple[str, str, str, str, str]


@dataclass(frozen=True)
class TaskRubric:
    """The rubric for one task (one tab of the rationale rubrics
    spreadsheet in docs/prompts). knowledge is None for tasks with no
    reference description."""

    general: str
    interpretation: Criterion
    visual_evidence: Criterion
    knowledge: Criterion | None = None
    consistency: Criterion = field(default_factory=lambda: _CONSISTENCY)


_CONSISTENCY = Criterion(
    "The rationale is consistent with itself and the final answer.",
    (
        "The rationale substantially contradicts itself and/or the final answer, its stated reasoning is incompatible with the conclusion entirely.",
        "The rationale contains major contradictions or inconsistencies with itself and/or the final answer.",
        "The rationale is mostly consistent with itself and the final answer, but contains noticeable contradictions or inconsistencies.",
        "The rationale is consistent with itself and the final answer, with only minor inconsistencies that do not substantially affect the overall reasoning.",
        "The rationale is fully consistent with itself and the final answer, with no identifiable contradictions or inconsistencies.",
    ),
)

_KNOWLEDGE = Criterion(
    "Knowledge assesses whether botanical claims agree with the reference and whether those "
    "claims are sufficiently characteristic of the target to provide meaningful support. "
    "Features not mentioned in the reference should not be penalised unless they contradict "
    "the reference or clearly describe a different species.",
    (
        "The stated features substantially contradict the reference description or clearly describe a different species.",
        "Some stated features in the rationale align with the reference description, but major contradictions or incorrect features are present.",
        "Most stated features in the rationale align with the reference description, but the rationale contains some contradictions, incorrect features, or relies primarily on generic features.",
        "The stated features closely align with the reference description, with only minor inaccuracies or reliance on some generic features.",
        "The stated features fully align with the reference description and include characteristics that meaningfully support identification of the target species, with no substantive contradictions.",
    ),
)

_VISUAL_EVIDENCE_PREAMBLE = (
    "Visual evidence assesses whether the visual claims made in the rationale are supported by "
    "clearly observable evidence in the image. Assess only claims explicitly made in the "
    "rationale. Treat a claim as contradictory only when visible evidence conflicts with it. "
    "If the image does not provide sufficient detail to verify a claim, classify that claim "
    "as unverifiable rather than incorrect. Do not infer fine-grained visual characteristics "
    "that cannot be reliably determined from the image."
)

_VISUAL_EVIDENCE_LEVELS = (
    "The rationale's visual claims are substantially contradicted by the image or refer to features that are clearly absent.",
    "Some visual claims in the rationale are supported, but one or more claims are contradicted by the image.",
    "Most visual claims are supported, but the rationale contains at least one noticeable contradiction or relies on unverifiable visual detail.",
    "The visual claims are supported, with only minor uncertainty or unverifiable detail and no major contradictions.",
    "All relevant visual claims are clearly supported by observable evidence, with no substantive contradictions or unverifiable details affecting the rationale.",
)


def _visual_evidence(focus: str, preamble_extra: str = "") -> Criterion:
    """The shared visual-evidence rubric with a task-specific focus sentence on
    every level."""
    levels = tuple(f"{level} {focus}" for level in _VISUAL_EVIDENCE_LEVELS)
    return Criterion(f"{_VISUAL_EVIDENCE_PREAMBLE} {preamble_extra}".strip(), levels)  # type: ignore[arg-type]


# Keyed by JudgeItem.task.
RUBRICS: dict[str, TaskRubric] = {
    "species_id": TaskRubric(
        general=(
            "Visual evidence is about whether the claimed morphology is visible; interpretation is "
            "whether those features support the identified species; knowledge compares claims "
            "against the species reference."
        ),
        interpretation=Criterion(
            "The VLM interprets the identified evidence to support the identified species.",
            (
                "The rationale's interpretation of the stated morphology, structure, colour, and growth habit is inconsistent with, or does not support, the conclusion.",
                "The rationale's interpretation of the stated morphology, structure, colour, and growth habit shows limited support for the conclusion, with major gaps or errors in how the observations are interpreted.",
                "The rationale's interpretation of the stated morphology, structure, colour, and growth habit provides reasonable support for the conclusion but contains noticeable gaps or errors in the interpretation of the observations.",
                "The rationale's interpretation of the stated morphology, structure, colour, and growth habit provides strong support for the conclusion, with only minor gaps, errors, or inconsistencies in the interpretation of the observations.",
                "The rationale clearly interprets the stated morphology, structure, colour, and growth habit to support the conclusion, with no substantive gaps or inconsistencies.",
            ),
        ),
        knowledge=_KNOWLEDGE,
        visual_evidence=_visual_evidence(
            "Focus on claimed morphological and visual features.",
            "Assess claims about visible morphology, structure, colour, and growth habit. Do not "
            "assess whether those features are botanically diagnostic of the identified species; "
            "that is assessed elsewhere.",
        ),
    ),
    "fine_grained": TaskRubric(
        general=(
            "Interpretation should specifically reward discriminative reasoning between "
            "candidate species, not merely correct generic statements."
        ),
        interpretation=Criterion(
            "Interpretation assesses whether the rationale uses relevant visual differences or "
            "similarities to support the selected answer. A visually correct but "
            "non-discriminative feature should not receive a high score if it does not help "
            "distinguish the alternatives.",
            (
                "The rationale's interpretation of the stated observations about the differences between the two specimens is inconsistent with, or does not support, the conclusion.",
                "The rationale's interpretation of the stated observations about the differences between the two specimens provides limited support for the conclusion, with major gaps or errors in how the observations are interpreted.",
                "The rationale's interpretation of the stated observations about the differences between the two specimens provides reasonable support for the conclusion but contains noticeable gaps or errors in the interpretation of the observations.",
                "The rationale's interpretation of the stated observations about the differences between the two specimens provides strong support for the conclusion, with only minor gaps, errors, or inconsistencies in the interpretation of the observations.",
                "The rationale clearly interprets the stated observations about the differences between the two specimens to support the conclusion, with no substantive gaps or inconsistencies.",
            ),
        ),
        visual_evidence=_visual_evidence(
            "Focus on the visual attributes used to determine whether the two weeds are the same or different species."
        ),
    ),
    "grounded_vqa": TaskRubric(
        general=(
            "Visual evidence is whether evidence relevant to the question is visible; interpretation "
            "is whether the evidence supports the particular answer."
        ),
        interpretation=Criterion(
            "The VLM interprets the identified evidence to support its answer to the question.",
            (
                "The stated evidence in the rationale is irrelevant to, does not support, or is inconsistent with the answer to the question.",
                "The rationale provides limited support for the answer, with major gaps or errors in how the relevant evidence is interpreted.",
                "The rationale provides reasonable support for the answer, but contains some noticeable gaps, interpretation errors, or irrelevant evidence.",
                "The rationale provides strong support for the answer using relevant evidence, with only minor gaps or interpretation errors.",
                "The rationale clearly interprets relevant visual and reference-based evidence to fully support the answer, with no substantive gaps.",
            ),
        ),
        knowledge=_KNOWLEDGE,
        visual_evidence=_visual_evidence("Focus only on visual claims relevant to answering the question."),
    ),
    "localisation": TaskRubric(
        general=(
            "Visual evidence concerns visual/spatial claims; the judge doesn't duplicate the existing "
            "localisation metrics."
        ),
        interpretation=Criterion(
            "Interpretation assesses whether the stated evidence explains why the proposed "
            "regions or instances correspond to the target, without evaluating the geometric "
            "accuracy of the localisation.",
            (
                "The rationale's interpretation of the stated observations is inconsistent with, or does not support, why the identified regions/instances correspond to the target(s).",
                "The rationale provides limited support for the conclusion, with major gaps or errors in how the identified regions/instances correspond to the target(s).",
                "The rationale provides reasonable support for the conclusion but contains noticeable gaps or errors in how the identified regions/instances correspond to the target(s).",
                "The rationale provides strong support for the conclusion, with only minor gaps, errors, or inconsistencies in how the identified regions/instances correspond to the target(s).",
                "The rationale clearly interprets the stated observations to support the conclusion, with no substantive gaps or inconsistencies in how the identified regions/instances correspond to the target(s).",
            ),
        ),
        knowledge=_KNOWLEDGE,
        visual_evidence=_visual_evidence(
            "Focus primarily on visual/spatial claims about the proposed target weed locations; "
            "localisation accuracy itself is evaluated quantitatively elsewhere."
        ),
    ),
    "density": TaskRubric(
        general=(
            "Visual evidence concerns claims about visible instances, overlap, distribution, etc; "
            "interpretation concerns whether those observations support the estimate. It does "
            "not judge numerical accuracy, as that is already accounted for."
        ),
        interpretation=Criterion(
            "Interpretation assesses whether the stated observations about visible instances, "
            "spatial distribution, overlap, occlusion, or coverage provide a reasonable basis "
            "for the reported estimate. It does not assess whether the numerical estimate "
            "matches the ground truth.",
            (
                "The rationale's interpretation of the stated observations is inconsistent with, or does not support, the conclusions about the stated count, coverage, and density estimates.",
                "The rationale provides limited support for the conclusion, with major gaps or errors in how the stated count, coverage, and density estimates are made.",
                "The rationale provides reasonable support for the conclusion but contains noticeable gaps or errors in how the stated count, coverage, and density estimates are made.",
                "The rationale provides strong support for the conclusion, with only minor gaps, errors, or inconsistencies in how the stated count, coverage, and density estimates are made.",
                "The rationale clearly interprets the stated observations to support the conclusions about the stated count, coverage, and density estimates, with no substantive gaps or inconsistencies.",
            ),
        ),
        visual_evidence=_visual_evidence(
            "Focus on claims concerning visible instances, distribution, overlap, or occlusion, "
            "rather than counting instances."
        ),
    ),
}

TEXTUAL_FIELDS = ("interpretation", "consistency")
# Scored only for questions that name a species, which have a reference description.
TEXTUAL_KNOWLEDGE_FIELD = "knowledge"
VISUAL_FIELDS = ("visual-evidence",)


def _render_criterion(name: str, criterion: Criterion) -> str:
    lines = [f"- {name.capitalize()}:" + (f" {criterion.preamble}" if criterion.preamble else "")]
    lines += [f"    {score}: {text}" for score, text in enumerate(criterion.levels)]
    return "\n".join(lines)


def _json_example(fields: tuple[str, ...]) -> str:
    body = ", ".join(f'"{f}": <int>' for f in fields)
    return f'```json\n{{"notes": "<string>", {body}}}\n```'


@cache
def build_system_prompt(task: str, kind: "JudgeKind") -> str:
    """The system prompt for one judge kind on one task: the shared
    preamble, the task's rubric for the criteria that judge scores, and the
    reply format. Knowledge appears only for tasks whose rubric has it."""
    rubric = RUBRICS.get(task)
    if rubric is None:
        raise ValueError(f"No rationale rubric for task: {task}")
    match kind:
        case "textual":
            criteria = {"interpretation": rubric.interpretation, "consistency": rubric.consistency}
            if rubric.knowledge is not None:
                criteria["knowledge"] = rubric.knowledge
            inputs = (
                "You have been given the question, the ground truth, and the text the VLM "
                "produced as its answer. You have explicitly not been given the source image, "
                "as you are only the textual judge, so do not assess whether the stated "
                "observations are true of the image; take them as given."
            )
        case "visual":
            criteria = {"visual-evidence": rubric.visual_evidence}
            inputs = (
                "You have been given the question, the ground truth, the source image, and "
                "the text the VLM produced as its answer."
            )
        case _:
            raise ValueError(f"Invalid judge kind: {kind}")
    fields = tuple(criteria)
    return (
        _RATIONALE_PREAMBLE
        + f"\nTask notes: {rubric.general}\n\n"
        + "\n".join(_render_criterion(name, c) for name, c in criteria.items())
        + "\n\n"
        + inputs
        + "\n\n"
        + _SCORING_GUIDANCE
        + f"\nReply with only a JSON object of the following form, where each score is an "
        f"integer from 0 to {SCORE_LEVELS - 1}:\n\n"
        + _json_example(fields)
        + "\n"
    )

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
        system = build_system_prompt(item.task, kind)
        match kind:
            case "textual":
                fields = TEXTUAL_FIELDS
                if RUBRICS[item.task].knowledge is not None:
                    fields += (TEXTUAL_KNOWLEDGE_FIELD,)
                return ReasoningPrompt(
                    system,
                    self.__load_user_text(item, visual=False),
                    fields,
                    attach_image=False,
                )
            case "visual":
                return ReasoningPrompt(
                    system,
                    self.__load_user_text(item, visual=True),
                    VISUAL_FIELDS,
                    attach_image=True,
                )
            case _:
                raise ValueError(f"Invalid judge kind: {kind}")
