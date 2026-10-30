# LLM-as-a-judge based reasoning evaluation

An LLM judges the post-hoc rationale that accompanies each sampled response, whether the response is correct or not. It scores the quality of the reasoning, not whether the final answer is right.

## Judges

Each response goes to two judges (`src/weedvlm/reasoning/prompt.py`):

- **Textual judge** sees the question, ground truth, response and rationale, but never the image. It scores `interpretation`, `consistency` and, for tasks with a reference species description, `knowledge`.
- **Visual judge** also gets the image (for localisation, rendered with the ground truth and fitted boxes). It scores `visual-evidence`.

## Rubrics

Every score is an integer from 0 (worst) to 4 (best). The rubrics come from `docs/prompts/WeedVLM rationale rubrics.xlsx`, one tab per task, transcribed into `RUBRICS` in `prompt.py` (keyed by `JudgeItem.task`). Each criterion has a preamble and one descriptor per score. The system prompt for a (task, judge) pair is rendered from the task's rubric by `build_system_prompt`.

- **Interpretation**: whether the stated observations support the conclusion. Task-specific (e.g. fine-grained rewards discriminative reasoning, density does not judge numerical accuracy).
- **Consistency**: whether the rationale agrees with itself and the final answer. Shared by all tasks.
- **Knowledge**: whether the stated features agree with the species reference description. Present only for `species_id`, `grounded_vqa` and `localisation`; the other tasks have no description, so the field is neither asked for nor stored.
- **Visual evidence** (`visual-evidence`): whether the visual claims are supported by the image. Shared wording, with a per-task focus sentence appended to every level (claims that cannot be verified are unverifiable, not wrong).

The judge replies with JSON: `notes` first, then one integer per criterion. `judge.py` rejects any score outside 0-4 and retries.
