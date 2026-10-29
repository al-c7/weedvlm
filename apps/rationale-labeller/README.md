# weedvlm rationale labeller

A local browser app for scoring reasoning by hand, so the LLM judges in `src/weedvlm/reasoning/` can
be checked against a human. It is a Python app (not Deno) because it reuses the pipeline's own code:
`JudgeBuilder` builds exactly the text and image the visual judge saw, so you mark the same thing.

Run it from the repo root, pointing at one or more judge runs:

    uv run apps/rationale-labeller/server.py --judgements reasoning-results/pilot3 [--judgements <other run>]

| Flag                 | Default                                          | What                                       |
| -------------------- | ------------------------------------------------ | ------------------------------------------ |
| `--judgements <dir>` | required, repeatable                             | A reasoning run directory.                 |
| `--config <file>`    | `reasoning.example.yaml`                         | Where the results, database and species descriptions are found. |
| `--labels-out <file>`| `rationale-labels.parquet` in this directory     | Where scores are saved.                    |
| `--limit <n>`        | `60`                                             | Responses to queue.                        |
| `--seed <n>`         | `0`                                              | Shuffle seed.                              |
| `--labeller <name>`  | `human`                                          | Recorded on every row.                     |
| `--port <n>`         | `8789`                                           |                                            |

## What you see

The queue is the responses judged in the given runs, dealt round-robin over (task, question type,
correct/incorrect) so a short session covers every task and includes wrong answers. Each is shown as
the visual judge's user text (question, ground truth, species description, response and rationale)
beside the image (the localisation render for localisation). The model that wrote the response is
hidden, as it is from the judges.

You give the same 0-10 scores as the judges: interpretation, consistency and knowledge (knowledge
only when the question names a species), observation and grounding. Seeing the image while scoring
the first three is the one difference from the textual judge. **Skip** records that an item can't be
judged and keeps it out of the agreement figures.

The judges' scores stay hidden until an item is saved, so they can't anchor yours; they are then
revealed beneath the form.

## Storage

One parquet row per response, keyed by `response_id`: the question id, task, question type and
correctness, the five scores (nullable integers; null when not asked or skipped), `notes`, `skipped`,
`labeller` and `labelled_at`. The file is rewritten whole through a temp file on every save, so a
crash can't corrupt it, and reloaded on the next run. Join to `judgements.jsonl` on `response_id`.

## Agreement

`/api/agreement` (shown under the form) compares your scores with each judge run field by field over
the items you both scored: Spearman correlation, mean absolute difference, share within one point,
and both means. Correlation needs more than two items to be defined, and is only meaningful with a
few dozen.
