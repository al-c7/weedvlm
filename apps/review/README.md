# weedvlm review

A local browser app for the review steps around the benchmark, in one place. Three views, switched
with the tabs at the top:

| View          | What it's for                                                                  |
| ------------- | ------------------------------------------------------------------------------ |
| **Datasets**  | Review the boxes in a WeedCOCO dataset: accept/reject images and single boxes. |
| **Questions** | Step through generated questions, reveal the answer, flag any that look wrong. |
| **Results**   | Browse evaluation runs: scores, wrong answers, and errors.                     |

Every data source is optional. A view whose source isn't available shows how to enable it, so you
can start the server with whatever you have.

## Prerequisites

- [Deno](https://deno.com) (any recent version).
- For **Questions**: generated questions (`src/weedvlm/scripts/generate-all-questions.py`).
- For **Results**: at least one run from `run-evaluation.py`. Scores need `score-evaluation.py` to
  have been run on it.

## Running

From this directory:

```sh
deno task serve
```

Then open the printed `http://localhost:8787`. `deno task dev` does the same with `--watch`. Views
are linkable (`#datasets`, `#questions`, `#results`).

| Flag                  | Default                                                | What                                             |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `--dataset <file>`    | none                                                   | A WeedCOCO JSON to review. Enables **Datasets**. |
| `--images <dir>`      | `<dataset dir>/images`, else that dir                  | Where the dataset's images are.                  |
| `--review-out <file>` | `<dataset dir>/review.json`                            | Where accept/reject decisions are saved.         |
| `--questions <file>`  | `<repo>/questions/all-questions.json` if it exists     | Questions to load; repeatable.                   |
| `--flags-out <file>`  | `question-flags.json` next to the first questions file | Where question flags are saved.                  |
| `--results <dir>`     | `<repo>/results` if it exists                          | Directory of run subdirectories.                 |
| `--port <n>`          | `8787`                                                 |                                                  |

For example, to review one dataset alongside everything else:

```sh
deno task serve --dataset ../../.datasets/CottonWeedDet12/weedcoco.json
```

Decisions and flags are saved as you go and reloaded next run, so a review can be paused and
resumed. They're the only files the app writes.

## Datasets

1. Tick the species to review and set how many images per species to sample, then **Build review
   queue**.
2. For each image, untick any box that's wrong, then **Accept** or **Reject** the image.

Keys: `G` accept, `B` reject, `←`/`→` navigate, `1`-`9` toggle that box's good/bad flag.

## Questions

1. Filter by dataset and question type (optionally to flagged questions only), then **Build preview
   queue**.
2. Read the question, form your own answer, then reveal the ground truth. Grounded VQA choices are
   box numbers; each has a swatch matching that box's colour in the image.
3. Mark **Looks good** or **Flag issue** (which asks for an optional note).

Keys: `Space` reveal, `G` looks good, `B` flag, `←`/`→` navigate.

## Results

Pick a run in the sidebar. Runs are read straight from `results/<run>/`:

| File                   | Used for                                                    |
| ---------------------- | ----------------------------------------------------------- |
| `responses.jsonl`      | Every response, including raw replies, errors, and retries. |
| `scores.json`          | The **Scores** tables.                                      |
| `scored-responses.csv` | Each response's outcome (correct, incorrect, ...).          |
| `progress.json`        | The run's state, and models stopped early by failures.      |
| `run.log`              | The log at the bottom of **Errors**.                        |

The app never re-scores anything: scores are exactly what `score-evaluation.py` wrote. If responses
were written after the last scoring, the run is marked "scores out of date", and those responses
show as `unscored` rather than with a stale outcome.

- **Scores**: outcome counts per model, accuracy per suite and model (bar = accuracy over answered
  questions, tick = random-guess accuracy; tick "Count failed responses as wrong" to include
  failures), and a per-class / per-dataset breakdown for the suite you click. Density suites add the
  count/coverage errors and confusion matrices.
- **Responses**: every response, filterable by model, suite, dataset, outcome and text. **Wrong
  answers** shows incorrect, invalid and unsure ones. Open one to see the image, the question with
  the expected and given answers marked, the model's reasoning, and the raw reply and request
  details. `←`/`→` step through the filtered list, `Esc` returns to it.
- **Errors**: failures per model, and every distinct error (with counts) alongside retries and
  warnings that didn't fail a response. **View responses** lists exactly the responses that failed
  that way.

Outcomes: `correct`, `incorrect`, `unsure` (the model declined), `invalid` (not one of the options),
`failed` (no usable response, see **Errors**), `unscored`.

Use **Refresh** to pick up a run that's still going.

## Development

```sh
deno task check   # type-check the server
deno task lint
deno task fmt
deno task gen-fixtures   # synthetic dataset + images under fixtures/, for trying Datasets
```
