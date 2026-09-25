# Benchmark Builder

Composes the _final_ evaluation benchmark from generated + QA'd questions: a controlled number of
questions per class (species, same/different-species, density category, ...) and a controlled number
of classes, per task. Configurable both in `benchmark-config.json` and live in a browser GUI.

This app sits downstream of the other two: `apps/weedcoco-review` (dataset review) →
`generate-all-questions.py` (question generation) → `apps/question-preview` (per-question QA, flag
good/bad) → **this app** (final selection + export).

## Prerequisites

- [Deno](https://deno.com) (any recent version).
- A generated `questions.json` -- see `generate-all-questions.py` in the repo root's `src/weedvlm/`.
- Optionally, a `question-flags.json` from `apps/question-preview` -- flagged-bad questions are
  excluded from selection by default (`exclude_flagged` in the config).

## Class keys

Every question in a generated file carries a `benchmark_class` field, computed once in Python at
generation time (see `src/weedvlm/types/questions.py` / `src/weedvlm/pipeline/*.py`), so this app
never has to re-derive it from an id or question text:

| Task                                                  | `benchmark_class`                                   |
| ----------------------------------------------------- | --------------------------------------------------- |
| `species_id_multiple_choice`, `species_id_open_ended` | the correct species' display name                   |
| `species_localisation`                                | the species being asked about                       |
| `fine_grained_id`                                     | `"same_species"` or `"different_species"`           |
| `density_estimation`                                  | the density category (`none`/`low`/`medium`/`high`) |

Density estimation's boxed and unannotated rows for the same source image count as one selectable
unit (selecting the image selects both rows), per `docs/WEED-DENSITY.md` -- every other task's rows
stand alone.

## Running

From this directory, with no arguments:

```sh
deno task serve
```

Defaults (matching `apps/question-preview`'s so the two apps share files naturally):

- `--questions`: `<repo root>/questions/all-questions.json` (repeatable).
- `--flags`: alongside the first `--questions` file, `question-flags.json` -- read-only here; flag
  questions from `apps/question-preview`.
- `--config`: `benchmark-config.json` in this directory (tracked in git -- this is the checked-in
  setup, edit it by hand or through the GUI's "Save config").
- `--out-dir`: alongside the first `--questions` file, `final-benchmark/` (already covered by the
  repo root `.gitignore`'s `questions/` entry).
- `--port`: `8790`.

`deno task dev` runs the same thing with `--watch`, for editing the app itself.

## Using the UI

1. Each task gets a card in the sidebar with two numbers: **questions / class** and **num classes**.
   Editing either live-recomputes the selection (seeded, so the same numbers always reproduce the
   same questions) and updates that card's stats line.
2. Click **Browse selection** on a task to see its class table (available vs. selected vs. target
   per class) in the main panel. Click a class row to see a thumbnail gallery of every selected
   image in that class -- a quick visual sanity pass over the whole class at once. Click a thumbnail
   to open the single-question viewer (image, question text, revealed ground truth) with Prev/Next.
   This is all read-only; flagging still happens in `apps/question-preview`.
3. **Save config** writes the current numbers to `benchmark-config.json`. **Export benchmark**
   writes `<out-dir>/<task>.json` (the selected question objects, same shape as
   `all-questions.json`) plus `<out-dir>/summary.json` for every task in the config.

Selection is v1-scope automatic: classes are ranked by how many questions are available and the top
`num_classes` are kept; there's no per-question manual lock/swap yet, beyond re-running with
different numbers or excluding questions via flags in `apps/question-preview`.
