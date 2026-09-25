# Question Preview

A local browser UI for QA-reviewing generated benchmark questions: step through them one at a time,
reveal the ground-truth answer, and flag anything that looks wrong. Reads one or more
`questions.json` files produced by weedvlm's Python question generators and writes flag decisions
back to disk as you go.

## Prerequisites

- [Deno](https://deno.com) (any recent version).
- At least one generated questions file -- see "Generating questions" below.

## Generating questions

The easiest way, from the repo root:

```sh
python src/weedvlm/generate-all-questions.py --out-dir questions/
```

This runs every task (species ID, fine-grained ID, density estimation, species localisation) in one
pass and, by default, pulls from **every** reviewed dataset (every `review-*.json` under
`apps/weedcoco-review`) -- no need to list them or run each generator separately. It writes one file
per task (`species-id.json`, `fine-grained.json`, `density.json`, `localisation.json`) plus
`all-questions.json` combining all of them, so the previewer can be pointed at either the whole set
or a single task. Pass `--review` (repeatably) to restrict to specific datasets instead of all of
them.

To generate a single task, or to tune task-specific options (area/colour similarity thresholds,
per-image caps, choice counts, etc) beyond what `generate-all-questions.py` exposes, use the
individual scripts instead -- they take the same `--review` default:

```sh
python src/weedvlm/generate-species-id-questions.py --out questions/species-id.json
python src/weedvlm/generate-fine-grained-questions.py --out questions/fine-grained.json
python src/weedvlm/generate-density-questions.py --out questions/density.json
python src/weedvlm/generate-localisation-questions.py --out questions/localisation.json
```

Every script prints which review file(s) it used, and writes a `rendered/` directory of
box-annotated images alongside its output. Run any script with `--help` for its full set of options.
Every question records which dataset its source image came from (`source.dataset_name`), so
combining datasets never loses that trail.

## Running the previewer

From this directory, with no arguments:

```sh
deno task serve
```

With no `--questions` given, this defaults to `<repo root>/questions/all-questions.json` -- i.e.
whatever `generate-all-questions.py` last wrote, covering every reviewed dataset and task. If that
file doesn't exist yet, the server prints a reminder to run `generate-all-questions.py` first and
exits.

Pass `--questions` (repeatably) to review specific files instead -- individual task files, files
from a restricted `--review` generation run, or several runs at once:

```sh
deno task serve \
  --questions ../../questions/species-id.json \
  --questions ../../questions/fine-grained.json
```

Other flags:

- `--out <path>` -- where flag decisions are saved (default: `question-flags.json` next to the first
  `--questions` file, or next to the default `all-questions.json`).
- `--port <n>` -- default `8788`.

Then open the printed `http://localhost:8788` (or your `--port`) in a browser. `deno task dev` runs
the same thing with `--watch`, for editing the app itself.

## Using the UI

1. Pick a dataset/question-type filter (optional) and click **Build preview queue**.
2. For each question: read the image and question text, form your own answer, then reveal the ground
   truth.
   - Multiple-choice questions (species ID, fine-grained ID, localisation) show the choice list
     directly; the correct one highlights on reveal.
   - Localisation choices are plain box numbers -- each gets a colour swatch matching that number's
     box colour in the image, so you can match a choice to a box without hunting for the tiny
     in-image label.
   - Open-ended and density-estimation questions show nothing until revealed (there's no fixed
     option list to pre-empt), then display the ground-truth answer / count / coverage / density
     category.
   - The badge next to the question type shows **boxed** or **unannotated**, i.e. whether the image
     carries ground-truth boxes.
3. Mark **Looks good** or **Flag issue** (an issue prompts for an optional note) -- either advances
   to the next question.

Keyboard shortcuts: `Space` reveal answer, `G` looks good, `B` flag issue, `←`/`→` navigate.

Flags are saved as you go to the `--out` file (a `question-flags.json` per review session,
gitignored) and reloaded on the next run, so a review pass can be paused and resumed.
