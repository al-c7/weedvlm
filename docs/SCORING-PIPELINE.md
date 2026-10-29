# Scoring pipeline: how the code works

This explains the internals of the code that marks stored responses and
aggregates them into metrics. It makes no API calls. It reads a finished
(or partial) evaluation run and the benchmark it was run against, from
either storage format (JSON files or a SQLite database, see
[DATABASE.md](DATABASE.md)).

Code: `src/weedvlm/evaluation/scoring.py`, `bboxes.py`, `metrics.py`, and
the orchestrating script `src/weedvlm/scripts/score-evaluation.py`.

## Structure

Scoring is a pure transformation in two layers, followed by a write:

```
ResponseRecord  +  Question  --score_response-->  ScoredResponse   (one row per response)
list[ScoredResponse]  --compute_metrics-->  nested dict of metrics  (per model, per suite)
both  --store.write_scores-->  files in the run directory, or database tables
```

All the judgement lives in `scoring.py` and `bboxes.py`, and all the
aggregation in `metrics.py`. This keeps marking rules separate from
statistics. Nothing is rendered: the script stores the per-response
outcomes and the metrics and prints one line saying where.

## Orchestrator (`scripts/score-evaluation.py`)

`main` runs these steps:

1. Open the run's store. A run directory gives a `RunStore` (JSON). A run
   name with a database path gives a `SqliteRunStore`. Read the run's
   settings with `read_run_info` and re-validate the stored `config` into an
   `EvaluationConfig`. The benchmark to score against is therefore whatever
   the run used (unless overridden). For a database run, the config's
   `storage` is pointed at the database that was passed, wherever the file
   was when the run was made.
2. `store.load_records()` gives the latest record per (model, question).
3. Build the alias table and the per-model box-format overrides. Overrides
   come from the model entries in the config and are then replaced by any
   command-line values.
4. `open_benchmark(config)` loads the answer key. This reuses the
   evaluation pipeline's loader and its validations.
5. `score_run` joins and scores. If any responses are stale (see below) it
   refuses unless told to skip them.
6. `compute_metrics`, then `store.write_scores(scored, summary)`. `summary`
   is the metrics plus when and how they were produced.

`write_scores` is the only format-specific step:

- `RunStore` writes `scores.json` (the summary), `scored-responses.csv` (the
  `ScoredResponse` dataclasses as a DataFrame) and
  `open-ended-mismatches.json` (open-ended rows whose outcome is
  `incorrect`, for alias review).
- `SqliteRunStore` replaces the run's rows in the `scores` table and writes
  the summary to `scoring_runs`. The mismatches are a query on the
  `response_details` view, so nothing separate is stored for them.

## Joining and stale detection (`score_run`)

The join key is `question_id`. `score_run` builds `by_id` from the
benchmark's questions and asks the benchmark for the current hash of every
question image (`Benchmark.image_hashes`). The JSON benchmark hashes the
files across a thread pool. The SQLite benchmark returns the hashes stored
in its `images` table. For each stored response it then drops it into one of
three buckets:

- **missing:** the id is no longer in the benchmark,
- **changed:** the record's `image_sha256` differs from the image's current
  hash,
- **scored:** otherwise, passed on to `score_response`.

In a database, "missing" can't happen, because a response row requires its
question row. "Changed" only arises for migrated responses that were
already stale.

The hash check exists because ids are reused when the benchmark is
regenerated while pictures may change. In grounded VQA the labels are
shuffled per generation, so an answer of "2" could point at a different
species now. Scoring against a mismatched key would produce wrong numbers
silently, so the script refuses by default.

## Per-response scoring (`scoring.py`)

### The `ScoredResponse` row

A flat dataclass with identifying fields, `suite`, `status`, `outcome`,
`expected`, `answer`, `predicted_label`, `chance`, then task-specific
columns that are `None` where they don't apply: density errors, localisation
metrics, and usage data. Being flat is deliberate. `vars(s)` turns it
straight into a DataFrame row, which both the metrics code and the CSV use.

`predicted_label` is the option or category the answer *mapped to*, or
`None` if it matched nothing. It is the input to macro-F1, as distinct from
`answer`, which is the raw text.

### Dispatch: `score_response`

First it assembles the common fields (`_base`, `_expected`, `_chance`). A
record with `status != "ok"` or no `parsed` becomes outcome `failed`
immediately. Otherwise a `match` on the question class picks the scorer:

| Question | Scorer | Outcomes |
| --- | --- | --- |
| `LocalisationQuestion` | `_score_localisation` | `scored`, `invalid` |
| `OpenEndedQuestion` | `open_ended_correct` | `correct`, `incorrect`, `unsure` |
| `DensityEstimationQuestion` | `_score_density` | `correct`, `incorrect`, `invalid` (+ numeric errors) |
| `MultipleChoiceQuestion` | `_score_choice` | `correct`, `incorrect`, `unsure`, `invalid` |

Localisation is checked first because its parsed dict has no `answer` key.

### Text normalisation

`_light` is the baseline normaliser: strip, remove surrounding quote
characters and full stops, collapse whitespace, case-fold. `_UNSURE` is a
set of phrases (`unsure`, `unknown`, `i don't know`, ...) treated as
declining to answer. `_number` accepts ints, floats, and numeric strings
with a trailing `%`, and rejects booleans (which are ints in Python).

### Multiple choice: `_score_choice`

Builds `{normalised option: original option}` and looks up the normalised
answer. A hit is `correct` if the matched option is the answer-index option,
else `incorrect`. A miss that is an "unsure" phrase is `unsure`. Any other
miss is `invalid`. Grounded VQA uses `_label` in place of `_light` on both
sides. This regex reduces `2`, `2.0`, `"Box 2"`, `"#2"` and the like to the
bare number. Unsure is not accepted for grounded VQA, because the prompt
doesn't offer it.

### Open ended: `open_ended_correct`

Free text needs more careful matching. The set of accepted names is the
truth plus its aliases (`load_aliases` case-folds the keys). The answer is
split into candidate parts at parentheses, brackets, commas and semicolons,
so "Common purslane (Portulaca oleracea)" gives three candidates. Then:

1. Any candidate equal to an accepted name is correct.
2. If the answer contains " or " or "/", it is a hedge and can only match by
   step 1, never by containment, so naming two options doesn't win.
3. Otherwise, an accepted name appearing as a whole phrase (regex with
   `(?<!\w)` / `(?!\w)` boundaries) anywhere in the text is correct. This
   accepts "Eclipta prostrata" for "Eclipta" but not a less specific
   genus-only answer.

### Density: `_score_density`

The category answer is normalised and checked against `DENSITY_CATEGORIES`
(`none`, `low`, `medium`, `high`). A match sets `predicted_label` and the
outcome, and `category_rank_error` is the absolute distance between the
two categories' indices in that ordered list. A non-category is `invalid`.
The numeric fields are scored independently of the category's validity, so a
bad category with a good count still contributes to count error. Errors are
always *estimate minus truth*, signed. Coverage truth is converted from a
fraction to percent to match the prompt.

### Localisation: `_score_localisation` and `bboxes.py`

A model's boxes are in an unknown format. The pipeline has to turn them into
pixel corners before it can measure overlap.

**Finding the format.** The format is a `BboxFormat(order, scale,
scale_known)`.

- If the model has a configured `bbox_format`, `parse_bbox_format` reads it
  (`<order>_<scale>`) and the reply's own description is ignored.
- Otherwise `infer_bbox_format` reads the `response_format` text. It strips
  non-alphanumerics and looks for order hints (`ymin xmin ymax xmax`,
  `x1 y1 x2 y2`, `cx cy w h`, ...). Scale is read from words: "1000",
  "percent"/"%", "normalised"/"0 to 1", or "pixel". A Gemini-style
  `box_2d` implies `yxyx` on 0-1000. If no scale is stated, `scale_known`
  is `False` and the scale is decided later from the values.

**Converting to pixels: `to_pixel_boxes`.**

1. Accept a list of boxes, or a bare four-number list (wrapped into one box).
2. `_coordinates` extracts four numbers and an order from each box. Boxes
   given as objects carry their own order via named-key layouts
   (`xmin/ymin/xmax/ymax`, `cx/cy/w/h`, ...) or a nested key like
   `box_2d`. A bare list needs the order from the format, and fails if it is
   unknown. Two `[x, y]` points are flattened to four numbers.
3. Resolve scale: if not stated, all values within 0-1 means fractions and
   anything larger means pixels. If "normalised" was stated but values exceed
   1.5, it is treated as 0-1000.
4. Convert each box to `(x1, y1, x2, y2)`: swap for `yxyx`, add width and
   height for `xywh`, expand around the centre for `cxcywh`. Multiply by
   `width/scale` and `height/scale`, sort so corners are ordered, and clip to
   the image.

Any `ValueError` in this process makes the response `invalid` with a
localisation score of 0. Unreadable boxes count against the model.

**Measuring overlap.**

- `iou` is the standard intersection over union of two corner boxes.
- `match_boxes` computes every truth/prediction IoU, sorts descending, and
  greedily pairs the best unused pairs, ignoring zero overlaps. This gives a
  one-to-one matching, so one big predicted box can't claim several plants.
- `score_boxes` converts the truth from `(x, y, w, h)` to corners, matches,
  and returns:
  - `localisation_score` = sum of paired IoU / `max(true count, predicted
    count)`. The denominator is the larger count, so every missing box and
    every extra box costs a full share. A perfect score needs the right
    number of boxes, each exactly on a plant.
  - `matched_iou` = mean IoU of the pairs. This isolates placement quality
    from counting.
  - `box_count_error` = predicted minus true.

The scored row stores the format actually used (`bbox_format`) and the
converted `pred_boxes` as JSON, so results can be audited.

### Suites (`suite_name`)

A suite is the reporting group: a task under one image condition and format.
The name comes from the question class, `annotated`, and the type.
`OpenEndedQuestion` gives `species_id/open_ended/{boxed,unannotated}`, and
density gives `density/{...}`. Localisation, fine-grained and grounded VQA
(`grounded_vqa`) are single suites. The remaining MC questions give
`species_id/multiple_choice/{...}`.

`chance` is `1/len(choices)` for MC, `1/len(density_choices)` for density,
and `None` for open-ended and localisation.

## Metrics (`metrics.py`)

`compute_metrics` builds a DataFrame from the scored rows and groups it by
model, then by suite. Each suite goes to `_suite_metrics`, which branches on
the suite name.

**The accuracy block.** Failed requests say more about run settings than
about model ability, so two accuracies are kept: `accuracy` = correct /
answered (`status == "ok"`), and `accuracy_all` = correct / all questions.
Unsure and invalid count as wrong in both. `wilson_interval` gives a 95%
interval for the proportion of answered questions. It is used because it
behaves sensibly near 0 and 1 and at small n.

**Macro-F1** (`macro_f1`) is computed for species-ID MC, fine-grained and
density only (`_F1_SUITES`), where "which class was predicted" is
meaningful. For each ground-truth class, it computes F1 from true
positives, false positives and false negatives, then takes the unweighted
mean. This addresses class imbalance. A `None` prediction (unsure, invalid)
matches no class, so it adds false negatives for the true class but no false
positives anywhere.

**Breakdowns.** `_breakdown` gives accuracy per `benchmark_class` and per
`dataset` among answered rows.

**Density block** (`_density_block`). For count and for coverage, it gives
mean absolute, median absolute and mean signed error (bias), computed over
rows where the estimate was parseable. It also gives mean category
distance and a truth-by-predicted confusion matrix over the four ordered
categories.

**Localisation block.** Localisation suites don't have accuracy.
`_localisation_block` reports mean and median `localisation_score` over
answered rows (invalid rows contribute 0), `mean_score_all` (divided by all
questions, so failures count as 0), mean matched IoU, the exact-count rate,
and count MAE and bias over the scored rows. Per-class and per-dataset mean
scores are also given.

**Per model**, outside the suites:

- `_usage_block` has request and failure counts, total and mean cost, mean
  tokens, mean latency, and how many requests needed more than one attempt.
- `_mc_vs_open_ended` merges the open-ended rows to their MC originals on
  `ablation_of == question_id`, keeps pairs where both were answered, and
  compares accuracy per image condition. Pairing means both numbers are over
  the *same* questions, so the gap shows how much of the MC score is
  guessing among the given options.
