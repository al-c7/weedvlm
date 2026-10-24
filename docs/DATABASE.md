# The SQLite storage format

A benchmark and its results can be stored either as JSON files (the
original layout) or in one SQLite database. This documents the database:
what it holds, the schema, and how the code reads and writes it.

Schema source: `src/weedvlm/storage/database.py` (`_SCHEMA`). The current
`SCHEMA_VERSION` is 1.

## What a database is

One database holds **one benchmark** and **any number of runs against it**:

- the questions, with their answer keys,
- the images those questions use, and only those,
- each evaluation run's settings and responses,
- each run's scores, once it has been scored.

Everything needed to review a model's answers is therefore joinable in one
place: question, image, expected answer, the model's answer and reasoning,
and the outcome.

Because responses point at questions, a benchmark that has been answered
can't be replaced. `write_benchmark` raises `BenchmarkInUseError` if the
`responses` table has any rows. A regenerated benchmark goes into a new
database, or the runs are deleted first.

The database is opened in WAL mode, so it can be read (for manual review)
while a run is writing to it. Foreign keys are enforced.

## Choosing the format

Both configs have a `storage` section (`storage/config.py`,
`StorageConfig`):

| Field | Meaning |
| --- | --- |
| `format` | `json` (default) or `sqlite`. |
| `database` | Path of the database. Default: `benchmark.db` in the questions directory (`out_dir` for generation, `questions_dir` for evaluation). |
| `store_images` | Generation only. Whether image bytes go into the database. Default `true`. |

- **Generation** reads `storage` from the generation config, with
  `--format` and `--database` overriding it.
- **Evaluation** reads `storage` from the evaluation config. With `sqlite`
  the questions come from the database and the responses go into it.
- **Scoring** takes a run directory for a JSON run, or a run name plus
  `--database` for a SQLite run.

The two formats are complete alternatives: a run reads its questions from
and writes its responses to the same format.

## Tables

Booleans are stored as `0`/`1`. Lists and objects are stored as JSON text
and are marked *JSON* below; SQLite's `json_extract` and `json_each` work on
them. Timestamps are ISO 8601 text in UTC.

### `meta`

Key-value facts about the database.

| Key | Value |
| --- | --- |
| `schema_version` | The schema version the file was created with. `connect` refuses any other version. |
| `generation_config` | The text that the JSON format writes to `generation-config.yaml`: a header with the timestamp and command line, then the generation config. The density prompt's rules are read from it. Absent if a migrated benchmark had no such file. |
| `benchmark_written_at` | When the questions were written. |
| `images_stored` | `1` if image bytes were stored, `0` if only paths and hashes. |

### `images`

One row per image used by at least one question.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `path` | TEXT, unique | The file's path when the benchmark was written. This is the `image_path` of the questions that use it. |
| `sha256` | TEXT | Hash of the file's bytes. Used to tell whether a response was given for this image. |
| `media_type` | TEXT | e.g. `image/jpeg`, guessed from the file name. |
| `size_bytes` | INTEGER | |
| `data` | BLOB, nullable | The image file's bytes, unchanged. `NULL` when `store_images` was off. |

Images are either rendered box-annotated pictures or plain source images,
depending on the question. An image is stored once however many questions
share it. Images that were rendered during generation but whose questions
weren't selected never reach the database, because rows are created only
while walking the selected questions.

With `store_images` on, the full benchmark's database is several GB (about
5 GB for the current 2,791 questions and 2,472 images).

### `questions`

One row per question. Columns that apply to only some question types are
`NULL` for the others.

| Column | Type | Applies to | Meaning |
| --- | --- | --- | --- |
| `id` | TEXT PK | all | The question id. |
| `task` | TEXT | all | `species_id`, `fine_grained`, `density`, `grounded_vqa` or `localisation`. |
| `position` | INTEGER | all | Order within the task, as generated. Unique per task. |
| `question_type` | TEXT | all | e.g. `species_id_multiple_choice`, `species_id_open_ended`. |
| `question_text` | TEXT | all | |
| `image_id` | INTEGER FK | all | The image shown, in `images`. |
| `annotated` | 0/1 | all | Whether boxes are drawn on the image. |
| `benchmark_class` | TEXT | all | The class the question counts against. |
| `dataset_name` | TEXT | all | Source dataset. |
| `source_image_id` | INTEGER | all | Image id within the source dataset. |
| `annotation_ids` | *JSON* list | all | Source annotation ids. |
| `expected_answer` | TEXT | all but localisation | The correct answer as text: the correct choice, the species name (open-ended), or the density category. |
| `choices` | *JSON* list | multiple choice, density | The options offered. For grounded VQA these are the labels `"1"`..`"N"`. For density, the category names. |
| `answer_index` | INTEGER | multiple choice | Index of the correct entry in `choices`. |
| `ablation_of` | TEXT FK | open-ended | The multiple-choice question this is an ablation of. |
| `true_weed_count` | INTEGER | density | |
| `true_weed_coverage_fraction` | REAL | density | 0 to 1. |
| `target_species` | TEXT | localisation | |
| `image_width`, `image_height` | INTEGER | localisation | The size the boxes are relative to. |
| `boxes` | *JSON* list | localisation | Ground truth, `[x, y, width, height]` in pixels. |

The mapping to and from the question models is `_question_row` and
`_question_from_row`. It is lossless: a question read back equals the one
written. Fields that hold the same thing under different names in the models
share a column (`answer_text` and `density_category` are both
`expected_answer`; `density_choices` is `choices`).

### `runs`

One row per evaluation run. This is the JSON format's `run.json`.

| Column | Type | Meaning |
| --- | --- | --- |
| `name` | TEXT PK | The run name. |
| `prompt_version` | INTEGER | |
| `config` | *JSON* | The evaluation config the run was last started with. |
| `model_fingerprints` | *JSON* | Model id to request-settings fingerprint, used to refuse resuming with changed settings. |
| `sessions` | *JSON* list | Start time of each session of the run. |

### `responses`

One row per (run, model, question): the model's latest answer.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `run` | TEXT FK | `runs.name`. Deleting a run deletes its responses. |
| `model` | TEXT | OpenRouter model id. |
| `question_id` | TEXT FK | `questions.id`. |
| `status` | TEXT | `ok` or `failed`. |
| `answer` | TEXT | The reply's `answer` field. For localisation, its `bboxes` as JSON. Non-text answers (a number) are stored as JSON. |
| `reasoning` | TEXT | The reply's `reasoning` field. |
| `parsed` | *JSON* | The whole JSON object the model returned, including density's estimates and localisation's `response_format`. |
| `raw_response` | TEXT | The reply verbatim, kept even when it couldn't be parsed. |
| `warnings` | *JSON* list | Problems that didn't make the reply unusable. |
| `error_kind`, `error_message`, `error_http_status` | TEXT, TEXT, INTEGER | Why a `failed` response failed. `error_kind` is `transient`, `request`, `fatal` or `parse`. |
| `attempts` | INTEGER | Requests made for this answer. |
| `retry_log` | *JSON* list | Why each retried attempt was retried. |
| `finish_reason` | TEXT | |
| `latency_seconds` | REAL | |
| `usage` | *JSON* | Token usage as OpenRouter reported it. |
| `total_tokens` | INTEGER | `usage.total_tokens`, pulled out for convenience. |
| `cost_usd` | REAL | |
| `generation_id`, `provider` | TEXT | |
| `image_sha256` | TEXT | Hash of the image the model was actually sent. |
| `prompt_version` | INTEGER | |
| `created_at` | TEXT | When the answer was recorded. |

`(run, model, question_id)` is unique. Differences from the JSON format:

- The JSON format's `responses.jsonl` is append-only and keeps every
  attempt, with the latest winning. Here a new answer **replaces** the old
  row, so earlier failed attempts are not kept.
- A response row doesn't repeat the question's fields (task, class,
  dataset, image path). They are joined from `questions` when a record is
  loaded.
- A response can't exist for a question that isn't in the database.

### `scores`

One row per scored response, written by the scoring script. This is the
JSON format's `scored-responses.csv`, minus the columns that `questions` and
`responses` already hold.

| Column | Type | Meaning |
| --- | --- | --- |
| `response_id` | INTEGER PK, FK | `responses.id`. Deleted with its response. |
| `suite` | TEXT | Reporting group, e.g. `species_id/multiple_choice/boxed`. |
| `outcome` | TEXT | `correct`, `incorrect`, `unsure`, `invalid`, `failed`, or `scored` (localisation). |
| `predicted_label` | TEXT | The option or category the answer mapped to, if any. |
| `chance` | REAL | Random-guess accuracy for the question. |
| `true_count`, `est_count`, `count_error` | | Density. Errors are estimate minus truth. |
| `true_coverage_pct`, `est_coverage_pct`, `coverage_error_pp` | REAL | Density, in percent and percentage points. |
| `category_rank_error` | INTEGER | Density: categories away from the truth. |
| `true_box_count`, `pred_box_count`, `box_count_error` | INTEGER | Localisation. |
| `matched_iou`, `localisation_score` | REAL | Localisation. |
| `bbox_format` | TEXT | Localisation: the format the model's boxes were read with. |
| `pred_boxes` | *JSON* | Localisation: the model's boxes as `[x1, y1, x2, y2]` pixels. |

Scoring a run replaces all of that run's score rows. When a response is
re-answered, its score row is deleted, since it described a different
answer. Responses skipped as stale get no score row.

### `scoring_runs`

One row per scored run. This is the JSON format's `scores.json`.

| Column | Type | Meaning |
| --- | --- | --- |
| `run` | TEXT PK, FK | |
| `scored_at` | TEXT | |
| `aliases_file` | TEXT | The alias file used, if any. |
| `skipped_missing_questions`, `skipped_changed_images` | INTEGER | Responses left out as stale. |
| `metrics` | *JSON* | The full metrics, keyed by model: usage, per-suite metrics, and the multiple-choice vs open-ended comparison. |

## Views

Two views cover the common manual-review queries.

**`response_details`** has one row per response, joined to its question,
image and score: `response_id`, `run`, `model`, `question_id`, `task`,
`suite`, `question_type`, `annotated`, `dataset_name`, `benchmark_class`,
`question_text`, `choices`, `expected_answer`, `status`, `answer`,
`reasoning`, `outcome`, `predicted_label`, `error_kind`, `error_message`,
`cost_usd`, `latency_seconds`, `image_id`, `image_path`. `suite`, `outcome`
and `predicted_label` are `NULL` until the run is scored.

**`suite_summary`** has one row per (run, model, suite), computed live from
`scores`: `n_total`, `n_answered`, `correct`, `incorrect`, `unsure`,
`invalid`, `accuracy` (correct / answered), `accuracy_all` (correct / all)
and `mean_localisation_score`. For the localisation suite the accuracy
columns are 0 and meaningless; read `mean_localisation_score`. Confidence
intervals, macro-F1 and the density breakdowns are only in
`scoring_runs.metrics`.

Examples:

```sql
-- Every wrong boxed species-ID answer from one run
SELECT question_id, expected_answer, answer, reasoning
FROM response_details
WHERE run = 'gemini-full-run' AND suite = 'species_id/multiple_choice/boxed'
  AND outcome = 'incorrect';

-- Open-ended answers judged wrong (what open-ended-mismatches.json holds)
SELECT question_id, model, expected_answer, answer
FROM response_details
WHERE suite LIKE 'species_id/open_ended%' AND outcome = 'incorrect';

-- Questions two runs disagree on
SELECT a.question_id, a.outcome, b.outcome
FROM response_details a JOIN response_details b USING (question_id)
WHERE a.run = 'run-a' AND b.run = 'run-b' AND a.outcome != b.outcome;

-- Remove a run and everything derived from it
DELETE FROM runs WHERE name = 'old-run';
```

`DELETE` cascades only when foreign keys are on for the connection
(`PRAGMA foreign_keys = ON`); the code always turns them on.

## How the code uses it

**`storage/database.py`** owns the schema and the benchmark half.

- `connect(path, create=False)` opens a database, sets the pragmas, creates
  the schema in a new file when `create` is set, and checks the schema
  version. Opening a file that isn't a benchmark database is an error.
- `write_benchmark` replaces the questions and images in one transaction,
  so a failure leaves the database as it was. It walks the selected
  questions in order. The first time it meets an image path it reads the
  file, hashes it and inserts the image row. Foreign keys are deferred to
  commit, because an ablation can appear before or after the question it
  refers to. Afterwards the write-ahead log is truncated, since it would
  otherwise stay as large as the images.
- `load_questions`, `image_hashes`, `read_image` and
  `unstored_image_paths` are the read side. `read_image` returns the stored
  bytes, or reads the file at `path` when none were stored.

**`evaluation/questions.py`** defines `Benchmark`, with `JsonBenchmark` and
`SqliteBenchmark` behind `open_benchmark(config)`. The runner and the scorer
use only this interface: `questions`, `read_image`, `image_hashes`,
`answer_is_current` and `density_config`. The two differ in how they detect
a stale answer. The JSON one compares the image file's modification time
with the answer's timestamp. The SQLite one compares the response's
`image_sha256` with `images.sha256`.

**`evaluation/store.py`** defines `RunStore` (JSON) and `SqliteRunStore`
behind `open_run_store(config)`, with one interface: `load_records`,
`read_run_info`, `write_run_info`, `append`, `write_consolidated`,
`write_scores`. `SqliteRunStore.append` upserts one response and commits,
which is the equivalent of the JSON store flushing a line. A run directory
is still created in SQLite mode, for `progress.json` and `run.log`.

## Migrating existing JSON data

`src/weedvlm/scripts/migrate-to-sqlite.py` copies an existing JSON
benchmark and its runs into a database, leaving the JSON files untouched.
It reuses the same code paths as normal operation, so a migrated database
is indistinguishable from one written directly.

1. **Questions.** Every task file present in the questions directory is
   parsed, and `write_benchmark` stores the questions, their images and
   `generation-config.yaml`.
2. **Runs.** For every directory under the results directory that has a
   `run.json`, the run's settings are written to `runs` and the latest
   response per (model, question) to `responses`. The stored config's
   `storage` section is set to point at the database.

Two things can't be carried over exactly:

- **Responses to questions that no longer exist** are left behind, because
  a response needs a question row. The script prints the count per run.
  This happens for runs made against an earlier generation of the
  benchmark.
- **Responses given for an image that has since changed** are copied, and
  remain detectable as stale through `image_sha256`. Scoring refuses them
  unless told to skip them, as it does for JSON runs.

Scores aren't copied. Scoring each migrated run against the database
recreates them, and for the same responses produces the same metrics as the
JSON format.

Question flags from the review app (`question-flags.json`) are not
migrated; the review app still reads the JSON layout only.
