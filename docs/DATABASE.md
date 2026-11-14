# Storage layout

The benchmark and its results are parquet files plus the images on disk.
There is no database. Any tool that reads parquet can query them (polars,
pandas, duckdb, or the review app via hyparquet).

```
questions/                     the generation out_dir / evaluation questions_dir
  questions.parquet            one row per question, every task
  generation-config.yaml       the config + command line that produced them
  rendered/...                 the images the questions use
results/<run_name>/            one directory per evaluation run
  responses.jsonl              append-only log, written as each answer lands
  responses.parquet            latest answer per (model, question), one flat row each
  run.json                     the run's config, model fingerprints and session starts
  progress.json, run.log       live status and errors
notebooks/weedvlm_scored_responses.parquet
                               responses x questions with scores, written by
                               notebooks/scoring.ipynb; reasoning judge scores are
                               added by append-reasoning-scores.py
```

Code: `src/weedvlm/storage/parquet.py` (questions) and
`src/weedvlm/evaluation/store.py` (`RunStore`, responses).

## questions.parquet

Written by `parquet.write_questions` from `generate-all-questions.py`, and
always replaced as a whole (written to a temp file, then renamed). Before
writing, it checks that ids are unique and that every open-ended ablation's
`ablation_of` points at a question in the file.

Columns that hold lists are JSON text. That way pandas, polars and duckdb
all read them the same way.

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | str | Question id, unique across tasks |
| `task` | str | `species_id`, `fine_grained`, `density`, `grounded_vqa`, `localisation` |
| `position` | int | Order within the task, as generated |
| `question_type` | str | A `QuestionType` value |
| `question_text` | str | |
| `image_path` | str | Absolute path of the image file |
| `image_sha256` | str | Hash of the image when the benchmark was written |
| `annotated` | bool | Whether the image has boxes drawn on it |
| `benchmark_class` | str | The species / class the question is balanced on |
| `dataset_name`, `source_image_id`, `annotation_ids` (JSON) | | Where the question came from |
| `expected_answer` | str | The correct option / species / density category. Null for localisation |
| `choices` | JSON | MC options, or density choices |
| `answer_index` | int | MC only |
| `ablation_of` | str | Open-ended only: the MC question it ablates |
| `true_weed_count`, `true_weed_coverage_fraction` | | Density only |
| `target_species`, `image_width`, `image_height`, `boxes` (JSON `[x, y, w, h]` px) | | Localisation only |

`parquet.load_questions` turns rows back into the question models in
`types/questions.py`.

## responses.parquet

While a run is going, `responses.jsonl` is the durable record. Parquet
can't be appended to row by row, so a resumed run reads the jsonl to know
what's already been done. When each session ends, `RunStore.write_consolidated`
writes the latest record per (model, question) to `responses.parquet`:

| Column | Meaning |
| --- | --- |
| `id` | int64 response id: stable across sessions and unique across runs. New runs derive it from (run, model, question_id). Runs migrated from SQLite keep their original ids. Re-consolidating never changes an id that was already given out |
| `run`, `model`, `question_id` | Which question was answered, by which model, in which run |
| `status` | `ok` / `failed` |
| `answer`, `reasoning` | Pulled out of `parsed`. For localisation, `answer` is the boxes as JSON |
| `parsed` | The model's whole JSON reply object, as JSON text |
| `raw_response` | The reply verbatim |
| `warnings`, `retry_log` | JSON lists |
| `error_kind`, `error_message`, `error_http_status` | When failed |
| `attempts`, `finish_reason`, `latency_seconds`, `usage` (JSON), `total_tokens`, `cost_usd`, `generation_id`, `provider` | Request details |
| `image_sha256` | Hash of the image actually sent |
| `prompt_version`, `created_at` | |

On resume, a stored answer only counts if its `image_sha256` matches the
hash in `questions.parquet`. That way a regenerated benchmark that reuses
ids never mixes in answers to old images.

## Querying

```python
import polars as pl
questions = pl.read_parquet("questions/questions.parquet")
responses = pl.read_parquet("results/*/responses.parquet")
joined = responses.join(questions.rename({"id": "question_id"}), on="question_id")
```

```sql
-- duckdb
SELECT r.run, r.model, q.task, avg((r.answer = q.expected_answer)::int)
FROM 'results/*/responses.parquet' r JOIN 'questions/questions.parquet' q ON q.id = r.question_id
GROUP BY ALL;
```

## Migrating an old SQLite benchmark

`src/weedvlm/scripts/migrate-sqlite-to-parquet.py <benchmark.db>` writes
`questions.parquet`, `generation-config.yaml` and a run directory for each
run, keeping the response ids. Any image whose bytes were only in the
database gets written back to its path. The database itself is only read.
