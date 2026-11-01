# Evaluation pipeline: how the code works

This explains the internals of the code that sends the benchmark to VLMs
and records their answers. It records answers only and does no marking; see
[SCORING-PIPELINE.md](SCORING-PIPELINE.md) for that.

Code: `src/weedvlm/evaluation/` (`config`, `questions`, `prompts`,
`openrouter`, `parsing`, `runner`, `store`, `progress`) and the entry script
`src/weedvlm/scripts/run-evaluation.py`.

Questions and results can live in JSON files or in a SQLite database. The
runner doesn't know which: it talks to a `Benchmark` (questions and images)
and a `RunStore` (results), each with a JSON and a SQLite implementation
chosen by the config's `storage.format`. The database itself is documented
in [DATABASE.md](DATABASE.md).

## Design goals

The code is built around a few properties, and most of the structure
follows from them:

- **Expensive and interruptible.** A full run is thousands of paid
  requests, so every result is persisted the instant it lands, and a rerun
  must pick up exactly where it stopped.
- **Comparable results.** Two answers in the same run must have been
  produced under the same prompt and request settings. The code refuses to
  mix them.
- **Failures are data.** A request that can't be completed is recorded, with
  its reason, rather than crashing or silently vanishing.
- **Ground truth never travels.** Responses reference questions by id only.
  The answer key is joined back in later, at scoring time.

## Control flow

`scripts/run-evaluation.py` loads the config, applies CLI overrides by
re-validating a merged dict, and calls `asyncio.run(run_evaluation(config))`.
It converts exceptions into exit codes and prints a summary from the
returned `RunResult`.

`run_evaluation` (in `runner.py`) does the setup, then hands the work to a
`Runner`:

1. `load_api_key`, then `open_benchmark`.
2. If density is enabled, load the density rules (see
   [Prompts](#prompts)).
3. `open_run_store` creates `<output_dir>/<run_name>/` and returns the
   store. Set up file logging to `run.log`.
4. Compute per-model fingerprints and check them against any previous
   session (`_check_resumable`).
5. `plan_jobs` builds the list of `Job`s still to do.
6. Write the run's settings (`write_run_info`), create the progress tracker
   and HTTP client, install the Ctrl-C handler, and run the workers.
7. In `finally`: stop the reporter, close the client, work out the final
   state, consolidate the responses, and close the store and benchmark.

The final state is one of `finished`, `stopped` (a model was dropped),
`interrupted` or `aborted` (fatal error).

## Config (`config.py`)

`EvaluationConfig` is a Pydantic model. Details worth knowing:

- `models` accepts bare id strings. A `mode="before"` validator turns each
  into `{"id": ...}`. A model-level validator rejects empty lists, duplicate
  ids, and configs with every task disabled.
- `ModelConfig` holds per-model overrides (`temperature`, `max_tokens`,
  `extra_body`, `bbox_format`). The accessors `temperature_for(model)` and
  `max_tokens_for(model)` apply the "model value, else default" fallback.
  `bbox_format` is validated at load time by calling
  `parse_bbox_format`, so a typo fails immediately and not during scoring.
- `tasks` reuses `TaskToggles` from the generation config, so the task
  names are shared between the two pipelines.
- `concurrency` is bounded at 32 by `MAX_CONCURRENCY`.
- `extra_body` is passed straight into the request body, so OpenRouter
  features (reasoning effort, provider routing) need no code support.
- `storage` is the shared `StorageConfig`. `resolved_database()` gives the
  database path, defaulting to `benchmark.db` in `questions_dir`.
- The whole config is stored with the run, and the scoring script
  re-validates it from there.

## Loading the benchmark (`questions.py`)

`open_benchmark(config)` returns a `Benchmark`: `JsonBenchmark` or
`SqliteBenchmark`. Its `questions` attribute is a flat `[(task, question)]`
list in generation order. The rest of the interface is what the runner and
the scorer need from the images and the generation settings:

| Method | JSON | SQLite |
| --- | --- | --- |
| `read_image(question)` | reads the file at `image_path` | the stored bytes, or the file if bytes weren't stored |
| `image_hashes()` | hashes every question's image file | the hashes in the `images` table |
| `answer_is_current(question, record)` | image file's mtime is not newer than the answer | the answer's `image_sha256` equals the image's stored hash |
| `density_config()` | from `generation-config.yaml` | from the `generation_config` entry in `meta` |

The question classes are parsed by `parse_question` in `types/questions.py`,
which maps each `QuestionType` to its Pydantic class (grounded VQA and
fine-grained are both `MultipleChoiceQuestion`; species ID covers both MC
and open-ended). `TASK_FILES`, in the same module, maps each task toggle to
its JSON file.

Both implementations validate eagerly, because failures here cost nothing
while failures after paid requests do:

- an enabled task with no question file (JSON) or no rows (SQLite) is an
  error,
- ids must be unique across all loaded tasks, and
- every image must be readable: the file must exist, unless its bytes are
  in the database.

The shared `_select` step then cuts each task's list down, in two stages.

`benchmark_fraction` (a number or a string like `1/3`, parsed with
`Fraction` in the config validator) is for running a model on an evenly
sampled share of the benchmark. `_stratified_subset` groups a task's
questions by `(question_type, annotated, benchmark_class)` and keeps
`floor(n * fraction + 0.5)` of each group, never fewer than one, so every
species, density category and image format keeps its share. Within a group:

- the members are shuffled by an RNG seeded with `seed`, the task and the
  group key, then cut to a prefix. The seed is per group, so a group's
  choice is the same whichever other tasks are enabled, and a larger
  fraction's prefix contains a smaller one's.
- open-ended groups are processed first, recording the id each kept
  ablation points at (`ablation_of`). When the multiple-choice groups are
  cut, those ids are moved to the front of the shuffled order (a stable
  sort), so an ablation is never kept without its original. The
  MC-versus-open-ended comparison in `metrics.py` joins on that pair.

The kept questions are returned in their original order. The scorer rebuilds
the same subset, because it opens the benchmark with the config saved in the
run info.

`max_questions_per_task` is applied after that. It takes a seeded
`rng.sample` of indices, sorted so the subset stays in order. This is a
smoke-test tool, applied per task, and it is not stratified.

## Prompts

`prompts.py` turns a question into a `Prompt` dataclass: `system`,
`user_text`, `required_fields`, `answer_field` and `answer_is_list`. The
image is attached separately by the runner.

`build_prompt` dispatches with a `match` on the question class:

| Question | Builder | Notes |
| --- | --- | --- |
| `DensityEstimationQuestion` | `_density_prompt` | needs the rules text; extra required numeric fields |
| `OpenEndedQuestion` | `_open_ended_prompt` | free-text species name; "Unsure" allowed |
| `LocalisationQuestion` | `_localisation_prompt` | `answer_field="bboxes"`, `answer_is_list=True` |
| `MultipleChoiceQuestion` of type grounded VQA | `_grounded_vqa_prompt` | explains numbering; no "Unsure" |
| other `MultipleChoiceQuestion` | `_mc_prompt` | appends an `"Unsure"` option |

The case order matters. `OpenEndedQuestion` and
`MultipleChoiceQuestion` are separate classes, but grounded VQA is a guard
on the MC case and must come before the generic MC case.

Every prompt ends with a response spec built by `_format_response_spec`: a
JSON object with `answer`, then `reasoning` (at most two sentences), then
any extra fields. Options are rendered with `json.dumps`, so models are
shown them quoted and can copy them character for character.

**Density rules.** `density_rules_text` writes a plain-language explanation
of how categories are decided. It is generated from the
`DensityConfig` the questions were created with (read from
`generation-config.yaml` by `load_density_config`), and the worked examples
are computed by calling the real `classify_density` and `density_score`.
The model is told the exact rule that produced the labels, and the text
can't go stale if thresholds change.

`PROMPT_VERSION` is a manually bumped integer. It is stored in every
response and folded into the resume fingerprint.

## Parsing replies (`parsing.py`)

`extract_json_object` is tolerant by design, because models told to return
only JSON often add fences or prose. It strips a leading and trailing
Markdown fence, tries `json.loads` on the whole text, and otherwise scans
for each `{` and tries `JSONDecoder.raw_decode` from there, returning the
first complete object that is a dict.

`check_fields` returns a problem string for each required field that is
missing or an empty string.

## The HTTP client (`openrouter.py`)

`OpenRouterClient.chat` makes one request and either returns a `ChatReply`
or raises `RequestFailure(kind, message, http_status, retry_after)`. The key
idea is that the client classifies *every* failure so the caller only
decides policy:

| Kind | Examples | Meaning |
| --- | --- | --- |
| `transient` | 408, 429, 5xx, timeouts, transport errors, non-JSON bodies, empty `choices` | worth retrying |
| `request` | 400, 403, 404, 413 | this request will never work as sent |
| `fatal` | 401, 402 | nothing will work until the user acts |

Details:

- OpenRouter can return HTTP 200 with an `error` object, either at the top
  level or on the choice. The client detects this and classifies it from the
  embedded `code` (or `transient` if there is none).
- A 429 sets a shared `_paused_until` timestamp (from `Retry-After`, else 1
  second). Every call begins with `_wait_out_cooldown`, so a rate limit
  pauses all workers and not just the one that hit it.
- The body always includes `usage: {include: true}`, which makes OpenRouter
  report the cost. The reply is returned as content text (string or joined
  content parts), `finish_reason`, usage, cost, generation id, provider and
  measured latency.
- Provider error text from `error.metadata.raw` is appended to messages, as
  that is where the upstream provider's real reason lives.

## The runner (`runner.py`)

### Job planning

`plan_jobs` produces one `Job(task, question, model)` per question and model
pair that isn't already done, iterating questions in the outer loop and
models in the inner. This question-major order means a run cut short has
covered all models evenly rather than finishing one model completely. It
loads existing records from the store. A pair is skipped only if its latest
record is `ok` **and** `benchmark.answer_is_current` confirms it was given
for the image the question has now. Regenerating the benchmark reuses ids
but can change pictures (grounded-VQA numbering), so such answers are
redone.

It also seeds per-model `ModelProgress` with how many were already done.

### Worker pool

`Runner.run_jobs` puts all jobs in an `asyncio.Queue` and starts
`min(concurrency, len(jobs))` worker tasks. Each worker loops until the
queue is empty or `stop` is set, skipping jobs for models in the `_tripped`
set. `_process` calls `_answer`, appends the resulting record to the store,
updates progress, and manages the consecutive-failure counter.

### One request: `_answer`

`_answer` builds the prompt, reads the image through the benchmark in a
thread (`asyncio.to_thread`, returning a base64 data URL and the image's
SHA-256), and builds the
messages: a system message, then a user message with an `image_url` part and
a text part. Then it loops:

```
attempt:
    reply = client.chat(...)
    RequestFailure:
        fatal      -> raise FatalRunError (stops the whole run)
        transient  -> if retries left: sleep(backoff), retry; else record failed
        request    -> record failed
    reply received:
        _interpret(reply) -> parsed?  -> record ok
        else: truncated (finish_reason == "length")? -> record failed
              parse retries left? -> short sleep, retry
              else record failed
```

Two separate retry counters (`transient_retries`, `parse_retries`) apply, so
a flaky network can't use up the parse budget and vice versa. Backoff is
`min(max, base * 2^(n-1))` with 0.5-1.0x jitter, floored by any
`Retry-After`. `_sleep` waits on the stop event, so a Ctrl-C interrupts a
backoff immediately, and `_answer` then returns `None`, leaving the pair
pending.

`_interpret` is the validity check for a reply. It extracts the JSON, and
if there is none it explains why (cut off at `max_tokens`, empty, or no
JSON). It then calls `check_fields` and enforces the answer's shape: a list
for localisation, a scalar otherwise. A problem with the *answer field*
makes the reply unusable. Problems with other fields become `warnings` on
an otherwise `ok` record.

A truncation (`finish_reason == "length"`) is not retried, because the same
prompt will truncate again. The error text tells the user to raise
`max_tokens`. That is why `max_tokens` is excluded from the fingerprint:
raising it and rerunning is the intended fix.

### Failure policy

- **Per model.** `_consecutive_failures[model]` is reset on any success and
  incremented on any failed record. Reaching `model_failure_threshold`
  adds the model to `_tripped`. Its remaining jobs are skipped, and stay
  pending, so a rerun resumes them. This catches typo'd model ids, models
  without image input, and outages without burning the whole queue.
- **Fatal.** A `FatalRunError` raised inside a worker records the message
  and sets `stop`. Other workers finish their current job and exit.
- **Interrupt.** `on_sigint` (installed with `loop.add_signal_handler`) sets
  `stop` on the first Ctrl-C. On the second it cancels the main task,
  abandoning in-flight requests.

### Resume safety

`_fingerprints` hashes `PROMPT_VERSION`, temperature and `extra_body` per
model (`store.model_fingerprint`, a truncated SHA-256 of sorted JSON).
`_check_resumable` compares these with the fingerprints saved with the run
and raises `FatalRunError` if any model's differ, since mixing them would
make the results incomparable. New fingerprints are merged into the run's
stored settings alongside a list of session start times.

## Storage (`store.py`)

`ResponseRecord` is the unit of storage. It has identifying fields
(question id, model, task, type, `annotated`, class, dataset, source image
id, image path, `image_sha256`, `prompt_version`), `status`
(`ok`/`failed`), `parsed`, `raw_response`, `warnings`, a `ResponseError`
(`kind` is a client failure kind or `"parse"`), attempts and retry log, and
usage data. It has no ground truth. The identifying fields are denormalised
copies so a response is useful alone, but the question id is the join key.

`open_run_store(config)` returns the store for the configured format. Both
have the same interface.

`RunStore` is the JSON format and manages the run directory:

- `append` writes one JSON line and flushes. This is the durable record.
- `load_records` replays `responses.jsonl` into a dict keyed by
  `(model, question_id)`, so the **latest line wins**. A truncated trailing
  line from a crash fails validation and is ignored, so that pair is
  simply asked again. This is what makes retries of failed pairs work
  without ever editing the file.
- `write_consolidated` produces `responses.json` from the latest records,
  ordered by the question-then-model order computed at start.
- `read_run_info` and `write_run_info` read and write `run.json`.
- `write_run_info` and `write_progress` use `_atomic_write` (write to a temp
  file, then `os.replace`), so a reader never sees a half-written file.

`SqliteRunStore` subclasses it and keeps the same data in the benchmark
database:

- `append` upserts one row into `responses` and commits, which gives the
  same durability as flushing a line. A new answer for the same model and
  question replaces the old row, so "latest wins" is built into the table
  and earlier failed attempts aren't kept. Any score for the replaced answer
  is deleted.
- `load_records` joins `responses` to `questions` and `images` to rebuild
  full `ResponseRecord`s, since a row doesn't repeat the question's fields.
- `read_run_info` and `write_run_info` use one row of the `runs` table.
- `write_consolidated` has nothing to write and just counts the answered
  pairs.
- `write_progress` and the log are inherited: `progress.json` and `run.log`
  still go in the run directory.

Both stores also have `write_scores`, used by the scoring script.

## Progress reporting (`progress.py`)

`ModelProgress` is a per-model counter dataclass (`total`, `already_done`,
`ok`, `failed`, `retries`, `in_flight`, `cost_usd`, `tokens`,
`stopped_early`). `Progress` is updated by workers through `started`,
`finished` and `abandoned`, and reports on a timer from a background task:

- A rolling 60-second window of completion timestamps gives the request
  rate, and so the ETA.
- On a terminal it redraws a block of lines in place using ANSI cursor
  moves. Otherwise it prints a summary line every interval.
- Every interval it also writes `progress.json` through the store, so other
  tools can poll a run.
- `detach_display` resets the line count so that a message printed in the
  middle of a run (e.g. the interrupt notice) is not overwritten.
