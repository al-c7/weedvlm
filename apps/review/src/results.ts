/**
 * Reads evaluation runs from a results directory: one subdirectory per run,
 * as written by run-evaluation.py (responses.jsonl, run.json, progress.json,
 * run.log).
 *
 * responses.jsonl is the source of truth for what the models answered -- the
 * consolidated responses.parquet is only rewritten when a session ends, so it
 * can lag behind. Scores come from the scored responses parquet that
 * notebooks/scoring.ipynb writes (all runs in one file, see src/scores.ts),
 * but only where they still describe the response on disk: a response
 * re-asked since scoring is "unscored", not silently given its old outcome.
 */
import { join } from "@std/path";
import { readParquet } from "./parquet.ts";
import { computeMetrics, SCORED_COLUMNS, scoredRowFromParquet, suiteName } from "./scores.ts";
import type { Outcome, ResponseRecord, ResponseSummary, RunSummary, ScoredRow } from "./types.ts";

const RESPONSES_JSONL = "responses.jsonl";
const RUN_FILES = [RESPONSES_JSONL, "run.json", "progress.json"];

export interface ProblemGroup {
  model: string;
  /** error kind, or "retry" / "warning" for things that didn't fail the response */
  kind: string;
  message: string;
  count: number;
  http_status?: number | null;
}

export interface LoadedRun {
  name: string;
  records: Map<string, ResponseRecord>;
  summaries: ResponseSummary[];
  /** question id -> the image a model was shown, from the responses */
  imagePaths: Map<string, string>;
  scores: Record<string, unknown> | null;
  scoredAt: string | null;
  runJson: Record<string, unknown> | null;
  progress: Record<string, unknown> | null;
  problems: ProblemGroup[];
  hasResponses: boolean;
  scoresStale: boolean;
}

/** (model, question) -> one key. Neither can contain a NUL. */
export const responseKey = (model: string, questionId: string) => `${model}\0${questionId}`;

async function readJson(path: string): Promise<Record<string, unknown> | null> {
  try {
    return JSON.parse(await Deno.readTextFile(path));
  } catch {
    return null;
  }
}

async function mtime(path: string): Promise<number | null> {
  try {
    return (await Deno.stat(path)).mtime?.getTime() ?? null;
  } catch {
    return null;
  }
}

/** Latest record per (model, question). A line cut short by a crash is skipped. */
async function readRecords(path: string): Promise<Map<string, ResponseRecord>> {
  const latest = new Map<string, ResponseRecord>();
  let text: string;
  try {
    text = await Deno.readTextFile(path);
  } catch {
    return latest;
  }
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const record = JSON.parse(line) as ResponseRecord;
      latest.set(responseKey(record.model, record.question_id), record);
    } catch {
      continue;
    }
  }
  return latest;
}

/** The scored responses parquet, parsed once per change of the file. */
interface ScoresFile {
  mtime: number | null;
  /** run -> (model, question) -> score */
  byRun: Map<string, Map<string, ScoredRow>>;
}

let scoresFile: ScoresFile = { mtime: null, byRun: new Map() };

async function loadScoresFile(path: string | null): Promise<ScoresFile> {
  const stamp = path ? await mtime(path) : null;
  if (stamp === scoresFile.mtime) return scoresFile;
  const byRun = new Map<string, Map<string, ScoredRow>>();
  if (path && stamp !== null) {
    try {
      for (const raw of await readParquet(await Deno.readFile(path), SCORED_COLUMNS)) {
        const row = scoredRowFromParquet(raw);
        if (!row) continue;
        const run = String(raw.run);
        if (!byRun.has(run)) byRun.set(run, new Map());
        byRun.get(run)!.set(responseKey(row.model, row.question_id), row);
      }
    } catch (err) {
      console.error(`Failed to read scores from ${path}: ${(err as Error).message}`);
      return scoresFile; // keep what we had; retried when the file changes again
    }
  }
  scoresFile = { mtime: stamp, byRun };
  return scoresFile;
}

/** A failed response, as a row the metrics count (as wrong, when failures are counted). */
function failedRow(record: ResponseRecord): ScoredRow {
  return {
    question_id: record.question_id,
    model: record.model,
    task: record.task,
    suite: suiteName(record.question_type, record.annotated),
    dataset: record.dataset_name,
    benchmark_class: record.benchmark_class,
    annotated: record.annotated,
    ablation_of: null,
    status: "failed",
    created_at: record.created_at,
    outcome: "failed",
    expected: "",
    answer: null,
    predicted_label: null,
    chance: null,
    count_error: null,
    coverage_error_pp: null,
    category_rank_error: null,
    localisation_score: null,
    matched_iou: null,
    box_count_error: null,
    pred_boxes: null,
    cost_usd: record.cost_usd ?? null,
    total_tokens: (record.usage?.total_tokens as number | undefined) ?? null,
    latency_seconds: record.latency_seconds ?? null,
    attempts: record.attempts,
  };
}

/** The answer as text, before scoring: localisation answers with `bboxes`, not `answer`. */
function rawAnswer(parsed: Record<string, unknown> | null | undefined): string | null {
  if (parsed?.answer !== undefined) return String(parsed.answer);
  if (parsed?.bboxes !== undefined) return JSON.stringify(parsed.bboxes);
  return null;
}

/**
 * `scored` is only passed when it describes this very response; `question`
 * is any score for the pair, whose question fields hold even when stale.
 */
function summarise(
  record: ResponseRecord | undefined,
  scored: ScoredRow | undefined,
  question: ScoredRow | undefined,
) {
  const current = scored;
  const base = record ?? scored;
  if (!base) throw new Error("summarise needs a record or a scored row");
  const outcome: Outcome = current?.outcome ?? (base.status === "failed" ? "failed" : "unscored");
  return {
    model: base.model,
    question_id: base.question_id,
    task: base.task,
    suite: current?.suite ?? "",
    dataset: record?.dataset_name ?? scored?.dataset ?? "",
    benchmark_class: base.benchmark_class,
    annotated: base.annotated,
    // A property of the question, so it holds even where the score is stale.
    ablation_of: question?.ablation_of ?? null,
    status: base.status,
    outcome,
    expected: current?.expected ?? null,
    answer: current?.answer ?? rawAnswer(record?.parsed),
    localisation_score: current?.localisation_score ?? null,
    pred_boxes: current?.pred_boxes ?? null,
    error_kind: record?.error?.kind ?? null,
    error_message: record?.error?.message ?? null,
    attempts: record?.attempts ?? scored?.attempts ?? 1,
    cost_usd: record?.cost_usd ?? scored?.cost_usd ?? null,
    latency_seconds: record?.latency_seconds ?? scored?.latency_seconds ?? null,
  } satisfies ResponseSummary;
}

function collectProblems(records: Iterable<ResponseRecord>): ProblemGroup[] {
  const groups = new Map<string, ProblemGroup>();
  const add = (model: string, kind: string, message: string, http_status?: number | null) => {
    // Collapse whitespace and drop the "(retry in 1.3s)" a provider appends, so
    // the same failure isn't split into a group per delay.
    const message1 = message.replace(/\s*\(retry in [\d.]+s\)/g, "").replace(/\s+/g, " ").trim();
    const key = `${model}\0${kind}\0${message1}`;
    const group = groups.get(key);
    if (group) group.count++;
    else groups.set(key, { model, kind, message: message1, count: 1, http_status });
  };
  for (const r of records) {
    if (r.error) add(r.model, r.error.kind, r.error.message, r.error.http_status);
    for (const reason of r.retry_log ?? []) add(r.model, "retry", reason);
    for (const warning of r.warnings ?? []) add(r.model, "warning", warning);
  }
  return [...groups.values()].sort((a, b) => b.count - a.count);
}

const cache = new Map<string, { signature: string; run: LoadedRun }>();

export async function loadRun(
  resultsDir: string,
  name: string,
  scoresPath: string | null,
): Promise<LoadedRun | null> {
  // A run is a direct child directory of the results dir.
  if (!name || name === "." || name === ".." || /[\\/]/.test(name)) return null;
  const dir = join(resultsDir, name);
  const mtimes = await Promise.all(RUN_FILES.map((f) => mtime(join(dir, f))));
  if (mtimes.every((m) => m === null)) return null;
  const scores = await loadScoresFile(scoresPath);
  const signature = [...mtimes, scores.mtime].join("|");
  const hit = cache.get(dir);
  if (hit?.signature === signature) return hit.run;

  const [records, runJson, progress] = await Promise.all([
    readRecords(join(dir, RESPONSES_JSONL)),
    readJson(join(dir, "run.json")),
    readJson(join(dir, "progress.json")),
  ]);
  const scoredRows = scores.byRun.get(name) ?? new Map<string, ScoredRow>();

  // Only trust a score for the exact response on disk (same created_at). A
  // run with no responses.jsonl (e.g. moved) shows its scored rows as they are.
  const usable = new Map<string, ScoredRow>();
  let stale = false;
  for (const [key, row] of scoredRows) {
    const record = records.get(key);
    if (!record || record.created_at === row.created_at) usable.set(key, row);
    else stale = true;
  }
  if (scoredRows.size > 0) {
    for (const [key, record] of records) {
      if (record.status === "ok" && !scoredRows.has(key)) stale = true;
    }
  }

  const summaries: ResponseSummary[] = [];
  const imagePaths = new Map<string, string>();
  for (const key of new Set([...records.keys(), ...usable.keys()])) {
    summaries.push(summarise(records.get(key), usable.get(key), scoredRows.get(key)));
  }
  for (const record of records.values()) {
    if (record.image_path && !imagePaths.has(record.question_id)) {
      imagePaths.set(record.question_id, record.image_path);
    }
  }

  const scoredAt = usable.size > 0 && scores.mtime !== null
    ? new Date(scores.mtime).toISOString()
    : null;
  let metrics: Record<string, unknown> | null = null;
  if (usable.size > 0) {
    const rows = [...usable.values()];
    for (const [key, record] of records) {
      if (record.status === "failed" && !usable.has(key)) rows.push(failedRow(record));
    }
    metrics = { scored_at: scoredAt, models: computeMetrics(rows) };
  }

  const run: LoadedRun = {
    name,
    records,
    summaries,
    imagePaths,
    scores: metrics,
    scoredAt,
    runJson,
    progress,
    problems: collectProblems(records.values()),
    hasResponses: summaries.length > 0,
    scoresStale: stale,
  };
  cache.set(dir, { signature, run });
  return run;
}

export async function listRuns(
  resultsDir: string,
  scoresPath: string | null,
): Promise<RunSummary[]> {
  const names: string[] = [];
  try {
    for await (const entry of Deno.readDir(resultsDir)) {
      if (entry.isDirectory) names.push(entry.name);
    }
  } catch {
    return [];
  }

  const runs: RunSummary[] = [];
  for (const name of names.sort()) {
    const run = await loadRun(resultsDir, name, scoresPath);
    if (!run) continue;
    const models = [...new Set(run.summaries.map((s) => s.model))].sort();
    const failed = run.summaries.filter((s) => s.status === "failed").length;
    runs.push({
      name,
      models,
      state: (run.progress?.state as string | undefined) ?? null,
      updated_at: (run.progress?.updated_at as string | undefined) ?? null,
      answered: run.summaries.length - failed,
      failed,
      cost_usd: run.summaries.reduce((sum, s) => sum + (s.cost_usd ?? 0), 0),
      scored_at: run.scoredAt,
      scores_stale: run.scoresStale,
      has_responses: run.hasResponses,
    });
  }
  // Most recently touched first.
  return runs.sort((a, b) =>
    (b.updated_at ?? b.scored_at ?? "").localeCompare(a.updated_at ?? a.scored_at ?? "")
  );
}

/** The last `count` lines of a run's run.log (none, if the run has no log). */
export async function readLogTail(
  resultsDir: string,
  name: string,
  count: number,
): Promise<{ total: number; lines: string[] } | null> {
  if (!name || name === "." || name === ".." || /[\\/]/.test(name)) return null;
  try {
    const text = await Deno.readTextFile(join(resultsDir, name, "run.log"));
    const lines = text.split("\n").filter((l) => l.trim());
    return { total: lines.length, lines: lines.slice(-count) };
  } catch {
    return { total: 0, lines: [] };
  }
}
