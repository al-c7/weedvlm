/**
 * Reads evaluation runs from a results directory: one subdirectory per run,
 * as written by run-evaluation.py (responses.jsonl, run.json, progress.json,
 * run.log) and score-evaluation.py (scores.json, scored-responses.csv).
 *
 * responses.jsonl is the source of truth for what the models answered -- the
 * consolidated responses.json is only rewritten when a session ends, so it
 * can lag behind. Scores come from scored-responses.csv, but only where they
 * still describe the response on disk (a response re-asked since scoring is
 * "unscored", not silently given its old outcome).
 */
import { parse as parseCsv } from "@std/csv/parse";
import { join } from "@std/path";
import type {
  Outcome,
  ResponseRecord,
  ResponseStatus,
  ResponseSummary,
  RunSummary,
  ScoredRow,
} from "./types.ts";

const RESPONSES_JSONL = "responses.jsonl";
const SCORES_JSON = "scores.json";
const SCORED_CSV = "scored-responses.csv";
const RUN_FILES = [RESPONSES_JSONL, "run.json", "progress.json", SCORES_JSON, SCORED_CSV];

const OUTCOMES: readonly Outcome[] = ["correct", "incorrect", "unsure", "invalid", "failed"];

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

function numberOrNull(value: string | undefined): number | null {
  if (value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

async function readScoredRows(path: string): Promise<Map<string, ScoredRow>> {
  const rows = new Map<string, ScoredRow>();
  let text: string;
  try {
    text = await Deno.readTextFile(path);
  } catch {
    return rows;
  }
  for (const raw of parseCsv(text, { skipFirstRow: true }) as Record<string, string>[]) {
    const outcome = OUTCOMES.find((o) => o === raw.outcome);
    if (!outcome) continue;
    rows.set(responseKey(raw.model, raw.question_id), {
      question_id: raw.question_id,
      model: raw.model,
      task: raw.task,
      suite: raw.suite,
      dataset: raw.dataset,
      benchmark_class: raw.benchmark_class,
      annotated: raw.annotated === "True",
      status: raw.status as ResponseStatus,
      outcome,
      expected: raw.expected,
      answer: raw.answer === "" ? null : raw.answer,
      cost_usd: numberOrNull(raw.cost_usd),
      latency_seconds: numberOrNull(raw.latency_seconds),
      attempts: numberOrNull(raw.attempts),
    });
  }
  return rows;
}

function summarise(record: ResponseRecord | undefined, scored: ScoredRow | undefined) {
  // A score describes a response only if it's for the same outcome-relevant
  // state: the record is no newer than the scoring, and agrees on status.
  const current = scored && (!record || record.status === scored.status) ? scored : undefined;
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
    status: base.status,
    outcome,
    expected: current?.expected ?? null,
    answer: current?.answer ??
      (record?.parsed?.answer !== undefined ? String(record.parsed.answer) : null),
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

export async function loadRun(resultsDir: string, name: string): Promise<LoadedRun | null> {
  // A run is a direct child directory of the results dir.
  if (!name || name === "." || name === ".." || /[\\/]/.test(name)) return null;
  const dir = join(resultsDir, name);
  const mtimes = await Promise.all(RUN_FILES.map((f) => mtime(join(dir, f))));
  if (mtimes.every((m) => m === null)) return null;
  const signature = mtimes.join("|");
  const hit = cache.get(dir);
  if (hit?.signature === signature) return hit.run;

  const [records, scoredRows, scores, runJson, progress] = await Promise.all([
    readRecords(join(dir, RESPONSES_JSONL)),
    readScoredRows(join(dir, SCORED_CSV)),
    readJson(join(dir, SCORES_JSON)),
    readJson(join(dir, "run.json")),
    readJson(join(dir, "progress.json")),
  ]);

  const scoredAt = typeof scores?.scored_at === "string"
    ? scores.scored_at
    : mtimes[RUN_FILES.indexOf(SCORED_CSV)] !== null
    ? new Date(mtimes[RUN_FILES.indexOf(SCORED_CSV)] as number).toISOString()
    : null;
  const scoredAtMs = scoredAt ? Date.parse(scoredAt) : null;

  // Only trust a score for a response that hasn't been re-answered since.
  const usable = new Map<string, ScoredRow>();
  for (const [key, row] of scoredRows) {
    const record = records.get(key);
    const newer = record && scoredAtMs !== null && Date.parse(record.created_at) > scoredAtMs;
    if (!newer) usable.set(key, row);
  }

  const summaries: ResponseSummary[] = [];
  const imagePaths = new Map<string, string>();
  for (const key of new Set([...records.keys(), ...usable.keys()])) {
    summaries.push(summarise(records.get(key), usable.get(key)));
  }
  for (const record of records.values()) {
    if (record.image_path && !imagePaths.has(record.question_id)) {
      imagePaths.set(record.question_id, record.image_path);
    }
  }

  const responsesMtime = mtimes[RUN_FILES.indexOf(RESPONSES_JSONL)];
  const run: LoadedRun = {
    name,
    records,
    summaries,
    imagePaths,
    scores,
    scoredAt,
    runJson,
    progress,
    problems: collectProblems(records.values()),
    hasResponses: summaries.length > 0,
    scoresStale: scoredAtMs !== null && responsesMtime !== null && responsesMtime > scoredAtMs,
  };
  cache.set(dir, { signature, run });
  return run;
}

export async function listRuns(resultsDir: string): Promise<RunSummary[]> {
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
    const run = await loadRun(resultsDir, name);
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
