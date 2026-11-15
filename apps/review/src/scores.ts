/**
 * Scores for the Results view, from the scored responses parquet that
 * notebooks/scoring.ipynb writes (one row per answered response: the
 * response, its question, and `match` / localisation columns).
 *
 * The notebook records correctness as `match`; the finer outcome is
 * derived here: a non-match answering "Unsure" is `unsure`, one that
 * isn't any of the options is `invalid`, the rest `incorrect`.
 * Localisation keeps the notebook's `outcome` (`scored` / `invalid`).
 *
 * computeMetrics then aggregates per model and suite, in the shape the
 * Scores tab renders. Pure functions, no file access.
 */
import type { Row } from "./parquet.ts";
import type { Outcome, ResponseStatus, ScoredRow } from "./types.ts";

export const DENSITY_CATEGORIES = ["none", "low", "medium", "high"];

/** The scored parquet columns this module reads. */
export const SCORED_COLUMNS = [
  "run",
  "model",
  "question_id",
  "task",
  "question_type",
  "annotated",
  "benchmark_class",
  "dataset_name",
  "ablation_of",
  "status",
  "created_at",
  "answer",
  "expected_answer",
  "choices",
  "boxes",
  "match",
  "outcome",
  "localisation_score",
  "matched_iou",
  "pred_boxes",
  "count_error",
  "coverage_error_pp",
  "cost_usd",
  "total_tokens",
  "latency_seconds",
  "attempts",
];

const light = (text: unknown) =>
  String(text ?? "").trim().replace(/^["'`.]+|["'`.]+$/g, "").trim().replace(/\s+/g, " ")
    .toLowerCase();

function jsonList(value: unknown): unknown[] | null {
  if (typeof value !== "string" || !value) return null;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

const num = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/** The group a question is reported in (as the old score-evaluation.py named them). */
export function suiteName(questionType: string, annotated: boolean): string {
  const condition = annotated ? "boxed" : "unannotated";
  switch (questionType) {
    case "species_id_open_ended":
      return `species_id/open_ended/${condition}`;
    case "density_estimation":
      return `density/${condition}`;
    case "localisation":
      return "localisation";
    case "grounded_vqa":
      return "grounded_vqa";
    case "fine_grained_id":
      return "fine_grained";
    default:
      return `species_id/multiple_choice/${condition}`;
  }
}

/** One scored parquet row as a ScoredRow, or null if it carries no score. */
export function scoredRowFromParquet(row: Row): ScoredRow | null {
  const questionType = String(row.question_type);
  const annotated = Boolean(row.annotated);
  const answer = row.answer == null ? null : String(row.answer);
  const choices = jsonList(row.choices) as string[] | null;
  const base = {
    question_id: String(row.question_id),
    model: String(row.model),
    task: String(row.task),
    suite: suiteName(questionType, annotated),
    dataset: String(row.dataset_name ?? ""),
    benchmark_class: String(row.benchmark_class ?? ""),
    annotated,
    ablation_of: (row.ablation_of as string | null) ?? null,
    status: (row.status ?? "ok") as ResponseStatus,
    created_at: (row.created_at as string | null) ?? null,
    answer,
    chance: null as number | null,
    predicted_label: null as string | null,
    category_rank_error: null as number | null,
    count_error: num(row.count_error),
    coverage_error_pp: num(row.coverage_error_pp),
    localisation_score: null as number | null,
    matched_iou: null as number | null,
    box_count_error: null as number | null,
    pred_boxes: null as number[][] | null,
    cost_usd: num(row.cost_usd),
    total_tokens: num(row.total_tokens),
    latency_seconds: num(row.latency_seconds),
    attempts: num(row.attempts),
  };

  if (questionType === "localisation") {
    const outcome = row.outcome === "scored" || row.outcome === "invalid" ? row.outcome : null;
    if (!outcome) return null;
    const truth = jsonList(row.boxes) ?? [];
    const predicted = jsonList(row.pred_boxes) as number[][] | null;
    return {
      ...base,
      outcome,
      expected: `${truth.length} box${truth.length === 1 ? "" : "es"}`,
      localisation_score: num(row.localisation_score) ?? 0,
      matched_iou: num(row.matched_iou),
      pred_boxes: predicted,
      box_count_error: outcome === "scored" && predicted ? predicted.length - truth.length : null,
    };
  }

  if (typeof row.match !== "boolean") return null; // not scored (e.g. a crop baseline)
  const expected = String(row.expected_answer ?? "");
  const given = light(answer);
  let predicted: string | null = null;
  if (questionType === "density_estimation") {
    predicted = DENSITY_CATEGORIES.includes(given) ? given : null;
  } else if (choices) {
    predicted = choices.find((c) => light(c) === given) ?? null;
  }
  let outcome: Outcome;
  if (row.match) outcome = "correct";
  else if (given === "unsure") outcome = "unsure";
  else if (questionType !== "species_id_open_ended" && predicted === null) outcome = "invalid";
  else outcome = "incorrect";

  const density = questionType === "density_estimation";
  return {
    ...base,
    outcome,
    expected,
    predicted_label: predicted,
    chance: questionType === "species_id_open_ended" || !choices ? null : 1 / choices.length,
    category_rank_error: density && predicted
      ? Math.abs(
        DENSITY_CATEGORIES.indexOf(predicted) - DENSITY_CATEGORIES.indexOf(light(expected)),
      )
      : null,
  };
}

// -- aggregation (a port of the old evaluation/metrics.py) ---------------

type Metrics = Record<string, unknown>;

const F1_SUITES = ["species_id/multiple_choice", "fine_grained", "density"];

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const numbers = (values: (number | null)[]) => values.filter((v): v is number => v !== null);

function groupBy<T>(items: T[], key: (item: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const k = key(item);
    const group = groups.get(k);
    if (group) group.push(item);
    else groups.set(k, [item]);
  }
  return new Map([...groups].sort(([a], [b]) => a.localeCompare(b)));
}

export function wilsonInterval(successes: number, n: number, z = 1.96): [number, number] | null {
  if (n === 0) return null;
  const p = successes / n;
  const denominator = 1 + z ** 2 / n;
  const centre = (p + z ** 2 / (2 * n)) / denominator;
  const half = (z * Math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denominator;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function macroF1(rows: ScoredRow[]): number | null {
  const classes = [...new Set(rows.map((r) => r.expected))];
  const scores = classes.map((cls) => {
    let tp = 0, fp = 0, fn = 0;
    for (const r of rows) {
      const isTrue = r.expected === cls, isPred = r.predicted_label === cls;
      if (isTrue && isPred) tp++;
      else if (!isTrue && isPred) fp++;
      else if (isTrue && !isPred) fn++;
    }
    const d = 2 * tp + fp + fn;
    return d ? (2 * tp) / d : 0;
  });
  return mean(scores);
}

const count = (rows: ScoredRow[], outcome: Outcome) =>
  rows.filter((r) => r.outcome === outcome).length;

function accuracyBlock(group: ScoredRow[]): Metrics {
  const answered = group.filter((r) => r.status === "ok");
  const correct = count(answered, "correct");
  const n = answered.length;
  return {
    n_total: group.length,
    n_answered: n,
    n_failed: group.length - n,
    correct,
    incorrect: count(answered, "incorrect"),
    unsure: count(answered, "unsure"),
    invalid: count(answered, "invalid"),
    accuracy: n ? correct / n : null,
    accuracy_ci95: wilsonInterval(correct, n),
    accuracy_all: group.length ? correct / group.length : null,
    unsure_rate: n ? count(answered, "unsure") / n : null,
  };
}

function breakdown(group: ScoredRow[], key: (r: ScoredRow) => string): Metrics {
  const answered = group.filter((r) => r.status === "ok");
  const result: Metrics = {};
  for (const [k, part] of groupBy(answered, key)) {
    const correct = count(part, "correct");
    result[k] = { n: part.length, correct, accuracy: correct / part.length };
  }
  return result;
}

function errorStats(values: number[]): Metrics {
  return {
    n: values.length,
    mean_abs_error: mean(values.map(Math.abs)),
    median_abs_error: median(values.map(Math.abs)),
    mean_signed_error: mean(values),
  };
}

function densityBlock(group: ScoredRow[]): Metrics {
  const answered = group.filter((r) => r.status === "ok");
  const confusion: Record<string, Record<string, number>> = {};
  for (const truth of DENSITY_CATEGORIES) {
    confusion[truth] = {};
    for (const pred of DENSITY_CATEGORIES) {
      confusion[truth][pred] = answered.filter(
        (r) => light(r.expected) === truth && r.predicted_label === pred,
      ).length;
    }
  }
  return {
    count: errorStats(numbers(answered.map((r) => r.count_error))),
    coverage_pp: errorStats(numbers(answered.map((r) => r.coverage_error_pp))),
    mean_category_distance: mean(numbers(answered.map((r) => r.category_rank_error))),
    confusion_truth_by_predicted: confusion,
  };
}

function localisationBlock(group: ScoredRow[]): Metrics {
  const answered = group.filter((r) => r.status === "ok");
  const scored = answered.filter((r) => r.outcome === "scored");
  const scores = answered.map((r) => r.localisation_score ?? 0);
  const countErrors = numbers(scored.map((r) => r.box_count_error));
  return {
    n_scored: scored.length,
    n_invalid: count(answered, "invalid"),
    mean_score: mean(scores),
    median_score: median(scores),
    mean_score_all: group.length ? scores.reduce((a, b) => a + b, 0) / group.length : null,
    mean_matched_iou: mean(numbers(scored.map((r) => r.matched_iou))),
    count_exact_rate: countErrors.length
      ? countErrors.filter((e) => e === 0).length / countErrors.length
      : null,
    count_mean_abs_error: mean(countErrors.map(Math.abs)),
    count_mean_signed_error: mean(countErrors),
  };
}

function localisationSuite(group: ScoredRow[]): Metrics {
  const answered = group.filter((r) => r.status === "ok");
  const by = (key: (r: ScoredRow) => string) => {
    const result: Metrics = {};
    for (const [k, part] of groupBy(answered, key)) {
      result[k] = { n: part.length, mean_score: mean(part.map((r) => r.localisation_score ?? 0)) };
    }
    return result;
  };
  return {
    n_total: group.length,
    n_answered: answered.length,
    n_failed: group.length - answered.length,
    localisation: localisationBlock(group),
    by_class: by((r) => r.benchmark_class),
    by_dataset: by((r) => r.dataset),
  };
}

function suiteMetrics(suite: string, group: ScoredRow[]): Metrics {
  if (suite.startsWith("localisation")) return localisationSuite(group);
  const answered = group.filter((r) => r.status === "ok");
  const metrics: Metrics = {
    ...accuracyBlock(group),
    chance_accuracy: mean(numbers(group.map((r) => r.chance))),
    macro_f1: F1_SUITES.some((s) => suite.startsWith(s)) && answered.length
      ? macroF1(answered)
      : null,
    by_class: breakdown(group, (r) => r.benchmark_class),
    by_dataset: breakdown(group, (r) => r.dataset),
  };
  if (suite.startsWith("density")) metrics.density = densityBlock(group);
  return metrics;
}

function mcVsOpenEnded(rows: ScoredRow[]): Metrics {
  const mc = new Map(
    rows.filter((r) => r.suite.startsWith("species_id/multiple_choice")).map((
      r,
    ) => [r.question_id, r]),
  );
  const pairs = rows
    .filter((r) => r.suite.startsWith("species_id/open_ended") && r.ablation_of)
    .map((oe) => [oe, mc.get(oe.ablation_of as string)] as const)
    .filter(([oe, m]) => m && oe.status === "ok" && m.status === "ok") as [ScoredRow, ScoredRow][];
  const result: Metrics = {};
  for (
    const [condition, part] of groupBy(pairs, ([oe]) => (oe.annotated ? "boxed" : "unannotated"))
  ) {
    result[condition] = {
      n_pairs: part.length,
      multiple_choice_accuracy: part.filter(([, m]) => m.outcome === "correct").length /
        part.length,
      open_ended_accuracy: part.filter(([oe]) => oe.outcome === "correct").length / part.length,
    };
  }
  return result;
}

function usageBlock(group: ScoredRow[]): Metrics {
  const costs = numbers(group.map((r) => r.cost_usd));
  const tokens = numbers(group.map((r) => r.total_tokens));
  return {
    requests: group.length,
    failed: group.filter((r) => r.status === "failed").length,
    total_cost_usd: costs.reduce((a, b) => a + b, 0),
    mean_cost_usd: mean(costs),
    mean_total_tokens: mean(tokens),
    mean_latency_seconds: mean(
      numbers(group.filter((r) => r.status === "ok").map((r) => r.latency_seconds)),
    ),
    retried_requests: group.filter((r) => (r.attempts ?? 1) > 1).length,
  };
}

/** Per model: usage, per-suite metrics, and the MC vs open-ended ablation. */
export function computeMetrics(rows: ScoredRow[]): Record<string, Metrics> {
  const models: Record<string, Metrics> = {};
  for (const [model, modelRows] of groupBy(rows, (r) => r.model)) {
    const suites: Metrics = {};
    for (const [suite, group] of groupBy(modelRows, (r) => r.suite)) {
      suites[suite] = suiteMetrics(suite, group);
    }
    models[model] = {
      usage: usageBlock(modelRows),
      suites,
      species_id_multiple_choice_vs_open_ended: mcVsOpenEnded(modelRows),
    };
  }
  return models;
}
