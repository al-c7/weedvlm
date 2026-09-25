/**
 * The one place the benchmark-composition sampling logic lives -- used by both the live preview
 * (/api/compose) and the final export (/api/export), so a preview always matches what export
 * would actually write.
 *
 * For each question_type ("task"), questions are grouped by their `benchmark_class` (the species
 * name, same/different-species label, or density category -- computed once in Python at
 * generation time, see src/weedvlm/types/questions.py). The `num_classes` classes with the most
 * available questions are kept, and within each kept class, up to `questions_per_class` are drawn
 * with a seeded shuffle, so the same config always reproduces the same selection.
 */

import type { BenchmarkConfig, TaskConfig } from "./config.ts";
import type { FlagStore, Question } from "./types.ts";

export interface ClassResult {
  class_key: string;
  /** Distinct selectable units available for this class (after excluding flagged questions). */
  available: number;
  /** Units actually selected -- min(available, questions_per_class). */
  selected: number;
  question_ids: string[];
}

export interface TaskResult {
  task: string;
  target_num_classes: number;
  target_questions_per_class: number;
  /** Every class that had >=1 available unit, not just the ones kept -- so the GUI can show
   * "kept 25 of 40 available classes". */
  total_classes_available: number;
  classes: ClassResult[];
  total_selected: number;
}

export type ComposeResult = Record<string, TaskResult>;

/** Groups rows that must be selected/rejected together as one unit. Density estimation asks the
 * same source image under both the boxed and unannotated conditions (per docs/WEED-DENSITY.md,
 * "the same source images will be used between each scenario") -- selecting one selects both, and
 * it only counts once against questions_per_class. Every other task's rows stand alone. */
function dedupeKey(q: Question, task: string): string {
  if (task === "density_estimation") return `${q.source.dataset_name}:${q.source.image_id}`;
  return q.id;
}

function classKey(q: Question): string {
  return q.benchmark_class ?? (q as { density_category?: string }).density_category ?? "unknown";
}

/** mulberry32 -- small, fast, deterministic PRNG from a 32-bit seed. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Cheap string hash (FNV-1a), used to derive a per-class seed from the run seed + class name so
 * that editing one class's config doesn't reshuffle every other class's selection. */
function fnv1a(str: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function seededShuffle<T>(items: T[], rand: () => number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function selectTask(
  questions: Question[],
  flags: FlagStore,
  task: string,
  taskConfig: TaskConfig,
  seed: number,
  excludeFlagged: boolean,
): TaskResult {
  const candidates = questions.filter((q) =>
    q.question_type === task &&
    (!excludeFlagged || flags.flags[q.id]?.flag !== "flagged")
  );

  const byClass = new Map<string, Question[]>();
  for (const q of candidates) {
    const key = classKey(q);
    (byClass.get(key) ?? byClass.set(key, []).get(key)!).push(q);
  }

  const rankedClasses = [...byClass.entries()]
    .map(([classKeyValue, rows]) => {
      const units = new Map<string, Question[]>();
      for (const row of rows) {
        const key = dedupeKey(row, task);
        (units.get(key) ?? units.set(key, []).get(key)!).push(row);
      }
      return { classKeyValue, units };
    })
    .sort((a, b) => b.units.size - a.units.size || a.classKeyValue.localeCompare(b.classKeyValue));

  const kept = rankedClasses.slice(0, Math.max(0, taskConfig.num_classes));

  const classes: ClassResult[] = kept.map(({ classKeyValue, units }) => {
    const classSeed = (seed + fnv1a(classKeyValue)) >>> 0;
    const shuffledUnitKeys = seededShuffle([...units.keys()], mulberry32(classSeed));
    const chosenUnitKeys = shuffledUnitKeys.slice(0, Math.max(0, taskConfig.questions_per_class));

    return {
      class_key: classKeyValue,
      available: units.size,
      selected: chosenUnitKeys.length,
      question_ids: chosenUnitKeys.flatMap((key) => units.get(key)!.map((q) => q.id)),
    };
  });

  classes.sort((a, b) => b.selected - a.selected || a.class_key.localeCompare(b.class_key));

  return {
    task,
    target_num_classes: taskConfig.num_classes,
    target_questions_per_class: taskConfig.questions_per_class,
    total_classes_available: byClass.size,
    classes,
    total_selected: classes.reduce((sum, c) => sum + c.question_ids.length, 0),
  };
}

export function compose(
  questions: Question[],
  flags: FlagStore,
  config: BenchmarkConfig,
): ComposeResult {
  const result: ComposeResult = {};
  for (const [task, taskConfig] of Object.entries(config.tasks)) {
    result[task] = selectTask(
      questions,
      flags,
      task,
      taskConfig,
      config.seed,
      config.exclude_flagged,
    );
  }
  return result;
}
