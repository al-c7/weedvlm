/**
 * Works out density rules from hand-labelled images, and measures how
 * well each agrees with the labels. Three rules are compared:
 *
 *   current  the pipeline's score rule with the thresholds it has now
 *   refit    the same score rule, its thresholds searched for the best fit
 *   tree     a small tree of threshold tests over every feature
 *
 * The score rule mirrors classify_density in
 * src/weedvlm/generation/density.py, so a refit's thresholds can be
 * pasted straight into the generation config. The tree is free to use
 * typical weed size and spatial spread too, which that rule can't.
 *
 * Categories are ordinal, held as 0 (none) to 3 (high).
 */
import { type FeatureId, FEATURES, type Features, SPREAD_GRID } from "./features.ts";

export const CLASSES = ["none", "low", "medium", "high"] as const;
export type DensityClass = typeof CLASSES[number];

export interface Row extends Features {
  label: number;
}

// -- Score rule ------------------------------------------------------------

export interface ScoreRule {
  count_scale: [number, number];
  coverage_scale: [number, number];
  medium_score: number;
  high_score: number;
  medium_min_count: number;
  high_min_count: number;
}

/** DensityConfig's defaults in src/weedvlm/generation/config.py. */
export const DEFAULT_SCORE_RULE: ScoreRule = {
  count_scale: [1, 40],
  coverage_scale: [0.01, 0.4],
  medium_score: 0.3,
  high_score: 0.6,
  medium_min_count: 2,
  high_min_count: 5,
};

function logScale(value: number, low: number, high: number): number {
  if (value <= low) return 0;
  return Math.min(1, Math.log(value / low) / Math.log(high / low));
}

export function classifyScore(rule: ScoreRule, count: number, coverage: number): number {
  if (count === 0) return 0;
  const score =
    (logScale(count, ...rule.count_scale) + logScale(coverage, ...rule.coverage_scale)) /
    2;
  if (score >= rule.high_score && count >= rule.high_min_count) return 3;
  if (score >= rule.medium_score && count >= rule.medium_min_count) return 2;
  return 1;
}

/** Deterministic PRNG, so the same labels always fit to the same rule. */
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

const sig2 = (value: number) => Number(value.toPrecision(2));
const round2 = (value: number) => Math.round(value * 100) / 100;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * Pulls a rule's numbers back into range and onto round values. The
 * search only ever scores rules in this form, so the thresholds it
 * reports are exactly the ones that were measured.
 */
function tidy(rule: ScoreRule): ScoreRule {
  const countHigh = clamp(Math.round(rule.count_scale[1]), 3, 300);
  const coverageLow = sig2(clamp(rule.coverage_scale[0], 0.0002, 0.1));
  const coverageHigh = sig2(clamp(rule.coverage_scale[1], coverageLow * 2, 1));
  const medium = round2(clamp(rule.medium_score, 0.02, 0.9));
  const high = round2(clamp(rule.high_score, medium + 0.02, 0.98));
  const mediumMin = clamp(Math.round(rule.medium_min_count), 1, 10);
  const highMin = clamp(Math.round(rule.high_min_count), mediumMin, 40);
  return {
    count_scale: [1, countHigh],
    coverage_scale: [coverageLow, coverageHigh],
    medium_score: medium,
    high_score: high,
    medium_min_count: mediumMin,
    high_min_count: highMin,
  };
}

export function scoreRuleLoss(rule: ScoreRule, rows: Row[]): number {
  let loss = 0;
  for (const row of rows) {
    loss += Math.abs(classifyScore(rule, row.count, row.coverage) - row.label);
  }
  return loss;
}

/**
 * Searches the score rule's thresholds for the lowest total category
 * error (so being two categories out costs twice being one out): a
 * random sweep of the plausible ranges, then small nudges to the best
 * rule found. `start` is always a candidate, so the result is never
 * worse than it on these rows.
 */
export function fitScoreRule(
  rows: Row[],
  start: ScoreRule,
  { sweeps = 6000, nudges = 3000, seed = 1 } = {},
): ScoreRule {
  const random = mulberry32(seed);
  const uniform = (low: number, high: number) => low + random() * (high - low);
  const logUniform = (low: number, high: number) =>
    Math.exp(uniform(Math.log(low), Math.log(high)));

  let best = tidy(start);
  let bestLoss = scoreRuleLoss(best, rows);
  const consider = (candidate: ScoreRule) => {
    const rule = tidy(candidate);
    const loss = scoreRuleLoss(rule, rows);
    if (loss < bestLoss) {
      best = rule;
      bestLoss = loss;
    }
  };

  for (let i = 0; i < sweeps; i++) {
    const medium = uniform(0.05, 0.7);
    const mediumMin = 1 + Math.floor(random() * 4);
    consider({
      count_scale: [1, logUniform(4, 200)],
      coverage_scale: [logUniform(0.0005, 0.05), logUniform(0.03, 1)],
      medium_score: medium,
      high_score: uniform(medium + 0.05, 0.95),
      medium_min_count: mediumMin,
      high_min_count: mediumMin + Math.floor(random() * 12),
    });
  }
  for (let i = 0; i < nudges; i++) {
    const factor = () => Math.exp(uniform(-0.25, 0.25));
    const step = () => Math.round(uniform(-1.5, 1.5));
    consider({
      count_scale: [1, best.count_scale[1] * (random() < 0.5 ? factor() : 1)],
      coverage_scale: [
        best.coverage_scale[0] * (random() < 0.5 ? factor() : 1),
        best.coverage_scale[1] * (random() < 0.5 ? factor() : 1),
      ],
      medium_score: best.medium_score + (random() < 0.5 ? uniform(-0.05, 0.05) : 0),
      high_score: best.high_score + (random() < 0.5 ? uniform(-0.05, 0.05) : 0),
      medium_min_count: best.medium_min_count + (random() < 0.3 ? step() : 0),
      high_min_count: best.high_min_count + (random() < 0.3 ? step() : 0),
    });
  }
  return best;
}

/** The rule as the `density:` keys of the generation config. */
export function scoreRuleYaml(rule: ScoreRule): string {
  return [
    `count_scale: [${rule.count_scale.join(", ")}]`,
    `coverage_scale: [${rule.coverage_scale.join(", ")}]`,
    `medium_score: ${rule.medium_score}`,
    `high_score: ${rule.high_score}`,
    `medium_min_count: ${rule.medium_min_count}`,
    `high_min_count: ${rule.high_min_count}`,
  ].join("\n");
}

// -- Threshold tree --------------------------------------------------------

export interface Leaf {
  label: number;
  /** How many training rows of each category ended up here. */
  counts: number[];
}

export interface Split {
  feature: FeatureId;
  threshold: number;
  /** Rows with feature < threshold. */
  below: TreeNode;
  /** Rows with feature >= threshold. */
  atOrAbove: TreeNode;
}

export type TreeNode = Leaf | Split;

const isLeaf = (node: TreeNode): node is Leaf => "label" in node;

function classCounts(rows: Row[]): number[] {
  const counts = CLASSES.map(() => 0);
  for (const row of rows) counts[row.label]++;
  return counts;
}

function gini(counts: number[], total: number): number {
  if (total === 0) return 0;
  let sumSquares = 0;
  for (const count of counts) sumSquares += (count / total) ** 2;
  return 1 - sumSquares;
}

/** The commonest category; a tie goes to whichever is nearer the middle of the rows. */
function majority(counts: number[]): number {
  const total = counts.reduce((a, b) => a + b, 0);
  const mean = counts.reduce((sum, count, label) => sum + count * label, 0) / Math.max(total, 1);
  let best = 0;
  for (let label = 1; label < counts.length; label++) {
    const better = counts[label] > counts[best] ||
      (counts[label] === counts[best] && Math.abs(label - mean) < Math.abs(best - mean));
    if (better) best = label;
  }
  return best;
}

/**
 * The thresholds worth testing on a feature: one between each pair of
 * neighbouring values that occur. Whole-number features split on the
 * value itself; fractions split on a round number between the two, so
 * the finished rule reads as "coverage < 2.5%" rather than "< 2.4871%".
 */
function candidateThresholds(rows: Row[], feature: FeatureId): number[] {
  const values = [...new Set(rows.map((row) => row[feature]))].sort((a, b) => a - b);
  const exact = feature === "count" || feature === "spread";
  const thresholds = new Set<number>();
  for (let i = 1; i < values.length; i++) {
    const low = values[i - 1];
    const high = values[i];
    if (exact || low <= 0) {
      thresholds.add(high);
      continue;
    }
    const middle = Math.sqrt(low * high);
    const rounded = [2, 3].map((digits) => Number(middle.toPrecision(digits)))
      .find((value) => value > low && value <= high);
    thresholds.add(rounded ?? high);
  }
  return [...thresholds];
}

function grow(rows: Row[], depth: number, minLeaf: number): TreeNode {
  const counts = classCounts(rows);
  const leaf: Leaf = { label: majority(counts), counts };
  const impurity = gini(counts, rows.length);
  if (depth === 0 || impurity === 0 || rows.length < 2 * minLeaf) return leaf;

  let best: { feature: FeatureId; threshold: number; impurity: number } | null = null;
  for (const { id: feature } of FEATURES) {
    for (const threshold of candidateThresholds(rows, feature)) {
      const below = CLASSES.map(() => 0);
      let belowTotal = 0;
      for (const row of rows) {
        if (row[feature] < threshold) {
          below[row.label]++;
          belowTotal++;
        }
      }
      const aboveTotal = rows.length - belowTotal;
      if (belowTotal < minLeaf || aboveTotal < minLeaf) continue;
      const above = counts.map((count, label) => count - below[label]);
      const split = (belowTotal * gini(below, belowTotal) + aboveTotal * gini(above, aboveTotal)) /
        rows.length;
      if (split < (best?.impurity ?? impurity) - 1e-9) {
        best = { feature, threshold, impurity: split };
      }
    }
  }
  if (!best) return leaf;

  const { feature, threshold } = best;
  const below = grow(rows.filter((row) => row[feature] < threshold), depth - 1, minLeaf);
  const atOrAbove = grow(rows.filter((row) => row[feature] >= threshold), depth - 1, minLeaf);
  // A split whose two sides answer the same decides nothing.
  if (isLeaf(below) && isLeaf(atOrAbove) && below.label === atOrAbove.label) return leaf;
  return { feature, threshold, below, atOrAbove };
}

/**
 * Grows a tree of at most `depth` tests deep. Each test is the single
 * threshold that leaves the two sides most uniform in category (lowest
 * Gini impurity), and no side may hold fewer than `minLeaf` images, so
 * a branch can't be carved out for one or two odd labels.
 */
export function fitTree(rows: Row[], depth: number, minLeaf?: number): TreeNode {
  return grow(rows, depth, minLeaf ?? Math.max(3, Math.ceil(rows.length * 0.03)));
}

export function classifyTree(node: TreeNode, features: Features): number {
  while (!isLeaf(node)) {
    node = features[node.feature] < node.threshold ? node.below : node.atOrAbove;
  }
  return node.label;
}

const capitalise = (text: string) => text[0].toUpperCase() + text.slice(1);

function thresholdText(feature: FeatureId, threshold: number): string {
  if (feature === "count") return String(threshold);
  if (feature === "spread") {
    const cells = SPREAD_GRID * SPREAD_GRID;
    return `${Math.round(threshold * cells)} of ${cells} grid cells`;
  }
  return `${Number((threshold * 100).toPrecision(3))}%`;
}

/** The tree written out as nested if/else lines. */
export function treeText(node: TreeNode, indent = ""): string {
  if (isLeaf(node)) {
    const seen = node.counts
      .map((count, label) => count ? `${count} ${CLASSES[label]}` : "")
      .filter(Boolean)
      .join(", ");
    return `${indent}-> ${capitalise(CLASSES[node.label])}   (labelled: ${seen})`;
  }
  const name = FEATURES.find((f) => f.id === node.feature)!.label.toLowerCase();
  return [
    `${indent}if ${name} < ${thresholdText(node.feature, node.threshold)}:`,
    treeText(node.below, indent + "    "),
    `${indent}else:`,
    treeText(node.atOrAbove, indent + "    "),
  ].join("\n");
}

// -- Agreement -------------------------------------------------------------

export interface Metrics {
  n: number;
  /** Fraction given exactly the labelled category. */
  exact: number;
  /** Fraction given the labelled category or one next to it. */
  within_one: number;
  /** Mean number of categories out. */
  mean_error: number;
  /** confusion[labelled][predicted] */
  confusion: number[][];
}

export function measure(labels: number[], predictions: number[]): Metrics {
  const confusion = CLASSES.map(() => CLASSES.map(() => 0));
  let exact = 0;
  let withinOne = 0;
  let error = 0;
  labels.forEach((label, i) => {
    const distance = Math.abs(label - predictions[i]);
    confusion[label][predictions[i]]++;
    if (distance === 0) exact++;
    if (distance <= 1) withinOne++;
    error += distance;
  });
  const n = Math.max(labels.length, 1);
  return {
    n: labels.length,
    exact: exact / n,
    within_one: withinOne / n,
    mean_error: error / n,
    confusion,
  };
}

const CV_FOLDS = 5;
/** Below this, held-out folds are too small for their agreement to mean anything. */
export const MIN_ROWS_FOR_CV = 30;

/**
 * Agreement on images the rule wasn't fitted to: the rows are dealt
 * into CV_FOLDS groups, and each group is predicted by a rule fitted to
 * the other groups. This is the honest estimate of how a fitted rule
 * will do on images that haven't been labelled.
 */
function crossValidate(rows: Row[], fit: (train: Row[]) => (row: Row) => number): Metrics | null {
  if (rows.length < MIN_ROWS_FOR_CV) return null;
  const random = mulberry32(7);
  const order = rows.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const predictions = new Array<number>(rows.length);
  for (let fold = 0; fold < CV_FOLDS; fold++) {
    const held = new Set(order.filter((_, position) => position % CV_FOLDS === fold));
    const predict = fit(rows.filter((_, i) => !held.has(i)));
    for (const i of held) predictions[i] = predict(rows[i]);
  }
  return measure(rows.map((row) => row.label), predictions);
}

export interface FittedRule {
  id: "current" | "refit" | "tree";
  name: string;
  summary: string;
  /** The rule itself, as text to read or paste. */
  text: string;
  metrics: Metrics;
  /** Null for the current rule (nothing was fitted) and when there are too few labels. */
  cross_validated: Metrics | null;
  /** Category index per row, in the order the rows were given. */
  predictions: number[];
}

export function fitAll(rows: Row[], current: ScoreRule, treeDepth: number): FittedRule[] {
  const labels = rows.map((row) => row.label);
  const scorePredictor = (rule: ScoreRule) => (row: Row) =>
    classifyScore(rule, row.count, row.coverage);

  const refit = fitScoreRule(rows, current);
  const tree = fitTree(rows, treeDepth);
  // The tree's minimum branch size is held at the full set's, so the held-out
  // trees are no freer to chase single images than the one being reported.
  const minLeaf = Math.max(3, Math.ceil(rows.length * 0.03));

  const describe = (
    id: FittedRule["id"],
    name: string,
    summary: string,
    text: string,
    predict: (row: Row) => number,
    crossValidated: Metrics | null,
  ): FittedRule => {
    const predictions = rows.map(predict);
    return {
      id,
      name,
      summary,
      text,
      metrics: measure(labels, predictions),
      cross_validated: crossValidated,
      predictions,
    };
  };

  return [
    describe(
      "current",
      "Current rule",
      "The count/coverage score rule with the thresholds in the generation config.",
      scoreRuleYaml(current),
      scorePredictor(current),
      null,
    ),
    describe(
      "refit",
      "Refitted score rule",
      "The same count/coverage score rule, its thresholds refitted to your labels. " +
        "These are the generation config's density keys.",
      scoreRuleYaml(refit),
      scorePredictor(refit),
      crossValidate(
        rows,
        (train) => scorePredictor(fitScoreRule(train, current, { sweeps: 2000, nudges: 1000 })),
      ),
    ),
    describe(
      "tree",
      "Threshold tree",
      `Up to ${treeDepth} threshold tests in a row, over all four measurements.`,
      treeText(tree),
      (row) => classifyTree(tree, row),
      crossValidate(rows, (train) => {
        const heldOutTree = fitTree(train, treeDepth, minLeaf);
        return (row) => classifyTree(heldOutTree, row);
      }),
    ),
  ];
}
