import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { computeFeatures, spreadFraction, unionArea } from "./features.ts";
import {
  classifyScore,
  classifyTree,
  DEFAULT_SCORE_RULE,
  fitAll,
  fitScoreRule,
  fitTree,
  measure,
  type Row,
  scoreRuleLoss,
} from "./fit.ts";

Deno.test("unionArea counts overlap once", () => {
  assertEquals(unionArea([]), 0);
  assertEquals(unionArea([[0, 0, 10, 10]]), 100);
  assertEquals(unionArea([[0, 0, 10, 10], [5, 5, 10, 10]]), 175);
  assertEquals(unionArea([[0, 0, 10, 10], [2, 2, 3, 3]]), 100);
  assertEquals(unionArea([[0, 0, 10, 10], [20, 0, 10, 10]]), 200);
});

Deno.test("spreadFraction counts the grid cells boxes reach", () => {
  assertEquals(spreadFraction([[0, 0, 100, 100]], 400, 400), 1 / 16);
  // Touching a cell's edge isn't reaching into it.
  assertEquals(spreadFraction([[0, 0, 100, 101]], 400, 400), 2 / 16);
  assertEquals(spreadFraction([[0, 0, 400, 400]], 400, 400), 1);
  assertEquals(spreadFraction([[10, 10, 20, 20], [310, 310, 20, 20]], 400, 400), 2 / 16);
});

Deno.test("computeFeatures", () => {
  const features = computeFeatures([[0, 0, 10, 10], [0, 0, 20, 20], [50, 50, 30, 30]], 100, 100);
  assertEquals(features.count, 3);
  assertAlmostEquals(features.coverage, 0.13);
  assertAlmostEquals(features.typical_size, 0.04);
  assertEquals(computeFeatures([], 100, 100).coverage, 0);
});

// Matches the worked examples beside the density thresholds in config.example.yaml.
Deno.test("classifyScore matches the pipeline's examples", () => {
  const rule = DEFAULT_SCORE_RULE;
  assertEquals(classifyScore(rule, 0, 0), 0);
  assertEquals(classifyScore(rule, 1, 1.0), 1);
  assertEquals(classifyScore(rule, 3, 0.3), 2);
  assertEquals(classifyScore(rule, 15, 0.03), 2);
  assertEquals(classifyScore(rule, 10, 0.25), 3);
  assertEquals(classifyScore(rule, 30, 0.03), 3);
});

/** Rows labelled by a known rule: high from 12 weeds, medium from 4. */
function syntheticRows(n: number): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const count = 1 + (i * 7) % 30;
    const coverage = 0.002 * (1 + (i * 13) % 90);
    rows.push({
      count,
      coverage,
      typical_size: coverage / count,
      spread: Math.min(16, count) / 16,
      label: count >= 12 ? 3 : count >= 4 ? 2 : 1,
    });
  }
  return rows;
}

Deno.test("fitTree recovers a threshold rule", () => {
  const rows = syntheticRows(200);
  const tree = fitTree(rows, 2);
  for (const row of rows) assertEquals(classifyTree(tree, row), row.label);
});

Deno.test("fitScoreRule never does worse than where it started", () => {
  const rows = syntheticRows(200);
  const refit = fitScoreRule(rows, DEFAULT_SCORE_RULE, { sweeps: 500, nudges: 300 });
  assert(scoreRuleLoss(refit, rows) <= scoreRuleLoss(DEFAULT_SCORE_RULE, rows));
});

Deno.test("fitAll reports each rule with held-out agreement", () => {
  const rules = fitAll(syntheticRows(120), DEFAULT_SCORE_RULE, 3);
  assertEquals(rules.map((rule) => rule.id), ["current", "refit", "tree"]);
  assertEquals(rules[0].cross_validated, null);
  assert(rules[2].cross_validated!.exact > 0.9);
  assert(rules[1].metrics.mean_error <= rules[0].metrics.mean_error);
});

Deno.test("measure", () => {
  const metrics = measure([1, 2, 3, 1], [1, 3, 1, 1]);
  assertEquals(metrics.exact, 0.5);
  assertEquals(metrics.within_one, 0.75);
  assertEquals(metrics.mean_error, 0.75);
  assertEquals(metrics.confusion[3][1], 1);
});
