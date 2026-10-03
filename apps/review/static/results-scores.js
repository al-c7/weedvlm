import { h, money, num, pct, seconds } from "./shared.js";

const OUTCOMES = ["correct", "incorrect", "unsure", "invalid", "failed", "unscored"];

/** Per-model totals, computed from the responses so they work for unscored runs too. */
function modelTotals(responses) {
  const totals = new Map();
  for (const r of responses) {
    let t = totals.get(r.model);
    if (!t) {
      t = {
        n: 0,
        retried: 0,
        cost: 0,
        latency: [],
        outcomes: Object.fromEntries(OUTCOMES.map((o) => [o, 0])),
      };
      totals.set(r.model, t);
    }
    t.n++;
    t.outcomes[r.outcome]++;
    if (r.attempts > 1) t.retried++;
    t.cost += r.cost_usd ?? 0;
    if (r.latency_seconds !== null) t.latency.push(r.latency_seconds);
  }
  return totals;
}

function overviewTable(ctx) {
  const totals = modelTotals(ctx.responses);
  const shown = OUTCOMES.filter((o) =>
    o !== "unscored" || [...totals.values()].some((t) => t.outcomes.unscored)
  );
  return h(
    "table",
    { class: "data" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", {}, "Model"),
        h("th", { class: "r" }, "Responses"),
        ...shown.map((o) => h("th", { class: "r" }, o)),
        h("th", { class: "r" }, "Retried"),
        h("th", { class: "r" }, "Cost"),
        h("th", { class: "r" }, "Mean latency"),
      ),
    ),
    h(
      "tbody",
      {},
      [...totals].sort().map(([model, t]) =>
        h(
          "tr",
          {},
          h("td", {}, model),
          h("td", { class: "r" }, t.n),
          ...shown.map((o) =>
            h(
              "td",
              { class: "r" },
              t.outcomes[o] === 0 ? h("span", { class: "muted" }, "0") : h(
                "button",
                {
                  class: "link",
                  title: `Show ${o} responses`,
                  onclick: () => ctx.showResponses({ model, outcomes: [o] }),
                },
                t.outcomes[o],
              ),
            )
          ),
          h("td", { class: "r" }, t.retried),
          h("td", { class: "r" }, money(t.cost)),
          h(
            "td",
            { class: "r" },
            seconds(
              t.latency.length ? t.latency.reduce((a, b) => a + b, 0) / t.latency.length : null,
            ),
          ),
        )
      ),
    ),
  );
}

/** A thin single-hue bar, with a tick where random guessing would land. */
function accuracyCell(suite, countFailures, onclick) {
  if (!suite) return h("td", { class: "muted" }, "--");
  const value = countFailures ? suite.accuracy_all : suite.accuracy;
  const ci = suite.accuracy_ci95;
  const detail = `${suite.n_answered}/${suite.n_total} answered` +
    (suite.n_failed ? ` · ${suite.n_failed} failed` : "") +
    (ci && !countFailures ? ` · CI ${pct(ci[0], 0)}–${pct(ci[1], 0)}` : "");
  return h(
    "td",
    { class: "acc-cell", onclick },
    h(
      "div",
      { class: "acc" },
      h(
        "div",
        { class: "acc-bar", title: `chance ${pct(suite.chance_accuracy, 0)}` },
        h("i", { style: `width:${(value ?? 0) * 100}%` }),
        suite.chance_accuracy !== null && suite.chance_accuracy !== undefined
          ? h("b", { class: "chance", style: `left:${suite.chance_accuracy * 100}%` })
          : null,
      ),
      h("span", { class: "acc-value" }, pct(value)),
    ),
    h("div", { class: "muted small" }, detail),
  );
}

function suiteMatrix(ctx, state, models) {
  const suites = [...new Set(models.flatMap(([, m]) => Object.keys(m.suites)))].sort();
  return h(
    "table",
    { class: "data matrix" },
    h(
      "thead",
      {},
      h("tr", {}, h("th", {}, "Suite"), ...models.map(([name]) => h("th", {}, name))),
    ),
    h(
      "tbody",
      {},
      suites.map((suite) =>
        h(
          "tr",
          { class: state.suite === suite ? "selected" : "" },
          h("th", { scope: "row" }, suite),
          ...models.map(([, m]) =>
            accuracyCell(m.suites[suite], state.countFailures, () => {
              state.suite = suite;
              ctx.rerender();
            })
          ),
        )
      ),
    ),
  );
}

function breakdownTable(models, suiteName, field, label) {
  const groups = new Set();
  for (const [, m] of models) {
    for (const key of Object.keys(m.suites[suiteName]?.[field] ?? {})) groups.add(key);
  }
  const accuracy = (m, key) => m.suites[suiteName]?.[field]?.[key];
  // Weakest first, so the classes worth looking at are at the top.
  const worst = (key) => Math.min(...models.map(([, m]) => accuracy(m, key)?.accuracy ?? Infinity));
  const rows = [...groups].sort((a, b) => worst(a) - worst(b) || a.localeCompare(b));
  return h(
    "table",
    { class: "data" },
    h(
      "thead",
      {},
      h("tr", {}, h("th", {}, label), ...models.map(([name]) => h("th", { class: "r" }, name))),
    ),
    h(
      "tbody",
      {},
      rows.map((key) =>
        h(
          "tr",
          {},
          h("th", { scope: "row" }, key),
          ...models.map(([, m]) => {
            const a = accuracy(m, key);
            return h(
              "td",
              { class: "r" },
              a ? `${pct(a.accuracy, 0)} ` : "--",
              a ? h("span", { class: "muted small" }, `n=${a.n}`) : null,
            );
          }),
        )
      ),
    ),
  );
}

function densityBlock(models, suiteName) {
  const withDensity = models.filter(([, m]) => m.suites[suiteName]?.density);
  if (withDensity.length === 0) return null;

  const errors = h(
    "table",
    { class: "data" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", {}, "Model"),
        h(
          "th",
          { class: "r", title: "0 = exact category, 1 = one category off" },
          "Category distance",
        ),
        h("th", { class: "r" }, "Count MAE"),
        h("th", { class: "r" }, "Count bias"),
        h("th", { class: "r" }, "Coverage MAE (pp)"),
        h("th", { class: "r" }, "Coverage bias (pp)"),
      ),
    ),
    h(
      "tbody",
      {},
      withDensity.map(([name, m]) => {
        const d = m.suites[suiteName].density;
        return h(
          "tr",
          {},
          h("td", {}, name),
          h("td", { class: "r" }, num(d.mean_category_distance)),
          h("td", { class: "r" }, num(d.count.mean_abs_error, 1)),
          h("td", { class: "r" }, num(d.count.mean_signed_error, 1)),
          h("td", { class: "r" }, num(d.coverage_pp.mean_abs_error, 1)),
          h("td", { class: "r" }, num(d.coverage_pp.mean_signed_error, 1)),
        );
      }),
    ),
  );

  const confusions = withDensity.map(([name, m]) => {
    const matrix = m.suites[suiteName].density.confusion_truth_by_predicted;
    const categories = Object.keys(matrix);
    return h(
      "div",
      { class: "confusion" },
      h("h4", {}, name),
      h(
        "table",
        {},
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", {}, "truth ↓ / predicted →"),
            ...categories.map((c) => h("th", {}, c)),
          ),
        ),
        h(
          "tbody",
          {},
          categories.map((truth) => {
            const rowTotal = categories.reduce((s, c) => s + matrix[truth][c], 0);
            return h(
              "tr",
              {},
              h("th", { scope: "row" }, truth),
              ...categories.map((c) => {
                const share = rowTotal ? matrix[truth][c] / rowTotal : 0;
                return h(
                  "td",
                  {
                    class: truth === c ? "diag" : "",
                    style: `--share:${share}`,
                    title: `${pct(share, 0)} of truth "${truth}"`,
                  },
                  matrix[truth][c],
                );
              }),
            );
          }),
        ),
      ),
    );
  });

  return h(
    "div",
    {},
    h("h4", {}, "Density estimates"),
    errors,
    h(
      "p",
      { class: "muted small" },
      "Confusion matrices: rows are the true category; shading is the share of that row.",
    ),
    h("div", { class: "confusion-row" }, ...confusions),
  );
}

function mcVsOpenEnded(models) {
  const rows = [];
  for (const [name, m] of models) {
    for (const [condition, v] of Object.entries(m.species_id_multiple_choice_vs_open_ended ?? {})) {
      rows.push({ name, condition, ...v });
    }
  }
  if (rows.length === 0) return null;
  return h(
    "div",
    {},
    h("h3", {}, "Species ID: multiple choice vs open-ended"),
    h(
      "table",
      { class: "data" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, "Model"),
          h("th", {}, "Images"),
          h("th", { class: "r" }, "Pairs"),
          h("th", { class: "r" }, "Multiple choice"),
          h("th", { class: "r" }, "Open-ended"),
          h("th", { class: "r" }, "Gap"),
        ),
      ),
      h(
        "tbody",
        {},
        rows.map((r) =>
          h(
            "tr",
            {},
            h("td", {}, r.name),
            h("td", {}, r.condition),
            h("td", { class: "r" }, r.n_pairs),
            h("td", { class: "r" }, pct(r.multiple_choice_accuracy)),
            h("td", { class: "r" }, pct(r.open_ended_accuracy)),
            h("td", { class: "r" }, pct(r.multiple_choice_accuracy - r.open_ended_accuracy)),
          )
        ),
      ),
    ),
  );
}

/** `state` outlives re-renders: which suite is expanded, and the accuracy basis. */
export function renderScores(container, ctx, state) {
  container.append(h("h3", {}, "Outcomes"), overviewTable(ctx));

  if (!ctx.detail.scores) {
    container.append(
      h(
        "div",
        { class: "notice" },
        h("strong", {}, "This run hasn't been scored. "),
        "Score it to get accuracy per suite: ",
        h("code", {}, `uv run src/weedvlm/scripts/score-evaluation.py results/${ctx.runName}`),
      ),
    );
    return;
  }

  const models = Object.entries(ctx.detail.scores.models).sort(([a], [b]) => a.localeCompare(b));
  container.append(
    h(
      "div",
      { class: "section-head" },
      h("h3", {}, "Accuracy by suite"),
      h(
        "label",
        { class: "row" },
        h("input", {
          type: "checkbox",
          checked: state.countFailures,
          onchange: (e) => {
            state.countFailures = e.target.checked;
            ctx.rerender();
          },
        }),
        "Count failed responses as wrong",
      ),
    ),
    h(
      "p",
      { class: "muted small" },
      "Bars show accuracy over answered questions; the tick marks random-guess accuracy. " +
        "Click a row to break it down.",
    ),
    suiteMatrix(ctx, state, models),
  );

  if (state.suite && models.some(([, m]) => m.suites[state.suite])) {
    container.append(
      h(
        "div",
        { class: "section-head" },
        h("h3", {}, state.suite),
        h(
          "button",
          {
            onclick: () =>
              ctx.showResponses({
                suite: state.suite,
                outcomes: ["incorrect", "invalid", "unsure", "failed"],
              }),
          },
          "Show wrong answers & failures",
        ),
      ),
      densityBlock(models, state.suite),
      h("h4", {}, "By class"),
      breakdownTable(models, state.suite, "by_class", "Class"),
      h("h4", {}, "By dataset"),
      breakdownTable(models, state.suite, "by_dataset", "Dataset"),
    );
  }
  container.append(mcVsOpenEnded(models));
}
