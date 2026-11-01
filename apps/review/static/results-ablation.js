import { h, pct, truncate } from "./shared.js";
import { outcomeBadge } from "./results-responses.js";

/** What `state.suite` holds while the ablation is the selected row of the suite matrix. */
export const ABLATION = "species_id: multiple choice vs open-ended";

const PAGE_SIZE = 100;
const AGREEMENTS = [
  ["both", "Both correct"],
  ["mc_only", "Multiple choice only"],
  ["oe_only", "Open-ended only"],
  ["neither", "Neither"],
  ["incomplete", "Not both answered"],
];

const condition = (r) => (r.annotated ? "boxed" : "unannotated");

/** As metrics._mc_vs_open_ended: a pair only counts where the model answered both. */
function agreement(mc, oe) {
  const answered = (r) => r.status === "ok" && r.outcome !== "unscored";
  if (!answered(mc) || !answered(oe)) return "incomplete";
  const mcCorrect = mc.outcome === "correct";
  const oeCorrect = oe.outcome === "correct";
  if (mcCorrect && oeCorrect) return "both";
  if (mcCorrect) return "mc_only";
  return oeCorrect ? "oe_only" : "neither";
}

/** Each open-ended response beside the same model's answer to the question it re-asks. */
export function ablationPairs(responses) {
  const byKey = new Map(responses.map((r) => [`${r.model}\0${r.question_id}`, r]));
  const pairs = [];
  for (const oe of responses) {
    if (!oe.ablation_of) continue;
    const mc = byKey.get(`${oe.model}\0${oe.ablation_of}`);
    if (mc) pairs.push({ mc, oe, agreement: agreement(mc, oe) });
  }
  return pairs;
}

const summaryOf = (m) => m.species_id_multiple_choice_vs_open_ended ?? {};

export const hasAblation = (models) => models.some(([, m]) => Object.keys(summaryOf(m)).length > 0);

/** The ablation's row of the suite matrix: paired accuracy both ways, per image condition. */
export function ablationRow(state, models, onclick) {
  return h(
    "tr",
    { class: state.suite === ABLATION ? "selected" : "" },
    h("th", { scope: "row" }, ABLATION),
    ...models.map(([, m]) => {
      const conditions = Object.entries(summaryOf(m)).sort(([a], [b]) => a.localeCompare(b));
      if (conditions.length === 0) return h("td", { class: "muted" }, "--");
      return h(
        "td",
        { class: "acc-cell", onclick },
        conditions.map(([name, v]) =>
          h(
            "div",
            {},
            `${pct(v.multiple_choice_accuracy)} → ${pct(v.open_ended_accuracy)} `,
            h("span", { class: "muted small" }, `${name} · ${v.n_pairs} pairs`),
          )
        ),
      );
    }),
  );
}

function summaryTable(models, pairs, setFilters) {
  const count = (model, images, key) =>
    pairs.filter((p) => p.mc.model === model && condition(p.oe) === images && p.agreement === key)
      .length;
  const counted = AGREEMENTS.filter(([key]) => key !== "incomplete");
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
        h("th", {}, "Images"),
        h("th", { class: "r" }, "Pairs"),
        h("th", { class: "r" }, "Multiple choice"),
        h("th", { class: "r" }, "Open-ended"),
        h("th", { class: "r" }, "Gap"),
        ...counted.map(([, label]) => h("th", { class: "r" }, label)),
      ),
    ),
    h(
      "tbody",
      {},
      models.flatMap(([name, m]) =>
        Object.entries(summaryOf(m)).sort(([a], [b]) => a.localeCompare(b)).map(([images, v]) =>
          h(
            "tr",
            {},
            h("td", {}, name),
            h("td", {}, images),
            h("td", { class: "r" }, v.n_pairs),
            h("td", { class: "r" }, pct(v.multiple_choice_accuracy)),
            h("td", { class: "r" }, pct(v.open_ended_accuracy)),
            h("td", { class: "r" }, pct(v.multiple_choice_accuracy - v.open_ended_accuracy)),
            ...counted.map(([key, label]) => {
              const n = count(name, images, key);
              return h(
                "td",
                { class: "r" },
                n === 0 ? h("span", { class: "muted" }, "0") : h(
                  "button",
                  {
                    class: "link",
                    title: `Show pairs: ${label.toLowerCase()}`,
                    onclick: () => setFilters({ model: name, images, agreement: key }),
                  },
                  n,
                ),
              );
            }),
          )
        )
      ),
    ),
  );
}

function answerCell(r) {
  return h(
    "td",
    {},
    outcomeBadge(r.outcome),
    " ",
    r.status === "failed"
      ? h(
        "span",
        { class: "error-text" },
        `[${r.error_kind ?? "error"}] ${truncate(r.error_message ?? "", 60)}`,
      )
      : truncate(r.answer ?? "", 60),
  );
}

function pairList(ctx, filters, pairs, update) {
  const models = [...new Set(pairs.map((p) => p.mc.model))].sort();
  const inScope = pairs.filter((p) =>
    (!filters.model || p.mc.model === filters.model) &&
    (!filters.images || condition(p.oe) === filters.images)
  );
  const shown = inScope.filter((p) => !filters.agreement || p.agreement === filters.agreement);
  const rows = shown.slice(0, filters.shown);

  const select = (label, field, values) =>
    h(
      "label",
      { class: "filter" },
      label,
      h(
        "select",
        { onchange: (e) => update({ [field]: e.target.value }) },
        h("option", { value: "" }, "All"),
        values.map((v) => h("option", { value: v, selected: v === filters[field] }, v)),
      ),
    );
  const chip = (key, label) =>
    h(
      "button",
      {
        class: `chip${filters.agreement === key ? " active" : ""}`,
        onclick: () => update({ agreement: key }),
      },
      label,
    );

  const wrap = h(
    "div",
    {},
    h(
      "div",
      { class: "filters" },
      h(
        "div",
        { class: "filter-row" },
        models.length > 1 ? select("Model", "model", models) : null,
        select("Images", "images", ["boxed", "unannotated"]),
      ),
      h(
        "div",
        { class: "filter-row" },
        chip("", "All"),
        AGREEMENTS.map(([key, label]) => {
          const n = inScope.filter((p) => p.agreement === key).length;
          return n > 0 ? chip(key, `${label} ${n}`) : null;
        }),
      ),
    ),
    h("p", { class: "muted" }, `${shown.length} of ${pairs.length} pairs`),
  );
  if (shown.length === 0) {
    wrap.append(h("p", { class: "empty-state" }, "No pairs match these filters."));
    return wrap;
  }

  const open = (p) => ctx.showResponses({ model: p.mc.model, pair: p.mc.question_id });
  wrap.append(
    h(
      "table",
      { class: "data clickable" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          models.length > 1 ? h("th", {}, "Model") : null,
          h("th", {}, "Images"),
          h("th", {}, "Question"),
          h("th", {}, "Expected"),
          h("th", {}, "Multiple choice"),
          h("th", {}, "Open-ended"),
        ),
      ),
      h(
        "tbody",
        {},
        rows.map((p) =>
          h(
            "tr",
            {
              tabindex: "0",
              title: "Show both responses",
              onclick: () => open(p),
              onkeydown: (e) => e.key === "Enter" && open(p),
            },
            models.length > 1 ? h("td", { class: "nowrap" }, p.mc.model) : null,
            h("td", { class: "muted" }, condition(p.oe)),
            h("td", { title: p.mc.question_id }, `${p.mc.dataset} · ${p.mc.benchmark_class}`),
            h("td", {}, p.mc.expected ?? p.oe.expected ?? h("span", { class: "muted" }, "--")),
            answerCell(p.mc),
            answerCell(p.oe),
          )
        ),
      ),
    ),
  );
  if (shown.length > rows.length) {
    wrap.append(
      h(
        "button",
        { onclick: () => update({ shown: filters.shown + PAGE_SIZE }, { keepPage: true }) },
        `Show ${Math.min(PAGE_SIZE, shown.length - rows.length)} more`,
      ),
    );
  }
  return wrap;
}

/** The section under the suite matrix while the ablation row is selected. */
export function renderAblation(ctx, state, models) {
  const filters = (state.ablation ??= { model: "", images: "", agreement: "", shown: PAGE_SIZE });
  const pairs = ablationPairs(ctx.responses);
  const update = (changes, { keepPage = false } = {}) => {
    Object.assign(filters, changes);
    if (!keepPage) filters.shown = PAGE_SIZE;
    ctx.rerender();
  };
  return h(
    "div",
    {},
    h("div", { class: "section-head" }, h("h3", {}, "Species ID: multiple choice vs open-ended")),
    h(
      "p",
      { class: "muted small" },
      "Multiple-choice questions that were also asked open-ended (same image and ground truth, " +
        "no options), over the pairs where the model answered both. The gap is how much of the " +
        "multiple-choice accuracy came from having the options. Click a count to list those pairs.",
    ),
    summaryTable(models, pairs, update),
    h("h4", {}, "Question pairs"),
    pairList(ctx, filters, pairs, update),
  );
}
