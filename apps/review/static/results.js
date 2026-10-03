import { getJson, h, money, when } from "./shared.js";
import { renderScores } from "./results-scores.js";
import { createResponsesView } from "./results-responses.js";
import { renderErrors } from "./results-errors.js";

const SUBTABS = [
  ["scores", "Scores"],
  ["responses", "Responses"],
  ["errors", "Errors"],
];

let runs = [];
let runName = null;
/** @type {{detail: any, responses: any[]} | null} */
let loaded = null;
let subtab = "scores";
let scoresState = { suite: null, countFailures: false };
let responsesView = null;
let loadToken = 0;

const listEl = () => document.getElementById("rs-run-list");
const mainEl = () => document.getElementById("rs-main");

export async function init() {
  document.getElementById("rs-refresh").addEventListener("click", refresh);
  runs = await getJson("/api/runs");
  renderRunList();
  if (runs.length > 0) await selectRun(runs[0].name);
  else {mainEl().replaceChildren(
      h("p", { class: "empty-state" }, "No runs in the results directory yet."),
    );}
}

export function activate() {}

export function onKey(e) {
  if (subtab === "responses") responsesView?.onKey(e);
}

/** Re-reads the run list and the selected run, e.g. while a run is still going. */
async function refresh() {
  runs = await getJson("/api/runs");
  renderRunList();
  if (runName && runs.some((r) => r.name === runName)) await selectRun(runName, { keepView: true });
}

function renderRunList() {
  listEl().replaceChildren(
    ...runs.map((run) =>
      h(
        "li",
        {},
        h(
          "button",
          {
            class: `run-item${run.name === runName ? " active" : ""}`,
            onclick: () => selectRun(run.name),
          },
          h("div", { class: "run-name" }, run.name),
          h("div", { class: "muted small" }, run.models.join(", ")),
          h(
            "div",
            { class: "run-stats small" },
            `${run.answered} answered`,
            run.failed ? h("span", { class: "error-text" }, ` · ${run.failed} failed`) : null,
            ` · ${money(run.cost_usd)}`,
          ),
          h(
            "div",
            {},
            run.state ? h("span", { class: `badge state-${run.state}` }, run.state) : null,
            run.scored_at
              ? h(
                "span",
                { class: `badge ${run.scores_stale ? "kind-warning" : "badge-muted"}` },
                run.scores_stale ? "scores out of date" : "scored",
              )
              : h("span", { class: "badge badge-muted" }, "not scored"),
          ),
        ),
      )
    ),
  );
}

async function selectRun(name, { keepView = false } = {}) {
  const token = ++loadToken;
  const previous = runName;
  runName = name;
  renderRunList();
  mainEl().replaceChildren(h("p", { class: "muted" }, "Loading…"));
  const base = `/api/runs/${encodeURIComponent(name)}`;
  try {
    const [detail, responses] = await Promise.all([getJson(base), getJson(`${base}/responses`)]);
    if (token !== loadToken) return; // another run was picked while this loaded
    loaded = { detail, responses };
  } catch (err) {
    if (token !== loadToken) return;
    mainEl().replaceChildren(
      h("p", { class: "error-text" }, `Failed to load ${name}: ${err.message}`),
    );
    return;
  }
  if (!keepView || previous !== name) {
    subtab = "scores";
    scoresState = { suite: null, countFailures: false };
    responsesView = null;
  }
  render();
}

/** Shared with the sub-views; reads the loaded run live so a refresh is picked up. */
const ctx = {
  get runName() {
    return runName;
  },
  get detail() {
    return loaded.detail;
  },
  get responses() {
    return loaded.responses;
  },
  rerender: () => render(),
  /** Jumps to the Responses view with these filters applied. */
  showResponses(filters) {
    responsesView ??= createResponsesView(ctx);
    responsesView.setFilters(filters);
    subtab = "responses";
    render();
  },
};

function header() {
  const { detail, responses } = loaded;
  const models = [...new Set(responses.map((r) => r.model))].sort();
  const progress = detail.progress;
  const banners = [];
  if (detail.scores_stale) {
    banners.push(
      h(
        "div",
        { class: "notice" },
        `Responses were written after this run was last scored (${
          when(detail.scored_at)
        }), so scores may be out of date. Re-run score-evaluation.py.`,
      ),
    );
  }
  if (progress?.state === "running") {
    banners.push(
      h(
        "div",
        { class: "notice" },
        `Run is marked running (last update ${when(progress.updated_at)}). Refresh for the latest.`,
      ),
    );
  }
  return h(
    "div",
    {},
    h(
      "div",
      { class: "viewer-header" },
      h("h2", { class: "run-title" }, detail.name),
      progress?.state
        ? h("span", { class: `badge state-${progress.state}` }, progress.state)
        : null,
      h("span", { class: "muted" }, models.join(", ")),
    ),
    detail.scored_at ? h("p", { class: "muted small" }, `Scored ${when(detail.scored_at)}`) : null,
    banners,
  );
}

function render() {
  if (!loaded) return;
  const { responses } = loaded;
  const failed = responses.filter((r) => r.status === "failed").length;
  const counts = { scores: null, responses: responses.length, errors: failed || null };

  const content = h("div", { class: "subview" });
  const tabs = h(
    "nav",
    { class: "subtabs" },
    SUBTABS.map(([key, label]) =>
      h(
        "button",
        {
          class: `subtab${key === subtab ? " active" : ""}`,
          onclick: () => {
            subtab = key;
            render();
          },
        },
        label,
        counts[key] !== null ? h("span", { class: "count" }, counts[key]) : null,
      )
    ),
  );
  mainEl().replaceChildren(header(), tabs, content);

  if (subtab === "scores") renderScores(content, ctx, scoresState);
  else if (subtab === "errors") renderErrors(content, ctx);
  else {
    responsesView ??= createResponsesView(ctx);
    responsesView.render(content);
  }
}
