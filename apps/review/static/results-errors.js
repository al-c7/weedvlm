import { getJson, h, pct, truncate } from "./shared.js";

/** Error kinds that mean a response got no answer; the rest are noise that didn't fail one. */
const NOT_FAILURES = new Set(["retry", "warning"]);

function modelTable(ctx) {
  const stats = new Map();
  for (const r of ctx.responses) {
    const s = stats.get(r.model) ?? { n: 0, failed: 0, retriedOk: 0 };
    s.n++;
    if (r.status === "failed") s.failed++;
    else if (r.attempts > 1) s.retriedOk++;
    stats.set(r.model, s);
  }
  const progressModels = ctx.detail.progress?.models ?? {};
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
        h("th", { class: "r" }, "Failed"),
        h("th", { class: "r" }, "Failure rate"),
        h("th", { class: "r", title: "Answered, but only after retrying" }, "Recovered by retry"),
        h("th", {}, ""),
      ),
    ),
    h(
      "tbody",
      {},
      [...stats].sort().map(([model, s]) =>
        h(
          "tr",
          {},
          h("td", {}, model),
          h("td", { class: "r" }, s.n),
          h("td", { class: "r" }, s.failed),
          h("td", { class: "r" }, pct(s.n ? s.failed / s.n : null)),
          h("td", { class: "r" }, s.retriedOk),
          h(
            "td",
            {},
            progressModels[model]?.stopped_early
              ? h("span", { class: "badge outcome-failed" }, "stopped early: too many failures")
              : null,
          ),
        )
      ),
    ),
  );
}

function problemTable(ctx) {
  const problems = ctx.detail.problems;
  if (problems.length === 0) {
    return h(
      "p",
      { class: "muted" },
      ctx.detail.has_records
        ? "No errors, retries or warnings recorded."
        : "No error details on disk for this run (its responses.jsonl is missing).",
    );
  }
  const multipleModels = new Set(problems.map((p) => p.model)).size > 1;
  return h(
    "table",
    { class: "data" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        multipleModels ? h("th", {}, "Model") : null,
        h("th", {}, "Kind"),
        h("th", {}, "Message"),
        h("th", { class: "r" }, "Count"),
        h("th", {}, ""),
      ),
    ),
    h(
      "tbody",
      {},
      problems.map((p) =>
        h(
          "tr",
          {},
          multipleModels ? h("td", {}, p.model) : null,
          h(
            "td",
            {},
            h(
              "span",
              { class: `badge kind-${NOT_FAILURES.has(p.kind) ? p.kind : "error"}` },
              p.kind,
            ),
            p.http_status ? h("span", { class: "muted small" }, ` HTTP ${p.http_status}`) : null,
          ),
          h("td", { class: "wrap", title: p.message }, truncate(p.message, 200)),
          h("td", { class: "r" }, p.count),
          h(
            "td",
            {},
            NOT_FAILURES.has(p.kind) ? null : h(
              "button",
              {
                onclick: () =>
                  ctx.showResponses({
                    model: p.model,
                    outcomes: ["failed"],
                    error: { kind: p.kind, message: p.message },
                  }),
              },
              "View responses",
            ),
          ),
        )
      ),
    ),
  );
}

const LEVELS = ["ERROR", "WARNING", "INFO"];

function logViewer(ctx) {
  const pre = h("pre", { class: "log" }, "Loading…");
  const status = h("span", { class: "muted small" });
  let lines = [];
  let level = "";

  const draw = () => {
    const shown = lines.filter((line) => {
      if (!level) return true;
      const found = LEVELS.find((l) => line.includes(` ${l} `));
      // Show this level and anything more severe.
      return found !== undefined && LEVELS.indexOf(found) <= LEVELS.indexOf(level);
    });
    pre.textContent = shown.length ? shown.join("\n") : "No log lines match.";
    pre.scrollTop = pre.scrollHeight;
  };

  getJson(`/api/runs/${encodeURIComponent(ctx.runName)}/log?lines=1000`)
    .then((log) => {
      lines = log.lines;
      if (log.total === 0) {
        pre.textContent = "This run has no run.log.";
        return;
      }
      status.textContent = `last ${log.lines.length} of ${log.total} lines`;
      draw();
    })
    .catch((err) => {
      pre.textContent = `Failed to load run.log: ${err.message}`;
    });

  return h(
    "div",
    {},
    h(
      "div",
      { class: "section-head" },
      h("h3", {}, "run.log"),
      h(
        "label",
        { class: "filter" },
        "Level",
        h(
          "select",
          {
            onchange: (e) => {
              level = e.target.value;
              draw();
            },
          },
          h("option", { value: "" }, "All"),
          h("option", { value: "WARNING" }, "Warnings and errors"),
          h("option", { value: "ERROR" }, "Errors only"),
        ),
      ),
      status,
    ),
    pre,
  );
}

export function renderErrors(container, ctx) {
  container.append(
    h("h3", {}, "By model"),
    modelTable(ctx),
    h("h3", {}, "Errors, retries and warnings"),
    h(
      "p",
      { class: "muted small" },
      "Errors are responses that got no usable answer. Retries and warnings didn't fail a response.",
    ),
    problemTable(ctx),
    logViewer(ctx),
  );
}
