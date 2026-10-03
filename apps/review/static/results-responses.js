import {
  getJson,
  h,
  isLabelChoices,
  labelSwatch,
  money,
  normalizeMessage,
  seconds,
  truncate,
  when,
} from "./shared.js";

const OUTCOMES = ["correct", "incorrect", "unsure", "invalid", "failed", "unscored"];
const WRONG = ["incorrect", "invalid", "unsure"];
const PAGE_SIZE = 100;

export const outcomeBadge = (outcome) => h("span", { class: `badge outcome-${outcome}` }, outcome);

const norm = (s) =>
  String(s).trim().replace(/^["'`]+|["'`.]+$/g, "").replace(/\s+/g, " ").toLowerCase();
/** Mirrors scoring._label: "2", "Box 2" and "#2" all mean box 2. */
const boxLabel = (s) =>
  /^(?:bounding box|box|label|number)?\s*#?\s*(\d+)(?:\.0+)?$/.exec(norm(s))?.[1] ?? norm(s);

/**
 * The Responses view: a filterable table of every response in the run, and
 * a detail page for one response (with prev/next through the filtered list).
 */
export function createResponsesView(ctx) {
  const state = {
    model: "",
    suite: "",
    dataset: "",
    outcomes: new Set(),
    text: "",
    // {kind, message}: only responses that failed with exactly this error
    error: null,
    shown: PAGE_SIZE,
    detail: -1,
  };
  let container;
  let filtered = [];

  function setFilters(filters) {
    state.model = filters.model ?? "";
    state.suite = filters.suite ?? "";
    state.dataset = filters.dataset ?? "";
    state.outcomes = new Set(filters.outcomes ?? []);
    state.text = filters.text ?? "";
    state.error = filters.error ?? null;
    state.shown = PAGE_SIZE;
    state.detail = -1;
  }

  const matches = (r, skipOutcome) =>
    (!state.model || r.model === state.model) &&
    (!state.suite || r.suite === state.suite) &&
    (!state.dataset || r.dataset === state.dataset) &&
    (skipOutcome || state.outcomes.size === 0 || state.outcomes.has(r.outcome)) &&
    (!state.error ||
      (r.error_kind === state.error.kind &&
        normalizeMessage(r.error_message ?? "") === state.error.message)) &&
    (!state.text || haystack(r).includes(state.text.toLowerCase()));

  const haystack = (r) =>
    [r.question_id, r.benchmark_class, r.expected, r.answer, r.error_message].join("\n")
      .toLowerCase();

  function render(target) {
    container = target;
    container.replaceChildren();
    if (state.detail >= 0 && filtered[state.detail]) return renderDetail();
    state.detail = -1;
    filtered = ctx.responses.filter((r) => matches(r, false));
    container.append(renderFilters(), renderTable());
  }

  function select(label, value, values, onchange) {
    return h(
      "label",
      { class: "filter" },
      label,
      h(
        "select",
        { onchange: (e) => onchange(e.target.value) },
        h("option", { value: "" }, "All"),
        values.filter(Boolean).map((v) => h("option", { value: v, selected: v === value }, v)),
      ),
    );
  }

  function renderFilters() {
    const distinct = (field) => [...new Set(ctx.responses.map((r) => r[field]))].sort();
    const models = distinct("model");
    const counts = Object.fromEntries(OUTCOMES.map((o) => [o, 0]));
    for (const r of ctx.responses) if (matches(r, true)) counts[r.outcome]++;

    const update = (fn) => {
      fn();
      state.shown = PAGE_SIZE;
      render(container);
    };
    const chip = (label, outcomes) => {
      const active = outcomes.length === state.outcomes.size &&
        outcomes.every((o) => state.outcomes.has(o));
      return h(
        "button",
        {
          class: `chip${active ? " active" : ""}`,
          onclick: () => update(() => (state.outcomes = new Set(outcomes))),
        },
        label,
      );
    };

    return h(
      "div",
      { class: "filters" },
      h(
        "div",
        { class: "filter-row" },
        models.length > 1
          ? select("Model", state.model, models, (v) => update(() => (state.model = v)))
          : null,
        select("Suite", state.suite, distinct("suite"), (v) => update(() => (state.suite = v))),
        select(
          "Dataset",
          state.dataset,
          distinct("dataset"),
          (v) => update(() => (state.dataset = v)),
        ),
        h(
          "label",
          { class: "filter grow" },
          "Search",
          h("input", {
            type: "search",
            placeholder: "question id, class, answer, error…",
            value: state.text,
            oninput: (e) => {
              state.text = e.target.value;
              state.shown = PAGE_SIZE;
              // Re-render only the table, so the search box keeps focus.
              filtered = ctx.responses.filter((r) => matches(r, false));
              container.lastChild.replaceWith(renderTable());
            },
          }),
        ),
      ),
      h(
        "div",
        { class: "filter-row" },
        chip("All", []),
        chip("Wrong answers", WRONG),
        chip("Failed (no answer)", ["failed"]),
        h("span", { class: "sep" }),
        OUTCOMES.filter((o) => counts[o] > 0).map((o) => chip(`${o} ${counts[o]}`, [o])),
      ),
      state.error
        ? h(
          "div",
          { class: "filter-row" },
          h(
            "button",
            {
              class: "chip active",
              title: "Clear",
              onclick: () => update(() => (state.error = null)),
            },
            `error: [${state.error.kind}] ${truncate(state.error.message, 100)} ✕`,
          ),
        )
        : null,
    );
  }

  function renderTable() {
    const multipleModels = new Set(ctx.responses.map((r) => r.model)).size > 1;
    const rows = filtered.slice(0, state.shown);
    const wrap = h("div", {});
    wrap.append(
      h("p", { class: "muted" }, `${filtered.length} of ${ctx.responses.length} responses`),
    );
    if (filtered.length === 0) {
      wrap.append(h("p", { class: "empty-state" }, "No responses match these filters."));
      return wrap;
    }
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
            h("th", {}, "Outcome"),
            multipleModels ? h("th", {}, "Model") : null,
            h("th", {}, "Suite"),
            h("th", {}, "Question"),
            h("th", {}, "Expected"),
            h("th", {}, "Answer / error"),
            h("th", { class: "r" }, "Tries"),
            h("th", { class: "r" }, "Time"),
          ),
        ),
        h(
          "tbody",
          {},
          rows.map((r, i) =>
            h(
              "tr",
              {
                tabindex: "0",
                onclick: () => open(i),
                onkeydown: (e) => e.key === "Enter" && open(i),
              },
              h("td", {}, outcomeBadge(r.outcome)),
              multipleModels ? h("td", { class: "nowrap" }, r.model) : null,
              h("td", { class: "muted" }, r.suite || r.task),
              h("td", { title: r.question_id }, `${r.dataset} · ${r.benchmark_class}`),
              h("td", {}, r.expected ?? h("span", { class: "muted" }, "--")),
              h(
                "td",
                {},
                r.status === "failed"
                  ? h(
                    "span",
                    { class: "error-text" },
                    `[${r.error_kind ?? "error"}] ${truncate(r.error_message ?? "", 80)}`,
                  )
                  : truncate(r.answer ?? "", 80),
              ),
              h("td", { class: "r" }, r.attempts),
              h("td", { class: "r" }, seconds(r.latency_seconds)),
            )
          ),
        ),
      ),
    );
    if (filtered.length > rows.length) {
      wrap.append(
        h(
          "button",
          {
            onclick: () => {
              state.shown += PAGE_SIZE;
              wrap.replaceWith(renderTable());
            },
          },
          `Show ${Math.min(PAGE_SIZE, filtered.length - rows.length)} more`,
        ),
      );
    }
    return wrap;
  }

  function open(index) {
    state.detail = index;
    render(container);
  }

  // -- detail ----------------------------------------------------------------

  async function renderDetail() {
    const summary = filtered[state.detail];
    const position = `${state.detail + 1} / ${filtered.length}`;
    const body = h("div", {}, h("p", { class: "muted" }, "Loading…"));
    container.append(
      h(
        "div",
        { class: "detail-nav" },
        h("button", { onclick: () => back() }, "← Back to list (Esc)"),
        h("button", { onclick: () => step(-1), disabled: state.detail === 0 }, "← Prev"),
        h("span", { class: "position" }, position),
        h(
          "button",
          { onclick: () => step(1), disabled: state.detail === filtered.length - 1 },
          "Next →",
        ),
      ),
      body,
    );

    const params = new URLSearchParams({ model: summary.model, question: summary.question_id });
    try {
      const data = await getJson(`/api/runs/${encodeURIComponent(ctx.runName)}/response?${params}`);
      // The user may have moved on while this loaded.
      if (filtered[state.detail] !== summary) return;
      body.replaceWith(detailBody(data));
    } catch (err) {
      body.replaceChildren(
        h("p", { class: "error-text" }, `Failed to load response: ${err.message}`),
      );
    }
  }

  function detailBody({ summary, record, question, has_image: hasImage }) {
    const parsed = record?.parsed ?? null;
    const imageUrl = `/api/runs/${encodeURIComponent(ctx.runName)}/image?question=${
      encodeURIComponent(summary.question_id)
    }`;

    const media = h("div", { class: "detail-media" });
    if (hasImage) {
      const img = h("img", { src: imageUrl, alt: "question image" });
      img.addEventListener(
        "error",
        () =>
          img.replaceWith(
            h("p", { class: "muted" }, "Image file is missing on disk (regenerated or moved?)."),
          ),
      );
      media.append(img);
    } else {
      media.append(h("p", { class: "muted" }, "No image recorded for this question."));
    }

    const info = h(
      "div",
      { class: "detail-info" },
      h(
        "div",
        { class: "viewer-header" },
        outcomeBadge(summary.outcome),
        h("strong", {}, summary.model),
        h("span", { class: "muted" }, summary.suite || summary.task),
      ),
      h("p", { class: "muted small" }, summary.question_id),
    );

    if (question) info.append(h("p", { class: "question-text" }, question.question_text));
    if (question?.choices) info.append(choiceList(question, summary));
    info.append(answerFacts(summary, question, parsed));

    if (parsed?.reasoning) {
      info.append(
        h("h4", {}, "Model's reasoning"),
        h("p", { class: "reasoning" }, String(parsed.reasoning)),
      );
    }
    if (record?.error || summary.status === "failed") info.append(errorBlock(record, summary));
    if (record?.warnings?.length) {
      info.append(h("h4", {}, "Warnings"), h("ul", {}, record.warnings.map((w) => h("li", {}, w))));
    }
    if (record?.retry_log?.length) {
      info.append(
        h("h4", {}, `Retries (${record.retry_log.length})`),
        h("ul", {}, record.retry_log.map((r) => h("li", {}, r))),
      );
    }
    if (record?.raw_response) {
      info.append(
        h(
          "details",
          { open: summary.status === "failed" || summary.outcome === "invalid" },
          h("summary", {}, "Raw response"),
          h("pre", { class: "raw" }, record.raw_response),
        ),
      );
    }
    if (record) info.append(metaTable(record));
    else {info.append(
        h(
          "p",
          { class: "muted small" },
          "responses.jsonl isn't on disk for this run, so only the scored summary is available.",
        ),
      );}

    return h("div", { class: "detail" }, media, info);
  }

  function choiceList(question, summary) {
    const labels = isLabelChoices(question.choices);
    const key = labels ? boxLabel : norm;
    const given = summary.answer === null ? null : key(summary.answer);
    return h(
      "ul",
      { class: "choice-list detail-choices" },
      question.choices.map((choice, i) => {
        const isExpected = i === question.answer_index;
        const isGiven = given !== null && key(choice) === given;
        return h(
          "li",
          {
            class: [isExpected ? "correct" : "", isGiven && !isExpected ? "wrong" : ""].join(" ")
              .trim(),
          },
          labels ? labelSwatch(choice) : null,
          choice,
          h(
            "span",
            { class: "tags" },
            isExpected ? h("span", { class: "tag" }, "expected") : null,
            isGiven ? h("span", { class: "tag" }, "model's answer") : null,
          ),
        );
      }),
    );
  }

  function answerFacts(summary, question, parsed) {
    const dl = h("dl", { class: "facts" });
    const fact = (label, value) => {
      if (value === null || value === undefined || value === "") return;
      dl.append(h("dt", {}, label), h("dd", {}, String(value)));
    };
    fact("Expected", summary.expected);
    fact("Model's answer", summary.answer);
    if (question && typeof question.true_weed_count === "number") {
      fact(
        "Weed count (true → estimated)",
        `${question.true_weed_count} → ${parsed?.estimated_weed_count ?? "--"}`,
      );
      fact(
        "Weed coverage (true → estimated)",
        `${(question.true_weed_coverage_fraction * 100).toFixed(1)}% → ${
          parsed?.estimated_weed_coverage_percent ?? "--"
        }%`,
      );
    }
    return dl;
  }

  function errorBlock(record, summary) {
    const error = record?.error;
    return h(
      "div",
      { class: "notice error" },
      h(
        "strong",
        {},
        `Error${
          error ? ` [${error.kind}${error.http_status ? ` ${error.http_status}` : ""}]` : ""
        }`,
      ),
      h(
        "p",
        {},
        error?.message ?? summary.error_message ?? "No answer was obtained for this question.",
      ),
      record?.finish_reason
        ? h("p", { class: "muted small" }, `finish_reason: ${record.finish_reason}`)
        : null,
    );
  }

  function metaTable(record) {
    const usage = record.usage ?? {};
    const rows = [
      ["Attempts", record.attempts],
      ["Finish reason", record.finish_reason],
      ["Latency", seconds(record.latency_seconds)],
      ["Cost", record.cost_usd === null ? null : money(record.cost_usd)],
      [
        "Tokens",
        usage.total_tokens
          ? `${usage.total_tokens} (${usage.prompt_tokens} prompt + ${usage.completion_tokens} completion)`
          : null,
      ],
      ["Provider", record.provider],
      ["Answered", when(record.created_at)],
      ["Prompt version", record.prompt_version],
      ["Image", record.image_path],
    ].filter(([, v]) => v !== null && v !== undefined);
    return h(
      "details",
      {},
      h("summary", {}, "Request details"),
      h(
        "dl",
        { class: "answer-panel" },
        rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))]),
      ),
    );
  }

  function step(delta) {
    const next = state.detail + delta;
    if (next < 0 || next >= filtered.length) return;
    state.detail = next;
    render(container);
  }

  function back() {
    const index = state.detail;
    state.detail = -1;
    // Show enough rows that the one just viewed is on screen.
    state.shown = Math.max(state.shown, Math.ceil((index + 1) / PAGE_SIZE) * PAGE_SIZE);
    render(container);
  }

  function onKey(e) {
    if (state.detail < 0) return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "Escape") back();
  }

  return { render, setFilters, onKey };
}
