import {
  capitalise,
  formatFeature,
  getJson,
  h,
  imageStage,
  pct,
  setChildren,
  svg,
} from "./shared.js";

let config;
/** The /api/fit response currently on screen. */
let data = null;
/** Key of the charted image whose preview is open. */
let selectedKey = null;

const RULE_NAMES = {
  current: "Current rule",
  refit: "Refitted score rule",
  tree: "Threshold tree",
};

const el = (id) => document.getElementById(id);

export function init(appConfig) {
  config = appConfig;
  for (const [id, initial] of [["fit-x", "count"], ["fit-y", "coverage"]]) {
    for (const feature of config.features) {
      el(id).append(
        h("option", { value: feature.id, selected: feature.id === initial }, feature.label),
      );
    }
    el(id).addEventListener("change", renderChart);
  }
  for (const [id, name] of Object.entries(RULE_NAMES)) {
    el("fit-rule").append(h("option", { value: id, selected: id === "refit" }, name));
  }
  el("fit-rule").addEventListener("change", renderChart);
  el("fit-disagreements").addEventListener("change", renderChart);
  el("fit-depth").addEventListener("change", load);
}

/** Refits every time the tab opens, so it always reflects the labels given so far. */
export function activate() {
  load();
}

export function onKey() {}

async function load() {
  try {
    data = await getJson(`/api/fit?depth=${el("fit-depth").value}`);
  } catch (err) {
    console.error(err);
    setChildren(el("fit-viewer"), h("p", { class: "error-text" }, `Fit failed: ${err.message}`));
    return;
  }
  render();
}

const className = (index) => capitalise(config.classes[index]);

function render() {
  const viewer = el("fit-viewer");
  if (data.n === 0) {
    setChildren(
      viewer,
      h("p", { class: "empty-state" }, "Nothing labelled yet. Label some images first."),
    );
    return;
  }

  // Only low/medium/high can be scarce: a weed-free image is always "none" and is never sampled.
  const scarce = config.classes.filter((name, i) => name !== "none" && data.class_counts[i] < 10);
  setChildren(
    viewer,
    h(
      "p",
      null,
      h("strong", null, `${data.n} labelled images`),
      `: ${data.class_counts.map((count, i) => `${count} ${config.classes[i]}`).join(", ")}`,
      data.skipped ? ` (${data.skipped} skipped images left out)` : "",
      ".",
    ),
    scarce.length > 0 &&
      h(
        "p",
        { class: "notice" },
        `Fewer than 10 images labelled ${scarce.join(" or ")}: the thresholds around `,
        scarce.length > 1 ? "those categories rest" : "that category rest",
        " on very few examples.",
      ),
    h("h2", null, "Agreement with your labels"),
    agreementTable(),
    h(
      "p",
      { class: "muted small" },
      "Held out is the figure to trust for a fitted rule: each image is predicted by a rule ",
      "fitted without it, so it shows how the rule does on images you haven't labelled. ",
      data.n < data.min_rows_for_cv
        ? `It needs at least ${data.min_rows_for_cv} labelled images.`
        : "A rule that scores far better on its own images than held out has been fitted too closely.",
    ),
    data.rules.map(ruleSection),
    h("h2", null, "Your labels across the measurements"),
    h("div", { id: "fit-chart" }),
    h("div", { id: "fit-preview" }),
  );
  renderChart();
}

function agreementTable() {
  const cell = (metrics, pick) => h("td", { class: "numeric" }, metrics ? pick(metrics) : "–");
  return h(
    "table",
    { class: "data-table" },
    h(
      "thead",
      null,
      h(
        "tr",
        null,
        h("th", null, "Rule"),
        h("th", { class: "numeric" }, "Same category"),
        h("th", { class: "numeric" }, "Same or next category"),
        h("th", { class: "numeric" }, "Same category, held out"),
        h("th", { class: "numeric" }, "Same or next, held out"),
      ),
    ),
    h(
      "tbody",
      null,
      data.rules.map((rule) =>
        h(
          "tr",
          null,
          h("td", null, rule.name),
          cell(rule.metrics, (m) => pct(m.exact)),
          cell(rule.metrics, (m) => pct(m.within_one)),
          cell(rule.cross_validated, (m) => pct(m.exact)),
          cell(rule.cross_validated, (m) => pct(m.within_one)),
        )
      ),
    ),
  );
}

function ruleSection(rule) {
  return h(
    "section",
    { class: "rule" },
    h("h3", null, rule.name),
    h("p", { class: "muted" }, rule.summary),
    h(
      "div",
      { class: "rule-body" },
      h("pre", null, rule.text),
      confusionTable(rule.metrics.confusion),
    ),
  );
}

/** Rows are the labels given, columns what the rule says. */
function confusionTable(confusion) {
  // "None" is left out unless it was actually labelled or predicted.
  const used = config.classes.map((_, i) =>
    confusion[i].some((count) => count > 0) || confusion.some((row) => row[i] > 0)
  );
  const shown = config.classes.map((_, i) => i).filter((i) => used[i]);
  return h(
    "table",
    { class: "data-table confusion" },
    h(
      "thead",
      null,
      h(
        "tr",
        null,
        h("th", null, "You said ↓ / rule says →"),
        shown.map((i) => h("th", { class: "numeric" }, className(i))),
      ),
    ),
    h(
      "tbody",
      null,
      shown.map((row) =>
        h(
          "tr",
          null,
          h("th", null, className(row)),
          shown.map((column) =>
            h(
              "td",
              {
                class: `numeric${row === column ? " agree" : ""}${
                  confusion[row][column] === 0 ? " zero" : ""
                }`,
              },
              confusion[row][column],
            )
          ),
        )
      ),
    ),
  );
}

// -- Chart -----------------------------------------------------------------

const WIDTH = 760;
const HEIGHT = 440;
const MARGIN = { left: 64, right: 20, top: 14, bottom: 46 };

/** Maps a feature's values onto [0, 1] along its axis, with the ticks to draw. */
function axisScale(feature, values) {
  if (feature.scale === "linear") {
    return { at: (value) => value, ticks: [0, 0.25, 0.5, 0.75, 1] };
  }
  const positive = values.map((value) => Math.max(value, 1e-5));
  const low = Math.log10(Math.min(...positive)) - 0.08;
  const high = Math.log10(Math.max(...positive)) + 0.08;
  const span = Math.max(high - low, 1e-9);
  const within = (tick) => Math.log10(tick) >= low && Math.log10(tick) <= high;
  const decades = [];
  for (let power = Math.floor(low); power <= Math.ceil(high); power++) decades.push(power);
  let ticks = decades.flatMap((power) => [1, 2, 5].map((step) => step * 10 ** power)).filter(
    within,
  );
  if (ticks.length > 8) ticks = decades.map((power) => 10 ** power).filter(within);
  return { at: (value) => (Math.log10(Math.max(value, 1e-5)) - low) / span, ticks };
}

/**
 * A steady offset in [-1, 1] per image and axis. Whole-number
 * measurements pile up on the same few positions; this fans each pile
 * out, the same way every time the chart is drawn.
 */
function jitter(key, salt) {
  let hash = salt;
  for (let i = 0; i < key.length; i++) hash = (Math.imul(hash, 31) + key.charCodeAt(i)) | 0;
  return ((hash >>> 0) % 2001) / 1000 - 1;
}

const isStepped = (feature) => feature.id === "count" || feature.id === "spread";

function renderChart() {
  const container = el("fit-chart");
  if (!container || !data || data.n === 0) return;

  const xFeature = config.features.find((f) => f.id === el("fit-x").value);
  const yFeature = config.features.find((f) => f.id === el("fit-y").value);
  const ruleId = el("fit-rule").value;
  const rows = data.rows.filter((row) =>
    !el("fit-disagreements").checked || row.predictions[ruleId] !== row.label
  );

  const x = axisScale(xFeature, data.rows.map((row) => row.features[xFeature.id]));
  const y = axisScale(yFeature, data.rows.map((row) => row.features[yFeature.id]));
  const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const px = (value) => MARGIN.left + x.at(value) * plotWidth;
  const py = (value) => MARGIN.top + (1 - y.at(value)) * plotHeight;

  const tooltip = h("div", { class: "chart-tooltip", hidden: true });
  const chart = svg(
    "svg",
    { class: "chart", viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img" },
    x.ticks.map((tick) => [
      svg("line", {
        class: "grid",
        x1: px(tick),
        x2: px(tick),
        y1: MARGIN.top,
        y2: MARGIN.top + plotHeight,
      }),
      svg(
        "text",
        { class: "tick", x: px(tick), y: MARGIN.top + plotHeight + 16, "text-anchor": "middle" },
        formatFeature(xFeature, tick),
      ),
    ]),
    y.ticks.map((tick) => [
      svg("line", {
        class: "grid",
        x1: MARGIN.left,
        x2: MARGIN.left + plotWidth,
        y1: py(tick),
        y2: py(tick),
      }),
      svg(
        "text",
        { class: "tick", x: MARGIN.left - 8, y: py(tick) + 4, "text-anchor": "end" },
        formatFeature(yFeature, tick),
      ),
    ]),
    svg(
      "text",
      {
        class: "axis-title",
        x: MARGIN.left + plotWidth / 2,
        y: HEIGHT - 8,
        "text-anchor": "middle",
      },
      xFeature.label,
    ),
    svg(
      "text",
      {
        class: "axis-title",
        transform: `translate(14 ${MARGIN.top + plotHeight / 2}) rotate(-90)`,
        "text-anchor": "middle",
      },
      yFeature.label,
    ),
    rows.map((row) => {
      const cx = px(row.features[xFeature.id]) + (isStepped(xFeature) ? jitter(row.key, 7) * 5 : 0);
      const cy = py(row.features[yFeature.id]) +
        (isStepped(yFeature) ? jitter(row.key, 13) * 5 : 0);
      return svg(
        "g",
        {
          class: `point${row.key === selectedKey ? " selected" : ""}`,
          onmouseenter: () => showTooltip(tooltip, row, ruleId, cx, cy),
          onmouseleave: () => (tooltip.hidden = true),
          onclick: () => select(row),
        },
        // A hit target larger than the dot itself.
        svg("circle", { class: "hit", cx, cy, r: 10 }),
        svg("circle", { class: `dot class-${row.label}`, cx, cy, r: 4.5 }),
      );
    }),
  );

  setChildren(
    container,
    h(
      "div",
      { class: "legend" },
      h("span", { class: "muted" }, "Your label"),
      config.classes.map((_, i) =>
        data.class_counts[i] > 0 &&
        h("span", null, h("i", { class: `swatch class-${i}` }), className(i))
      ),
      h(
        "span",
        { class: "muted" },
        el("fit-disagreements").checked
          ? `${rows.length} of ${data.n} images, where the ${
            RULE_NAMES[ruleId].toLowerCase()
          } says otherwise`
          : "Click a point to see its image",
      ),
    ),
    h("div", { class: "chart-wrap" }, chart, tooltip),
  );
}

function showTooltip(tooltip, row, ruleId, cx, cy) {
  setChildren(
    tooltip,
    h("strong", null, `You: ${className(row.label)}`),
    h("div", null, `${RULE_NAMES[ruleId]}: ${className(row.predictions[ruleId])}`),
    config.features.map((feature) =>
      h(
        "div",
        { class: "muted" },
        `${feature.label}: ${formatFeature(feature, row.features[feature.id])}`,
      )
    ),
    h("div", { class: "muted" }, row.dataset),
  );
  // Placed in the chart's own coordinates, flipping sides near the right edge.
  tooltip.style.top = `${(cy / HEIGHT) * 100}%`;
  if (cx > WIDTH * 0.65) {
    tooltip.style.left = "";
    tooltip.style.right = `${(1 - cx / WIDTH) * 100 + 2}%`;
  } else {
    tooltip.style.right = "";
    tooltip.style.left = `${(cx / WIDTH) * 100 + 2}%`;
  }
  tooltip.hidden = false;
}

async function select(row) {
  selectedKey = row.key;
  renderChart();
  const record = await getJson(`/api/item?key=${encodeURIComponent(row.key)}`);
  if (selectedKey !== row.key) return;
  setChildren(
    el("fit-preview"),
    h(
      "div",
      { class: "viewer-header" },
      h("span", { class: "badge" }, record.dataset),
      h("span", { class: "muted" }, record.file_name),
    ),
    h(
      "p",
      { class: "measurements" },
      h("span", null, "You ", h("strong", null, className(row.label))),
      data.rules.map((rule) =>
        h("span", null, `${rule.name} `, h("strong", null, className(row.predictions[rule.id])))
      ),
    ),
    h(
      "p",
      { class: "measurements" },
      config.features.map((feature) =>
        h(
          "span",
          { title: feature.description },
          `${feature.label} `,
          h("strong", null, formatFeature(feature, record.features[feature.id])),
        )
      ),
    ),
    imageStage(record, true),
  );
}
