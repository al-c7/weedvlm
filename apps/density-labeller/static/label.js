import { capitalise, formatFeature, getJson, h, imageStage, setChildren } from "./shared.js";

let config;
/** key -> {label, file_name, labelled_at} */
let labels = {};

/** Image records from /api/sample, in the order they'll be shown. */
let queue = [];
let currentIndex = -1;
let saveStatus = "";

const LABEL_KEYS = { "1": "none", "2": "low", "3": "medium", "4": "high" };

const el = (id) => document.getElementById(id);

export async function init(appConfig) {
  config = appConfig;
  labels = (await getJson("/api/labels")).labels;

  const list = el("lb-dataset-list");
  for (const dataset of config.datasets) {
    list.append(
      h(
        "li",
        null,
        h(
          "label",
          null,
          h("input", { type: "checkbox", checked: true, "data-dataset": dataset.name }),
          h("span", null, dataset.name),
        ),
        h("span", { class: "count" }, `${dataset.with_weeds} img`),
      ),
    );
  }

  el("lb-select-all").addEventListener("click", () => setAllDatasets(true));
  el("lb-select-none").addEventListener("click", () => setAllDatasets(false));
  el("lb-build-queue").addEventListener("click", (e) => buildQueue(e, false));
  el("lb-revisit").addEventListener("click", (e) => buildQueue(e, true));
  // Blurred after use so the keyboard shortcuts keep working without a click back on the image.
  for (const id of ["lb-show-boxes", "lb-show-measurements"]) {
    el(id).addEventListener("change", (e) => {
      e.target.blur();
      render();
    });
  }
  updateProgress();
}

export function activate() {}

function setAllDatasets(checked) {
  for (const box of document.querySelectorAll("#lb-dataset-list input")) box.checked = checked;
}

async function buildQueue(event, revisit) {
  event.target.blur();
  const selected = [...document.querySelectorAll("#lb-dataset-list input:checked")]
    .map((box) => box.dataset.dataset);
  if (selected.length === 0) {
    alert("Select at least one dataset first.");
    return;
  }
  const params = new URLSearchParams({ datasets: selected.join(",") });
  if (revisit) {
    params.set("labelled", "1");
  } else {
    params.set("n", String(Math.max(1, Number(el("lb-sample-n").value) || 1)));
    if (el("lb-unlabelled-only").checked) params.set("unlabelled", "1");
  }

  queue = await getJson(`/api/sample?${params}`);
  currentIndex = queue.length ? 0 : -1;
  saveStatus = "";
  if (queue.length === 0) {
    setChildren(
      el("lb-viewer"),
      h(
        "p",
        { class: "empty-state" },
        revisit
          ? "Nothing has been labelled in these datasets yet."
          : 'No images match this selection (try unticking "Only unlabelled images").',
      ),
    );
  } else {
    render();
  }
  updateProgress();
}

function render() {
  if (currentIndex < 0) return;
  const record = queue[currentIndex];
  const existing = labels[record.key]?.label;

  const labelButton = (key, value) =>
    h(
      "button",
      {
        class: `label-button${existing === value ? " selected" : ""}`,
        onclick: () => setLabel(value),
      },
      h("kbd", null, key),
      capitalise(value),
    );

  setChildren(
    el("lb-viewer"),
    h(
      "div",
      { class: "viewer-header" },
      h("span", { class: "position" }, `${currentIndex + 1} / ${queue.length}`),
      h("span", { class: "badge" }, record.dataset),
      h("span", { class: "muted" }, record.file_name),
      h("span", { class: "muted" }, saveStatus),
    ),
    imageStage(record, el("lb-show-boxes").checked),
    el("lb-show-measurements").checked &&
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
    h(
      "div",
      { class: "actions" },
      Object.entries(LABEL_KEYS).map(([key, value]) => labelButton(key, value)),
      h(
        "button",
        {
          class: `label-button skip${existing === "skip" ? " selected" : ""}`,
          onclick: () => setLabel("skip"),
        },
        h("kbd", null, "S"),
        "Skip",
      ),
      h("span", { class: "spacer" }),
      h("button", { onclick: () => navigate(-1), disabled: currentIndex === 0 }, "← Prev"),
      h(
        "button",
        { onclick: () => navigate(1), disabled: currentIndex === queue.length - 1 },
        "Next →",
      ),
    ),
    h(
      "p",
      { class: "muted small" },
      "How dense are the weeds in this image? Skip sets aside an image that can't be judged ",
      "(wrong boxes, unreadable photo); it's left out of the fit.",
      existing &&
        h("button", { class: "link clear", onclick: () => setLabel(null) }, "Clear label"),
    ),
  );
}

async function setLabel(label) {
  const record = queue[currentIndex];
  try {
    const res = await fetch("/api/labels", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ key: record.key, label }),
    });
    if (!res.ok) throw new Error(await res.text());
  } catch (err) {
    console.error(err);
    saveStatus = "Save failed — see console";
    render();
    return;
  }
  if (label === null) delete labels[record.key];
  else labels[record.key] = { label, file_name: record.file_name };
  saveStatus = `Saved ${new Date().toLocaleTimeString()}`;
  updateProgress();
  // Clearing a label stays put; giving one moves on.
  if (label !== null && currentIndex < queue.length - 1) currentIndex++;
  render();
}

function navigate(delta) {
  if (currentIndex < 0) return;
  currentIndex = Math.min(Math.max(currentIndex + delta, 0), queue.length - 1);
  saveStatus = "";
  render();
}

function updateProgress() {
  const done = queue.filter((record) => labels[record.key]).length;
  el("lb-progress").textContent = queue.length
    ? `${done} / ${queue.length} labelled`
    : "No queue yet";
  el("lb-progress-fill").style.width = `${queue.length ? (done / queue.length) * 100 : 0}%`;

  const tally = {};
  for (const { label } of Object.values(labels)) tally[label] = (tally[label] ?? 0) + 1;
  const total = Object.values(labels).length - (tally.skip ?? 0);
  el("lb-totals").textContent = `${total} labelled in all: ` +
    config.classes.map((name) => `${tally[name] ?? 0} ${name}`).join(", ") +
    (tally.skip ? ` (+${tally.skip} skipped)` : "");
}

function toggle(id) {
  el(id).checked = !el(id).checked;
  render();
}

export function onKey(e) {
  if (e.key === " ") {
    // Also stops Space from pressing whichever button was clicked last.
    e.preventDefault();
    toggle("lb-show-boxes");
    return;
  }
  if (e.key === "i" || e.key === "I") return toggle("lb-show-measurements");
  if (currentIndex < 0) return;

  if (LABEL_KEYS[e.key]) setLabel(LABEL_KEYS[e.key]);
  else if (e.key === "s" || e.key === "S") setLabel("skip");
  else if (e.key === "ArrowRight") navigate(1);
  else if (e.key === "ArrowLeft") navigate(-1);
}
