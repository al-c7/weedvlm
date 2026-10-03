import { isLabelChoices, labelSwatch, shuffled } from "./shared.js";

/** @type {any[]} */
let questions = [];
/** @type {{flags: Record<string, any>}} */
let flagStore = { flags: {} };

let queue = [];
let currentIndex = -1;
let revealed = false;

/** Loads the questions and flags, and wires up the sidebar. */
export async function init() {
  const [questionsRes, flagsRes] = await Promise.all([
    fetch("/api/questions"),
    fetch("/api/flags"),
  ]);
  questions = await questionsRes.json();
  flagStore = await flagsRes.json();

  populateFilters();

  document.getElementById("qp-build-queue").addEventListener("click", buildQueue);
}

export function activate() {}

function populateFilters() {
  const datasets = [...new Set(questions.map((q) => q.source.dataset_name))].sort();
  const types = [...new Set(questions.map((q) => q.question_type))].sort();
  fillSelect(document.getElementById("qp-filter-dataset"), "All datasets", datasets);
  fillSelect(document.getElementById("qp-filter-type"), "All types", types);
}

function fillSelect(select, allLabel, values) {
  select.replaceChildren(
    new Option(allLabel, ""),
    ...values.map((value) => new Option(value, value)),
  );
}

function buildQueue() {
  const dataset = document.getElementById("qp-filter-dataset").value;
  const type = document.getElementById("qp-filter-type").value;
  const flaggedOnly = document.getElementById("qp-filter-flagged-only").checked;
  const shuffle = document.getElementById("qp-shuffle-order").checked;

  const filtered = questions.filter((q) =>
    (!dataset || q.source.dataset_name === dataset) &&
    (!type || q.question_type === type) &&
    (!flaggedOnly || flagStore.flags[q.id]?.flag === "flagged")
  );

  queue = shuffle ? shuffled(filtered) : filtered;
  currentIndex = 0;

  if (queue.length === 0) {
    document.getElementById("qp-viewer").innerHTML =
      '<p class="empty-state">No questions match this filter.</p>';
    updateProgress();
    return;
  }

  mountViewer();
  showCurrent();
}

function mountViewer() {
  const viewer = document.getElementById("qp-viewer");
  const template = document.getElementById("qp-viewer-template");
  viewer.innerHTML = "";
  viewer.appendChild(template.content.cloneNode(true));

  document.getElementById("qp-prev").addEventListener("click", () => navigate(-1));
  document.getElementById("qp-next").addEventListener("click", () => navigate(1));
  document.getElementById("qp-reveal").addEventListener("click", toggleReveal);
  document.getElementById("qp-flag").addEventListener("click", () => setFlag("flagged"));
  document.getElementById("qp-unflag").addEventListener("click", () => setFlag("ok"));
}

function currentQuestion() {
  return queue[currentIndex];
}

function showCurrent() {
  const q = currentQuestion();
  revealed = false;

  document.getElementById("qp-position").textContent = `${currentIndex + 1} / ${queue.length}`;
  document.getElementById("qp-badge").textContent = q.question_type;
  document.getElementById("qp-annotated").textContent = q.annotated ? "boxed" : "unannotated";
  document.getElementById("qp-source").textContent =
    `${q.source.dataset_name} · image ${q.source.image_id} · annotations [${
      q.source.annotation_ids.join(", ")
    }]` + (q.ablation_of ? ` · open-ended ablation of ${q.ablation_of}` : "");

  const existing = flagStore.flags[q.id];
  document.getElementById("qp-save-status").textContent = existing
    ? `previously marked ${existing.flag}${existing.note ? ` — ${existing.note}` : ""}`
    : "";

  document.getElementById("qp-image").src = `/question-images/${encodeURIComponent(q.id)}`;
  document.getElementById("qp-question").textContent = q.question_text;

  renderAnswer(q);
  highlightDecisionButtons(existing?.flag);
  updateProgress();
}

/**
 * Renders either the multiple-choice list (species ID, fine-grained ID,
 * grounded VQA) or the ground-truth answer panel (open-ended, density
 * estimation) -- the latter only ever shown on reveal, since those tasks
 * never give the VLM a fixed set of options to pick between.
 */
function renderAnswer(q) {
  const list = document.getElementById("qp-choice-list");
  const panel = document.getElementById("qp-answer-panel");

  if (Array.isArray(q.choices)) {
    panel.hidden = true;
    list.hidden = false;
    list.innerHTML = "";
    const showSwatches = isLabelChoices(q.choices);
    q.choices.forEach((choice, i) => {
      const li = document.createElement("li");
      if (showSwatches) li.appendChild(labelSwatch(choice));
      li.appendChild(document.createTextNode(choice));
      if (revealed && i === q.answer_index) li.classList.add("correct");
      list.appendChild(li);
    });
    return;
  }

  list.hidden = true;
  panel.hidden = false;
  panel.innerHTML = "";

  if (!revealed) {
    panel.innerHTML = '<p class="muted">Press Space to reveal the ground truth.</p>';
    return;
  }

  for (const [label, value] of answerFields(q)) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    panel.append(dt, dd);
  }
}

function answerFields(q) {
  if (typeof q.answer_text === "string") {
    return [["Answer", q.answer_text]];
  }
  if (typeof q.true_weed_count === "number") {
    return [
      ["Category options shown to VLM", (q.density_choices ?? []).join(", ")],
      ["Weed count", String(q.true_weed_count)],
      ["Weed coverage", `${(q.true_weed_coverage_fraction * 100).toFixed(1)}%`],
      ["Correct density category", q.density_category],
    ];
  }
  return [];
}

function toggleReveal() {
  revealed = !revealed;
  renderAnswer(currentQuestion());
}

function highlightDecisionButtons(flag) {
  document.getElementById("qp-unflag").style.outline = flag === "ok" ? "3px solid black" : "";
  document.getElementById("qp-flag").style.outline = flag === "flagged" ? "3px solid black" : "";
}

async function setFlag(flag) {
  const q = currentQuestion();
  const statusEl = document.getElementById("qp-save-status");

  let note;
  if (flag === "flagged") {
    note = globalThis.prompt("Optional note about the issue:") || undefined;
  }

  statusEl.textContent = "Saving…";
  try {
    const res = await fetch("/api/flags", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ questionId: q.id, flag, note }),
    });
    if (!res.ok) throw new Error(await res.text());
    flagStore.flags[q.id] = { flag, note, flagged_at: new Date().toISOString() };
    statusEl.textContent = `Saved ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    statusEl.textContent = "Save failed — see console";
    console.error(err);
    return;
  }

  highlightDecisionButtons(flag);
  updateProgress();
  navigate(1);
}

function navigate(delta) {
  if (queue.length === 0) return;
  currentIndex = Math.min(Math.max(currentIndex + delta, 0), queue.length - 1);
  showCurrent();
}

function updateProgress() {
  const flaggedCount = queue.filter((q) => flagStore.flags[q.id]?.flag === "flagged").length;
  document.getElementById("qp-progress").textContent = queue.length
    ? `${flaggedCount} flagged / ${queue.length}`
    : "No queue yet";
  const pct = queue.length ? (flaggedCount / queue.length) * 100 : 0;
  document.getElementById("qp-progress-fill").style.width = `${pct}%`;
}

export function onKey(e) {
  if (currentIndex < 0) return;

  if (e.key === " ") {
    e.preventDefault();
    toggleReveal();
  } else if (e.key === "g" || e.key === "G") setFlag("ok");
  else if (e.key === "b" || e.key === "B") setFlag("flagged");
  else if (e.key === "ArrowRight") navigate(1);
  else if (e.key === "ArrowLeft") navigate(-1);
}
