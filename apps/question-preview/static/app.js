/** @type {any[]} */
let questions = [];
/** @type {{flags: Record<string, any>}} */
let flagStore = { flags: {} };

// Mirrors weedvlm's src/weedvlm/pipeline/render.py LABEL_PALETTE_HEX
// exactly (same order) -- a numbered-box question's choice N gets the
// same colour swatch here as box N has in the rendered image. Keep the
// two in sync if this changes.
const LABEL_PALETTE = [
  "#e6194b", // red
  "#4363d8", // blue
  "#f58231", // orange
  "#911eb4", // purple
  "#42d4f4", // cyan
  "#f032e6", // magenta
  "#ffe119", // yellow
  "#000075", // navy
  "#fabed4", // pink
  "#469990", // teal
];

function labelColour(label) {
  return LABEL_PALETTE[(label - 1) % LABEL_PALETTE.length];
}

/** True for choice sets that are just box labels ("1", "2", ...), i.e. species localisation. */
function isLabelChoices(choices) {
  return choices.every((c) => /^\d+$/.test(c));
}

let queue = [];
let currentIndex = -1;
let revealed = false;

async function main() {
  const [questionsRes, flagsRes] = await Promise.all([
    fetch("/api/questions"),
    fetch("/api/flags"),
  ]);
  questions = await questionsRes.json();
  flagStore = await flagsRes.json();

  renderSummary();
  populateFilters();

  document.getElementById("build-queue").addEventListener("click", buildQueue);
  document.addEventListener("keydown", onKeyDown);
}

function renderSummary() {
  document.getElementById("summary").textContent = `${questions.length} question${
    questions.length === 1 ? "" : "s"
  }`;
}

function populateFilters() {
  const datasets = [...new Set(questions.map((q) => q.source.dataset_name))].sort();
  const types = [...new Set(questions.map((q) => q.question_type))].sort();

  const datasetSelect = document.getElementById("filter-dataset");
  datasetSelect.innerHTML = '<option value="">All datasets</option>' +
    datasets.map((d) => `<option value="${d}">${d}</option>`).join("");

  const typeSelect = document.getElementById("filter-type");
  typeSelect.innerHTML = '<option value="">All types</option>' +
    types.map((t) => `<option value="${t}">${t}</option>`).join("");
}

function shuffled(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function buildQueue() {
  const dataset = document.getElementById("filter-dataset").value;
  const type = document.getElementById("filter-type").value;
  const flaggedOnly = document.getElementById("filter-flagged-only").checked;
  const shuffle = document.getElementById("shuffle-order").checked;

  const filtered = questions.filter((q) =>
    (!dataset || q.source.dataset_name === dataset) &&
    (!type || q.question_type === type) &&
    (!flaggedOnly || flagStore.flags[q.id]?.flag === "flagged")
  );

  queue = shuffle ? shuffled(filtered) : filtered;
  currentIndex = 0;

  if (queue.length === 0) {
    document.getElementById("viewer").innerHTML =
      '<p class="empty-state">No questions match this filter.</p>';
    updateProgress();
    return;
  }

  mountViewer();
  showCurrent();
}

function mountViewer() {
  const viewer = document.getElementById("viewer");
  const template = document.getElementById("viewer-template");
  viewer.innerHTML = "";
  viewer.appendChild(template.content.cloneNode(true));

  document.getElementById("btn-prev").addEventListener("click", () => navigate(-1));
  document.getElementById("btn-next").addEventListener("click", () => navigate(1));
  document.getElementById("btn-reveal").addEventListener("click", toggleReveal);
  document.getElementById("btn-flag").addEventListener("click", () => setFlag("flagged"));
  document.getElementById("btn-unflag").addEventListener("click", () => setFlag("ok"));
}

function currentQuestion() {
  return queue[currentIndex];
}

function showCurrent() {
  const q = currentQuestion();
  revealed = false;

  document.getElementById("viewer-position").textContent = `${currentIndex + 1} / ${queue.length}`;
  document.getElementById("viewer-badge").textContent = q.question_type;
  document.getElementById("viewer-annotated").textContent = q.annotated ? "boxed" : "unannotated";
  document.getElementById("viewer-source").textContent =
    `${q.source.dataset_name} · image ${q.source.image_id} · annotations [${
      q.source.annotation_ids.join(", ")
    }]` + (q.ablation_of ? ` · open-ended ablation of ${q.ablation_of}` : "");

  const existing = flagStore.flags[q.id];
  document.getElementById("save-status").textContent = existing
    ? `previously marked ${existing.flag}${existing.note ? ` — ${existing.note}` : ""}`
    : "";

  document.getElementById("viewer-image").src = `/images/${encodeURIComponent(q.id)}`;
  document.getElementById("viewer-question").textContent = q.question_text;

  renderAnswer(q);
  highlightDecisionButtons(existing?.flag);
  updateProgress();
}

/**
 * Renders either the multiple-choice list (species ID, fine-grained ID,
 * localisation) or the ground-truth answer panel (open-ended, density
 * estimation) -- the latter only ever shown on reveal, since those tasks
 * never give the VLM a fixed set of options to pick between.
 */
function renderAnswer(q) {
  const list = document.getElementById("choice-list");
  const panel = document.getElementById("answer-panel");

  if (Array.isArray(q.choices)) {
    panel.hidden = true;
    list.hidden = false;
    list.innerHTML = "";
    const showSwatches = isLabelChoices(q.choices);
    q.choices.forEach((choice, i) => {
      const li = document.createElement("li");
      if (showSwatches) {
        const swatch = document.createElement("span");
        swatch.className = "label-swatch";
        swatch.style.background = labelColour(Number(choice));
        li.appendChild(swatch);
      }
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
  document.getElementById("btn-unflag").style.outline = flag === "ok" ? "3px solid black" : "";
  document.getElementById("btn-flag").style.outline = flag === "flagged" ? "3px solid black" : "";
}

async function setFlag(flag) {
  const q = currentQuestion();
  const statusEl = document.getElementById("save-status");

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
  document.getElementById("progress").textContent = queue.length
    ? `${flaggedCount} flagged / ${queue.length}`
    : "No queue yet";
  const pct = queue.length ? (flaggedCount / queue.length) * 100 : 0;
  document.getElementById("progress-fill").style.width = `${pct}%`;
}

function onKeyDown(e) {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (currentIndex < 0) return;

  if (e.key === " ") {
    e.preventDefault();
    toggleReveal();
  } else if (e.key === "g" || e.key === "G") setFlag("ok");
  else if (e.key === "b" || e.key === "B") setFlag("flagged");
  else if (e.key === "ArrowRight") navigate(1);
  else if (e.key === "ArrowLeft") navigate(-1);
}

main();
