/** @type {any[]} */
let questions = [];
/** @type {Map<string, any>} */
let questionById = new Map();
/** @type {{flags: Record<string, any>}} */
let flagStore = { flags: {} };
/** @type {any} */
let config = { seed: 42, exclude_flagged: true, tasks: {} };
/** @type {Record<string, any>} */
let composeResult = {};

let currentTask = null;
/** @type {{ task: string, classKey: string, ids: string[], index: number, mode: "gallery" | "detail" } | null} */
let browse = null;

// Mirrors src/weedvlm/pipeline/render.py LABEL_PALETTE_HEX exactly (same order), same as
// apps/question-preview/static/app.js -- keep all three in sync if this changes.
const LABEL_PALETTE = [
  "#e6194b",
  "#4363d8",
  "#f58231",
  "#911eb4",
  "#42d4f4",
  "#f032e6",
  "#ffe119",
  "#000075",
  "#fabed4",
  "#469990",
];

function labelColour(label) {
  return LABEL_PALETTE[(label - 1) % LABEL_PALETTE.length];
}

function isLabelChoices(choices) {
  return Array.isArray(choices) && choices.every((c) => /^\d+$/.test(c));
}

const TASK_LABELS = {
  species_id_multiple_choice: "Species ID — multiple choice",
  species_id_open_ended: "Species ID — open ended",
  fine_grained_id: "Fine-grained ID",
  species_localisation: "Species localisation",
  density_estimation: "Density estimation",
};

function taskLabel(task) {
  return TASK_LABELS[task] ?? task.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

async function main() {
  const [questionsRes, flagsRes, configRes] = await Promise.all([
    fetch("/api/questions"),
    fetch("/api/flags"),
    fetch("/api/config"),
  ]);
  questions = await questionsRes.json();
  questionById = new Map(questions.map((q) => [q.id, q]));
  flagStore = await flagsRes.json();
  config = await configRes.json();

  document.getElementById("summary").textContent = `${questions.length} question${
    questions.length === 1 ? "" : "s"
  } loaded`;
  document.getElementById("cfg-seed").value = config.seed;
  document.getElementById("cfg-exclude-flagged").checked = config.exclude_flagged;

  document.getElementById("cfg-seed").addEventListener("change", (e) => {
    config.seed = Number(e.target.value) || 0;
    recompute();
  });
  document.getElementById("cfg-exclude-flagged").addEventListener("change", (e) => {
    config.exclude_flagged = e.target.checked;
    recompute();
  });
  document.getElementById("btn-save-config").addEventListener("click", saveConfig);
  document.getElementById("btn-export").addEventListener("click", exportBenchmark);

  currentTask = Object.keys(config.tasks)[0] ?? null;
  await recompute();
}

let recomputeTimer = null;
function debouncedRecompute() {
  clearTimeout(recomputeTimer);
  recomputeTimer = setTimeout(recompute, 300);
}

async function recompute() {
  const res = await fetch("/api/compose", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(config),
  });
  composeResult = await res.json();
  renderTaskCards();
  renderMain();
}

function renderTaskCards() {
  const container = document.getElementById("task-cards");
  const template = document.getElementById("task-card-template");
  container.innerHTML = "";

  for (const task of Object.keys(config.tasks)) {
    const node = template.content.cloneNode(true);
    const card = node.querySelector(".task-card");
    card.dataset.task = task;
    if (task === currentTask) card.classList.add("active");

    node.querySelector(".task-name").textContent = taskLabel(task);

    const perClassInput = node.querySelector(".input-per-class");
    perClassInput.value = config.tasks[task].questions_per_class;
    perClassInput.addEventListener("input", (e) => {
      config.tasks[task].questions_per_class = Number(e.target.value) || 0;
      debouncedRecompute();
    });

    const numClassesInput = node.querySelector(".input-num-classes");
    numClassesInput.value = config.tasks[task].num_classes;
    numClassesInput.addEventListener("input", (e) => {
      config.tasks[task].num_classes = Number(e.target.value) || 0;
      debouncedRecompute();
    });

    const result = composeResult[task];
    node.querySelector(".task-stats").textContent = result
      ? `${result.classes.length} of ${result.total_classes_available} classes kept · ` +
        `${result.total_selected} questions selected`
      : "";

    node.querySelector(".btn-browse").addEventListener("click", () => {
      currentTask = task;
      browse = null;
      renderTaskCards();
      renderMain();
    });

    container.appendChild(node);
  }
}

function renderMain() {
  if (browse?.mode === "gallery") {
    renderClassGallery();
    return;
  }
  if (browse?.mode === "detail") {
    renderClassBrowser();
    return;
  }
  renderClassTable();
}

function renderClassTable() {
  const main = document.getElementById("main");
  const result = composeResult[currentTask];

  if (!result || result.classes.length === 0) {
    main.innerHTML = `<p class="empty-state">No ${
      taskLabel(currentTask ?? "").toLowerCase()
    } questions available yet.</p>`;
    return;
  }

  main.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent = taskLabel(currentTask);
  main.appendChild(heading);

  const table = document.createElement("table");
  table.className = "class-table";
  table.innerHTML = `
    <thead>
      <tr><th>Class</th><th>Available</th><th>Selected</th><th>Target / class</th></tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  for (const c of result.classes) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${c.class_key}</td>
      <td>${c.available}</td>
      <td>${c.selected}${c.selected < result.target_questions_per_class ? " ⚠" : ""}</td>
      <td>${result.target_questions_per_class}</td>
    `;
    tr.addEventListener("click", () => {
      browse = {
        task: currentTask,
        classKey: c.class_key,
        ids: c.question_ids,
        index: 0,
        mode: "gallery",
      };
      renderMain();
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  main.appendChild(table);
}

/** A thumbnail grid of every selected image in a class, so a whole class can be sanity-checked
 * at a glance instead of stepping through one image at a time. Click a thumbnail to open it in
 * the single-question viewer. */
function renderClassGallery() {
  const main = document.getElementById("main");
  main.innerHTML = "";

  const backBtn = document.createElement("button");
  backBtn.textContent = "← Back to classes";
  backBtn.addEventListener("click", () => {
    browse = null;
    renderMain();
  });
  main.appendChild(backBtn);

  const heading = document.createElement("h2");
  heading.textContent = `${browse.classKey} (${browse.ids.length})`;
  main.appendChild(heading);

  const grid = document.createElement("div");
  grid.className = "thumb-grid";
  browse.ids.forEach((id, i) => {
    const q = questionById.get(id);
    if (!q) return;

    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = q.question_text;
    img.src = `/images/${encodeURIComponent(id)}`;
    const caption = document.createElement("div");
    caption.className = "thumb-caption";
    caption.textContent = `${q.source.dataset_name} · ${q.source.image_id}`;
    if (flagStore.flags[id]?.flag === "flagged") thumb.classList.add("thumb-flagged");
    thumb.append(img, caption);
    thumb.addEventListener("click", () => {
      browse.mode = "detail";
      browse.index = i;
      renderMain();
    });
    grid.appendChild(thumb);
  });
  main.appendChild(grid);
}

function renderClassBrowser() {
  const main = document.getElementById("main");
  const q = questionById.get(browse.ids[browse.index]);

  main.innerHTML = "";

  const backBtn = document.createElement("button");
  backBtn.textContent = "← Back to gallery";
  backBtn.addEventListener("click", () => {
    browse.mode = "gallery";
    renderMain();
  });
  main.appendChild(backBtn);

  if (!q) {
    main.appendChild(document.createTextNode("Question not found."));
    return;
  }

  const header = document.createElement("div");
  header.className = "viewer-header";
  header.innerHTML = `
    <span>${browse.index + 1} / ${browse.ids.length}</span>
    <span class="badge">${browse.classKey}</span>
    <span class="badge badge-muted">${q.annotated ? "boxed" : "unannotated"}</span>
    <span class="muted">${q.source.dataset_name} · image ${q.source.image_id}</span>
    ${
    flagStore.flags[q.id]?.flag === "flagged"
      ? '<span class="badge badge-flagged">flagged</span>'
      : ""
  }
  `;
  main.appendChild(header);

  const stage = document.createElement("div");
  stage.className = "image-stage";
  const img = document.createElement("img");
  img.alt = "question image";
  img.src = `/images/${encodeURIComponent(q.id)}`;
  stage.appendChild(img);
  main.appendChild(stage);

  const questionText = document.createElement("p");
  questionText.className = "question-text";
  questionText.textContent = q.question_text;
  main.appendChild(questionText);

  main.appendChild(renderAnswer(q));

  const nav = document.createElement("div");
  nav.className = "decision-bar";
  const prevBtn = document.createElement("button");
  prevBtn.textContent = "← Prev";
  prevBtn.disabled = browse.index === 0;
  prevBtn.addEventListener("click", () => {
    browse.index = Math.max(0, browse.index - 1);
    renderMain();
  });
  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Next →";
  nextBtn.disabled = browse.index === browse.ids.length - 1;
  nextBtn.addEventListener("click", () => {
    browse.index = Math.min(browse.ids.length - 1, browse.index + 1);
    renderMain();
  });
  nav.append(prevBtn, nextBtn);
  main.appendChild(nav);
}

function renderAnswer(q) {
  if (Array.isArray(q.choices)) {
    const list = document.createElement("ul");
    list.className = "choice-list";
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
      if (i === q.answer_index) li.classList.add("correct");
      list.appendChild(li);
    });
    return list;
  }

  const dl = document.createElement("dl");
  dl.className = "answer-panel";
  for (const [label, value] of answerFields(q)) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.append(dt, dd);
  }
  return dl;
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

async function saveConfig() {
  const status = document.getElementById("config-status");
  status.textContent = "Saving…";
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(await res.text());
    status.textContent = `Saved ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    status.textContent = "Save failed — see console";
    console.error(err);
  }
}

async function exportBenchmark() {
  if (
    !globalThis.confirm(
      "Write the current selection to final-benchmark/? This overwrites any previous export.",
    )
  ) {
    return;
  }
  const status = document.getElementById("config-status");
  status.textContent = "Exporting…";
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(await res.text());
    const { out_dir, written } = await res.json();
    const counts = Object.entries(written).map(([task, n]) => `${task}: ${n}`).join(", ");
    status.textContent = `Exported to ${out_dir} (${counts})`;
  } catch (err) {
    status.textContent = "Export failed — see console";
    console.error(err);
  }
}

main();
