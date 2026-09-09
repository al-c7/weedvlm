const PALETTE = [
  "#e6194b",
  "#3cb44b",
  "#4363d8",
  "#f58231",
  "#911eb4",
  "#46f0f0",
  "#f032e6",
  "#bcf60c",
  "#fabebe",
  "#008080",
  "#e6beff",
  "#9a6324",
  "#808000",
  "#ffd8b1",
  "#000075",
];

/** @type {{categories: any[], images: any[], annotations: any[]}} */
let dataset;
/** @type {Record<string, any>} */
let reviewStore = { decisions: {} };

let categoriesById = new Map();
let imagesById = new Map();
let annotationsByImage = new Map(); // imageId -> annotation[]
let imageIdsByCategory = new Map(); // categoryId -> Set<imageId>
const colorByCategory = new Map();

let queue = [];
let currentIndex = -1;
/** annotationId -> "good"|"bad" for the image currently on screen */
let currentAnnotationDecisions = {};

async function main() {
  const [datasetRes, reviewRes] = await Promise.all([
    fetch("/api/dataset"),
    fetch("/api/review"),
  ]);
  dataset = await datasetRes.json();
  reviewStore = await reviewRes.json();

  buildIndices();
  renderSummary();
  renderCategoryList();

  document.getElementById("select-all").addEventListener("click", () => setAllCategories(true));
  document.getElementById("select-none").addEventListener("click", () => setAllCategories(false));
  document.getElementById("build-queue").addEventListener("click", buildQueue);
  document.addEventListener("keydown", onKeyDown);
}

function buildIndices() {
  categoriesById = new Map(dataset.categories.map((c) => [c.id, c]));
  imagesById = new Map(dataset.images.map((img) => [img.id, img]));

  annotationsByImage = new Map();
  imageIdsByCategory = new Map();

  for (const ann of dataset.annotations) {
    if (!annotationsByImage.has(ann.image_id)) annotationsByImage.set(ann.image_id, []);
    annotationsByImage.get(ann.image_id).push(ann);

    if (!imageIdsByCategory.has(ann.category_id)) {
      imageIdsByCategory.set(ann.category_id, new Set());
    }
    imageIdsByCategory.get(ann.category_id).add(ann.image_id);
  }

  const sortedCatIds = [...categoriesById.keys()].sort((a, b) => a - b);
  sortedCatIds.forEach((id, i) => colorByCategory.set(id, PALETTE[i % PALETTE.length]));
}

function renderSummary() {
  document.getElementById("dataset-summary").textContent =
    `${dataset.images.length} images · ${dataset.annotations.length} annotations · ${dataset.categories.length} species`;
}

function renderCategoryList() {
  const list = document.getElementById("category-list");
  list.innerHTML = "";
  const sorted = [...categoriesById.values()].sort((a, b) => a.name.localeCompare(b.name));

  for (const cat of sorted) {
    const li = document.createElement("li");
    const imageCount = (imageIdsByCategory.get(cat.id) ?? new Set()).size;

    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.categoryId = String(cat.id);

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colorByCategory.get(cat.id);

    const name = document.createElement("span");
    name.textContent = cat.name;

    label.append(checkbox, swatch, name);

    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${imageCount} img`;

    li.append(label, count);
    list.appendChild(li);
  }
}

function setAllCategories(checked) {
  document.querySelectorAll("#category-list input[type=checkbox]").forEach((cb) => {
    cb.checked = checked;
  });
}

function getSelectedCategoryIds() {
  return [...document.querySelectorAll("#category-list input[type=checkbox]:checked")]
    .map((cb) => Number(cb.dataset.categoryId));
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
  const selected = getSelectedCategoryIds();
  if (selected.length === 0) {
    alert("Select at least one species first.");
    return;
  }
  const n = Math.max(1, Number(document.getElementById("sample-n").value) || 1);
  const unreviewedOnly = document.getElementById("unreviewed-only").checked;

  const picked = new Set();
  for (const catId of selected) {
    const candidates = [...(imageIdsByCategory.get(catId) ?? [])].filter(
      (imgId) => !unreviewedOnly || !reviewStore.decisions[String(imgId)],
    );
    const sample = shuffled(candidates).slice(0, n);
    for (const imgId of sample) picked.add(imgId);
  }

  queue = shuffled([...picked]);
  currentIndex = 0;

  if (queue.length === 0) {
    document.getElementById("viewer").innerHTML =
      '<p class="empty-state">No images match this selection (try unchecking "unreviewed only").</p>';
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
  document.getElementById("btn-accept").addEventListener("click", () => decide("good"));
  document.getElementById("btn-reject").addEventListener("click", () => decide("bad"));

  const img = document.getElementById("viewer-image");
  img.addEventListener("load", drawBoxes);
  globalThis.addEventListener("resize", drawBoxes);
}

function currentImage() {
  return imagesById.get(queue[currentIndex]);
}

function showCurrent() {
  const image = currentImage();
  const existing = reviewStore.decisions[String(image.id)];
  currentAnnotationDecisions = existing ? { ...existing.annotations } : {};

  document.getElementById("viewer-position").textContent = `${currentIndex + 1} / ${queue.length}`;
  document.getElementById("viewer-filename").textContent = image.file_name;
  document.getElementById("save-status").textContent = existing
    ? `previously marked ${existing.decision}`
    : "";

  const img = document.getElementById("viewer-image");
  img.src = `/images/${encodeURIComponent(image.file_name)}`;

  renderAnnotationList(image);
  highlightDecisionButtons(existing?.decision);
  updateProgress();
}

function renderAnnotationList(image) {
  const list = document.getElementById("annotation-list");
  list.innerHTML = "";
  const anns = annotationsByImage.get(image.id) ?? [];

  anns.forEach((ann, i) => {
    const cat = categoriesById.get(ann.category_id);
    const decision = currentAnnotationDecisions[String(ann.id)] ?? "good";

    const li = document.createElement("li");
    li.className = decision === "bad" ? "bad" : "";

    const key = document.createElement("span");
    key.className = "key";
    key.textContent = i < 9 ? String(i + 1) : "";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = decision === "good";
    checkbox.addEventListener("change", () => {
      setAnnotationDecision(ann.id, checkbox.checked ? "good" : "bad");
    });

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colorByCategory.get(ann.category_id);

    const name = document.createElement("span");
    name.textContent = cat ? cat.name : `category ${ann.category_id}`;

    li.append(key, checkbox, swatch, name);
    list.appendChild(li);
  });
}

function setAnnotationDecision(annId, decision) {
  currentAnnotationDecisions[String(annId)] = decision;
  const image = currentImage();
  renderAnnotationList(image);
  drawBoxes();
}

function drawBoxes() {
  const image = currentImage();
  if (!image) return;
  const imgEl = document.getElementById("viewer-image");
  const canvas = document.getElementById("viewer-canvas");
  if (!imgEl.clientWidth) return;

  canvas.width = imgEl.clientWidth;
  canvas.height = imgEl.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = imgEl.clientWidth / image.width;
  const scaleY = imgEl.clientHeight / image.height;

  for (const ann of annotationsByImage.get(image.id) ?? []) {
    const [x, y, w, h] = ann.bbox;
    const decision = currentAnnotationDecisions[String(ann.id)] ?? "good";
    ctx.strokeStyle = colorByCategory.get(ann.category_id);
    ctx.lineWidth = 2;
    ctx.setLineDash(decision === "bad" ? [6, 4] : []);
    ctx.globalAlpha = decision === "bad" ? 0.45 : 1;
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
  }
  ctx.globalAlpha = 1;
}

function highlightDecisionButtons(decision) {
  document.getElementById("btn-accept").style.outline = decision === "good"
    ? "3px solid black"
    : "";
  document.getElementById("btn-reject").style.outline = decision === "bad" ? "3px solid black" : "";
}

async function decide(decision) {
  const image = currentImage();
  const statusEl = document.getElementById("save-status");
  statusEl.textContent = "Saving…";

  const payload = {
    imageId: image.id,
    fileName: image.file_name,
    decision,
    annotations: currentAnnotationDecisions,
  };

  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    reviewStore.decisions[String(image.id)] = {
      file_name: image.file_name,
      decision,
      annotations: currentAnnotationDecisions,
      reviewed_at: new Date().toISOString(),
    };
    statusEl.textContent = `Saved ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    statusEl.textContent = "Save failed — see console";
    console.error(err);
    return;
  }

  highlightDecisionButtons(decision);
  updateProgress();
  navigate(1);
}

function navigate(delta) {
  if (queue.length === 0) return;
  currentIndex = Math.min(Math.max(currentIndex + delta, 0), queue.length - 1);
  showCurrent();
}

function updateProgress() {
  const reviewed = queue.filter((id) => reviewStore.decisions[String(id)]).length;
  document.getElementById("progress").textContent = queue.length
    ? `${reviewed} / ${queue.length} reviewed`
    : "No queue yet";
  const pct = queue.length ? (reviewed / queue.length) * 100 : 0;
  document.getElementById("progress-fill").style.width = `${pct}%`;
}

function onKeyDown(e) {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (currentIndex < 0) return;

  if (e.key === "g" || e.key === "G") decide("good");
  else if (e.key === "b" || e.key === "B") decide("bad");
  else if (e.key === "ArrowRight") navigate(1);
  else if (e.key === "ArrowLeft") navigate(-1);
  else if (/^[1-9]$/.test(e.key)) {
    const anns = annotationsByImage.get(currentImage().id) ?? [];
    const ann = anns[Number(e.key) - 1];
    if (ann) {
      const current = currentAnnotationDecisions[String(ann.id)] ?? "good";
      setAnnotationDecision(ann.id, current === "good" ? "bad" : "good");
    }
  }
}

main();
