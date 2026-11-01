import { getJson } from "./shared.js";
import * as datasets from "./datasets.js";
import * as questions from "./questions.js";
import * as results from "./results.js";

/**
 * Each view is a module exporting init() (called once, the first time its
 * tab opens), activate() (every time it opens) and onKey(event) (keyboard
 * shortcuts, only while it's the visible tab).
 */
const VIEWS = {
  datasets: {
    module: datasets,
    available: (c) => c.dataset,
    hint: "Start the server with --dataset <weedcoco.json> to review a dataset.",
    summary: (c) =>
      `${c.dataset.images} images · ${c.dataset.annotations} annotations · ${c.dataset.categories} species`,
  },
  questions: {
    module: questions,
    available: (c) => c.questions,
    hint: "No questions loaded. Run generate-all-questions.py, or start the server with " +
      "--questions <questions.json>.",
    summary: (c) => `${c.questions.count} question${c.questions.count === 1 ? "" : "s"}`,
  },
  results: {
    module: results,
    available: (c) => c.results,
    hint: "No results directory found. Start the server with --results <dir>.",
    summary: (c) => c.results.dir,
  },
};

const initialised = new Set();
let config;
let current = null;

async function show(name) {
  const view = VIEWS[name];
  if (!view) return;
  current = name;
  history.replaceState(null, "", `#${name}`);

  for (const tab of document.querySelectorAll(".tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  }
  for (const key of Object.keys(VIEWS)) {
    document.getElementById(`panel-${key}`).hidden = key !== name;
  }

  const unavailable = document.getElementById("unavailable");
  const summary = document.getElementById("summary");
  if (!view.available(config)) {
    unavailable.textContent = view.hint;
    unavailable.hidden = false;
    summary.textContent = "";
    document.getElementById(`panel-${name}`).hidden = true;
    return;
  }
  unavailable.hidden = true;
  summary.textContent = view.summary(config);

  if (!initialised.has(name)) {
    initialised.add(name);
    try {
      await view.module.init(config);
    } catch (err) {
      initialised.delete(name);
      console.error(err);
      unavailable.textContent = `Failed to load ${name}: ${err.message}`;
      unavailable.hidden = false;
      document.getElementById(`panel-${name}`).hidden = true;
      return;
    }
  }
  if (current === name) view.module.activate();
}

function onKeyDown(e) {
  if (!current || !VIEWS[current].available(config)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  VIEWS[current].module.onKey(e);
}

async function main() {
  config = await getJson("/api/config");

  for (const tab of document.querySelectorAll(".tab")) {
    const view = VIEWS[tab.dataset.tab];
    tab.classList.toggle("unavailable", !view.available(config));
    tab.addEventListener("click", () => show(tab.dataset.tab));
  }
  document.addEventListener("keydown", onKeyDown);

  const fromHash = location.hash.slice(1);
  const first = Object.keys(VIEWS).find((name) => VIEWS[name].available(config)) ?? "datasets";
  await show(VIEWS[fromHash] ? fromHash : first);
}

main();
