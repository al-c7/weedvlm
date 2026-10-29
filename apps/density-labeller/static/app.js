import { getJson } from "./shared.js";
import * as label from "./label.js";
import * as fit from "./fit.js";

/**
 * Each view is a module exporting init(config) (called once, the first
 * time its tab opens), activate() (every time it opens) and
 * onKey(event) (keyboard shortcuts, only while it's the visible tab).
 */
const VIEWS = { label, fit };

const initialised = new Set();
let config;
let current = null;

async function show(name) {
  if (!VIEWS[name]) return;
  current = name;
  history.replaceState(null, "", `#${name}`);

  for (const tab of document.querySelectorAll(".tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  }
  for (const key of Object.keys(VIEWS)) {
    document.getElementById(`panel-${key}`).hidden = key !== name;
  }

  if (!initialised.has(name)) {
    initialised.add(name);
    await VIEWS[name].init(config);
  }
  if (current === name) VIEWS[name].activate();
}

function onKeyDown(e) {
  if (!current) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  VIEWS[current].onKey(e);
}

async function main() {
  config = await getJson("/api/config");
  const images = config.datasets.reduce((sum, d) => sum + d.with_weeds, 0);
  document.getElementById("summary").textContent =
    `${config.datasets.length} datasets · ${images} images with weeds`;

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => show(tab.dataset.tab));
  }
  document.addEventListener("keydown", onKeyDown);

  const fromHash = location.hash.slice(1);
  await show(VIEWS[fromHash] ? fromHash : "label");
}

main();
