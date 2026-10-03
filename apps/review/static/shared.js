/** Fetches JSON, throwing with the server's message on a non-2xx response. */
export async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

export function shuffled(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Builds an element: h("div", {class: "x", onclick: fn}, "text", child, ...).
 * Strings become text nodes, so untrusted text is never parsed as HTML.
 */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (key === "style") el.style.cssText = value;
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// Mirrors weedvlm's src/weedvlm/generation/render.py LABEL_PALETTE_HEX
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

export function labelColour(label) {
  return LABEL_PALETTE[(label - 1) % LABEL_PALETTE.length];
}

/** True for choice sets that are just box labels ("1", "2", ...), i.e. grounded weed VQA. */
export function isLabelChoices(choices) {
  return choices.every((c) => /^\d+$/.test(c));
}

/** A coloured square for a numbered box's label. */
export function labelSwatch(label) {
  const swatch = h("span", { class: "label-swatch" });
  swatch.style.background = labelColour(Number(label));
  return swatch;
}

// -- Formatting ------------------------------------------------------------

export const pct = (fraction, digits = 1) =>
  fraction === null || fraction === undefined ? "--" : `${(fraction * 100).toFixed(digits)}%`;

export const num = (value, digits = 2) =>
  value === null || value === undefined ? "--" : Number(value).toFixed(digits);

export function money(usd) {
  if (usd === null || usd === undefined) return "--";
  return usd < 0.01 && usd > 0 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}

export function seconds(value) {
  if (value === null || value === undefined) return "--";
  return value < 100 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`;
}

export function when(iso) {
  return iso ? new Date(iso).toLocaleString() : "--";
}

export function truncate(text, max = 90) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/**
 * Normalises an error message the way the server groups them (see
 * collectProblems in src/results.ts): whitespace collapsed, and the
 * "(retry in 1.3s)" a provider appends dropped.
 */
export function normalizeMessage(message) {
  return message.replace(/\s*\(retry in [\d.]+s\)/g, "").replace(/\s+/g, " ").trim();
}
