/** Fetches JSON, throwing with the server's message on a non-2xx response. */
export async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

/** Appends children, flattening nested arrays and dropping false/null/undefined ones. */
function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === undefined || child === null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

/** Replaces an element's contents, accepting the same children as h(). */
export function setChildren(el, ...children) {
  el.replaceChildren();
  return appendChildren(el, children);
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
  return appendChildren(el, children);
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** h() for SVG elements. */
export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  return appendChildren(el, children);
}

/** An image with, optionally, its weed boxes ([x, y, width, height]) outlined over it. */
export function imageStage(record, showBoxes) {
  const stage = h(
    "div",
    { class: "image-stage" },
    h("img", { src: `/images/${encodeURIComponent(record.key)}`, alt: record.file_name }),
  );
  if (showBoxes) {
    stage.append(
      svg(
        "svg",
        {
          class: "box-overlay",
          viewBox: `0 0 ${record.width} ${record.height}`,
          preserveAspectRatio: "none",
        },
        record.boxes.map(([x, y, width, height]) => svg("rect", { x, y, width, height })),
      ),
    );
  }
  return stage;
}

export const capitalise = (text) => text[0].toUpperCase() + text.slice(1);

export const pct = (fraction, digits = 0) => `${(fraction * 100).toFixed(digits)}%`;

/** A measurement as it reads: a whole number, or a percentage to two significant figures. */
export function formatFeature(feature, value) {
  if (feature.kind === "count") return String(value);
  return `${Number((value * 100).toPrecision(2))}%`;
}
