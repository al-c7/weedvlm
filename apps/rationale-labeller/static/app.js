const $ = (id) => document.getElementById(id);
const getJson = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (value !== false && value != null) el.setAttribute(key, value);
  }
  el.append(...children.flat().filter((c) => c != null && c !== false));
  return el;
}

let items = [];
let index = 0;
let current = null;
let scores = {};

async function load(i) {
  index = Math.max(0, Math.min(i, items.length - 1));
  current = await getJson(`/api/item/${items[index].response_id}`);
  scores = { ...(current.label ?? {}) };
  render();
}

async function refreshSummary() {
  const summary = await getJson("/api/items");
  items = summary.items;
  const done = items.filter((x) => x.state).length;
  $("progress").textContent = `${done} of ${items.length} done · ${summary.labels_path}`;
  $("fill").style.width = `${(100 * done) / Math.max(items.length, 1)}%`;
}

function render() {
  const item = h("div", {},
    h("p", { class: "muted" }, `${index + 1} / ${items.length} · ${current.task} / ${current.question_type} · response ${current.response_id}`),
    h("pre", {}, current.text),
    h("img", { src: `/image/${current.response_id}`, alt: "" }),
  );
  $("item").replaceChildren(item);

  const form = $("form");
  const fields = current.fields.map((field) =>
    h("div", { class: "field" },
      h("b", {}, field),
      h("div", { class: "scale" },
        Array.from({ length: 11 }, (_, n) =>
          h("button", {
            class: scores[field] === n ? "on" : "",
            onclick: () => { scores[field] = n; render(); },
          }, String(n)))),
    ));
  const notes = h("textarea", { placeholder: "Notes (optional)" });
  notes.value = current.label?.notes ?? "";
  form.replaceChildren(
    h("p", { class: "muted" }, "Scored as the judges are: interpretation, consistency and knowledge from the text, observation and grounding against the image."),
    ...fields,
    notes,
    h("div", { class: "row" },
      h("button", { onclick: () => load(index - 1) }, "←"),
      h("button", { class: "primary", onclick: () => save(notes.value, false) }, "Save & next"),
      h("button", { onclick: () => save(notes.value, true) }, "Skip"),
      h("button", { onclick: () => load(index + 1) }, "→"),
    ),
    current.judges && judgeBox(),
  );
}

function judgeBox() {
  return h("div", { class: "judge" },
    h("b", {}, "Judges"),
    Object.entries(current.judges).map(([run, kinds]) =>
      h("div", {},
        h("div", { class: "muted" }, run),
        Object.values(kinds).map((k) =>
          h("div", {}, Object.entries(k.scores).map(([f, v]) => `${f} ${v}`).join(" · "), k.notes && h("div", { class: "muted" }, k.notes))),
      )),
  );
}

async function save(notes, skipped) {
  if (!skipped) {
    const missing = current.fields.filter((f) => typeof scores[f] !== "number");
    if (missing.length) return alert(`Score ${missing.join(", ")} first (or Skip).`);
  }
  const res = await fetch("/api/label", {
    method: "POST",
    body: JSON.stringify({ response_id: current.response_id, scores, notes, skipped }),
  });
  if (!res.ok) return alert(await res.text());
  await refreshSummary();
  await showAgreement();
  if (skipped || !items[index].state) return load(index + 1);
  // Reload this item so the judges' scores are revealed; "→" moves on.
  await load(index);
}

async function showAgreement() {
  const data = await getJson("/api/agreement");
  const blocks = Object.entries(data).map(([run, fields]) =>
    h("div", {},
      h("h3", {}, `Agreement with ${run}`),
      h("table", {},
        h("tr", {}, ["field", "n", "spearman", "mean |diff|", "within 1", "you", "judge"].map((c) => h("th", {}, c))),
        Object.entries(fields).map(([f, s]) =>
          h("tr", {}, [f, s.n, s.spearman ?? "–", s.mean_abs_diff, s.within_one, s.mean_human, s.mean_judge].map((c, i) => h("td", { style: i === 0 ? "text-align:left" : "" }, String(c))))))));
  $("agreement").replaceChildren(...blocks);
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") load(index + 1);
  if (e.key === "ArrowLeft") load(index - 1);
});

await refreshSummary();
await showAgreement();
const firstOpen = items.findIndex((x) => !x.state);
await load(firstOpen === -1 ? 0 : firstOpen);
