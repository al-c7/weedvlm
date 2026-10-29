/**
 * Local app for working out the weed density rule from hand labels. Two views:
 *
 *   Label  step through sampled images and give each a density category
 *   Fit    rules fitted to those labels, and how well each agrees with them
 *
 * Usage (no "--" needed before the flags — deno task forwards args as-is):
 *   deno task serve [--dataset path/to/weedcoco.json [--dataset more.json]]
 *     [--datasets-dir .datasets] [--labels-out density-labels.json]
 *     [--config config.example.yaml] [--port 8788]
 *
 * Defaults: every <repo root>/.datasets/<name>/weedcoco.json, labels saved to
 * density-labels.json beside this file, and the current rule's thresholds read
 * from <repo root>/config.example.yaml.
 */
import { parseArgs } from "@std/cli/parse-args";
import { basename, dirname, join, resolve } from "@std/path";
import { parse as parseYaml } from "@std/yaml";
import { loadJson, resolveWithin, saveJson, serveFile } from "./src/http.ts";
import { type Box, computeFeatures, FEATURES, type Features } from "./src/features.ts";
import {
  CLASSES,
  DEFAULT_SCORE_RULE,
  type DensityClass,
  fitAll,
  MIN_ROWS_FOR_CV,
  type ScoreRule,
} from "./src/fit.ts";

const scriptDir = dirname(new URL(import.meta.url).pathname);
const staticDir = join(scriptDir, "static");
const repoRoot = join(scriptDir, "..", "..");

const rawArgs = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
const args = parseArgs(rawArgs, {
  string: ["dataset", "datasets-dir", "labels-out", "config", "port"],
  collect: ["dataset"],
});

function fail(message: string): never {
  console.error(message);
  Deno.exit(1);
}

const port = args.port ? Number(args.port) : 8788;

// -- Images ----------------------------------------------------------------

interface ImageRecord {
  /** "<dataset>:<image id>", the same pair the pipeline's question ids start with. */
  key: string;
  dataset: string;
  image_id: number;
  file_name: string;
  width: number;
  height: number;
  /** Weed boxes only. */
  boxes: Box[];
  features: Features;
}

interface DatasetInfo {
  name: string;
  dir: string;
  images: number;
  /** Images with at least one weed box: the ones there is anything to label. */
  with_weeds: number;
}

async function discoverDatasets(dir: string): Promise<string[]> {
  const found: string[] = [];
  try {
    for await (const entry of Deno.readDir(dir)) {
      if (!entry.isDirectory) continue;
      const path = join(dir, entry.name, "weedcoco.json");
      try {
        if ((await Deno.stat(path)).isFile) found.push(path);
      } catch {
        // not a dataset directory
      }
    }
  } catch {
    // no datasets directory at all
  }
  return found.sort();
}

const records = new Map<string, ImageRecord>();
const datasets: DatasetInfo[] = [];

async function loadDataset(path: string): Promise<void> {
  let data;
  try {
    data = JSON.parse(await Deno.readTextFile(path));
  } catch (err) {
    fail(`Failed to read/parse dataset at ${path}: ${(err as Error).message}`);
  }
  for (const field of ["images", "annotations", "categories"]) {
    if (!Array.isArray(data[field])) fail(`${path} is missing required array field "${field}"`);
  }
  const dir = dirname(path);
  const name = basename(dir);
  // WeedCOCO categories are named "<role>: <name>".
  const weedCategories = new Set(
    data.categories.filter((c: { name: string }) => /^weed:/.test(c.name.trim())).map((
      c: { id: number },
    ) => c.id),
  );
  const boxesByImage = new Map<number, Box[]>();
  for (const annotation of data.annotations) {
    if (!weedCategories.has(annotation.category_id)) continue;
    if (!boxesByImage.has(annotation.image_id)) boxesByImage.set(annotation.image_id, []);
    boxesByImage.get(annotation.image_id)!.push(annotation.bbox);
  }
  let withWeeds = 0;
  for (const image of data.images) {
    const boxes = boxesByImage.get(image.id) ?? [];
    if (boxes.length > 0) withWeeds++;
    const key = `${name}:${image.id}`;
    records.set(key, {
      key,
      dataset: name,
      image_id: image.id,
      file_name: image.file_name,
      width: image.width,
      height: image.height,
      boxes,
      features: computeFeatures(boxes, image.width, image.height),
    });
  }
  datasets.push({ name, dir, images: data.images.length, with_weeds: withWeeds });
}

const datasetPaths = ((args.dataset as string[] | undefined) ?? []).map((p) => resolve(p));
if (datasetPaths.length === 0) {
  datasetPaths.push(
    ...await discoverDatasets(resolve(args["datasets-dir"] ?? join(repoRoot, ".datasets"))),
  );
}
if (datasetPaths.length === 0) {
  fail("No datasets found. Pass --dataset <weedcoco.json> or --datasets-dir <dir>.");
}
for (const path of datasetPaths) await loadDataset(path);
console.log(`Datasets:  ${datasets.length} (${records.size} images)`);
for (const d of datasets) console.log(`  ${d.name}: ${d.with_weeds} of ${d.images} with weeds`);

/**
 * WeedCOCO exports disagree on what file_name is relative to: most give
 * it relative to <dataset>/images/, some relative to the dataset folder.
 */
async function serveImage(record: ImageRecord): Promise<Response> {
  const dir = datasets.find((d) => d.name === record.dataset)!.dir;
  for (const base of [join(dir, "images"), dir]) {
    const path = resolveWithin(base, record.file_name);
    if (!path) continue;
    const response = await serveFile(path, "Image not found");
    if (response.ok) return response;
  }
  return new Response("Image not found", { status: 404 });
}

// -- Labels ----------------------------------------------------------------

type LabelValue = DensityClass | "skip";

interface LabelStore {
  updated_at: string;
  /** Keyed by image key. "skip" marks an image set aside as unusable. */
  labels: Record<string, { label: LabelValue; file_name: string; labelled_at: string }>;
}

const labelsPath = resolve(args["labels-out"] ?? join(scriptDir, "density-labels.json"));
const labelStore: LabelStore = await loadJson(labelsPath, () => ({
  updated_at: new Date().toISOString(),
  labels: {},
}));
console.log(`Labels:    ${labelsPath} (${Object.keys(labelStore.labels).length} so far)`);

// Writes are queued behind each other so two quick labels can't interleave on disk.
let pendingSave: Promise<void> = Promise.resolve();
function saveLabels(): Promise<void> {
  labelStore.updated_at = new Date().toISOString();
  pendingSave = pendingSave.catch(() => {}).then(() => saveJson(labelsPath, labelStore));
  return pendingSave;
}

const isCategory = (label: string): label is DensityClass =>
  (CLASSES as readonly string[]).includes(label);

/** Every image given a category (not skipped) that is still in a loaded dataset. */
function labelledRecords(): { record: ImageRecord; label: number }[] {
  const rows = [];
  for (const [key, entry] of Object.entries(labelStore.labels)) {
    const record = records.get(key);
    if (record && isCategory(entry.label)) {
      rows.push({ record, label: CLASSES.indexOf(entry.label) });
    }
  }
  return rows;
}

// -- Sampling --------------------------------------------------------------

function shuffled<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

const COUNT_BIN_EDGES = [2, 3, 5, 9, 17, 33];
const COVERAGE_BIN_EDGES = [0.005, 0.02, 0.05, 0.15, 0.4];
const bin = (value: number, edges: number[]) => edges.filter((edge) => value >= edge).length;

/**
 * Picks up to n images spread evenly over the count/coverage plane
 * rather than in proportion to how common they are: most images are a
 * weed or two, and a rule's thresholds are decided by the scarcer busy
 * ones. Images are grouped into count x coverage cells and dealt out
 * one cell at a time, each cell in turn rotating through its datasets
 * so no single dataset supplies a whole cell.
 */
function sample(candidates: ImageRecord[], n: number): ImageRecord[] {
  const cells = new Map<string, Map<string, ImageRecord[]>>();
  for (const record of candidates) {
    const cell = `${bin(record.features.count, COUNT_BIN_EDGES)}/${
      bin(record.features.coverage, COVERAGE_BIN_EDGES)
    }`;
    if (!cells.has(cell)) cells.set(cell, new Map());
    const byDataset = cells.get(cell)!;
    if (!byDataset.has(record.dataset)) byDataset.set(record.dataset, []);
    byDataset.get(record.dataset)!.push(record);
  }

  const queues = shuffled([...cells.values()]).map((byDataset) => {
    const piles = shuffled([...byDataset.values()].map((pile) => shuffled(pile)));
    const interleaved: ImageRecord[] = [];
    for (let round = 0; piles.some((pile) => round < pile.length); round++) {
      for (const pile of piles) if (round < pile.length) interleaved.push(pile[round]);
    }
    return interleaved;
  });

  const picked: ImageRecord[] = [];
  for (let round = 0; picked.length < n && queues.some((q) => round < q.length); round++) {
    for (const queue of queues) {
      if (round < queue.length && picked.length < n) picked.push(queue[round]);
    }
  }
  return shuffled(picked);
}

// -- Current rule ----------------------------------------------------------

const configPath = resolve(args.config ?? join(repoRoot, "config.example.yaml"));

/** The density thresholds in the generation config, falling back to the pipeline's defaults. */
async function currentRule(): Promise<ScoreRule> {
  let density: Record<string, unknown> = {};
  try {
    const parsed = parseYaml(await Deno.readTextFile(configPath)) as Record<string, unknown> | null;
    density = (parsed?.density as Record<string, unknown>) ?? {};
  } catch {
    // no config: the defaults stand
  }
  const rule = { ...DEFAULT_SCORE_RULE };
  for (const key of Object.keys(rule) as (keyof ScoreRule)[]) {
    if (density[key] !== undefined && density[key] !== null) {
      (rule[key] as unknown) = density[key];
    }
  }
  return rule;
}
console.log(`Config:    ${configPath}`);

// -- HTTP ------------------------------------------------------------------

function csvCell(value: string | number): string {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function handle(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const { pathname } = url;

  if (pathname === "/api/config" && req.method === "GET") {
    return Response.json({
      datasets: datasets.map(({ name, images, with_weeds }) => ({ name, images, with_weeds })),
      features: FEATURES,
      classes: CLASSES,
      labels_path: labelsPath,
      config_path: configPath,
    });
  }

  if (pathname === "/api/labels") {
    if (req.method === "GET") return Response.json(labelStore);
    if (req.method === "POST") {
      const { key, label } = (await req.json()) ?? {};
      const record = typeof key === "string" ? records.get(key) : undefined;
      if (!record) return new Response("Unknown image", { status: 400 });
      if (label === null) {
        delete labelStore.labels[key];
      } else if (label === "skip" || (typeof label === "string" && isCategory(label))) {
        labelStore.labels[key] = {
          label,
          file_name: record.file_name,
          labelled_at: new Date().toISOString(),
        };
      } else {
        return new Response("Invalid label", { status: 400 });
      }
      await saveLabels();
      return Response.json({ ok: true });
    }
  }

  // /api/sample?n=&datasets=a,b&unlabelled=1   or   /api/sample?labelled=1&datasets=a,b
  if (pathname === "/api/sample" && req.method === "GET") {
    const wanted = new Set((url.searchParams.get("datasets") ?? "").split(",").filter(Boolean));
    const inDatasets = (record: ImageRecord) => wanted.size === 0 || wanted.has(record.dataset);
    if (url.searchParams.get("labelled")) {
      // Revisiting: everything already labelled, skipped ones included.
      const revisit = Object.keys(labelStore.labels)
        .map((key) => records.get(key))
        .filter((record): record is ImageRecord => !!record && inDatasets(record));
      return Response.json(revisit);
    }
    const n = Math.max(1, Math.min(Number(url.searchParams.get("n")) || 100, 5000));
    const unlabelledOnly = url.searchParams.get("unlabelled") === "1";
    // A weed-free image is "none" under any rule, so there's nothing to learn from labelling one.
    const candidates = [...records.values()].filter((record) =>
      record.features.count > 0 && inDatasets(record) &&
      !(unlabelledOnly && labelStore.labels[record.key])
    );
    return Response.json(sample(candidates, n));
  }

  if (pathname === "/api/item" && req.method === "GET") {
    const record = records.get(url.searchParams.get("key") ?? "");
    return record ? Response.json(record) : new Response("Unknown image", { status: 404 });
  }

  if (pathname.startsWith("/images/")) {
    const record = records.get(decodeURIComponent(pathname.slice("/images/".length)));
    return record ? serveImage(record) : new Response("Unknown image", { status: 404 });
  }

  // /api/fit?depth=3
  if (pathname === "/api/fit" && req.method === "GET") {
    const depth = Math.max(1, Math.min(Number(url.searchParams.get("depth")) || 3, 5));
    const labelled = labelledRecords();
    const rows = labelled.map(({ record, label }) => ({ ...record.features, label }));
    const rules = rows.length > 0 ? fitAll(rows, await currentRule(), depth) : [];
    return Response.json({
      n: rows.length,
      skipped: Object.values(labelStore.labels).filter((entry) => entry.label === "skip").length,
      class_counts: CLASSES.map((_, i) => rows.filter((row) => row.label === i).length),
      min_rows_for_cv: MIN_ROWS_FOR_CV,
      rules: rules.map(({ predictions: _, ...rule }) => rule),
      rows: labelled.map(({ record, label }, i) => ({
        key: record.key,
        dataset: record.dataset,
        file_name: record.file_name,
        features: record.features,
        label,
        predictions: Object.fromEntries(rules.map((rule) => [rule.id, rule.predictions[i]])),
      })),
    });
  }

  // Every labelled image with its measurements, for analysis elsewhere.
  if (pathname === "/api/labels.csv" && req.method === "GET") {
    const header = ["dataset", "image_id", "file_name", "label", ...FEATURES.map((f) => f.id)];
    const lines = labelledRecords().map(({ record, label }) =>
      [
        record.dataset,
        record.image_id,
        record.file_name,
        CLASSES[label],
        ...FEATURES.map((f) => record.features[f.id]),
      ].map(csvCell).join(",")
    );
    return new Response([header.join(","), ...lines].join("\n") + "\n", {
      headers: {
        "content-type": "text/csv; charset=utf-8",
        "content-disposition": 'attachment; filename="density-labels.csv"',
      },
    });
  }

  // Static UI
  const rel = pathname === "/" ? "index.html" : pathname.slice(1);
  const path = resolveWithin(staticDir, rel);
  return path ? serveFile(path) : new Response("Forbidden", { status: 403 });
}

Deno.serve({ port }, async (req) => {
  try {
    return await handle(req);
  } catch (err) {
    console.error(`${req.method} ${req.url}:`, err);
    return new Response(`Internal error: ${(err as Error).message}`, { status: 500 });
  }
});

console.log(`\nOpen http://localhost:${port} in your browser.`);
