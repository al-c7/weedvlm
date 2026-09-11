/**
 * Local review server for WeedCOCO-format datasets.
 *
 * Serves the reviewer UI, the dataset JSON, the referenced images, and an
 * API for persisting accept/reject decisions to a review file on disk.
 *
 * Usage (no "--" needed before the flags — deno task forwards args as-is):
 *   deno task serve --dataset path/to/weedcoco.json [--images path/to/images] [--out path/to/review.json] [--port 8787]
 */
import { parseArgs } from "@std/cli/parse-args";
import { dirname, extname, join, normalize, resolve } from "@std/path";
import type { ReviewStore, WeedCocoDataset } from "./src/types.ts";

// Tolerate a leading "--" (npm-run habit) even though deno task doesn't need
// or want one — it would otherwise make parseArgs treat everything as
// positional text instead of flags.
const rawArgs = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
const args = parseArgs(rawArgs, {
  string: ["dataset", "images", "out", "port"],
});

if (!args.dataset) {
  console.error(
    "Usage: deno task serve --dataset <weedcoco.json> [--images <dir>] [--out <review.json>] [--port 8787]",
  );
  Deno.exit(1);
}

async function defaultImagesDir(datasetDir: string): Promise<string> {
  const candidate = join(datasetDir, "images");
  try {
    const stat = await Deno.stat(candidate);
    if (stat.isDirectory) return candidate;
  } catch {
    // fall through
  }
  return datasetDir;
}

const datasetPath = resolve(args.dataset);
const imagesDir = resolve(args.images ?? await defaultImagesDir(dirname(datasetPath)));
const outPath = resolve(args.out ?? join(dirname(datasetPath), "review.json"));
const port = args.port ? Number(args.port) : 8787;

const staticDir = join(dirname(new URL(import.meta.url).pathname), "static");

let dataset: WeedCocoDataset;
try {
  dataset = JSON.parse(await Deno.readTextFile(datasetPath));
} catch (err) {
  console.error(`Failed to read/parse dataset at ${datasetPath}: ${(err as Error).message}`);
  Deno.exit(1);
}
for (const field of ["images", "annotations", "categories"] as const) {
  if (!Array.isArray(dataset[field])) {
    console.error(`Dataset is missing required array field "${field}"`);
    Deno.exit(1);
  }
}

console.log(
  `Dataset:  ${datasetPath} (${dataset.images.length} images, ${dataset.annotations.length} annotations)`,
);
console.log(`Images:   ${imagesDir}`);
console.log(`Review:   ${outPath}`);

async function loadReviewStore(): Promise<ReviewStore> {
  try {
    const text = await Deno.readTextFile(outPath);
    return JSON.parse(text);
  } catch {
    return { dataset: datasetPath, updated_at: new Date().toISOString(), decisions: {} };
  }
}

async function saveReviewStore(store: ReviewStore): Promise<void> {
  const tmp = `${outPath}.tmp`;
  await Deno.writeTextFile(tmp, JSON.stringify(store, null, 2));
  await Deno.rename(tmp, outPath);
}

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
  ".webp": "image/webp",
};

function contentTypeFor(path: string): string {
  return CONTENT_TYPES[extname(path).toLowerCase()] ?? "application/octet-stream";
}

async function serveStatic(pathname: string): Promise<Response> {
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const filePath = join(staticDir, normalize(relative));
  if (!filePath.startsWith(staticDir)) return new Response("Forbidden", { status: 403 });
  try {
    const body = await Deno.readFile(filePath);
    return new Response(body, { headers: { "content-type": contentTypeFor(filePath) } });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}

async function serveImage(fileName: string): Promise<Response> {
  const decoded = decodeURIComponent(fileName);
  const filePath = join(imagesDir, normalize(decoded));
  if (!filePath.startsWith(imagesDir)) return new Response("Forbidden", { status: 403 });
  try {
    const body = await Deno.readFile(filePath);
    return new Response(body, { headers: { "content-type": contentTypeFor(filePath) } });
  } catch {
    return new Response("Image not found", { status: 404 });
  }
}

Deno.serve({ port }, async (req) => {
  const url = new URL(req.url);

  if (url.pathname === "/api/dataset" && req.method === "GET") {
    return Response.json(dataset);
  }

  if (url.pathname === "/api/review" && req.method === "GET") {
    return Response.json(await loadReviewStore());
  }

  if (url.pathname === "/api/review" && req.method === "POST") {
    const body = await req.json();
    const { imageId, fileName, decision, annotations } = body ?? {};
    if (typeof imageId !== "number" || (decision !== "good" && decision !== "bad")) {
      return new Response("Invalid review payload", { status: 400 });
    }
    const store = await loadReviewStore();
    store.decisions[String(imageId)] = {
      file_name: fileName,
      decision,
      annotations: annotations ?? {},
      reviewed_at: new Date().toISOString(),
    };
    store.updated_at = new Date().toISOString();
    await saveReviewStore(store);
    return Response.json({ ok: true });
  }

  if (url.pathname.startsWith("/images/")) {
    return serveImage(url.pathname.slice("/images/".length));
  }

  return serveStatic(url.pathname);
});

console.log(`\nOpen http://localhost:${port} in your browser.`);
