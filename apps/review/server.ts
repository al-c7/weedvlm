/**
 * Local review server for the weedvlm benchmark. One app, three views:
 *
 *   Datasets   review the boxes in a WeedCOCO dataset, accept/reject per image
 *   Questions  step through generated questions, flag any that look wrong
 *   Results    browse evaluation runs: scores, wrong answers, and errors
 *
 * Every source is optional -- a view whose source isn't available is
 * disabled in the UI, so you can run the server with whatever you have.
 *
 * Usage (no "--" needed before the flags — deno task forwards args as-is):
 *   deno task serve [--dataset path/to/weedcoco.json [--images dir] [--review-out review.json]]
 *     [--questions questions.json [--questions more.json]] [--flags-out question-flags.json]
 *     [--results results/] [--port 8787]
 *
 * Defaults: --questions is <repo root>/questions/all-questions.json and
 * --results is <repo root>/results, each used only if it exists. There is no
 * default dataset.
 */
import { parseArgs } from "@std/cli/parse-args";
import { dirname, join, resolve } from "@std/path";
import { loadJson, resolveWithin, saveJson, serveFile } from "./src/http.ts";
import { listRuns, loadRun, readLogTail, responseKey } from "./src/results.ts";
import type { FlagStore, QuestionBase, ReviewStore, WeedCocoDataset } from "./src/types.ts";

const scriptDir = dirname(new URL(import.meta.url).pathname);
const staticDir = join(scriptDir, "static");
const repoRoot = join(scriptDir, "..", "..");

// Tolerate a leading "--" (npm-run habit) even though deno task doesn't need
// or want one — it would otherwise make parseArgs treat everything as
// positional text instead of flags.
const rawArgs = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
const args = parseArgs(rawArgs, {
  string: ["dataset", "images", "review-out", "questions", "flags-out", "results", "port"],
  collect: ["questions"],
});

function fail(message: string): never {
  console.error(message);
  Deno.exit(1);
}

async function exists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

const port = args.port ? Number(args.port) : 8787;

// -- Datasets ------------------------------------------------------------

async function defaultImagesDir(datasetDir: string): Promise<string> {
  const candidate = join(datasetDir, "images");
  try {
    if ((await Deno.stat(candidate)).isDirectory) return candidate;
  } catch {
    // fall through
  }
  return datasetDir;
}

let dataset: WeedCocoDataset | null = null;
let datasetPath = "";
let imagesDir = "";
let reviewPath = "";

if (args.dataset) {
  datasetPath = resolve(args.dataset);
  imagesDir = resolve(args.images ?? await defaultImagesDir(dirname(datasetPath)));
  reviewPath = resolve(args["review-out"] ?? join(dirname(datasetPath), "review.json"));
  try {
    dataset = JSON.parse(await Deno.readTextFile(datasetPath));
  } catch (err) {
    fail(`Failed to read/parse dataset at ${datasetPath}: ${(err as Error).message}`);
  }
  for (const field of ["images", "annotations", "categories"] as const) {
    if (!Array.isArray(dataset![field])) fail(`Dataset is missing required array field "${field}"`);
  }
  console.log(
    `Dataset:   ${datasetPath} (${dataset!.images.length} images, ${
      dataset!.annotations.length
    } annotations)`,
  );
  console.log(`Review:    ${reviewPath}`);
} else {
  console.log("Dataset:   none (pass --dataset <weedcoco.json> to enable the Datasets view)");
}

// -- Questions -----------------------------------------------------------

const defaultQuestionsPath = join(repoRoot, "questions", "all-questions.json");
let questionPaths: string[] = ((args.questions as string[] | undefined) ?? []).map((p) =>
  resolve(p)
);
if (questionPaths.length === 0 && await exists(defaultQuestionsPath)) {
  questionPaths = [defaultQuestionsPath];
}

const questions: QuestionBase[] = [];
for (const path of questionPaths) {
  try {
    const parsed = JSON.parse(await Deno.readTextFile(path));
    if (!Array.isArray(parsed)) throw new Error("expected a JSON array of questions");
    for (const q of parsed) questions.push(q);
  } catch (err) {
    fail(`Failed to read/parse questions at ${path}: ${(err as Error).message}`);
  }
}
const questionImagePaths = new Map(questions.map((q) => [q.id, q.image_path]));
const flagsPath = resolve(
  args["flags-out"] ??
    join(dirname(questionPaths[0] ?? defaultQuestionsPath), "question-flags.json"),
);

if (questionPaths.length > 0) {
  console.log(`Questions: ${questions.length} loaded from ${questionPaths.length} file(s)`);
  console.log(`Flags:     ${flagsPath}`);
} else {
  console.log(
    "Questions: none (run generate-all-questions.py, or pass --questions <questions.json>)",
  );
}

// -- Results -------------------------------------------------------------

const resultsDir = resolve(args.results ?? join(repoRoot, "results"));
const hasResults = await exists(resultsDir);
console.log(
  hasResults
    ? `Results:   ${resultsDir}`
    : `Results:   none (${resultsDir} doesn't exist; pass --results <dir>)`,
);

// -- HTTP ----------------------------------------------------------------

const notConfigured = (what: string) => new Response(`${what} isn't configured`, { status: 404 });

/** Serves a run's image for a question: the one the model was shown, else the question's own. */
async function serveResultImage(runName: string, questionId: string): Promise<Response> {
  const run = await loadRun(resultsDir, runName);
  if (!run) return new Response("Unknown run", { status: 404 });
  const path = run.imagePaths.get(questionId) ?? questionImagePaths.get(questionId);
  if (!path) return new Response("No image recorded for this question", { status: 404 });
  return serveFile(path, "Image not found");
}

async function handleRuns(url: URL, req: Request): Promise<Response> {
  if (!hasResults) return notConfigured("Results directory");
  const segments = url.pathname.split("/").filter(Boolean).slice(2).map(decodeURIComponent);
  // /api/runs
  if (segments.length === 0) return Response.json(await listRuns(resultsDir));

  const [name, part] = segments;
  const run = await loadRun(resultsDir, name);
  if (!run) return new Response("Unknown run", { status: 404 });
  if (req.method !== "GET") return new Response("Method not allowed", { status: 405 });

  switch (part) {
    // /api/runs/:name
    case undefined:
      return Response.json({
        name,
        run: run.runJson,
        progress: run.progress,
        scores: run.scores,
        scored_at: run.scoredAt,
        scores_stale: run.scoresStale,
        problems: run.problems,
        has_responses: run.hasResponses,
        has_records: run.records.size > 0,
      });
    // /api/runs/:name/responses
    case "responses":
      return Response.json(run.summaries);
    // /api/runs/:name/response?model=&question=
    case "response": {
      const model = url.searchParams.get("model") ?? "";
      const questionId = url.searchParams.get("question") ?? "";
      const summary = run.summaries.find((s) => s.model === model && s.question_id === questionId);
      if (!summary) return new Response("Unknown response", { status: 404 });
      return Response.json({
        summary,
        record: run.records.get(responseKey(model, questionId)) ?? null,
        question: questions.find((q) => q.id === questionId) ?? null,
        has_image: run.imagePaths.has(questionId) || questionImagePaths.has(questionId),
      });
    }
    // /api/runs/:name/log
    case "log": {
      const count = Math.min(Number(url.searchParams.get("lines")) || 300, 5000);
      const tail = await readLogTail(resultsDir, name, count);
      return tail ? Response.json(tail) : new Response("No run.log", { status: 404 });
    }
    // /api/runs/:name/image?question=
    case "image":
      return serveResultImage(name, url.searchParams.get("question") ?? "");
  }
  return new Response("Not found", { status: 404 });
}

async function handle(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const { pathname } = url;

  if (pathname === "/api/config" && req.method === "GET") {
    return Response.json({
      dataset: dataset && {
        path: datasetPath,
        images: dataset.images.length,
        annotations: dataset.annotations.length,
        categories: dataset.categories.length,
      },
      questions: questionPaths.length > 0
        ? { files: questionPaths, count: questions.length }
        : null,
      results: hasResults ? { dir: resultsDir } : null,
    });
  }

  // Datasets
  if (pathname === "/api/dataset" && req.method === "GET") {
    return dataset ? Response.json(dataset) : notConfigured("Dataset");
  }
  if (pathname === "/api/review") {
    if (!dataset) return notConfigured("Dataset");
    const empty = (): ReviewStore => ({
      dataset: datasetPath,
      updated_at: new Date().toISOString(),
      decisions: {},
    });
    if (req.method === "GET") return Response.json(await loadJson(reviewPath, empty));
    if (req.method === "POST") {
      const { imageId, fileName, decision, annotations } = (await req.json()) ?? {};
      if (typeof imageId !== "number" || (decision !== "good" && decision !== "bad")) {
        return new Response("Invalid review payload", { status: 400 });
      }
      const store = await loadJson(reviewPath, empty);
      store.decisions[String(imageId)] = {
        file_name: fileName,
        decision,
        annotations: annotations ?? {},
        reviewed_at: new Date().toISOString(),
      };
      store.updated_at = new Date().toISOString();
      await saveJson(reviewPath, store);
      return Response.json({ ok: true });
    }
  }
  if (pathname.startsWith("/dataset-images/") && dataset) {
    const path = resolveWithin(
      imagesDir,
      decodeURIComponent(pathname.slice("/dataset-images/".length)),
    );
    return path ? serveFile(path, "Image not found") : new Response("Forbidden", { status: 403 });
  }

  // Questions
  if (pathname === "/api/questions" && req.method === "GET") {
    return questionPaths.length > 0 ? Response.json(questions) : notConfigured("Questions");
  }
  if (pathname === "/api/flags") {
    if (questionPaths.length === 0) return notConfigured("Questions");
    const empty = (): FlagStore => ({ updated_at: new Date().toISOString(), flags: {} });
    if (req.method === "GET") return Response.json(await loadJson(flagsPath, empty));
    if (req.method === "POST") {
      const { questionId, flag, note } = (await req.json()) ?? {};
      if (typeof questionId !== "string" || (flag !== "flagged" && flag !== "ok")) {
        return new Response("Invalid flag payload", { status: 400 });
      }
      const store = await loadJson(flagsPath, empty);
      store.flags[questionId] = {
        flag,
        note: note || undefined,
        flagged_at: new Date().toISOString(),
      };
      store.updated_at = new Date().toISOString();
      await saveJson(flagsPath, store);
      return Response.json({ ok: true });
    }
  }
  if (pathname.startsWith("/question-images/")) {
    const path = questionImagePaths.get(
      decodeURIComponent(pathname.slice("/question-images/".length)),
    );
    return path
      ? serveFile(path, "Image not found")
      : new Response("Unknown question id", { status: 404 });
  }

  // Results
  if (pathname === "/api/runs" || pathname.startsWith("/api/runs/")) return handleRuns(url, req);

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
