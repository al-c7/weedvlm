/**
 * Local preview server for generated multiple-choice questions.
 *
 * Serves a browsing UI over one or more questions.json files (produced by
 * weedvlm's Python question generators, e.g. generate-all-questions.py),
 * the images they reference, and an API for leaving QA flags on individual
 * questions.
 *
 * With no --questions given, defaults to <repo root>/questions/all-questions.json
 * (generate-all-questions.py's combined output covering every reviewed dataset).
 *
 * Usage (no "--" needed before the flags — deno task forwards args as-is):
 *   deno task serve [--questions path/to/questions.json [--questions path/to/more.json]]
 *     [--out path/to/question-flags.json] [--port 8788]
 */
import { parseArgs } from "@std/cli/parse-args";
import { dirname, extname, join, normalize, resolve } from "@std/path";
import type { FlagStore, QuestionBase } from "./src/types.ts";

const scriptDir = dirname(new URL(import.meta.url).pathname);
const staticDir = join(scriptDir, "static");
const defaultQuestionsPath = join(scriptDir, "..", "..", "questions", "all-questions.json");

const rawArgs = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
const args = parseArgs(rawArgs, {
  string: ["questions", "out", "port"],
  collect: ["questions"],
});

const questionsPaths = (args.questions as string | string[] | undefined) ?? [];
let normalizedPaths = Array.isArray(questionsPaths) ? questionsPaths : [questionsPaths];
let usedDefault = false;

if (normalizedPaths.length === 0) {
  try {
    await Deno.stat(defaultQuestionsPath);
    normalizedPaths = [defaultQuestionsPath];
    usedDefault = true;
  } catch {
    console.error(
      `No --questions given, and the default (${defaultQuestionsPath}) doesn't exist yet.\n` +
        "Run `python src/weedvlm/generate-all-questions.py --out-dir questions/` from the repo " +
        "root first, or pass --questions explicitly.\n\n" +
        "Usage: deno task serve [--questions <questions.json> [--questions <more.json>]] " +
        "[--out <question-flags.json>] [--port 8788]",
    );
    Deno.exit(1);
  }
}

const port = args.port ? Number(args.port) : 8788;

let questions: QuestionBase[] = [];
for (const rawPath of normalizedPaths) {
  const path = resolve(rawPath);
  try {
    const parsed = JSON.parse(await Deno.readTextFile(path));
    if (!Array.isArray(parsed)) throw new Error("expected a JSON array of questions");
    questions = questions.concat(parsed);
  } catch (err) {
    console.error(`Failed to read/parse questions at ${path}: ${(err as Error).message}`);
    Deno.exit(1);
  }
}

const outPath = resolve(
  args.out ?? join(dirname(resolve(normalizedPaths[0])), "question-flags.json"),
);
const imagePathById = new Map(questions.map((q) => [q.id, q.image_path]));

console.log(
  `Questions: ${questions.length} loaded from ${normalizedPaths.length} file(s)` +
    (usedDefault ? " (default: questions/all-questions.json)" : ""),
);
console.log(`Flags:     ${outPath}`);

async function loadFlagStore(): Promise<FlagStore> {
  try {
    const text = await Deno.readTextFile(outPath);
    return JSON.parse(text);
  } catch {
    return { updated_at: new Date().toISOString(), flags: {} };
  }
}

async function saveFlagStore(store: FlagStore): Promise<void> {
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

async function serveImage(questionId: string): Promise<Response> {
  const path = imagePathById.get(questionId);
  if (!path) return new Response("Unknown question id", { status: 404 });
  try {
    const body = await Deno.readFile(path);
    return new Response(body, { headers: { "content-type": contentTypeFor(path) } });
  } catch {
    return new Response("Image not found", { status: 404 });
  }
}

Deno.serve({ port }, async (req) => {
  const url = new URL(req.url);

  if (url.pathname === "/api/questions" && req.method === "GET") {
    return Response.json(questions);
  }

  if (url.pathname === "/api/flags" && req.method === "GET") {
    return Response.json(await loadFlagStore());
  }

  if (url.pathname === "/api/flags" && req.method === "POST") {
    const body = await req.json();
    const { questionId, flag, note } = body ?? {};
    if (typeof questionId !== "string" || (flag !== "flagged" && flag !== "ok")) {
      return new Response("Invalid flag payload", { status: 400 });
    }
    const store = await loadFlagStore();
    store.flags[questionId] = {
      flag,
      note: note || undefined,
      flagged_at: new Date().toISOString(),
    };
    store.updated_at = new Date().toISOString();
    await saveFlagStore(store);
    return Response.json({ ok: true });
  }

  if (url.pathname.startsWith("/images/")) {
    return serveImage(decodeURIComponent(url.pathname.slice("/images/".length)));
  }

  return serveStatic(url.pathname);
});

console.log(`\nOpen http://localhost:${port} in your browser.`);
