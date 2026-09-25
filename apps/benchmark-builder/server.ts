/**
 * Local server for composing the final benchmark from generated + QA'd questions.
 *
 * Reads the same questions.json / question-flags.json that apps/question-preview reads and
 * writes (question-flags.json is treated read-only here -- flagging still happens in
 * question-preview), plus a benchmark-config.json describing, per task, how many questions to
 * keep per class and how many classes to keep. The GUI edits that config live and can export the
 * resulting selection to a final-benchmark/ directory.
 *
 * Usage (no "--" needed before the flags -- deno task forwards args as-is):
 *   deno task serve [--questions path/to/questions.json [--questions path/to/more.json]]
 *     [--flags path/to/question-flags.json] [--config path/to/benchmark-config.json]
 *     [--out-dir path/to/final-benchmark] [--port 8790]
 */
import { parseArgs } from "@std/cli/parse-args";
import { dirname, extname, join, normalize, resolve } from "@std/path";
import { loadConfig, saveConfig } from "./src/config.ts";
import { compose } from "./src/select.ts";
import type { BenchmarkConfig } from "./src/config.ts";
import type { FlagStore, Question } from "./src/types.ts";

const scriptDir = dirname(new URL(import.meta.url).pathname);
const staticDir = join(scriptDir, "static");
const defaultQuestionsPath = join(scriptDir, "..", "..", "questions", "all-questions.json");
const defaultConfigPath = join(scriptDir, "benchmark-config.json");

const rawArgs = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
const args = parseArgs(rawArgs, {
  string: ["questions", "flags", "config", "out-dir", "port"],
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
        "[--flags <question-flags.json>] [--config <benchmark-config.json>] " +
        "[--out-dir <final-benchmark/>] [--port 8790]",
    );
    Deno.exit(1);
  }
}

const port = args.port ? Number(args.port) : 8790;
const firstQuestionsDir = dirname(resolve(normalizedPaths[0]));
const flagsPath = resolve(args.flags ?? join(firstQuestionsDir, "question-flags.json"));
const configPath = resolve(args.config ?? defaultConfigPath);
const outDir = resolve(args["out-dir"] ?? join(firstQuestionsDir, "final-benchmark"));

let questions: Question[] = [];
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

const imagePathById = new Map(questions.map((q) => [q.id, q.image_path]));

console.log(
  `Questions: ${questions.length} loaded from ${normalizedPaths.length} file(s)` +
    (usedDefault ? " (default: questions/all-questions.json)" : ""),
);
console.log(`Flags:     ${flagsPath} (read-only here -- flag from apps/question-preview)`);
console.log(`Config:    ${configPath}`);
console.log(`Export to: ${outDir}`);

async function loadFlagStore(): Promise<FlagStore> {
  try {
    const text = await Deno.readTextFile(flagsPath);
    return JSON.parse(text);
  } catch {
    return { updated_at: new Date().toISOString(), flags: {} };
  }
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
    // This is a local dev tool actively edited between runs -- never let the browser serve a
    // stale cached copy of the JS/CSS on a plain refresh.
    return new Response(body, {
      headers: { "content-type": contentTypeFor(filePath), "cache-control": "no-store" },
    });
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

function isValidConfig(value: unknown): value is BenchmarkConfig {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  if (typeof v.seed !== "number" || typeof v.exclude_flagged !== "boolean") return false;
  if (typeof v.tasks !== "object" || v.tasks === null) return false;
  return Object.values(v.tasks as Record<string, unknown>).every((t) => {
    if (typeof t !== "object" || t === null) return false;
    const tc = t as Record<string, unknown>;
    return typeof tc.questions_per_class === "number" && typeof tc.num_classes === "number";
  });
}

Deno.serve({ port }, async (req) => {
  const url = new URL(req.url);

  if (url.pathname === "/api/questions" && req.method === "GET") {
    return Response.json(questions);
  }

  if (url.pathname === "/api/flags" && req.method === "GET") {
    return Response.json(await loadFlagStore());
  }

  if (url.pathname === "/api/config" && req.method === "GET") {
    return Response.json(await loadConfig(configPath));
  }

  if (url.pathname === "/api/config" && req.method === "POST") {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      return new Response("Invalid JSON body", { status: 400 });
    }
    if (!isValidConfig(body)) return new Response("Invalid config shape", { status: 400 });
    await saveConfig(configPath, body);
    return Response.json(body);
  }

  if (url.pathname === "/api/compose" && req.method === "POST") {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      return new Response("Invalid JSON body", { status: 400 });
    }
    const config = isValidConfig(body) ? body : await loadConfig(configPath);
    const flags = await loadFlagStore();
    return Response.json(compose(questions, flags, config));
  }

  if (url.pathname === "/api/export" && req.method === "POST") {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      return new Response("Invalid JSON body", { status: 400 });
    }
    const config = isValidConfig(body) ? body : await loadConfig(configPath);
    const flags = await loadFlagStore();
    const result = compose(questions, flags, config);

    await Deno.mkdir(outDir, { recursive: true });
    const written: Record<string, number> = {};
    for (const [task, taskResult] of Object.entries(result)) {
      const ids = new Set(taskResult.classes.flatMap((c) => c.question_ids));
      const selectedQuestions = questions.filter((q) => ids.has(q.id));
      await Deno.writeTextFile(
        join(outDir, `${task}.json`),
        JSON.stringify(selectedQuestions, null, 2),
      );
      written[task] = selectedQuestions.length;
    }
    const summary = Object.fromEntries(
      Object.entries(result).map(([task, taskResult]) => [
        task,
        {
          target_num_classes: taskResult.target_num_classes,
          target_questions_per_class: taskResult.target_questions_per_class,
          total_classes_available: taskResult.total_classes_available,
          total_selected: taskResult.total_selected,
          classes: taskResult.classes.map((c) => ({
            class_key: c.class_key,
            available: c.available,
            selected: c.selected,
          })),
        },
      ]),
    );
    await Deno.writeTextFile(join(outDir, "summary.json"), JSON.stringify(summary, null, 2));

    return Response.json({ out_dir: outDir, written });
  }

  if (url.pathname.startsWith("/images/")) {
    return serveImage(decodeURIComponent(url.pathname.slice("/images/".length)));
  }

  return serveStatic(url.pathname);
});

console.log(`\nOpen http://localhost:${port} in your browser.`);
