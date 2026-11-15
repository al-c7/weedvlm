import { extname, isAbsolute, join, normalize, relative, resolve } from "@std/path";

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

export function contentTypeFor(path: string): string {
  return CONTENT_TYPES[extname(path).toLowerCase()] ?? "application/octet-stream";
}

/** `base`/`rel`, or null if that would escape `base`. */
export function resolveWithin(base: string, rel: string): string | null {
  const path = resolve(join(base, normalize(rel)));
  const back = relative(base, path);
  return back.startsWith("..") || isAbsolute(back) ? null : path;
}

export async function serveFile(path: string, notFound = "Not found"): Promise<Response> {
  try {
    const body = await Deno.readFile(path);
    return new Response(body, { headers: { "content-type": contentTypeFor(path) } });
  } catch {
    return new Response(notFound, { status: 404 });
  }
}

export async function loadJson<T>(path: string, fallback: () => T): Promise<T> {
  try {
    return JSON.parse(await Deno.readTextFile(path));
  } catch {
    return fallback();
  }
}

export async function saveJson(path: string, data: unknown): Promise<void> {
  const tmp = `${path}.tmp`;
  await Deno.writeTextFile(tmp, JSON.stringify(data, null, 2));
  await Deno.rename(tmp, path);
}
