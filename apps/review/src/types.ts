/**
 * Minimal WeedCOCO shapes. WeedCOCO is a COCO-derived format used by Weed-AI
 * datasets: https://weed-ai.sydney.edu.au/. We only rely on the fields below
 * and pass everything else through untouched (`[key: string]: unknown`)
 * rather than depending on a COCO library.
 */

export interface WeedCocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
  [key: string]: unknown;
}

export interface WeedCocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  /** [x, y, width, height] in pixels, top-left origin — standard COCO bbox. */
  bbox: [number, number, number, number];
  [key: string]: unknown;
}

export interface WeedCocoCategory {
  id: number;
  name: string;
  supercategory?: string;
  [key: string]: unknown;
}

export interface WeedCocoDataset {
  info?: unknown;
  images: WeedCocoImage[];
  annotations: WeedCocoAnnotation[];
  categories: WeedCocoCategory[];
  [key: string]: unknown;
}

export type Decision = "good" | "bad";

export interface ImageReview {
  file_name: string;
  decision: Decision;
  /** annotation id -> decision; absent entries default to "good" */
  annotations: Record<string, Decision>;
  reviewed_at: string;
}

export interface ReviewStore {
  dataset: string;
  updated_at: string;
  decisions: Record<string, ImageReview>;
}

// -- Generated questions (weedvlm's Python question generators) ---------
//
// We only rely on the fields below and pass everything else through
// untouched.

export interface QuestionSource {
  dataset_name: string;
  image_id: number;
  annotation_ids: number[];
  [key: string]: unknown;
}

export interface QuestionBase {
  id: string;
  question_type: string;
  question_text: string;
  /** Absolute path on disk to the (possibly box-annotated) question image. */
  image_path: string;
  source: QuestionSource;
  /** Whether image_path carries ground-truth boxes, or is the plain source image. */
  annotated: boolean;
  benchmark_class?: string;
  [key: string]: unknown;
}

export type Flag = "flagged" | "ok";

export interface QuestionFlag {
  flag: Flag;
  note?: string;
  flagged_at: string;
}

export interface FlagStore {
  updated_at: string;
  flags: Record<string, QuestionFlag>;
}

// -- Evaluation runs (results/<run>/, written by run-evaluation.py and
// score-evaluation.py) --------------------------------------------------

export type ResponseStatus = "ok" | "failed";

/**
 * How a response scored. Everything but "unscored" comes from
 * score-evaluation.py; "unscored" means the run hasn't been scored, or
 * the response changed since it was.
 */
export type Outcome = "correct" | "incorrect" | "unsure" | "invalid" | "failed" | "unscored";

export interface ResponseError {
  /** transient / request / fatal, or "parse" for a reply with no usable JSON answer. */
  kind: string;
  message: string;
  http_status?: number | null;
}

/** One line of responses.jsonl. */
export interface ResponseRecord {
  question_id: string;
  model: string;
  task: string;
  question_type: string;
  annotated: boolean;
  benchmark_class: string;
  dataset_name: string;
  source_image_id: number;
  image_path: string;
  image_sha256?: string | null;
  prompt_version: number;
  status: ResponseStatus;
  parsed?: Record<string, unknown> | null;
  raw_response?: string | null;
  warnings: string[];
  error?: ResponseError | null;
  attempts: number;
  retry_log: string[];
  finish_reason?: string | null;
  latency_seconds?: number | null;
  usage?: Record<string, unknown> | null;
  cost_usd?: number | null;
  provider?: string | null;
  created_at: string;
}

/** A row of scored-responses.csv, with the columns the app uses parsed. */
export interface ScoredRow {
  question_id: string;
  model: string;
  task: string;
  suite: string;
  dataset: string;
  benchmark_class: string;
  annotated: boolean;
  status: ResponseStatus;
  outcome: Outcome;
  expected: string;
  answer: string | null;
  cost_usd: number | null;
  latency_seconds: number | null;
  attempts: number | null;
}

/** What the responses table shows for one (model, question) pair. */
export interface ResponseSummary {
  model: string;
  question_id: string;
  task: string;
  /** Empty when the run hasn't been scored. */
  suite: string;
  dataset: string;
  benchmark_class: string;
  annotated: boolean;
  status: ResponseStatus;
  outcome: Outcome;
  expected: string | null;
  answer: string | null;
  error_kind: string | null;
  error_message: string | null;
  attempts: number;
  cost_usd: number | null;
  latency_seconds: number | null;
}

/** A row of the runs list. */
export interface RunSummary {
  name: string;
  models: string[];
  /** progress.json's own summary of the latest session, when present. */
  state: string | null;
  updated_at: string | null;
  /** Responses on disk with a usable answer / with none (see the run's errors). */
  answered: number;
  failed: number;
  cost_usd: number | null;
  scored_at: string | null;
  /** Responses were written after the run was last scored. */
  scores_stale: boolean;
  has_responses: boolean;
}
