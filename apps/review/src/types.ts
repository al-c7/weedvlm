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

// -- Evaluation runs (results/<run>/, written by run-evaluation.py; scores
// from notebooks/weedvlm_scored_responses.parquet) ----------------------

export type ResponseStatus = "ok" | "failed";

/**
 * How a response scored, from the scored responses parquet written by
 * notebooks/scoring.ipynb (see src/scores.ts). "unscored" means the
 * response isn't in it, or was re-asked since. Localisation responses are
 * "scored" (on box count and IoU) rather than correct or incorrect.
 */
export type Outcome =
  | "correct"
  | "incorrect"
  | "unsure"
  | "invalid"
  | "scored"
  | "failed"
  | "unscored";

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

/** One scored response (src/scores.ts), with what the metrics need. */
export interface ScoredRow {
  question_id: string;
  model: string;
  task: string;
  suite: string;
  dataset: string;
  benchmark_class: string;
  annotated: boolean;
  /** Open-ended ablations only: the multiple-choice question this re-asks. */
  ablation_of: string | null;
  status: ResponseStatus;
  /** The scored response's created_at, to tell whether it's still the one on disk. */
  created_at: string | null;
  outcome: Outcome;
  expected: string;
  answer: string | null;
  /** The option / density category the answer maps to, null if none. */
  predicted_label: string | null;
  /** Random-guess accuracy (null for open-ended and localisation). */
  chance: number | null;
  /** Density only: estimate minus truth, and category distance. */
  count_error: number | null;
  coverage_error_pp: number | null;
  category_rank_error: number | null;
  /** Localisation only: the score, and the model's boxes as [x1, y1, x2, y2] in image pixels. */
  localisation_score: number | null;
  matched_iou: number | null;
  box_count_error: number | null;
  pred_boxes: number[][] | null;
  cost_usd: number | null;
  total_tokens: number | null;
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
  /** Open-ended ablations only, once scored: the multiple-choice question this re-asks. */
  ablation_of: string | null;
  status: ResponseStatus;
  outcome: Outcome;
  expected: string | null;
  answer: string | null;
  /** Localisation only, once scored. */
  localisation_score: number | null;
  pred_boxes: number[][] | null;
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
