/**
 * Minimal shapes for the questions produced by weedvlm's Python question
 * generators (e.g. generate-species-id-questions.py, generate-fine-grained-
 * questions.py, generate-density-questions.py, generate-localisation-
 * questions.py). We only rely on the fields below and pass everything else
 * through untouched.
 */

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
  /** The class this question counts against when composing the final benchmark
   * (apps/benchmark-builder) -- optional since older, un-regenerated files won't have it. */
  benchmark_class?: string;
  [key: string]: unknown;
}

/** Species ID, fine-grained ID, and species localisation all use this shape. */
export interface MultipleChoiceQuestion extends QuestionBase {
  choices: string[];
  answer_index: number;
}

/** A free-response ablation of a MultipleChoiceQuestion (species ID only, for now). */
export interface OpenEndedQuestion extends QuestionBase {
  answer_text: string;
  ablation_of: string;
}

/** Density estimation: the VLM picks a category from density_choices and estimates the rest. */
export interface DensityEstimationQuestion extends QuestionBase {
  density_choices: string[];
  true_weed_count: number;
  true_weed_coverage_fraction: number;
  density_category: string;
}

export type Question = MultipleChoiceQuestion | OpenEndedQuestion | DensityEstimationQuestion;

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
