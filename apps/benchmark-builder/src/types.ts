/**
 * Question/flag shapes, mirrored from apps/question-preview/src/types.ts (kept in sync by hand --
 * both apps read the same generated JSON). Only the fields this app actually uses are declared;
 * everything else passes through untouched.
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
  image_path: string;
  source: QuestionSource;
  annotated: boolean;
  /** The class this question counts against when composing the benchmark -- e.g. a species
   * name, "same_species"/"different_species", or a density category. Optional since files
   * generated before this field existed won't have it. */
  benchmark_class?: string;
  [key: string]: unknown;
}

export interface MultipleChoiceQuestion extends QuestionBase {
  choices: string[];
  answer_index: number;
}

export interface OpenEndedQuestion extends QuestionBase {
  answer_text: string;
  ablation_of: string;
}

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
