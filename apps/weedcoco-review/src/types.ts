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
