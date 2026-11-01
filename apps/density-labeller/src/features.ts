/**
 * The per-image measurements a density heuristic can be built from, all
 * computed from an image's weed boxes alone. Count and coverage are the
 * two the pipeline's current rule uses, and are computed the same way as
 * src/weedvlm/generation/density.py so a rule fitted here carries over.
 */

/** [x, y, width, height] in image pixels, as in WeedCOCO. */
export type Box = [number, number, number, number];

export interface Features {
  /** Number of weed boxes. */
  count: number;
  /** Fraction of the image inside at least one weed box. */
  coverage: number;
  /** The median weed box's area, as a fraction of the image. */
  typical_size: number;
  /** Fraction of the cells of a SPREAD_GRID x SPREAD_GRID grid that a weed box reaches into. */
  spread: number;
}

export type FeatureId = keyof Features;

export const SPREAD_GRID = 4;

export interface FeatureInfo {
  id: FeatureId;
  label: string;
  /** How the value is spread out on a chart axis. */
  scale: "log" | "linear";
  /** "count" is a whole number, "fraction" is shown as a percentage. */
  kind: "count" | "fraction";
  description: string;
}

export const FEATURES: FeatureInfo[] = [
  {
    id: "count",
    label: "Weed count",
    scale: "log",
    kind: "count",
    description: "Number of weed boxes in the image.",
  },
  {
    id: "coverage",
    label: "Weed coverage",
    scale: "log",
    kind: "fraction",
    description: "Share of the image inside at least one weed box.",
  },
  {
    id: "typical_size",
    label: "Typical weed size",
    scale: "log",
    kind: "fraction",
    description: "The median weed box's area as a share of the image: how close the camera was.",
  },
  {
    id: "spread",
    label: "Spatial spread",
    scale: "linear",
    kind: "fraction",
    description:
      `Share of a ${SPREAD_GRID}x${SPREAD_GRID} grid's cells that a weed box reaches into: ` +
      "weeds throughout the frame versus in one clump.",
  },
];

/**
 * Exact area covered by the union of the boxes. Between two neighbouring
 * box edges on the x axis the set of boxes overhead doesn't change, so
 * each such strip only needs its boxes' y ranges merged.
 */
export function unionArea(boxes: Box[]): number {
  const xs = [...new Set(boxes.flatMap(([x, , w]) => [x, x + w]))].sort((a, b) => a - b);
  let total = 0;
  for (let i = 0; i < xs.length - 1; i++) {
    const x0 = xs[i];
    const x1 = xs[i + 1];
    const spans = boxes
      .filter(([x, , w, h]) => x <= x0 && x + w >= x1 && h > 0)
      .map(([, y, , h]) => [y, y + h])
      .sort((a, b) => a[0] - b[0]);
    let covered = 0;
    let end = -Infinity;
    for (const [y0, y1] of spans) {
      if (y1 <= end) continue;
      covered += y1 - Math.max(y0, end);
      end = y1;
    }
    total += covered * (x1 - x0);
  }
  return total;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Fraction of the grid's cells that at least one box overlaps. */
export function spreadFraction(boxes: Box[], width: number, height: number): number {
  const cellW = width / SPREAD_GRID;
  const cellH = height / SPREAD_GRID;
  const hit = new Set<number>();
  for (const [x, y, w, h] of boxes) {
    if (w <= 0 || h <= 0) continue;
    // The last cell a box reaches is the one holding the point just inside its far edge.
    const lastCol = Math.ceil((x + w) / cellW) - 1;
    const lastRow = Math.ceil((y + h) / cellH) - 1;
    const clamp = (cell: number) => Math.min(SPREAD_GRID - 1, Math.max(0, cell));
    for (let row = clamp(Math.floor(y / cellH)); row <= clamp(lastRow); row++) {
      for (let col = clamp(Math.floor(x / cellW)); col <= clamp(lastCol); col++) {
        hit.add(row * SPREAD_GRID + col);
      }
    }
  }
  return hit.size / (SPREAD_GRID * SPREAD_GRID);
}

export function computeFeatures(boxes: Box[], width: number, height: number): Features {
  const area = width * height;
  if (area === 0 || boxes.length === 0) {
    return { count: boxes.length, coverage: 0, typical_size: 0, spread: 0 };
  }
  return {
    count: boxes.length,
    coverage: unionArea(boxes) / area,
    typical_size: median(boxes.map(([, , w, h]) => w * h)) / area,
    spread: spreadFraction(boxes, width, height),
  };
}
