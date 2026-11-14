/**
 * Reads the benchmark's parquet files (docs/DATABASE.md) with hyparquet.
 * Pure functions over bytes -- no file access -- so they can be tested
 * outside Deno.
 */
import { parquetReadObjects, parquetSchema, parquetMetadata } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import type { QuestionBase } from "./types.ts";

export type Row = Record<string, unknown>;

/** BigInt (parquet int64) -> number; everything else as-is. */
function plain(value: unknown): unknown {
  return typeof value === "bigint" ? Number(value) : value;
}

/**
 * The rows of a parquet file, keeping only `columns` that the file has
 * (so older files missing a column still load). Int64s become numbers.
 */
export async function readParquet(bytes: Uint8Array, columns?: string[]): Promise<Row[]> {
  const file = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  let wanted = columns;
  if (columns) {
    const present = new Set(
      parquetSchema(parquetMetadata(file)).children.map((c) => c.element.name),
    );
    wanted = columns.filter((c) => present.has(c));
  }
  const rows = await parquetReadObjects({ file, compressors, columns: wanted });
  for (const row of rows) {
    for (const key of Object.keys(row)) row[key] = plain(row[key]);
  }
  return rows;
}

function json(value: unknown): unknown {
  return typeof value === "string" ? JSON.parse(value) : value ?? null;
}

/**
 * A questions.parquet row back in the shape the Python question models
 * dump to (weedvlm.storage.parquet.question_from_row), which is what the
 * Questions view reads.
 */
export function questionFromRow(row: Row): QuestionBase {
  const question: QuestionBase = {
    id: row.id as string,
    question_type: row.question_type as string,
    question_text: row.question_text as string,
    image_path: row.image_path as string,
    source: {
      dataset_name: row.dataset_name as string,
      image_id: row.source_image_id as number,
      annotation_ids: json(row.annotation_ids) as number[],
    },
    annotated: Boolean(row.annotated),
    benchmark_class: row.benchmark_class as string,
    task: row.task,
  };
  switch (row.question_type) {
    case "species_id_open_ended":
      return { ...question, answer_text: row.expected_answer, ablation_of: row.ablation_of };
    case "density_estimation":
      return {
        ...question,
        density_choices: json(row.choices),
        true_weed_count: row.true_weed_count,
        true_weed_coverage_fraction: row.true_weed_coverage_fraction,
        density_category: row.expected_answer,
      };
    case "localisation":
      return {
        ...question,
        target_species: row.target_species,
        image_width: row.image_width,
        image_height: row.image_height,
        boxes: json(row.boxes),
      };
    default:
      return { ...question, choices: json(row.choices), answer_index: row.answer_index };
  }
}

/** Every question in a questions.parquet, task by task in generation order. */
export async function readQuestions(bytes: Uint8Array): Promise<QuestionBase[]> {
  const rows = await readParquet(bytes);
  rows.sort((a, b) =>
    String(a.task).localeCompare(String(b.task)) || (a.position as number) - (b.position as number)
  );
  return rows.map(questionFromRow);
}
