"""
Loads a generated question set (the output of generate-all-questions.py
or one of the individual generate-<task>-questions.py scripts) into a
DuckDB database, as a starting point for querying/joining/exporting the
benchmark without re-parsing JSON every time.

Creates two tables:

- `questions` -- one row per question, read straight from the JSON with
  `union_by_name=true` so the different task shapes (choices/
  answer_index for MC, density_choices/true_weed_count/... for density,
  answer_text/ablation_of for open-ended, ...) all land in one table,
  with the columns that don't apply to a given row's question_type left
  NULL. `source` (dataset_name/image_id/annotation_ids) comes through
  as a nested STRUCT rather than being flattened.
- `images` -- one row per distinct image_path referenced by `questions`
  (both source and rendered/box-annotated images), with width/height
  read off the file itself via Pillow, plus the dataset_name/image_id
  and whether it's a rendered (annotated) or source image.

Usage:
    python src/weedvlm/load_questions_to_duckdb.py \
        --questions questions/all-questions.json \
        --db questions/questions.duckdb

This is a starting point, not the final schema -- e.g. it doesn't yet
store image bytes (only paths, so the database isn't portable off this
machine) or split `choices`/`density_choices` into a normalised table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
from PIL import Image as PILImage

CREATE_QUESTIONS_SQL = """
    CREATE OR REPLACE TABLE questions AS
    SELECT * FROM read_json_auto(?, union_by_name=true)
"""

# annotated=true rows come from rendered/box-annotated copies of the
# source image; the (dataset_name, image_id) pair is what ties a row
# back to the original reviewed dataset regardless of which image_path
# it's shown under.
CREATE_IMAGES_SQL = """
    CREATE OR REPLACE TABLE images AS
    SELECT
        image_path,
        first(source.dataset_name) AS dataset_name,
        first(source.image_id) AS image_id,
        first(annotated) AS annotated,
        count(*) AS question_count,
    FROM questions
    GROUP BY image_path
"""


def _add_image_dimensions(con: duckdb.DuckDBPyConnection) -> None:
    """Pillow, not DuckDB, reads image dimensions -- fetch the distinct
    paths, open each once, and write the results back as new columns."""
    paths = [row[0] for row in con.execute("SELECT image_path FROM images").fetchall()]

    dimensions = []
    for path_str in paths:
        path = Path(path_str)
        try:
            with PILImage.open(path) as im:
                width, height = im.size
        except (FileNotFoundError, OSError):
            width = height = None
        dimensions.append((path_str, width, height))

    con.execute("ALTER TABLE images ADD COLUMN width INTEGER")
    con.execute("ALTER TABLE images ADD COLUMN height INTEGER")
    con.executemany(
        "UPDATE images SET width = $2, height = $3 WHERE image_path = $1", dimensions
    )

    missing = sum(1 for _, width, _ in dimensions if width is None)
    if missing:
        print(f"Warning: {missing}/{len(dimensions)} image(s) could not be opened (missing width/height)")


def load_questions_into_duckdb(questions_path: Path, db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(CREATE_QUESTIONS_SQL, [str(questions_path)])
        con.execute(CREATE_IMAGES_SQL)
        _add_image_dimensions(con)

        num_questions = con.execute("SELECT count(*) FROM questions").fetchone()[0]
        num_images = con.execute("SELECT count(*) FROM images").fetchone()[0]
        print(f"Loaded {num_questions} questions and {num_images} images into {db_path}")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to a questions JSON file (e.g. questions/all-questions.json, or a single "
        "task's <task>.json)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the DuckDB database file to create/overwrite",
    )
    args = parser.parse_args()

    load_questions_into_duckdb(args.questions, args.db)


if __name__ == "__main__":
    main()
