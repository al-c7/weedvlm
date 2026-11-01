"""
Where a benchmark and its results are kept, shared by the generation and
evaluation configs (the `storage` section of each):

    json    the original layout -- one JSON file of questions per task in
            the questions directory, and a directory of JSON files per run
    sqlite  one SQLite database holding the questions, the images they use,
            and every run's responses and scores (see docs/DATABASE.md)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

StorageFormat = Literal["json", "sqlite"]

DEFAULT_DATABASE_NAME = "benchmark.db"


class StorageConfig(BaseModel):
    format: StorageFormat = "json"
    # sqlite only. Default: benchmark.db inside the questions directory
    # (generation's out_dir, evaluation's questions_dir).
    database: Path | None = None
    # sqlite, generation only: also store each used image's bytes, so the
    # database is self-contained. Without it only the image's path and
    # hash are stored, and the file must stay where it is.
    store_images: bool = True

    def resolved_database(self, questions_dir: Path) -> Path:
        return self.database or questions_dir / DEFAULT_DATABASE_NAME
