"""
Process-parallel map for the pipeline's image-heavy steps (rendering,
fine-grained's mean-colour pass). Decoding/encoding images is CPU-bound
and holds the GIL for long stretches, so this uses processes rather
than threads.

Results always come back in input order, so anything downstream that
consumes the shared RNG does so in the same order it would have
serially -- a seeded run's output doesn't depend on the worker count.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor


def resolve_workers(workers: int | None) -> int:
    """workers if given, else every available CPU."""
    if workers is not None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        return workers
    return os.process_cpu_count() or 1


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int | None) -> list[R]:
    """[fn(item) for item in items], spread across `workers` processes.
    fn must be a picklable module-level function (or a functools.partial
    of one). Falls back to a plain serial loop for one worker or too few
    items to be worth starting a pool for."""
    workers = min(resolve_workers(workers), len(items))
    if workers <= 1:
        return [fn(item) for item in items]

    # A few chunks per worker keeps per-item IPC overhead down while
    # still balancing load when some images are much slower than others.
    chunksize = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))
