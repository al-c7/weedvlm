"""
Live progress tracking. Counts everything as it happens and reports it
two ways: an in-place terminal display (or, when stderr isn't a terminal,
a plain log line every progress_interval), and progress.json for anything
that wants to poll a run from outside -- e.g. a future web UI.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

RATE_WINDOW_SECONDS = 60.0
REFRESH_SECONDS = 0.5


@dataclass
class ModelProgress:
    total: int  # questions this model has to answer, including already-done ones
    already_done: int = 0  # answered by an earlier session of this run
    ok: int = 0
    failed: int = 0
    retries: int = 0
    in_flight: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    # Set when the model's failure threshold tripped; its remaining
    # questions were left pending rather than attempted.
    stopped_early: bool = False

    @property
    def answered(self) -> int:
        return self.already_done + self.ok

    @property
    def attempted(self) -> int:
        return self.ok + self.failed


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


class Progress:
    def __init__(self, models: dict[str, ModelProgress], *, write_snapshot: Callable[[dict], None]):
        self.models = models
        self._write_snapshot = write_snapshot
        self._started = time.monotonic()
        self._started_at = datetime.now(UTC).isoformat()
        self._completions: deque[float] = deque()
        self._state = "running"
        self._drawn_lines = 0

    # -- updates from workers ----------------------------------------

    def started(self, model: str) -> None:
        self.models[model].in_flight += 1

    def finished(
        self, model: str, *, ok: bool, retries: int, cost_usd: float | None, tokens: int | None
    ) -> None:
        m = self.models[model]
        m.in_flight -= 1
        m.ok += ok
        m.failed += not ok
        m.retries += retries
        m.cost_usd += cost_usd or 0.0
        m.tokens += tokens or 0
        self._completions.append(time.monotonic())

    def abandoned(self, model: str) -> None:
        """A question given up on mid-flight because the run is stopping;
        it stays pending."""
        self.models[model].in_flight -= 1

    def detach_display(self) -> None:
        """Call before printing anything else to the terminal, so the next
        redraw starts below it instead of overwriting it."""
        self._drawn_lines = 0

    # -- reporting ----------------------------------------------------

    def _rate(self) -> float:
        now = time.monotonic()
        while self._completions and now - self._completions[0] > RATE_WINDOW_SECONDS:
            self._completions.popleft()
        if not self._completions:
            return 0.0
        window = min(RATE_WINDOW_SECONDS, now - self._started)
        return len(self._completions) / window if window > 0 else 0.0

    def snapshot(self) -> dict[str, Any]:
        total = sum(m.total for m in self.models.values())
        answered = sum(m.answered for m in self.models.values())
        failed = sum(m.failed for m in self.models.values())
        stopped = sum(
            m.total - m.answered - m.failed for m in self.models.values() if m.stopped_early
        )
        remaining = total - answered - failed - stopped
        rate = self._rate()
        return {
            "state": self._state,
            "started_at": self._started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self._started, 1),
            "total": total,
            "answered": answered,
            "failed": failed,
            "remaining": remaining,
            "in_flight": sum(m.in_flight for m in self.models.values()),
            "retries": sum(m.retries for m in self.models.values()),
            "requests_per_second": round(rate, 3),
            "eta_seconds": round(remaining / rate) if rate > 0 and remaining > 0 else None,
            "cost_usd": round(sum(m.cost_usd for m in self.models.values()), 6),
            "models": {
                name: {
                    "total": m.total,
                    "already_done": m.already_done,
                    "ok": m.ok,
                    "failed": m.failed,
                    "retries": m.retries,
                    "in_flight": m.in_flight,
                    "cost_usd": round(m.cost_usd, 6),
                    "tokens": m.tokens,
                    "stopped_early": m.stopped_early,
                }
                for name, m in self.models.items()
            },
        }

    def summary_line(self, snap: dict[str, Any] | None = None) -> str:
        s = snap or self.snapshot()
        done = s["answered"] + s["failed"]
        pct = 100 * done / s["total"] if s["total"] else 100
        return (
            f"{done}/{s['total']} ({pct:.0f}%) | ok {s['answered']} failed {s['failed']} "
            f"retries {s['retries']} in-flight {s['in_flight']} | {s['requests_per_second']:.1f}/s "
            f"eta {_duration(s['eta_seconds'])} | ${s['cost_usd']:.4f}"
        )

    def _render(self, snap: dict[str, Any]) -> list[str]:
        width = max((len(name) for name in self.models), default=0)
        lines = [f"{self._state.upper():>8}  {_duration(snap['elapsed_seconds'])}  {self.summary_line(snap)}"]
        for name, m in snap["models"].items():
            done = m["already_done"] + m["ok"] + m["failed"]
            note = "  (stopped: too many failures)" if m["stopped_early"] else ""
            lines.append(
                f"  {name:<{width}}  {done:>6}/{m['total']:<6} failed {m['failed']:<4} "
                f"retries {m['retries']:<4} ${m['cost_usd']:.4f}{note}"
            )
        return lines

    def draw(self, *, final: bool = False) -> None:
        """Redraws the terminal display in place."""
        snap = self.snapshot()
        lines = self._render(snap)
        out = sys.stderr
        if self._drawn_lines:
            out.write(f"\033[{self._drawn_lines}F")  # back to the start of the last draw
        for line in lines:
            out.write(f"\033[2K{line}\n")
        out.flush()
        self._drawn_lines = 0 if final else len(lines)

    async def run(self, interval: float) -> None:
        """Reports until cancelled. Call finish() afterwards for the last
        update."""
        interactive = sys.stderr.isatty()
        last_snapshot = 0.0
        last_log_line = 0.0
        while True:
            now = time.monotonic()
            if now - last_snapshot >= interval:
                self._write_snapshot(self.snapshot())
                last_snapshot = now
            if interactive:
                self.draw()
            elif now - last_log_line >= interval:
                print(self.summary_line(), file=sys.stderr, flush=True)
                last_log_line = now
            await asyncio.sleep(REFRESH_SECONDS)

    def finish(self, state: str) -> None:
        self._state = state
        self._write_snapshot(self.snapshot())
        if sys.stderr.isatty():
            self.draw(final=True)
        else:
            print(f"{state}: {self.summary_line()}", file=sys.stderr, flush=True)
