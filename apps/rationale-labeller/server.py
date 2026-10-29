"""
Local app for hand-scoring the reasoning judges' sample, so the judges can be
checked against a human. Each response from one or more judge runs is shown
with exactly what the visual judge saw (its user text and image, model name
hidden), and you give the same 0-10 scores the judges give. Scores are saved to
a parquet file as each one is given; once an item is saved the judges' scores
for it are revealed, and /api/agreement summarises how each judge run compares.

Usage (from the repo root):
    uv run apps/rationale-labeller/server.py --judgements reasoning-results/pilot3 \
        [--judgements reasoning-results/other] [--config reasoning.example.yaml] \
        [--labels-out apps/rationale-labeller/rationale-labels.parquet] \
        [--limit 60] [--seed 0] [--labeller alex] [--port 8789]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from weedvlm.reasoning.config import load_reasoning_config, resolve_reasoning_path
from weedvlm.reasoning.judge import JUDGEMENTS_JSONL
from weedvlm.reasoning.prompt import (
    TEXTUAL_FIELDS,
    TEXTUAL_KNOWLEDGE_FIELD,
    VISUAL_FIELDS,
    JudgeBuilder,
    load_species_descriptions,
)
from weedvlm.reasoning.runner import JudgeItem, to_items
from weedvlm.storage.database import connect

HERE = Path(__file__).resolve().parent
SCORE_FIELDS = (*TEXTUAL_FIELDS, TEXTUAL_KNOWLEDGE_FIELD, *VISUAL_FIELDS)
LABEL_COLUMNS = [
    "response_id", "run", "question_id", "task", "question_type", "match",
    *SCORE_FIELDS, "notes", "skipped", "labeller", "labelled_at",
]


def load_judgements(run_dirs: list[Path]) -> dict[str, dict[tuple[int, str], dict]]:
    """run directory name -> {(response_id, kind): last record with status ok}."""
    runs: dict[str, dict[tuple[int, str], dict]] = {}
    for run_dir in run_dirs:
        records: dict[tuple[int, str], dict] = {}
        for line in (run_dir / JUDGEMENTS_JSONL).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records[(record["response_id"], record["kind"])] = record
        runs[run_dir.name] = {k: r for k, r in records.items() if r["status"] == "ok"}
    return runs


def order_items(items: list[JudgeItem], limit: int | None, seed: int) -> list[JudgeItem]:
    """Round-robin over (task, question type, correct/incorrect) so a short
    session touches every task and includes incorrect responses."""
    groups: dict[tuple, list[JudgeItem]] = {}
    for item in items:
        groups.setdefault((item.task, item.question_type, item.match), []).append(item)
    rng = random.Random(seed)
    for queue in groups.values():
        rng.shuffle(queue)
    ordered: list[JudgeItem] = []
    while any(groups.values()) and (limit is None or len(ordered) < limit):
        for queue in groups.values():
            if queue and (limit is None or len(ordered) < limit):
                ordered.append(queue.pop())
    return ordered


class Labels:
    """The labels parquet: one row per response, rewritten whole on every save
    (a few hundred rows at most) through a temp file so a crash can't corrupt it."""

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.df = pd.read_parquet(path)
        else:
            self.df = pd.DataFrame({c: pd.Series(dtype=object) for c in LABEL_COLUMNS})
        self.df = self.df.set_index("response_id", drop=False)

    def get(self, response_id: int) -> dict | None:
        if response_id not in self.df.index:
            return None
        row = self.df.loc[response_id]
        return {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in row.items()}

    def set(self, row: dict) -> None:
        self.df.loc[row["response_id"]] = pd.Series(row)
        self._write()

    def delete(self, response_id: int) -> None:
        if response_id in self.df.index:
            self.df = self.df.drop(index=response_id)
            self._write()

    def _write(self) -> None:
        self.df.index = self.df["response_id"].astype(int)
        out = self.df.reset_index(drop=True)[LABEL_COLUMNS].copy()
        for field in SCORE_FIELDS:
            out[field] = pd.to_numeric(out[field]).astype("Int64")
        out["response_id"] = out["response_id"].astype("int64")
        out["skipped"] = out["skipped"].astype(bool)
        tmp = self.path.with_suffix(".parquet.tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(self.path)


class App:
    def __init__(self, args: argparse.Namespace):
        config = load_reasoning_config(resolve_reasoning_path(args.config))
        self.judgements = load_judgements([Path(p) for p in args.judgements])
        ids = sorted({rid for records in self.judgements.values() for rid, _ in records})
        results = pd.read_parquet(config.results_file)
        results = results[results["id"].isin(ids)].drop_duplicates("id")
        found = set(results["id"])
        if missing := set(ids) - found:
            raise SystemExit(f"{len(missing)} judged responses aren't in {config.results_file}")
        self.builder = JudgeBuilder(
            connect(config.resolved_database()),
            str(config.resolved_generation_config()),
            load_species_descriptions(config.species_descriptions),
        )
        items = order_items(to_items(results), args.limit, args.seed)
        self.items = {item.response_id: item for item in items}
        self.order = [item.response_id for item in items]
        self.labels = Labels(Path(args.labels_out))
        self.labeller = args.labeller

    def fields_for(self, item: JudgeItem) -> list[str]:
        wanted = [
            *self.builder.build(item, "textual").required_fields,
            *self.builder.build(item, "visual").required_fields,
        ]
        return [f for f in SCORE_FIELDS if f in wanted]

    def summary(self) -> dict:
        labels = {rid: self.labels.get(rid) for rid in self.order}
        return {
            "items": [
                {
                    "response_id": rid,
                    "task": self.items[rid].task,
                    "question_type": self.items[rid].question_type,
                    "state": None if labels[rid] is None
                    else "skipped" if labels[rid]["skipped"] else "scored",
                }
                for rid in self.order
            ],
            "labels_path": str(self.labels.path),
            "judge_runs": list(self.judgements),
        }

    def item(self, response_id: int) -> dict:
        item = self.items[response_id]
        label = self.labels.get(response_id)
        visual = self.builder.build(item, "visual")
        out = {
            "response_id": response_id,
            "task": item.task,
            "question_type": item.question_type,
            "text": visual.user_text,
            "fields": self.fields_for(item),
            "label": label,
            # The judges' scores stay hidden until this item has been scored.
            "judges": None,
        }
        if label is not None:
            out["judges"] = {
                run: {
                    kind: {"scores": rec["scores"], "notes": rec["notes"]}
                    for kind in ("textual", "visual")
                    if (rec := records.get((response_id, kind)))
                }
                for run, records in self.judgements.items()
            }
        return out

    def save(self, body: dict) -> None:
        response_id = int(body["response_id"])
        item = self.items[response_id]
        if body.get("clear"):
            self.labels.delete(response_id)
            return
        skipped = bool(body.get("skipped"))
        fields = self.fields_for(item)
        scores = {f: None for f in SCORE_FIELDS}
        if not skipped:
            for f in fields:
                value = body["scores"].get(f)
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10:
                    raise ValueError(f"{f} must be an integer from 0 to 10")
                scores[f] = value
        self.labels.set({
            "response_id": response_id,
            "run": item.run,
            "question_id": item.question_id,
            "task": item.task,
            "question_type": item.question_type,
            "match": item.match,
            **scores,
            "notes": (body.get("notes") or "").strip() or None,
            "skipped": skipped,
            "labeller": self.labeller,
            "labelled_at": datetime.now(UTC).isoformat(),
        })

    def agreement(self) -> dict:
        """Per judge run and score field: how many items you both scored, Spearman
        correlation, mean absolute difference, and the share within one point."""
        df = self.labels.df
        df = df[~df["skipped"].astype(bool)] if len(df) else df
        out: dict[str, dict] = {}
        for run, records in self.judgements.items():
            per_field: dict[str, dict] = {}
            for field in SCORE_FIELDS:
                pairs = []
                for rid, row in df.iterrows():
                    kind = "textual" if field in (*TEXTUAL_FIELDS, TEXTUAL_KNOWLEDGE_FIELD) else "visual"
                    rec = records.get((int(rid), kind))
                    human = row[field]
                    if rec and field in rec["scores"] and pd.notna(human):
                        pairs.append((float(human), float(rec["scores"][field])))
                if not pairs:
                    continue
                human, judge = (pd.Series(x) for x in zip(*pairs))
                diff = (human - judge).abs()
                rho = human.corr(judge, method="spearman") if len(pairs) > 2 else float("nan")
                per_field[field] = {
                    "n": len(pairs),
                    "spearman": None if pd.isna(rho) else round(float(rho), 3),
                    "mean_abs_diff": round(float(diff.mean()), 2),
                    "within_one": round(float((diff <= 1).mean()), 3),
                    "mean_human": round(float(human.mean()), 2),
                    "mean_judge": round(float(judge.mean()), 2),
                }
            out[run] = per_field
        return out


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/api/items":
                    return self._json(app.summary())
                if path.startswith("/api/item/"):
                    return self._json(app.item(int(path.rsplit("/", 1)[1])))
                if path == "/api/agreement":
                    return self._json(app.agreement())
                if path.startswith("/image/"):
                    data, media_type = app.builder.image_for(app.items[int(path.rsplit("/", 1)[1])])
                    return self._send(200, data, media_type)
                static = {"/": "index.html", "/app.js": "app.js"}.get(path)
                if static:
                    type_ = "text/html; charset=utf-8" if static.endswith("html") else "text/javascript"
                    return self._send(200, (HERE / "static" / static).read_bytes(), type_)
                self._send(404, b"Not found", "text/plain")
            except (KeyError, ValueError) as error:
                self._send(400, str(error).encode(), "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/label":
                return self._send(404, b"Not found", "text/plain")
            try:
                app.save(json.loads(self.rfile.read(int(self.headers["content-length"]))))
                self._json({"ok": True})
            except (KeyError, ValueError) as error:
                self._send(400, str(error).encode(), "text/plain")

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judgements", action="append", required=True, help="a reasoning run directory (repeatable)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--labels-out", default=str(HERE / "rationale-labels.parquet"))
    parser.add_argument("--limit", type=int, default=60, help="responses to queue (default 60)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labeller", default="human")
    parser.add_argument("--port", type=int, default=8789)
    args = parser.parse_args()
    app = App(args)
    print(f"Items:  {len(app.order)} from {', '.join(app.judgements)}")
    print(f"Labels: {app.labels.path} ({len(app.labels.df)} so far)")
    print(f"\nOpen http://localhost:{args.port} in your browser.")
    try:
        HTTPServer(("127.0.0.1", args.port), make_handler(app)).serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
