"""Real-model evaluation in a disposable synthetic workspace, never user history."""

import argparse
import json
from pathlib import Path
import statistics
import time
from uuid import uuid4
from .database import connect
from .embeddings import Embedder
from .engine import Engine
from .models import Matter, TimeEntry, Decision
from . import store
from .scoring import demo_scorer

ROOT = Path(__file__).resolve().parents[1]


def run(output, limit=None):
    data = json.loads((ROOT / "fixtures/classification-corpus.json").read_text())
    ws = "evaluation-" + uuid4().hex
    embedder = Embedder()
    engine = Engine(embedder)
    store.migrate()
    with connect() as conn:
        store.workspace(conn, ws)
    store.import_catalog(ws, [Matter.model_validate(m) for m in data["catalog"]])
    start = time.perf_counter()
    while store.prepare_batch(ws, embedder):
        print(
            "Preparing synthetic catalog",
            store.progress(ws, embedder.model)["prepared"],
            flush=True,
        )
    preparation_ms = (time.perf_counter() - start) * 1000
    cases = [c for c in data["cases"] if c["split"] != "feedback"]
    if limit:
        cases = cases[:limit]
    results = []
    for i, case in enumerate(cases):
        snapshot = engine.classify(ws, case["entry"], persist=False)
        candidates = [c for c in snapshot["candidates"] if c["decision"] != "reject"]
        labels = [[c["ClientId"], c["MatterId"]] for c in candidates]
        results.append(
            {
                "id": case["entry"]["EntryId"],
                "category": case["category"],
                "split": case["split"],
                "expected": case["expected"],
                "should_abstain": case["should_abstain"],
                "top1_correct": labels[0] == case["expected"]
                if labels and case["expected"]
                else None,
                "top5_correct": case["expected"] in labels[:5]
                if case["expected"]
                else None,
                "candidate_covered": case["expected"] in labels
                if case["expected"]
                else None,
                "abstained": snapshot["low_confidence"],
                "reasons": snapshot["reasons"],
                "top": labels[:5],
                "latency_ms": snapshot["timing_ms"]["total"],
                "retrieval": snapshot["retrieval"],
            }
        )
        if (i + 1) % 25 == 0:
            print(f"Evaluated {i + 1}/{len(cases)}", flush=True)

    def metrics(rows):
        answerable = [r for r in rows if r["expected"]]
        ambiguous = [r for r in rows if r["should_abstain"]]
        suggested = [r for r in rows if not r["abstained"]]

        def rate(items, key):
            return sum(bool(r[key]) for r in items) / len(items) if items else None

        return {
            "cases": len(rows),
            "answerable": len(answerable),
            "ambiguous_or_unsupported": len(ambiguous),
            "top1_accuracy": rate(answerable, "top1_correct"),
            "top5_accuracy": rate(answerable, "top5_correct"),
            "candidate_coverage": rate(answerable, "candidate_covered"),
            "abstention_recall": rate(ambiguous, "abstained"),
            "coverage": len(suggested) / len(rows) if rows else 0,
            "incorrect_nonabstained": sum(not r["top1_correct"] for r in suggested),
            "p50_ms": statistics.median(r["latency_ms"] for r in rows) if rows else 0,
            "p95_ms": sorted(r["latency_ms"] for r in rows)[
                min(len(rows) - 1, int(0.95 * len(rows)))
            ]
            if rows
            else 0,
        }

    feedback = []
    for pair in data["feedback_pairs"][:3]:
        before = engine.classify(ws, pair["probe"])
        store.import_entries(ws, [TimeEntry.model_validate(pair["source"])])
        with connect() as conn:
            source = conn.execute(
                "SELECT * FROM pt_entries WHERE workspace_id=%s AND external_id=%s",
                (ws, pair["source"]["EntryId"]),
            ).fetchone()
            matter = conn.execute(
                "SELECT id FROM matters WHERE workspace_id=%s AND client_id=%s AND matter_id=%s",
                (ws, *pair["assignment"]),
            ).fetchone()
        store.decide(
            ws,
            str(source["id"]),
            Decision(
                revision=source["revision"],
                action="accept",
                candidate_id=str(matter["id"]),
            ),
            embedder,
        )
        comparison = engine.comparison(ws, before["id"])
        change = next(
            c for c in comparison["changes"] if c["CandidateId"] == str(matter["id"])
        )
        feedback.append(
            {
                "source": pair["source"]["EntryId"],
                "probe": pair["probe"]["EntryId"],
                "change": change,
                "same_scorer": comparison["same_scorer"],
            }
        )
    report = {
        "status": "complete",
        "workspace": ws,
        "model": embedder.model,
        "scorer_version": demo_scorer()["version"],
        "preparation_ms": preparation_ms,
        "catalog_records": len(data["catalog"]),
        "summary": metrics(results),
        "splits": {
            s: metrics([r for r in results if r["split"] == s])
            for s in ("development", "heldout")
        },
        "categories": {
            c: metrics([r for r in results if r["category"] == c])
            for c in sorted({r["category"] for r in results})
        },
        "feedback": feedback,
        "cases": results,
        "limitations": data["notes"]
        + [
            "These timings include mixed cache states and are not a cold-cache benchmark.",
            "No weights or thresholds were tuned on held-out results.",
            "Accuracy here is measured on 108 catalog matters; the separate 50k benchmark measures scale and ANN recall.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"report": str(output), "summary": report["summary"]}, indent=2),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=ROOT / ".pointtwo/python/evaluation.json"
    )
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    run(a.output, a.limit)
