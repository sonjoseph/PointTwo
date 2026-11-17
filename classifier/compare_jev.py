"""Paired synthetic evaluation. Offline by default; --execute opts into TypeSafe."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from uuid import uuid4

from . import store
from .database import connect
from .embeddings import Embedder
from .engine import Engine
from .jev import MODEL, PROMPT_VERSION, JevClient, interpret, request, shortlist
from .models import Decision, Matter, TimeEntry
from .scoring import demo_scorer, review_policy

ROOT = Path(__file__).resolve().parents[1]


def measure(labels, abstained, case, latency_ms, *, reasons=(), error=None):
    expected = case["expected"]
    return {
        "top": labels[:5],
        "top1_correct": bool(labels and labels[0] == expected) if expected else None,
        "top5_correct": expected in labels[:5] if expected else None,
        "abstained": abstained,
        "latency_ms": latency_ms,
        "reasons": list(reasons),
        "error": error,
    }


def metrics(rows, pipeline):
    selected = [(r, r[pipeline]) for r in rows if r[pipeline] is not None]
    if not selected:
        return None
    answerable = [v for r, v in selected if r["expected"]]
    ambiguous = [v for r, v in selected if r["should_abstain"]]
    ready = [(r, v) for r, v in selected if not v["abstained"]]
    times = sorted(v["latency_ms"] for _, v in selected if v["latency_ms"] is not None)
    return {
        "cases": len(selected),
        "answerable": len(answerable),
        "top1_accuracy": sum(v["top1_correct"] for v in answerable) / len(answerable)
        if answerable
        else None,
        "top5_accuracy": sum(v["top5_correct"] for v in answerable) / len(answerable)
        if answerable
        else None,
        "abstention_recall": sum(v["abstained"] for v in ambiguous) / len(ambiguous)
        if ambiguous
        else None,
        "coverage": len(ready) / len(selected),
        "incorrect_nonabstained": sum(not v["top1_correct"] for _, v in ready),
        "errors": sum(bool(v["error"]) for _, v in selected),
        "p50_ms": statistics.median(times) if times else None,
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))] if times else None,
    }


def retrieval_metrics(rows):
    answerable = [r for r in rows if r["expected"]]
    if not answerable:
        return None
    return {
        "answerable": len(answerable),
        "full_candidate_coverage": sum(r["full_candidate_covered"] for r in answerable)
        / len(answerable),
        "shortlist_coverage": sum(r["shortlist_covered"] for r in answerable)
        / len(answerable),
        "catalog_cosine_top1": sum(
            r["catalog_cosine_order"][:1] == [r["expected"]] for r in answerable
        )
        / len(answerable),
        "catalog_cosine_top5": sum(
            r["expected"] in r["catalog_cosine_order"][:5] for r in answerable
        )
        / len(answerable),
    }


def run(
    output,
    *,
    execute=False,
    limit=None,
    size=10,
    minimum_probability=0.7,
    minimum_margin=0.1,
    minimum_support=0.7,
):
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    if not 1 <= size <= 50:
        raise ValueError("Shortlist size must be between 1 and 50")
    for value in (minimum_probability, minimum_margin, minimum_support):
        if not 0 <= value <= 1:
            raise ValueError("Review thresholds must be between 0 and 1")
    client = (
        JevClient() if execute else None
    )  # Fail before DB work if access is missing.
    try:
        return _run(
            output,
            client,
            limit,
            size,
            {
                "minimum_probability": minimum_probability,
                "minimum_margin": minimum_margin,
                "minimum_support": minimum_support,
            },
        )
    finally:
        if client:
            client.close()


def _run(output, client, limit, size, gates):
    fixture = ROOT / "fixtures/classification-corpus.json"
    data = json.loads(fixture.read_text())
    ws = "jev-evaluation-" + uuid4().hex
    embedder = Embedder()
    engine = Engine(embedder)
    store.migrate()
    with connect() as conn:
        store.workspace(conn, ws)
    store.import_catalog(ws, [Matter.model_validate(m) for m in data["catalog"]])
    while store.prepare_batch(ws, embedder):
        pass
    cache = ROOT / ".pointtwo/jev/responses"
    cache.mkdir(parents=True, exist_ok=True)
    prepared = []

    def compare(case, phase="blind"):
        snapshot = engine.classify(ws, case["entry"], persist=False)
        candidates = shortlist(snapshot, size)
        linear = sorted(
            candidates,
            key=lambda c: (c["decision"] != "accept", -c["Score"], c["CandidateId"]),
        )
        low, reasons, _ = review_policy(linear, snapshot["scorer"])
        labels = lambda cs: [[c["ClientId"], c["MatterId"]] for c in cs]
        full = [c for c in snapshot["candidates"] if c["decision"] != "reject"]
        retrieval_ms = snapshot["timing_ms"]["total"]
        row = {
            "id": case["entry"]["EntryId"],
            "narrative": case["entry"]["Narrative"],
            "phase": phase,
            "split": case["split"],
            "category": case["category"],
            "expected": case["expected"],
            "should_abstain": case["should_abstain"],
            "candidate_count": len(candidates),
            "candidate_labels": labels(candidates),
            "candidate_evidence": [
                {
                    key: c[key]
                    for key in (
                        "ClientId",
                        "MatterId",
                        "ClientName",
                        "MatterName",
                        "catalog_text",
                        "features",
                        "Score",
                        "examples",
                    )
                }
                for c in candidates
            ],
            # Ranking-only diagnostic; cosine alone has no review policy here.
            "catalog_cosine_order": labels(
                sorted(
                    candidates,
                    key=lambda c: (
                        -c["features"]["catalog_similarity"],
                        c["ClientId"],
                        c["MatterId"],
                    ),
                )
            ),
            "full_candidate_covered": case["expected"] in labels(full)
            if case["expected"]
            else None,
            "shortlist_covered": case["expected"] in labels(candidates)
            if case["expected"]
            else None,
            "baseline_full": measure(
                labels(full), snapshot["low_confidence"], case, retrieval_ms
            ),
            "baseline_shortlist": measure(
                labels(linear), low, case, retrieval_ms, reasons=reasons
            ),
            "jev": None,
            "jev_request_ms": None,
            "cached_response": False,
        }
        if not candidates:
            if client:
                row["jev"] = measure(
                    [], True, case, retrieval_ms, reasons=["No retrieved candidates"]
                )
            return row
        payload, mapping = request(snapshot, candidates)
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256((PROMPT_VERSION + canonical).encode()).hexdigest()
        row["request_hash"] = digest
        prepared.append(
            {
                "id": row["id"],
                "phase": phase,
                "request_hash": digest,
                "request": payload,
            }
        )
        if client:
            path = cache / (digest + ".json")
            try:
                if path.exists():
                    saved = json.loads(path.read_text())
                    raw, request_ms = saved["response"], saved["latency_ms"]
                    row["cached_response"] = True
                else:
                    raw, request_ms = client.decide(payload)
                judged = interpret(raw, mapping, **gates)
                if not path.exists():
                    path.write_text(
                        json.dumps(
                            {"response": raw, "latency_ms": request_ms}, indent=2
                        )
                        + "\n"
                    )
                row["jev"] = measure(
                    judged["labels"],
                    judged["abstained"],
                    case,
                    None if row["cached_response"] else retrieval_ms + request_ms,
                    reasons=judged["reasons"],
                )
                row["jev"].update(
                    {
                        "probabilities": judged["probabilities"],
                        "selected_option": judged["selected_option"],
                        "sufficient_evidence": judged["sufficient_evidence"],
                    }
                )
                row["jev_request_ms"] = request_ms
                row["usage"] = raw.get("usage", {})
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
                row["jev"] = measure(
                    [],
                    True,
                    case,
                    None,
                    reasons=["Jev evaluation failed"],
                    error=type(error).__name__,
                )
        return row

    cases = [c for c in data["cases"] if c["split"] != "feedback"]
    if limit:
        cases = cases[:limit]
    rows = []
    for case in cases:
        rows.append(compare(case))
        if len(rows) % 25 == 0:
            print(f"Compared {len(rows)}/{len(cases)} synthetic cases", flush=True)
    feedback = []
    for pair in data["feedback_pairs"][:3]:
        case = {
            "entry": pair["probe"],
            "expected": pair["assignment"],
            "split": "feedback",
            "category": "feedback",
            "should_abstain": False,
        }
        before = compare(case, "before_feedback")
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
        after = compare(case, "after_feedback")
        feedback.append(
            {"source": pair["source"]["EntryId"], "before": before, "after": after}
        )
        # Keep each probe's history isolated from earlier experiments.
        updated = store.get_entry(ws, str(source["id"]))
        store.decide(
            ws,
            str(source["id"]),
            Decision(revision=updated["revision"], action="undo"),
            embedder,
        )
    pipelines = ("baseline_full", "baseline_shortlist", "jev")
    all_rows = rows + [r for f in feedback for r in (f["before"], f["after"])]
    report = {
        "status": "prepared_not_run"
        if client is None
        else (
            "complete_with_errors"
            if any(r["jev"]["error"] for r in all_rows)
            else "complete"
        ),
        "workspace": ws,
        "corpus_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "model": embedder.model,
        "scorer_version": demo_scorer()["version"],
        "jev_model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "shortlist_size": size,
        "thresholds": {**gates, "validated": False},
        "summary": {p: metrics(rows, p) for p in pipelines},
        "retrieval": {
            "all": retrieval_metrics(rows),
            **{
                s: retrieval_metrics([r for r in rows if r["split"] == s])
                for s in ("development", "heldout")
            },
        },
        "splits": {
            s: {p: metrics([r for r in rows if r["split"] == s], p) for p in pipelines}
            for s in ("development", "heldout")
        },
        "categories": {
            s: {
                p: metrics([r for r in rows if r["category"] == s], p)
                for p in pipelines
            }
            for s in sorted({r["category"] for r in rows})
        },
        "cases": rows,
        "feedback": feedback,
        "limitations": [
            "Synthetic fixtures only; not a real legal-domain accuracy claim.",
            "Shortlist selection is lexical-first then maximum catalog/example similarity; independent of linear weights and gold labels.",
            "Same shortlist for both rerankers; full baseline also reported to expose shortlist pruning losses.",
            "Jev probabilities are conditional on the supplied options; review thresholds are unvalidated presets.",
            "Previously inspected held-out set is a regression set, not a fresh blind release gate.",
            "Primary cases contain no approved history; three separate feedback probes exercise confirmed examples.",
            "Inference errors count as abstentions and incorrect answerable rankings, and are reported separately.",
            "End-to-end Jev time includes shared retrieval/linear preparation plus HTTP inference; cached responses excluded from latency aggregates.",
            "No production app switch, weekly Jev fine-tuning, or 50k accuracy claim.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    requests_path = output.with_suffix(".requests.json")
    requests_path.write_text(json.dumps(prepared, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "requests": str(requests_path),
                "status": report["status"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".pointtwo/python/jev-comparison.json"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send synthetic narratives/catalog/examples to TypeSafe (requires TYPESAFE_API_KEY)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shortlist", type=int, default=10)
    parser.add_argument("--minimum-probability", type=float, default=0.7)
    parser.add_argument("--minimum-margin", type=float, default=0.1)
    parser.add_argument("--minimum-support", type=float, default=0.7)
    args = parser.parse_args()
    try:
        result = run(
            args.output,
            execute=args.execute,
            limit=args.limit,
            size=args.shortlist,
            minimum_probability=args.minimum_probability,
            minimum_margin=args.minimum_margin,
            minimum_support=args.minimum_support,
        )
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    if result["status"] == "complete_with_errors":
        parser.exit(1, "Some Jev requests failed; inspect the comparison report.\n")
