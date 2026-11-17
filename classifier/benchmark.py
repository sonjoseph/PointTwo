"""Measure catalog-neighbor recall against exact search in a prepared workspace."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .database import connect
from .embeddings import INSTRUCTION, Embedder
from .store import NotReady, progress


def measure(workspace, model, queries, *, k=30, effort=400):
    """Queries are (identifier, embedding) pairs; no records or labels are changed."""
    if not 1 <= k <= 200 or not k <= effort <= 1000:
        raise ValueError("Require 1 <= k <= 200 and k <= effort <= 1000")
    query = """SELECT source_id,embedding <=> %(q)s AS distance FROM pt_vectors
        WHERE workspace_id=%(ws)s AND kind='catalog' AND model_id=%(model)s
        ORDER BY embedding <=> %(q)s LIMIT %(k)s"""
    exact_query = query.replace(
        "ORDER BY embedding <=> %(q)s", "ORDER BY (embedding <=> %(q)s)+0.0"
    )
    rows = []
    with connect() as conn:
        count = conn.execute(
            "SELECT count(*) n FROM pt_vectors WHERE workspace_id=%s AND kind='catalog' AND model_id=%s",
            (workspace, model),
        ).fetchone()["n"]
        if not count:
            raise ValueError("No prepared catalog vectors for this workspace and model")
        for ident, vector in queries:
            v = np.asarray(vector, dtype=np.float32)
            if v.shape != (1024,) or not np.isfinite(v).all() or not np.any(v):
                raise ValueError(
                    "Query must be a finite, nonzero 1,024-dimensional vector"
                )
            args = {"q": v, "ws": workspace, "model": model, "k": k}
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('hnsw.ef_search',%s,true)", (str(effort),)
                )
                conn.execute("SET LOCAL hnsw.iterative_scan=strict_order")
                plan = conn.execute("EXPLAIN (FORMAT JSON) " + query, args).fetchone()[
                    "QUERY PLAN"
                ]
                start = time.perf_counter()
                ann = conn.execute(query, args).fetchall()
                elapsed = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            exact = conn.execute(exact_query, args).fetchall()
            exact_ms = (time.perf_counter() - start) * 1000
            ids = {r["source_id"] for r in exact}
            recall = len(ids & {r["source_id"] for r in ann}) / len(ids)
            rows.append(
                {
                    "entry": ident,
                    "recall_at_k": recall,
                    "ann_ms": elapsed,
                    "exact_ms": exact_ms,
                    "within_exact_distance_threshold": sum(
                        r["distance"] <= exact[-1]["distance"] + 1e-7 for r in ann
                    )
                    / len(exact),
                    "plan": plan,
                }
            )
    if not rows:
        raise ValueError("Provide at least one query")
    return {
        "workspace": workspace,
        "model": model,
        "records": count,
        "k": k,
        "ef_search": effort,
        "queries": rows,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in rows) / len(rows),
        "limitations": [
            "Measures prepared catalog retrieval, not assignment accuracy or inference latency.",
            "Mixed cache states; query order is ANN then exact, without an OS cold-cache reset.",
            "The planner may use exact search for small scopes; inspect each query plan.",
            "Duplicate vectors can make exact neighbor identities interchangeable.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="local-review")
    parser.add_argument(
        "--queries", type=Path, default=Path("fixtures/time-entries.json")
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--effort", type=int, default=400)
    parser.add_argument(
        "--output", type=Path, default=Path(".pointtwo/python/benchmark.json")
    )
    args = parser.parse_args()
    embedder = Embedder()
    if progress(args.workspace, embedder.model)["state"] != "ready":
        raise NotReady("Prepare the workspace catalog before benchmarking")
    entries = json.loads(args.queries.read_text())
    queries = [
        (e["EntryId"], embedder.embed(INSTRUCTION + e["Narrative"])) for e in entries
    ]
    report = measure(
        args.workspace, embedder.model, queries, k=args.k, effort=args.effort
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["records", "k", "ef_search", "mean_recall_at_k"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
