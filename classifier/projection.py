"""Read-only, bounded projections of immutable scoring evidence.

Fit PCA to the baseline only. Reuse that basis for the comparison so accepting
or relabeling another entry does not falsely appear to move existing vectors.
Actual cosine distances always use the full normalized embedding space.
"""

import hashlib
from uuid import UUID

import numpy as np

from .database import connect
from .embeddings import INSTRUCTION
from .store import Conflict, Missing, NotReady


def cache_key(model, text):
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()


def project_vectors(baseline, all_vectors):
    base = np.asarray(baseline, dtype=np.float64)
    all_points = np.asarray(all_vectors, dtype=np.float64)
    if base.ndim != 2 or not len(base):
        raise ValueError("Baseline vectors required")
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    all_points = all_points / np.linalg.norm(all_points, axis=1, keepdims=True)
    center = base.mean(axis=0)
    _, singular, vt = np.linalg.svd(base - center, full_matrices=False)
    basis = np.zeros((2, base.shape[1]))
    for i in range(min(2, len(vt))):
        if singular[i] > 1e-12:
            basis[i] = vt[i]
            # Fix SVD's arbitrary sign for repeatable views.
            if basis[i, np.argmax(np.abs(basis[i]))] < 0:
                basis[i] *= -1
    energy = float(np.sum(singular**2))
    return (all_points - center) @ basis.T, float(
        np.sum(singular[:2] ** 2) / energy
    ) if energy > 1e-15 else 0.0


def graph(workspace, run_id, after_id=None):
    with connect() as conn:

        def read(ident):
            row = conn.execute(
                "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
                (workspace, UUID(ident)),
            ).fetchone()
            if not row:
                raise Missing("Scoring run not found in this workspace")
            return row["snapshot"]

        before = read(run_id)
        after = read(after_id) if after_id else None
        if after and (
            before["model"] != after["model"] or before["input"] != after["input"]
        ):
            raise Conflict(
                "Compare the same probe and embedding model to share projection axes"
            )
        model = before["model"]
        points = {}
        query_text = INSTRUCTION + before["input"]["Narrative"]
        query_key = cache_key(model, query_text)
        points["query"] = {
            "id": "query",
            "kind": "query",
            "cache_key": query_key,
            "text": before["input"]["Narrative"],
            "before": {"label": "Entry being classified", "matter_id": None},
            "after": {"label": "Entry being classified", "matter_id": None}
            if after
            else None,
        }
        for phase, run in [("before", before), ("after", after)]:
            if run is None:
                continue
            # A small neighborhood, not a misleading map of the entire catalog.
            for c in run["candidates"][:8]:
                label = f"{c['ClientName']} / {c['MatterName']}"
                catalog_key = cache_key(model, c["catalog_text"])
                ident = "catalog:" + c["CandidateId"] + ":" + catalog_key
                node = points.setdefault(
                    ident,
                    {
                        "id": ident,
                        "kind": "catalog",
                        "cache_key": catalog_key,
                        "source_id": c["CandidateId"],
                        "text": c["catalog_text"],
                        "before": None,
                        "after": None,
                    },
                )
                node[phase] = {
                    "label": label,
                    "matter_id": c["CandidateId"],
                    "score": c["Score"],
                    "rank": run["candidates"].index(c) + 1,
                }
                for e in c["examples"][:3]:
                    key = cache_key(model, e["narrative"])
                    ident = "example:" + e["entry_id"] + ":" + key
                    node = points.setdefault(
                        ident,
                        {
                            "id": ident,
                            "kind": "example",
                            "cache_key": key,
                            "source_id": e["entry_id"],
                            "text": e["narrative"],
                            "before": None,
                            "after": None,
                        },
                    )
                    node[phase] = {
                        "label": label,
                        "matter_id": c["CandidateId"],
                        "entry_id": e["entry_id"],
                        "timekeeper": e["timekeeper"],
                    }
        keys = list({p["cache_key"] for p in points.values()})
        cached = {
            r["content_hash"]: r["embedding"]
            for r in conn.execute(
                "SELECT content_hash,embedding FROM pt_embedding_cache WHERE model_id=%s AND content_hash=ANY(%s)",
                (model, keys),
            )
        }
        # Older imported vectors may predate the shared cache. Only use exactly
        # matching historical text/model; never substitute a new source revision.
        for p in points.values():
            if p["cache_key"] in cached or p["kind"] == "query":
                continue
            row = conn.execute(
                """SELECT embedding FROM pt_vectors WHERE workspace_id=%s AND source_id=%s
                AND kind=%s AND model_id=%s AND body=%s LIMIT 1""",
                (
                    workspace,
                    UUID(p["source_id"]),
                    "catalog" if p["kind"] == "catalog" else "example",
                    model,
                    p["text"],
                ),
            ).fetchone()
            if row:
                cached[p["cache_key"]] = row["embedding"]
    if query_key not in cached:
        raise NotReady(
            "The saved query vector is unavailable. Analyze the narrative again to create a new baseline."
        )
    available = [p for p in points.values() if p["cache_key"] in cached]
    base = [cached[p["cache_key"]] for p in available if p["before"]]
    vectors = np.asarray([cached[p["cache_key"]] for p in available], dtype=np.float64)
    coords, variance = project_vectors(base, vectors)
    query = np.asarray(cached[query_key], dtype=np.float64)
    query /= np.linalg.norm(query)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for p, xy, v in zip(available, coords, normalized):
        p["x"], p["y"] = map(float, xy)
        p["cosine_similarity"] = float(np.clip(query @ v, -1, 1))
        p["cosine_distance"] = 1 - p["cosine_similarity"]
        p.pop("cache_key", None)
    return {
        "baseline_id": run_id,
        "after_id": after_id,
        "nodes": available,
        "explained_variance": variance,
        "method": "PCA of normalized baseline evidence; fixed basis for comparisons",
        "dimensions": 1024,
        "candidate_limit": 8,
        "example_limit_per_matter": 3,
        "missing_points": len(points) - len(available),
        "baseline_candidate_count": len(before["candidates"]),
        "notes": [
            "2D distances are approximate; inspect exact 1,024-dimensional cosine distances.",
            "Connections indicate confirmed assignments, not HNSW graph edges.",
            "Feedback changes labels and retrieved evidence, not the embedding coordinates.",
            "Absent points may not have been retrieved in that run; absence does not imply deletion.",
        ],
    }
