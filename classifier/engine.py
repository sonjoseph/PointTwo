"""Candidate retrieval, transparent scoring, and immutable debug snapshots."""

import time
from uuid import UUID, uuid4
from psycopg.types.json import Jsonb
from .database import connect
from .embeddings import INSTRUCTION
from .scoring import contains, normalize, feature_values, score, review_policy
from .store import NotReady, Missing, lock, progress


class Engine:
    def __init__(self, embedder):
        self.embedder = embedder

    def classify(self, ws, entry, entry_id=None, persist=True, exact=False):
        start = time.perf_counter()
        model = self.embedder.model
        prepared = progress(ws, model)
        if prepared["state"] != "ready":
            raise NotReady("Catalog must finish preparation before classification")
        query = self.embedder.embed(INSTRUCTION + entry["Narrative"])
        embedded = time.perf_counter()
        with connect() as conn, conn.transaction():
            # A consistent snapshot of catalog, feedback and scorer; serialize only DB work.
            lock(conn, ws)
            state = conn.execute(
                "SELECT * FROM pt_workspaces WHERE id=%s", (ws,)
            ).fetchone()
            missing = conn.execute(
                """SELECT 1 FROM matters m WHERE m.workspace_id=%s AND NOT EXISTS
                (SELECT 1 FROM pt_vectors v WHERE v.workspace_id=m.workspace_id AND v.source_id=m.id
                AND v.kind='catalog' AND v.model_id=%s AND v.source_revision=m.revision) LIMIT 1""",
                (ws, model),
            ).fetchone()
            if missing:
                raise NotReady(
                    "Catalog changed during inference; wait for preparation and retry"
                )
            conn.execute("SET LOCAL hnsw.ef_search=400")
            conn.execute("SET LOCAL hnsw.iterative_scan=strict_order")
            conn.execute("SET LOCAL hnsw.max_scan_tuples=100000")
            conn.execute("SET LOCAL hnsw.scan_mem_multiplier=4")
            params = {"ws": ws, "model": model, "query": query, "user": entry["UserId"]}
            order = (
                "(v.embedding <=> %(query)s) + 0.0"
                if exact
                else "v.embedding <=> %(query)s"
            )
            # The complete-catalog gate above guarantees current vectors. Materialize
            # ANN candidates before metadata joins so the planner retains the distance index.
            catalog_sql = f"""WITH nearest AS MATERIALIZED (
                SELECT v.source_id,v.matter_id,v.matter_revision,v.embedding <=> %(query)s AS distance
                FROM pt_vectors v WHERE v.workspace_id=%(ws)s AND v.model_id=%(model)s AND v.kind='catalog'
                ORDER BY {order} LIMIT 50)
                SELECT n.source_id,n.matter_id,n.distance FROM nearest n JOIN matters m
                ON m.workspace_id=%(ws)s AND m.id=n.matter_id AND m.revision=n.matter_revision
                ORDER BY n.distance"""
            catalog = conn.execute(catalog_sql, params).fetchall()
            example_sql = f"""SELECT v.source_id,v.matter_id,v.body,v.normalized,v.embedding <=> %(query)s AS distance
                FROM pt_vectors v JOIN matters m ON m.workspace_id=v.workspace_id AND m.id=v.matter_id AND m.revision=v.matter_revision
                JOIN pt_entries e ON e.workspace_id=v.workspace_id AND e.id=v.source_id AND e.source_revision=v.source_revision AND e.accepted_id=v.matter_id
                WHERE v.workspace_id=%(ws)s AND v.model_id=%(model)s AND v.kind='example' AND v.timekeeper=%(user)s
                ORDER BY {order} LIMIT 100"""
            history = conn.execute(example_sql, params).fetchall()
            plan = conn.execute(
                "EXPLAIN (FORMAT JSON) " + catalog_sql, params
            ).fetchone()["QUERY PLAN"]

            def indexes(node):
                found = []
                if isinstance(node, dict):
                    if node.get("Index Name"):
                        found.append(node["Index Name"])
                    for val in node.values():
                        found.extend(indexes(val))
                if isinstance(node, list):
                    for val in node:
                        found.extend(indexes(val))
                return found

            indexes_used = indexes(plan)
            decisions = conn.execute(
                """SELECT d.* FROM pt_decisions d JOIN pt_entries e
                ON e.workspace_id=d.workspace_id AND e.id=d.entry_id AND e.source_revision=d.entry_revision
                LEFT JOIN matters m ON m.workspace_id=d.workspace_id AND m.id=d.matter_id
                WHERE d.workspace_id=%s AND d.timekeeper=%s AND d.normalized=%s
                AND (d.action='undo' OR m.revision=d.matter_revision) ORDER BY d.id""",
                (ws, entry["UserId"], normalize(entry["Narrative"])),
            ).fetchall()
            explicit = {}
            for d in decisions:
                if d["action"] == "undo":
                    explicit.clear()
                elif d["action"] == "accept":
                    explicit = {k: v for k, v in explicit.items() if v != "accept"}
                    explicit[str(d["matter_id"])] = "accept"
                else:
                    explicit[str(d["matter_id"])] = "reject"
            # Literal substring prefilter, then Unicode-aware boundary verification.
            # No wildcard interpretation of client-controlled names.
            lexical = conn.execute(
                """SELECT id,client_name,matter_id FROM matters WHERE workspace_id=%s
                AND (strpos(lower(%s),lower(client_name))>0 OR strpos(lower(%s),lower(matter_id))>0)
                ORDER BY id LIMIT 1000""",
                (ws, entry["Narrative"], entry["Narrative"]),
            ).fetchall()
            lexical_ids = {
                str(r["id"])
                for r in lexical
                if contains(entry["Narrative"], r["client_name"])
                or contains(entry["Narrative"], r["matter_id"])
            }
            ids = (
                {str(r["matter_id"]) for r in catalog + history}
                | lexical_ids
                | set(explicit)
            )
            matters = (
                conn.execute(
                    """SELECT m.*,v.body,v.embedding <=> %s AS distance FROM matters m JOIN pt_vectors v
                ON v.workspace_id=m.workspace_id AND v.source_id=m.id AND v.kind='catalog' AND v.source_revision=m.revision AND v.model_id=%s
                WHERE m.workspace_id=%s AND m.id=ANY(%s)""",
                    (query, model, ws, [UUID(i) for i in ids]),
                ).fetchall()
                if ids
                else []
            )
            by_matter = {}
            for h in history:
                group = by_matter.setdefault(str(h["matter_id"]), {})
                key = h["normalized"]
                if key not in group:
                    group[key] = {
                        "entry_id": str(h["source_id"]),
                        "narrative": h["body"],
                        "similarity": float(1 - h["distance"]),
                        "timekeeper": entry["UserId"],
                    }
            catalog_ids = {str(c["matter_id"]) for c in catalog}
            candidates = []
            for m in matters:
                ident = str(m["id"])
                examples = list(by_matter.get(ident, {}).values())[:3]
                features = feature_values(entry, m, float(1 - m["distance"]), examples)
                value, contributions = score(features, state["scorer"])
                sources = []
                if ident in catalog_ids:
                    sources.append("catalog")
                if examples:
                    sources.append("confirmed examples")
                if ident in lexical_ids:
                    sources.append("explicit reference")
                if ident in explicit:
                    sources.append("review decision")
                candidates.append(
                    {
                        "CandidateId": ident,
                        "ClientId": m["client_id"],
                        "MatterId": m["matter_id"],
                        "ClientName": m["client_name"],
                        "MatterName": m["matter_name"],
                        "Status": m["status"],
                        "MatchesCurrent": entry["ClientName"] == m["client_name"]
                        and entry["MatterName"] == m["matter_name"],
                        "Score": value,
                        "features": features,
                        "contributions": contributions,
                        "examples": examples,
                        "catalog_text": m["body"],
                        "catalog_revision": m["revision"],
                        "sources": sources,
                        "decision": explicit.get(ident),
                        "Rationale": f"Catalog similarity {features['catalog_similarity']:.3f}; example similarity {features['example_similarity']:.3f}. Evidence: {', '.join(sources)}.",
                    }
                )
            candidates.sort(
                key=lambda c: (
                    c["decision"] == "reject",
                    c["decision"] != "accept",
                    -c["Score"],
                    c["CandidateId"],
                )
            )
            low, reasons, margin = review_policy(candidates, state["scorer"])
            eligible = [c for c in candidates if c["decision"] != "reject"]
            public = [
                {
                    k: c[k]
                    for k in (
                        "CandidateId",
                        "ClientName",
                        "MatterName",
                        "Score",
                        "Rationale",
                        "Status",
                        "MatchesCurrent",
                    )
                }
                for c in eligible[:5]
            ]
            ident = str(uuid4())
            snapshot = {
                "id": ident,
                "input": entry,
                "generation": state["generation"],
                "scorer": state["scorer"],
                "model": model,
                "candidates": candidates,
                "suggestions": public,
                "low_confidence": low,
                "reasons": reasons,
                "margin": margin,
                "retrieval": {
                    "settings": {
                        "ef_search": 400,
                        "catalog_limit": 50,
                        "example_limit": 100,
                        "exact": exact,
                    },
                    "catalog_hits": len(catalog),
                    "example_hits": len(history),
                    "candidate_count": len(candidates),
                    "indexes_used": indexes_used,
                    "mode": "exact"
                    if exact
                    else (
                        "hnsw"
                        if any(i.endswith("_catalog") for i in indexes_used)
                        else "planner-selected exact"
                    ),
                },
                "timing_ms": {
                    "embedding": (embedded - start) * 1000,
                    "retrieval_scoring": (time.perf_counter() - embedded) * 1000,
                    "total": (time.perf_counter() - start) * 1000,
                },
            }
            if persist:
                conn.execute(
                    "INSERT INTO pt_runs(workspace_id,id,entry_id,snapshot) VALUES(%s,%s,%s,%s)",
                    (
                        ws,
                        UUID(ident),
                        UUID(entry_id) if entry_id else None,
                        Jsonb(snapshot),
                    ),
                )
            return snapshot

    def process_entry(self, ws, row):
        result = self.classify(ws, row["input"], str(row["id"]))
        with connect() as conn, conn.transaction():
            lock(conn, ws)
            conn.execute(
                """UPDATE pt_entries SET suggestions=%s,low_confidence=%s,state=%s,error=NULL,run_id=%s,generation=%s
                WHERE workspace_id=%s AND id=%s AND revision=%s AND accepted_id IS NULL
                AND %s=(SELECT generation FROM pt_workspaces WHERE id=%s)""",
                (
                    Jsonb(result["suggestions"]),
                    result["low_confidence"],
                    "low_confidence" if result["low_confidence"] else "review",
                    UUID(result["id"]),
                    result["generation"],
                    ws,
                    row["id"],
                    row["revision"],
                    result["generation"],
                    ws,
                ),
            )
        return result

    def comparison(self, ws, run_id):
        with connect() as conn:
            previous = conn.execute(
                "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
                (ws, UUID(run_id)),
            ).fetchone()
        if not previous:
            raise Missing("Baseline not found")
        before = previous["snapshot"]
        after = self.classify(ws, before["input"])
        old = {c["CandidateId"]: c for c in before["candidates"]}
        new = {c["CandidateId"]: c for c in after["candidates"]}
        changes = []
        for ident in old.keys() | new.keys():
            a = old.get(ident)
            b = new.get(ident)
            changes.append(
                {
                    "CandidateId": ident,
                    "ClientName": (b or a)["ClientName"],
                    "MatterName": (b or a)["MatterName"],
                    "before": a["Score"] if a else None,
                    "after": b["Score"] if b else None,
                    "delta": b["Score"] - a["Score"] if a and b else None,
                    "features": {
                        f: {
                            "before": a["features"][f] if a else None,
                            "after": b["features"][f] if b else None,
                        }
                        for f in after["scorer"]["features"]
                    },
                    "decision_before": a["decision"] if a else None,
                    "decision_after": b["decision"] if b else None,
                }
            )
        changes.sort(key=lambda c: abs(c["delta"] or 0), reverse=True)
        return {
            "before": before,
            "after": after,
            "changes": changes,
            "same_scorer": before["scorer"] == after["scorer"],
            "same_retrieval": before["retrieval"].get("settings")
            == after["retrieval"].get("settings"),
            "same_model": before["model"] == after["model"],
            "generation_changed": before["generation"] != after["generation"],
        }
