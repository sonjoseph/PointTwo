"""Transactional review state and reusable, partitioned search preparation."""

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from psycopg import sql
from psycopg.types.json import Jsonb

from .database import connect
from .models import Matter
from .scoring import demo_scorer, normalize

ROOT = Path(__file__).resolve().parents[1]


class Conflict(Exception):
    pass


class Missing(Exception):
    pass


class NotReady(Exception):
    pass


def migrate():
    with connect() as conn:
        for path in sorted((ROOT / "database").glob("*.sql")):
            conn.execute(path.read_text())


def lock(conn, workspace):
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (workspace,))


def workspace(conn, name):
    with conn.transaction():
        lock(conn, name)
        conn.execute(
            "INSERT INTO pt_workspaces(id,scorer) VALUES (%s,%s) ON CONFLICT DO NOTHING",
            (name, Jsonb(demo_scorer())),
        )
        previous = conn.execute(
            "SELECT scorer FROM pt_workspaces WHERE id=%s", (name,)
        ).fetchone()["scorer"]
        retired = {"has_example_match", "timekeeper_involvement"}
        if retired.intersection(previous["features"]) or retired.intersection(
            previous["weights"]
        ):
            is_demo = previous.get("version") in {
                "random-42-v1",
                "random-42-v2",
            } and not previous.get("trained")
            replacement = {
                **previous,
                "version": "random-42-v3"
                if is_demo
                else previous["version"] + "-without-retired-features",
                "features": [f for f in previous["features"] if f not in retired],
                "weights": {
                    f: w for f, w in previous["weights"].items() if f not in retired
                },
                "calibrated": False,
                "policy": {**previous["policy"], "thresholds_validated": False},
            }
            conn.execute(
                "UPDATE pt_workspaces SET scorer=%s,generation=generation+1 WHERE id=%s",
                (Jsonb(replacement), name),
            )
            conn.execute(
                "UPDATE pt_entries SET state='waiting',error=NULL WHERE workspace_id=%s AND accepted_id IS NULL",
                (name,),
            )
        # Physical partitions isolate small workspaces from unrelated benchmark vectors.
        table = "pt_v_" + hashlib.sha256(name.encode()).hexdigest()[:20]
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} PARTITION OF pt_vectors FOR VALUES IN ({})"
            ).format(sql.Identifier(table), sql.Literal(name))
        )
        for kind in ("catalog", "example"):
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops) WITH (m=16,ef_construction=128) WHERE kind={}"
                ).format(
                    sql.Identifier(table + "_" + kind),
                    sql.Identifier(table),
                    sql.Literal(kind),
                )
            )


def bump(conn, ws):
    conn.execute("UPDATE pt_workspaces SET generation=generation+1 WHERE id=%s", (ws,))


def configure_decisions(conn, ws, configuration):
    """Version a pipeline switch once, retaining explicit reviews and old runs."""
    with conn.transaction():
        lock(conn, ws)
        previous = conn.execute(
            "SELECT decision_config FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()["decision_config"]
        if previous == configuration:
            return 0
        conn.execute(
            "UPDATE pt_workspaces SET decision_config=%s WHERE id=%s",
            (Jsonb(configuration), ws),
        )
        if previous is None and configuration == {"kind": "weighted"}:
            return 0
        bump(conn, ws)
        return conn.execute(
            "UPDATE pt_entries SET state='waiting',error=NULL WHERE workspace_id=%s AND accepted_id IS NULL",
            (ws,),
        ).rowcount


def import_catalog(ws, items):
    if not 1 <= len(items) <= 50000:
        raise ValueError("Import 1–50,000 catalog records")
    pairs = [(m.ClientId, m.MatterId) for m in items]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate client/matter IDs")
    changed = 0
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        for m in items:
            record = m.model_dump()
            found = conn.execute(
                "SELECT * FROM matters WHERE workspace_id=%s AND client_id=%s AND matter_id=%s FOR UPDATE",
                (ws, m.ClientId, m.MatterId),
            ).fetchone()
            if found and found["metadata"] == record:
                continue
            ident = (
                found["id"]
                if found
                else uuid5(NAMESPACE_URL, json.dumps([ws, m.ClientId, m.MatterId]))
            )
            conn.execute(
                """INSERT INTO matters(workspace_id,id,client_id,matter_id,client_name,matter_name,description,metadata,status,revision)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
                ON CONFLICT(workspace_id,id) DO UPDATE SET client_name=excluded.client_name,matter_name=excluded.matter_name,
                description=excluded.description,metadata=excluded.metadata,status=excluded.status,revision=matters.revision+1""",
                (
                    ws,
                    ident,
                    m.ClientId,
                    m.MatterId,
                    m.ClientName,
                    m.MatterName,
                    m.MatterDescription,
                    Jsonb(record),
                    m.Status,
                ),
            )
            changed += 1
        if changed:
            bump(conn, ws)
            conn.execute(
                "UPDATE pt_workspaces SET preparation_error=NULL WHERE id=%s", (ws,)
            )
    return {"changed": changed, "unchanged": len(items) - changed}


def import_entries(ws, items):
    if not 1 <= len(items) <= 10000:
        raise ValueError("Import 1–10,000 entries")
    ids = [e.EntryId for e in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate EntryId")
    counts = {"imported": 0, "updated": 0, "unchanged": 0}
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        for e in items:
            old = conn.execute(
                "SELECT * FROM pt_entries WHERE workspace_id=%s AND external_id=%s FOR UPDATE",
                (ws, e.EntryId),
            ).fetchone()
            if old:
                if old["input"] == e.record():
                    counts["unchanged"] += 1
                    continue
                raise Conflict(
                    f"{e.EntryId} already exists with different content; edit it with its revision"
                )
            conn.execute(
                "INSERT INTO pt_entries(workspace_id,id,external_id,timekeeper,input) VALUES(%s,%s,%s,%s,%s)",
                (ws, uuid4(), e.EntryId, e.UserId, Jsonb(e.record())),
            )
            counts["imported"] += 1
    return counts


def row_entry(row, generation):
    return {
        "id": str(row["id"]),
        "input": row["input"],
        "revision": row["revision"],
        "state": row["state"],
        "suggestions": row["suggestions"],
        "accepted_id": str(row["accepted_id"]) if row["accepted_id"] else None,
        "low_confidence": row["low_confidence"],
        "error": row["error"],
        "run_id": str(row["run_id"]) if row["run_id"] else None,
        "outdated": row["generation"] is not None and row["generation"] != generation,
        "index_model": None,
        "capture_id": None,
    }


def get_entry(ws, ident):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s",
            (ws, UUID(ident)),
        ).fetchone()
        if not row:
            raise Missing("Entry not found")
        gen = conn.execute(
            "SELECT generation FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()["generation"]
        run = (
            conn.execute(
                "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
                (ws, row["run_id"]),
            ).fetchone()
            if row["run_id"]
            else None
        )
    snapshot = run["snapshot"] if run else {}
    return {
        **row_entry(row, gen),
        "decision": snapshot.get("decision"),
        "review_reasons": snapshot.get("reasons", []),
        "score_kind": snapshot.get("scorer", {}).get("kind", "weighted"),
    }


def progress(ws, model):
    with connect() as conn:
        row = conn.execute(
            """SELECT count(*) total,count(v.source_id) prepared FROM matters m LEFT JOIN pt_vectors v
            ON v.workspace_id=m.workspace_id AND v.source_id=m.id AND v.kind='catalog' AND v.model_id=%s AND v.source_revision=m.revision
            WHERE m.workspace_id=%s""",
            (model, ws),
        ).fetchone()
        state = conn.execute(
            "SELECT * FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()
    return {
        **row,
        "generation": state["generation"],
        "error": state["preparation_error"],
        "failed": int(bool(state["preparation_error"])),
        "state": "failed"
        if state["preparation_error"]
        else "empty"
        if not row["total"]
        else "ready"
        if row["total"] == row["prepared"]
        else "building",
    }


def prepare_batch(ws, embedder):
    model = embedder.model
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        conn.execute(
            "UPDATE pt_workspaces SET active_model=%s,generation=generation+CASE WHEN active_model IS NULL THEN 0 ELSE 1 END WHERE id=%s AND active_model IS DISTINCT FROM %s",
            (model, ws, model),
        )
    with connect() as conn:
        rows = conn.execute(
            """SELECT m.* FROM matters m LEFT JOIN pt_vectors v
            ON v.workspace_id=m.workspace_id AND v.source_id=m.id AND v.kind='catalog' AND v.model_id=%s AND v.source_revision=m.revision
            WHERE m.workspace_id=%s AND v.source_id IS NULL ORDER BY m.id LIMIT 16""",
            (model, ws),
        ).fetchall()
    if not rows:
        return False
    texts = [Matter.model_validate(r["metadata"]).text() for r in rows]
    vectors = embedder.many(texts)
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        for row, body, vector in zip(rows, texts, vectors):
            current = conn.execute(
                "SELECT revision FROM matters WHERE workspace_id=%s AND id=%s",
                (ws, row["id"]),
            ).fetchone()
            if not current or current["revision"] != row["revision"]:
                continue
            put_vector(
                conn,
                ws,
                row["id"],
                "catalog",
                model,
                row["revision"],
                row["id"],
                row["revision"],
                "",
                body,
                vector,
            )
        conn.execute(
            "UPDATE pt_workspaces SET preparation_error=NULL WHERE id=%s", (ws,)
        )
    return True


def put_vector(
    conn,
    ws,
    source,
    kind,
    model,
    revision,
    matter,
    matter_revision,
    timekeeper,
    body,
    vector,
):
    conn.execute(
        """INSERT INTO pt_vectors(workspace_id,source_id,kind,model_id,source_revision,matter_id,matter_revision,timekeeper,normalized,body,embedding)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(workspace_id,source_id,kind,model_id) DO UPDATE SET source_revision=excluded.source_revision,
        matter_id=excluded.matter_id,matter_revision=excluded.matter_revision,timekeeper=excluded.timekeeper,
        normalized=excluded.normalized,body=excluded.body,embedding=excluded.embedding""",
        (
            ws,
            source,
            kind,
            model,
            revision,
            matter,
            matter_revision,
            timekeeper,
            normalize(body),
            body,
            vector,
        ),
    )


def edit_entry(ws, ident, edit):
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s FOR UPDATE",
            (ws, UUID(ident)),
        ).fetchone()
        if not row:
            raise Missing("Entry not found")
        if row["revision"] != edit.revision:
            raise Conflict("Entry changed; refresh before editing")
        if row["external_id"] != edit.input.EntryId:
            raise ValueError("EntryId cannot change")
        if not edit.reset_decision:
            raise ValueError("Editing must explicitly reset review decisions")
        conn.execute(
            "INSERT INTO pt_entry_history VALUES(%s,%s,%s,%s,now())",
            (ws, row["id"], row["revision"], Jsonb(row["input"])),
        )
        conn.execute(
            "DELETE FROM pt_vectors WHERE workspace_id=%s AND source_id=%s AND kind='example'",
            (ws, row["id"]),
        )
        conn.execute(
            "UPDATE pt_entries SET input=%s,timekeeper=%s,revision=revision+1,source_revision=source_revision+1,state='waiting',accepted_id=NULL,suggestions='[]',run_id=NULL,error=NULL,updated_at=now() WHERE workspace_id=%s AND id=%s",
            (Jsonb(edit.input.record()), edit.input.UserId, ws, row["id"]),
        )
        bump(conn, ws)


def decide(ws, ident, decision, embedder):
    # Inference happens before the transaction; revision is rechecked under lock.
    vector = None
    source = get_entry(ws, ident)
    if decision.action == "accept":
        vector = embedder.embed(source["input"]["Narrative"])
    with connect() as conn, conn.transaction():
        lock(conn, ws)
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s FOR UPDATE",
            (ws, UUID(ident)),
        ).fetchone()
        if not row:
            raise Missing("Entry not found")
        if (
            row["revision"] != decision.revision
            or source["revision"] != row["revision"]
        ):
            raise Conflict("Entry changed; refresh before deciding")
        if decision.action == "retry":
            if row["accepted_id"]:
                raise Conflict("Undo the assignment before reclassifying")
            conn.execute(
                "UPDATE pt_entries SET state='waiting',error=NULL WHERE workspace_id=%s AND id=%s",
                (ws, row["id"]),
            )
            return
        matter = None
        if decision.action in ("accept", "reject"):
            if not decision.candidate_id:
                raise ValueError("Candidate required")
            matter = conn.execute(
                "SELECT * FROM matters WHERE workspace_id=%s AND id=%s",
                (ws, UUID(decision.candidate_id)),
            ).fetchone()
            if not matter:
                raise Missing("Matter not found in this workspace")
        conn.execute(
            """INSERT INTO pt_decisions(workspace_id,entry_id,entry_revision,timekeeper,normalized,matter_id,matter_revision,action,run_id)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                ws,
                row["id"],
                row["source_revision"],
                row["timekeeper"],
                normalize(row["input"]["Narrative"]),
                matter["id"] if matter else None,
                matter["revision"] if matter else None,
                decision.action,
                row["run_id"],
            ),
        )
        accepted = matter["id"] if decision.action == "accept" else row["accepted_id"]
        if decision.action == "undo" or (
            decision.action == "reject" and accepted == matter["id"]
        ):
            accepted = None
        if decision.action == "accept" or not accepted:
            conn.execute(
                "DELETE FROM pt_vectors WHERE workspace_id=%s AND source_id=%s AND kind='example'",
                (ws, row["id"]),
            )
        if decision.action == "accept":
            put_vector(
                conn,
                ws,
                row["id"],
                "example",
                embedder.model,
                row["source_revision"],
                matter["id"],
                matter["revision"],
                row["timekeeper"],
                row["input"]["Narrative"],
                vector,
            )
        conn.execute(
            "UPDATE pt_entries SET accepted_id=%s,state=%s,revision=revision+1,updated_at=now(),error=NULL WHERE workspace_id=%s AND id=%s",
            (accepted, "accepted" if accepted else "waiting", ws, row["id"]),
        )
        bump(conn, ws)
