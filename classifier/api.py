"""Loopback-only review API. Workspace identity comes from server configuration."""

import hmac
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import store
from .database import connect
from .embeddings import Embedder
from .jev_engine import DecisionUnavailable
from .models import Decision, Edit, Matter, TimeEntry
from .runtime import create_engine

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("POINTTWO_DATA", str(ROOT / ".pointtwo/python")))
WS = os.getenv("POINTTWO_WORKSPACE", "local-review")
embedder = Embedder()
engine = None
stop = threading.Event()
_token = ""


def work():
    while not stop.is_set():
        try:
            if store.prepare_batch(WS, embedder):
                continue
            with connect() as conn:
                row = conn.execute(
                    "SELECT * FROM pt_entries WHERE workspace_id=%s AND state='waiting' ORDER BY updated_at,id LIMIT 1",
                    (WS,),
                ).fetchone()
            if row:
                try:
                    engine.process_entry(WS, row)
                except store.NotReady:
                    stop.wait(1)
                except Exception:
                    log.exception("Entry classification failed")
                    with connect() as conn:
                        conn.execute(
                            "UPDATE pt_entries SET state='failed',error='Classification failed; check local service logs and retry' WHERE workspace_id=%s AND id=%s AND revision=%s AND accepted_id IS NULL",
                            (WS, row["id"], row["revision"]),
                        )
                continue
        except Exception:
            log.exception("Preparation failed")
            with connect() as conn:
                conn.execute(
                    "UPDATE pt_workspaces SET preparation_error='Local embedding preparation failed; check Ollama and retry' WHERE id=%s",
                    (WS,),
                )
        stop.wait(1)


@asynccontextmanager
async def lifespan(app):
    global _token, engine
    DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
    tokenfile = DATA / "access-token"
    if not tokenfile.exists():
        fd = os.open(tokenfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(48))
    _token = tokenfile.read_text().strip()
    if len(_token) < 32:
        raise ValueError("Access token must have at least 32 characters")
    store.migrate()
    owned_engine = engine is None
    if owned_engine:
        engine = create_engine(embedder, DATA)
    with connect() as conn:
        store.workspace(conn, WS)
        store.configure_decisions(conn, WS, engine.configuration)
    stop.clear()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(timeout=65)
    if owned_engine:
        if hasattr(engine, "client"):
            engine.client.close()
        engine = None


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def authorized(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer ") or not hmac.compare_digest(
        authorization[7:], _token
    ):
        raise HTTPException(401, "Use the local workspace access token")
    return WS


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(
        {
            "error": "Invalid request fields",
            "details": [
                {"field": ".".join(str(x) for x in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ],
        },
        422,
    )


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({"error": exc.detail}, exc.status_code)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"error": str(exc)}, 400)


for exception, status in [
    (store.Conflict, 409),
    (store.Missing, 404),
    (store.NotReady, 409),
    (DecisionUnavailable, 502),
]:

    def handler(request, exc, status=status):
        return JSONResponse({"error": str(exc)}, status)

    app.add_exception_handler(exception, handler)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "engine": "python-pgvector",
        "decision_pipeline": engine.configuration["kind"] if engine else "starting",
    }


@app.get("/api/preparation")
def preparation(ws=Depends(authorized)):
    try:
        return store.progress(ws, embedder.model)
    except (RuntimeError, __import__("httpx").HTTPError):
        return {
            "state": "failed",
            "total": 0,
            "prepared": 0,
            "failed": 1,
            "generation": 0,
            "error": "Local Qwen model unavailable",
        }


@app.post("/api/catalog")
def catalog(items: list[Matter], ws=Depends(authorized)):
    return store.import_catalog(ws, items)


@app.post("/api/catalog/rebuild")
def rebuild(ws=Depends(authorized)):
    with connect() as conn, conn.transaction():
        store.lock(conn, ws)
        conn.execute(
            "DELETE FROM pt_vectors WHERE workspace_id=%s AND kind='catalog'", (ws,)
        )
        conn.execute(
            "UPDATE pt_workspaces SET preparation_error=NULL WHERE id=%s", (ws,)
        )
        store.bump(conn, ws)
        queued = conn.execute(
            "UPDATE pt_entries SET state='waiting',error=NULL WHERE workspace_id=%s AND accepted_id IS NULL",
            (ws,),
        ).rowcount
    return {"queued": True, "entries_queued": queued}


@app.get("/api/matters")
def matters(
    q: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    ws=Depends(authorized),
):
    with connect() as conn:
        where = "workspace_id=%s AND (%s='' OR strpos(lower(client_name||' '||matter_name||' '||client_id||' '||matter_id),lower(%s))>0)"
        args = (ws, q, q)
        count = conn.execute(
            "SELECT count(*) n FROM matters WHERE " + where, args
        ).fetchone()["n"]
        rows = conn.execute(
            "SELECT *,description AS matter_description FROM matters WHERE "
            + where
            + " ORDER BY client_name,matter_name,id OFFSET %s LIMIT %s",
            args + (offset, limit),
        ).fetchall()
    return {"items": rows, "total": count}


@app.get("/api/matters/{ident}")
def matter(ident: str, ws=Depends(authorized)):
    with connect() as conn:
        row = conn.execute(
            "SELECT *,description AS matter_description FROM matters WHERE workspace_id=%s AND id=%s",
            (ws, UUID(ident)),
        ).fetchone()
    if not row:
        raise store.Missing("Matter not found")
    return row


@app.post("/api/time-entries")
def entries_import(items: list[TimeEntry], ws=Depends(authorized)):
    return store.import_entries(ws, items)


@app.get("/api/time-entries")
def entries(
    q: str = "",
    entry_id: str = "",
    timekeeper: str = "",
    date: str = "",
    state: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
    ws=Depends(authorized),
):
    with connect() as conn:
        where = "workspace_id=%s AND (%s='' OR strpos(lower(external_id||' '||(input->>'Narrative')),lower(%s))>0) AND (%s='' OR timekeeper=%s) AND (%s='' OR input->>'EntryDate'=%s) AND (%s='' OR state=%s)"
        where += " AND (%s='' OR external_id=%s)"
        args = (
            ws,
            q,
            q,
            timekeeper,
            timekeeper,
            date,
            date,
            state,
            state,
            entry_id,
            entry_id,
        )
        total = conn.execute(
            "SELECT count(*) n FROM pt_entries WHERE " + where, args
        ).fetchone()["n"]
        gen = conn.execute(
            "SELECT generation FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()["generation"]
        rows = conn.execute(
            "SELECT * FROM pt_entries WHERE "
            + where
            + " ORDER BY external_id,id OFFSET %s LIMIT %s",
            args + (offset, limit),
        ).fetchall()
        users = conn.execute(
            "SELECT DISTINCT timekeeper FROM pt_entries WHERE workspace_id=%s ORDER BY timekeeper",
            (ws,),
        ).fetchall()
    return {
        "items": [store.row_entry(r, gen) for r in rows],
        "total": total,
        "timekeepers": [r["timekeeper"] for r in users],
    }


@app.post("/api/time-entries/refresh-open")
def refresh_open(ws=Depends(authorized)):
    """Queue stale unresolved results; the worker reuses prepared vectors."""
    with connect() as conn, conn.transaction():
        store.lock(conn, ws)
        result = conn.execute(
            """UPDATE pt_entries SET state='waiting',error=NULL
            WHERE workspace_id=%s AND accepted_id IS NULL AND state!='waiting'
              AND (generation IS DISTINCT FROM (SELECT generation FROM pt_workspaces WHERE id=%s)
                   OR state='failed')""",
            (ws, ws),
        )
        return {"queued": result.rowcount}


@app.post("/api/time-entries/{ident}/debug")
def debug_entry(ident: str, ws=Depends(authorized)):
    row = store.get_entry(ws, ident)
    # Fresh evidence with the actual EntryId/metadata; does not change the review.
    return engine.classify(ws, row["input"], ident)


@app.get("/api/time-entries/{ident}")
def entry(ident: str, ws=Depends(authorized)):
    return store.get_entry(ws, ident)


@app.patch("/api/time-entries/{ident}")
def edit(ident: str, body: Edit, ws=Depends(authorized)):
    store.edit_entry(ws, ident, body)
    return store.get_entry(ws, ident)


@app.post("/api/time-entries/{ident}/decisions")
def decision(ident: str, body: Decision, ws=Depends(authorized)):
    store.decide(ws, ident, body, embedder)
    return store.get_entry(ws, ident)


@app.post("/api/classify")
def classify(body: TimeEntry, ws=Depends(authorized)):
    result = engine.classify(ws, body.record())
    public = [
        {k: s[k] for k in ("ClientName", "MatterName", "Score", "Rationale")}
        for s in result["suggestions"]
    ]
    return JSONResponse(
        public,
        headers={
            "X-PointTwo-Low-Confidence": str(result["low_confidence"]).lower(),
            "X-PointTwo-Run-Id": result["id"],
        },
    )


@app.post("/api/debug/classify")
def debug_classify(body: TimeEntry, ws=Depends(authorized)):
    return engine.classify(ws, body.record())


@app.get("/api/debug/runs/{ident}")
def debug_run(ident: str, ws=Depends(authorized)):
    with connect() as conn:
        row = conn.execute(
            "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
            (ws, UUID(ident)),
        ).fetchone()
    if not row:
        raise store.Missing("Run not found")
    return row["snapshot"]


@app.get("/api/debug/runs/{ident}/projection")
def debug_projection(ident: str, after: str | None = None, ws=Depends(authorized)):
    from .projection import graph

    return graph(ws, ident, after)


@app.post("/api/debug/runs/{ident}/compare")
def debug_compare(ident: str, ws=Depends(authorized)):
    return engine.comparison(ws, ident)


@app.get("/api/scorer")
def scorer(ws=Depends(authorized)):
    if engine.configuration["kind"] == "jev":
        return engine.configuration
    with connect() as conn:
        return conn.execute(
            "SELECT scorer FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()["scorer"]


web = ROOT / "web/dist"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
