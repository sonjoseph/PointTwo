"""HTTP contract and authorization checks against isolated Postgres data."""

import hashlib
import os
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from classifier import api
from classifier.database import connect
from classifier.engine import Engine
from tests.test_python_classifier import FixedEmbedder, entry


@pytest.fixture
def client(tmp_path, monkeypatch, request):
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set DATABASE_URL")
    ws = "test-api-" + uuid4().hex
    embedder = FixedEmbedder()
    monkeypatch.setattr(api, "WS", ws)
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(api, "embedder", embedder)
    monkeypatch.setattr(api, "engine", Engine(embedder))
    if getattr(request, "param", True) is False:
        monkeypatch.setattr(api, "work", lambda: None)
    with TestClient(api.app) as c:
        token = (tmp_path / "access-token").read_text()
        c.headers["Authorization"] = "Bearer " + token
        yield c
    with connect() as conn, conn.transaction():
        for table in [
            "pt_vectors",
            "pt_decisions",
            "pt_entry_history",
            "pt_runs",
            "pt_entries",
            "matters",
            "pt_workspaces",
        ]:
            column = "id" if table == "pt_workspaces" else "workspace_id"
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE {}=%s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (ws,),
            )
        conn.execute(
            sql.SQL("DROP TABLE IF EXISTS {}").format(
                sql.Identifier("pt_v_" + hashlib.sha256(ws.encode()).hexdigest()[:20])
            )
        )


def test_auth_and_validation(client):
    for path in ["/api/time-entries", "/api/scorer"]:
        assert (
            client.get(path, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
    bad = client.post("/api/time-entries", json=[{**entry().record(), "Hours": -2}])
    assert bad.status_code == 422 and "error" in bad.json()
    assert client.get("/api/matters?limit=100000").status_code == 422
    assert client.post("/api/catalog", json=[]).status_code == 400


def test_review_debug_contract(client):
    cat = [
        {
            "ClientId": "a",
            "ClientName": "Acme",
            "MatterId": "finance",
            "MatterName": "Financing",
            "MatterDescription": "investor rights",
        }
    ]
    assert client.post("/api/catalog", json=cat).status_code == 200
    assert (
        client.post(
            "/api/time-entries", json=[entry("Acme investor rights").record()]
        ).status_code
        == 200
    )
    for _ in range(50):
        row = client.get("/api/time-entries").json()["items"][0]
        if row["run_id"]:
            break
        time.sleep(0.1)
    assert row["run_id"], row
    result = client.post("/api/classify", json=entry().record())
    assert result.status_code == 200
    assert set(result.json()[0]) == {"ClientName", "MatterName", "Score", "Rationale"}
    baseline = client.get("/api/debug/runs/" + row["run_id"]).json()
    assert baseline["scorer"]["version"] == "random-42-v3"
    candidate = row["suggestions"][0]["CandidateId"]
    assert (
        client.post(
            "/api/time-entries/" + row["id"] + "/decisions",
            json={
                "revision": row["revision"],
                "action": "accept",
                "candidate_id": candidate,
            },
        ).status_code
        == 200
    )
    conflict = client.post(
        "/api/time-entries/" + row["id"] + "/decisions",
        json={"revision": row["revision"], "action": "undo"},
    )
    assert conflict.status_code == 409
    comparison = client.post(
        "/api/debug/runs/" + row["run_id"] + "/compare", json={}
    ).json()
    assert comparison["same_scorer"] and comparison["generation_changed"]
    assert comparison["after"]["candidates"][0]["features"]["has_example_match"] == 1
    assert client.get("/api/debug/runs/" + str(uuid4())).status_code == 404


def test_projection_auth_missing_and_valid(client):
    from classifier.embeddings import INSTRUCTION
    from tests.test_projection import seed_cache

    path = "/api/debug/runs/" + str(uuid4()) + "/projection"
    assert (
        client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert client.get(path).status_code == 404
    probe = entry("An investor narrative").record()
    # Prepare one catalog record before requesting a saved run.
    from classifier import store

    client.post(
        "/api/catalog",
        json=[
            {
                "ClientId": "projection",
                "ClientName": "Projection",
                "MatterId": "finance",
                "MatterName": "Financing",
                "MatterDescription": "investor rights",
            }
        ],
    )
    while store.prepare_batch(api.WS, api.embedder):
        pass
    seed_cache(api.embedder, INSTRUCTION + probe["Narrative"])
    run = client.post("/api/debug/classify", json=probe)
    assert run.status_code == 200, run.text
    result = client.get("/api/debug/runs/" + run.json()["id"] + "/projection")
    assert result.status_code == 200, result.text
    assert len(result.json()["nodes"]) == 2
    assert result.json()["nodes"][0]["kind"] == "query"


@pytest.mark.parametrize("client", [False], indirect=True)
def test_exact_entry_debug_and_refresh_only_open_results(client):
    from classifier import store
    from classifier.models import Decision

    cat = [
        {
            "ClientId": "a",
            "ClientName": "Acme",
            "MatterId": "finance",
            "MatterName": "Financing",
            "MatterDescription": "investor rights",
        }
    ]
    assert client.post("/api/catalog", json=cat).status_code == 200
    while store.prepare_batch(api.WS, api.embedder):
        pass
    client.post(
        "/api/time-entries",
        json=[
            entry("Investor rights", ident="e1").record(),
            entry("Investor vesting", ident="e10").record(),
        ],
    )
    rows = client.get("/api/time-entries").json()["items"]
    for row in rows:
        with connect() as conn:
            raw = conn.execute(
                "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s",
                (api.WS, row["id"]),
            ).fetchone()
        api.engine.process_entry(api.WS, raw)
    exact = client.get("/api/time-entries?entry_id=e1").json()
    assert exact["total"] == 1 and exact["items"][0]["input"]["EntryId"] == "e1"
    assert client.get("/api/time-entries?entry_id=missing").json()["total"] == 0
    row = exact["items"][0]
    saved = client.post("/api/time-entries/" + row["id"] + "/debug", json={}).json()
    assert saved["input"] == row["input"] and saved["input"]["EntryId"] == "e1"
    target = saved["candidates"][0]["CandidateId"]
    store.decide(
        api.WS,
        row["id"],
        Decision(revision=row["revision"], action="accept", candidate_id=target),
        api.embedder,
    )
    queued = client.post("/api/time-entries/refresh-open", json={}).json()
    assert queued["queued"] == 1
    # Repeated clicks coalesce into one durable queue item.
    assert client.post("/api/time-entries/refresh-open", json={}).json()["queued"] == 0
    accepted = client.get("/api/time-entries/" + row["id"]).json()
    assert accepted["accepted_id"] == target and accepted["state"] == "accepted"
    pending = client.get("/api/time-entries?entry_id=e10").json()["items"][0]
    assert pending["state"] == "waiting"
    with connect() as conn:
        raw = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s",
            (api.WS, pending["id"]),
        ).fetchone()
    api.engine.process_entry(api.WS, raw)
    fresh = client.get("/api/time-entries/" + pending["id"]).json()
    assert not fresh["outdated"] and fresh["run_id"] != pending["run_id"]
    snapshot = client.get("/api/debug/runs/" + fresh["run_id"]).json()
    assert snapshot["candidates"][0]["features"]["has_example_match"] == 1
    assert (
        client.post("/api/time-entries/" + str(uuid4()) + "/debug", json={}).status_code
        == 404
    )


@pytest.mark.parametrize("client", [False], indirect=True)
def test_rebuild_also_refreshes_open_scores(client):
    from classifier import store

    cat = [
        {
            "ClientId": "a",
            "ClientName": "Acme",
            "MatterId": "finance",
            "MatterName": "Financing",
            "MatterDescription": "investor rights",
        }
    ]
    client.post("/api/catalog", json=cat)
    while store.prepare_batch(api.WS, api.embedder):
        pass
    client.post(
        "/api/time-entries",
        json=[entry(ident="open").record(), entry(ident="approved").record()],
    )
    rows = client.get("/api/time-entries").json()["items"]
    for row in rows:
        with connect() as conn:
            raw = conn.execute(
                "SELECT * FROM pt_entries WHERE workspace_id=%s AND id=%s",
                (api.WS, row["id"]),
            ).fetchone()
        api.engine.process_entry(api.WS, raw)
    approved = next(r for r in rows if r["input"]["EntryId"] == "approved")
    candidate = client.get("/api/time-entries/" + approved["id"]).json()["suggestions"][
        0
    ]["CandidateId"]
    client.post(
        "/api/time-entries/" + approved["id"] + "/decisions",
        json={
            "revision": approved["revision"],
            "action": "accept",
            "candidate_id": candidate,
        },
    )
    result = client.post("/api/catalog/rebuild", json={})
    assert result.json()["entries_queued"] == 1
    assert (
        client.get("/api/time-entries?entry_id=open").json()["items"][0]["state"]
        == "waiting"
    )
    assert (
        client.get("/api/time-entries/" + approved["id"]).json()["accepted_id"]
        == candidate
    )
    while store.prepare_batch(api.WS, api.embedder):
        pass
    with connect() as conn:
        raw = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND external_id='open'",
            (api.WS,),
        ).fetchone()
    api.engine.process_entry(api.WS, raw)
    assert not client.get("/api/time-entries/" + str(raw["id"])).json()["outdated"]


@pytest.mark.parametrize("client", [False], indirect=True)
def test_jev_api_returns_alternatives_and_uncertainty(client, monkeypatch):
    from classifier import store
    from classifier.jev_engine import JevEngine
    from tests.test_jev_engine import StubJev

    monkeypatch.setattr(api, "engine", JevEngine(api.embedder, StubJev(uncertain=True)))
    assert client.get("/api/health").json()["decision_pipeline"] == "jev"
    assert client.get("/api/scorer").json()["kind"] == "jev"
    client.post(
        "/api/catalog",
        json=[
            {
                "ClientId": "a",
                "ClientName": "Acme",
                "MatterId": "finance",
                "MatterName": "Financing",
                "MatterDescription": "Investor rights",
            }
        ],
    )
    while store.prepare_batch(api.WS, api.embedder):
        pass
    result = client.post("/api/classify", json=entry().record())
    assert result.status_code == 200 and len(result.json()) == 1
    assert result.headers["X-PointTwo-Low-Confidence"] == "true"
    assert set(result.json()[0]) == {"ClientName", "MatterName", "Score", "Rationale"}
    run = client.get("/api/debug/runs/" + result.headers["X-PointTwo-Run-Id"]).json()
    assert run["decision"]["selected_label"] == "Insufficient evidence"
    assert run["decision"]["thresholds"]["minimum_probability"] == 0.7
    assert run["suggestions"][0]["Score"] == 0.3
    assert run["decision_request"]["state"]["narrative"] == entry().Narrative
