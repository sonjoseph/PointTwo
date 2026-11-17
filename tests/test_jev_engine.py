"""Jev application integration with real Postgres and a deterministic remote stub."""

import json

import pytest

from classifier import store
from classifier.database import connect
from classifier.jev import MODEL, NONE
from classifier.jev_engine import DecisionUnavailable, JevEngine
from classifier.models import Decision
from classifier.runtime import create_engine
from tests.test_python_classifier import entry
from tests.test_python_classifier import system as system_fixture

system = system_fixture


class StubJev:
    def __init__(self, *, uncertain=False, on_request=None):
        self.requests = []
        self.uncertain = uncertain
        self.on_request = on_request

    def decide(self, payload):
        self.requests.append(payload)
        if self.on_request:
            self.on_request()
        options = payload["questions"]["assignment"]["criteria"]
        target = next(
            k
            for k, c in options.items()
            if isinstance(c, dict) and c["client_id"] == "a"
        )
        probs = {k: 0.0 for k in options}
        probs[target] = 0.3 if self.uncertain else 0.95
        probs[NONE] = 0.7 if self.uncertain else 0.05
        return {
            "model": MODEL,
            "answers": {
                "assignment": {
                    "type": "choice",
                    "choice": NONE if self.uncertain else target,
                    "probabilities": probs,
                },
                "sufficient_evidence": {
                    "type": "noul",
                    "noul": 0.2 if self.uncertain else 0.95,
                },
            },
        }, 12.5


@pytest.mark.parametrize("uncertain", [False, True])
def test_jev_saves_probabilities_alternatives_and_review_reasons(system, uncertain):
    ws, emb, _, ids, _ = system
    client = StubJev(uncertain=uncertain)
    engine = JevEngine(emb, client)
    store.import_entries(ws, [entry()])
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s", (ws,)
        ).fetchone()
    result = engine.process_entry(ws, row)
    saved = store.get_entry(ws, str(row["id"]))
    assert result["scorer"]["kind"] == saved["score_kind"] == "jev"
    assert saved["state"] == ("low_confidence" if uncertain else "review")
    assert saved["low_confidence"] is uncertain
    assert len(saved["suggestions"]) == 2  # Abstention does not discard alternatives.
    assert saved["suggestions"][0]["CandidateId"] == ids["a"]
    assert saved["suggestions"][0]["Score"] == (0.3 if uncertain else 0.95)
    assert saved["decision"]["insufficient_evidence_probability"] == (
        0.7 if uncertain else 0.05
    )
    assert result["timing_ms"]["decision"] == 12.5
    assert all(not c["contributions"] for c in result["candidates"])
    assert result["decision_request"] == client.requests[0]
    assert "Wrong" not in json.dumps(client.requests[0])
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_runs WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 1
        )


def test_jev_receives_confirmed_text_and_excludes_query_itself(system):
    ws, emb, _, ids, _ = system
    source = entry("Prepared investor agreements", ident="source")
    store.import_entries(ws, [source])
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s", (ws,)
        ).fetchone()
    store.decide(
        ws,
        str(row["id"]),
        Decision(revision=row["revision"], action="accept", candidate_id=ids["a"]),
        emb,
    )
    client = StubJev()
    engine = JevEngine(emb, client)
    result = engine.classify(ws, entry("Revised investor agreements").record())
    assert result["candidates"][0]["examples"][0]["narrative"] == source.Narrative
    assert source.Narrative in json.dumps(
        client.requests[-1]["questions"]["assignment"]["criteria"]
    )
    same = engine.classify(ws, source.record())
    assert all(not c["examples"] for c in same["candidates"])
    assert same["decision"]["explicit_override"]
    assert same["candidates"][0]["Score"] == 0.95


def test_jev_rejects_response_after_concurrent_evidence_change(system):
    ws, emb, _, _, _ = system

    def change():
        with connect() as conn:
            store.bump(conn, ws)

    engine = JevEngine(emb, StubJev(on_request=change))
    with pytest.raises(store.NotReady, match="Evidence changed"):
        engine.classify(ws, entry().record())
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_runs WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 0
        )


def test_jev_failure_does_not_save_a_weighted_fallback(system):
    ws, emb, _, _, _ = system

    def fail():
        raise RuntimeError("secret provider response")

    with pytest.raises(DecisionUnavailable) as exc:
        JevEngine(emb, StubJev(on_request=fail)).classify(ws, entry().record())
    assert "secret" not in str(exc.value)
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_runs WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 0
        )


def test_pipeline_switch_requeues_only_open_entries_once(system):
    ws, emb, baseline, ids, _ = system
    store.import_entries(ws, [entry(ident="open"), entry(ident="approved")])
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s", (ws,)
        ).fetchall()
    for row in rows:
        baseline.process_entry(ws, row)
    approved = next(r for r in rows if r["external_id"] == "approved")
    store.decide(
        ws,
        str(approved["id"]),
        Decision(revision=approved["revision"], action="accept", candidate_id=ids["a"]),
        emb,
    )
    with connect() as conn:
        assert store.configure_decisions(conn, ws, JevEngine.configuration) == 1
        assert store.configure_decisions(conn, ws, JevEngine.configuration) == 0
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_runs WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 2
        )
        assert (
            conn.execute(
                "SELECT state FROM pt_entries WHERE workspace_id=%s AND external_id='open'",
                (ws,),
            ).fetchone()["state"]
            == "waiting"
        )
    assert store.get_entry(ws, str(approved["id"]))["accepted_id"] == ids["a"]


def test_runtime_cloud_opt_in_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("POINTTWO_DECISION", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY_FILE", raising=False)
    assert create_engine(None, tmp_path).configuration == {"kind": "weighted"}
    monkeypatch.setenv("POINTTWO_DECISION", "jev")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        create_engine(None, tmp_path)
    keyfile = tmp_path / "private-key"
    keyfile.write_text("test-private-key")
    monkeypatch.setenv("TYPESAFE_API_KEY_FILE", str(keyfile))
    engine = create_engine(None, tmp_path)
    assert engine.configuration["kind"] == "jev"
    engine.client.close()
