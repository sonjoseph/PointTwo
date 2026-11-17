"""Integration tests use real Postgres/HNSW; deterministic embeddings isolate logic."""

import os
from uuid import uuid4
import numpy as np
import pytest
from psycopg import sql
from classifier import store
from classifier.database import connect
from classifier.engine import Engine
from classifier.models import Matter, TimeEntry, Decision, Edit
from classifier.scoring import demo_scorer, score, contains


class FixedEmbedder:
    model = "test-v1:1024"

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        vector = np.zeros(1024, dtype=np.float32)
        words = ("investor", "patent", "employee", "privacy", "tax", "vesting")
        for i, word in enumerate(words):
            vector[i] = float(word in text.lower())
        vector[7] = 0.1
        return vector / np.linalg.norm(vector)

    def many(self, texts):
        return [self.embed(t) for t in texts]


def entry(text="Reviewed investor rights", user="sam", ident="e1"):
    return TimeEntry(
        UserId=user,
        EntryId=ident,
        Narrative=text,
        Hours="0.123456789",
        ClientName="Wrong",
        MatterName="Wrong",
        EntryDate="2026-11-16",
    )


@pytest.fixture
def system():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set DATABASE_URL for isolated Postgres integration tests")
    store.migrate()
    ws = "test-" + uuid4().hex
    emb = FixedEmbedder()
    engine = Engine(emb)
    with connect() as conn:
        store.workspace(conn, ws)
    catalog = [
        Matter(
            ClientId=c,
            ClientName=name,
            MatterId="finance",
            MatterName="Financing",
            MatterDescription="investor rights",
            Status="Open",
        )
        for c, name in [("a", "Acme"), ("b", "Birch")]
    ]
    store.import_catalog(ws, catalog)
    while store.prepare_batch(ws, emb):
        pass
    with connect() as conn:
        ids = {
            r["client_id"]: str(r["id"])
            for r in conn.execute("SELECT * FROM matters WHERE workspace_id=%s", (ws,))
        }
    yield ws, emb, engine, ids, catalog
    import hashlib

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


def source(ws, e):
    store.import_entries(ws, [e])
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND external_id=%s",
            (ws, e.EntryId),
        ).fetchone()


def decide(ws, row, emb, action, target=None):
    current = store.get_entry(ws, str(row["id"]))
    store.decide(
        ws,
        str(row["id"]),
        Decision(revision=current["revision"], action=action, candidate_id=target),
        emb,
    )


def test_seed_and_arithmetic():
    a = demo_scorer()
    b = demo_scorer()
    assert a == b
    assert "has_example_match" not in a["weights"]
    assert sum(a["weights"].values()) == pytest.approx(0.5221642770336326)
    value, parts = score(dict.fromkeys(a["features"], 1), a)
    assert value == pytest.approx(sum(parts.values()) + a["bias"])
    assert value == pytest.approx(sum(a["weights"].values()))
    assert contains("ACME: investor rights", "Acme")
    assert not contains("Acmeology", "Acme")


def test_cold_start_separate_features(system):
    ws, emb, engine, ids, _ = system
    result = engine.classify(ws, entry("Acme investor rights").record())
    assert result["candidates"][0]["CandidateId"] == ids["a"]
    assert all(c["features"]["has_example_match"] == 0 for c in result["candidates"])
    assert result["scorer"]["calibrated"] is False
    assert sum(result["candidates"][0]["contributions"].values()) == pytest.approx(
        result["candidates"][0]["Score"]
    )


def test_feedback_comparison_correction_undo(system):
    ws, emb, engine, ids, _ = system
    row = source(ws, entry("Investor rights and vesting"))
    probe = entry("Reviewed investor vesting schedules", ident="probe").record()
    before = engine.classify(ws, probe)
    decide(ws, row, emb, "accept", ids["a"])
    comparison = engine.comparison(ws, before["id"])
    assert comparison["same_scorer"] and comparison["generation_changed"]
    acme = next(c for c in comparison["changes"] if c["CandidateId"] == ids["a"])
    assert acme["delta"] > 0
    assert acme["features"]["example_similarity"]["before"] == 0
    assert acme["features"]["example_similarity"]["after"] > 0
    decide(ws, row, emb, "accept", ids["b"])
    after = engine.classify(ws, probe)
    assert (
        next(c for c in after["candidates"] if c["CandidateId"] == ids["a"])["examples"]
        == []
    )
    assert next(c for c in after["candidates"] if c["CandidateId"] == ids["b"])[
        "examples"
    ]
    decide(ws, row, emb, "undo")
    undone = engine.classify(ws, probe)
    assert all(not c["examples"] for c in undone["candidates"])
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_decisions WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 3
        )
        old = conn.execute(
            "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
            (ws, before["id"]),
        ).fetchone()["snapshot"]
        assert old == before


def test_exact_override_is_not_fake_probability(system):
    ws, emb, engine, ids, _ = system
    e = entry("Acme investor rights")
    row = source(ws, e)
    decide(ws, row, emb, "accept", ids["b"])
    result = engine.classify(ws, e.record())
    assert result["candidates"][0]["CandidateId"] == ids["b"]
    assert result["candidates"][0]["decision"] == "accept"
    assert result["candidates"][0]["Score"] == pytest.approx(
        sum(result["candidates"][0]["contributions"].values())
    )
    decide(ws, row, emb, "reject", ids["b"])
    rejected = engine.classify(ws, e.record())
    assert ids["b"] not in [c["CandidateId"] for c in rejected["suggestions"]]


def test_timekeeper_isolation(system):
    ws, emb, engine, ids, _ = system
    e = entry()
    row = source(ws, e)
    decide(ws, row, emb, "accept", ids["a"])
    other = engine.classify(ws, entry(user="other").record())
    assert all(not c["examples"] and not c["decision"] for c in other["candidates"])
    with pytest.raises(store.Missing):
        engine.comparison("other-workspace", other["id"])


def test_catalog_revision_invalidates_examples(system):
    ws, emb, engine, ids, catalog = system
    e = entry()
    row = source(ws, e)
    decide(ws, row, emb, "accept", ids["a"])
    catalog[0].MatterDescription = "investor rights with updated scope"
    store.import_catalog(ws, catalog)
    with pytest.raises(store.NotReady):
        engine.classify(ws, e.record())
    while store.prepare_batch(ws, emb):
        pass
    result = engine.classify(ws, e.record())
    assert all(not c["examples"] and not c["decision"] for c in result["candidates"])


def test_source_edit_revision_conflict_and_history(system):
    ws, emb, engine, ids, _ = system
    e = entry()
    row = source(ws, e)
    decide(ws, row, emb, "accept", ids["a"])
    with pytest.raises(store.Conflict):
        store.decide(ws, str(row["id"]), Decision(revision=1, action="undo"), emb)
    current = store.get_entry(ws, str(row["id"]))
    changed = entry("Discussed employee contracts")
    store.edit_entry(
        ws,
        str(row["id"]),
        Edit(revision=current["revision"], input=changed, reset_decision=True),
    )
    assert store.get_entry(ws, str(row["id"]))["accepted_id"] is None
    result = engine.classify(ws, e.record())
    assert all(not c["decision"] and not c["examples"] for c in result["candidates"])
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_entry_history WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 1
        )


def test_reuse_idempotency_and_atomic_import(system):
    ws, emb, engine, ids, catalog = system
    before = emb.calls
    assert store.import_catalog(ws, catalog)["changed"] == 0
    assert not store.prepare_batch(ws, emb)
    assert emb.calls == before
    source(ws, entry())
    assert store.import_entries(ws, [entry()])["unchanged"] == 1
    with pytest.raises(store.Conflict):
        store.import_entries(ws, [entry(ident="new"), entry("Changed")])
    with connect() as conn:
        assert (
            conn.execute(
                "SELECT count(*) n FROM pt_entries WHERE workspace_id=%s", (ws,)
            ).fetchone()["n"]
            == 1
        )


@pytest.mark.parametrize("hours", ["-1", "0.0000000001", "NaN", "1000001"])
def test_hours_validation(hours):
    with pytest.raises(ValueError):
        TimeEntry.model_validate({**entry().record(), "Hours": hours})


def test_duplicate_history_is_not_multiple_votes(system):
    ws, emb, engine, ids, _ = system
    first = source(ws, entry("Investor vesting rights", ident="first"))
    decide(ws, first, emb, "accept", ids["a"])
    probe = entry("Discussed investor vesting", ident="probe").record()
    before = engine.classify(ws, probe)
    second = source(ws, entry("Investor vesting rights", ident="second"))
    decide(ws, second, emb, "accept", ids["a"])
    after = engine.classify(ws, probe)
    a = next(c for c in before["candidates"] if c["CandidateId"] == ids["a"])
    b = next(c for c in after["candidates"] if c["CandidateId"] == ids["a"])
    assert a["Score"] == pytest.approx(b["Score"])
    assert len(b["examples"]) == 1


def test_old_classification_does_not_overwrite_source_edit(system):
    ws, emb, engine, ids, _ = system
    old = source(ws, entry())
    store.edit_entry(
        ws,
        str(old["id"]),
        Edit(
            revision=old["revision"],
            input=entry("Employee advice"),
            reset_decision=True,
        ),
    )
    engine.process_entry(ws, old)
    current = store.get_entry(ws, str(old["id"]))
    assert current["input"]["Narrative"] == "Employee advice"
    assert current["state"] == "waiting" and current["suggestions"] == []


def test_model_change_marks_saved_results_outdated(system):
    ws, emb, engine, ids, _ = system
    row = source(ws, entry())
    engine.process_entry(ws, row)
    assert not store.get_entry(ws, str(row["id"]))["outdated"]
    emb.model = "test-v2:1024"
    while store.prepare_batch(ws, emb):
        pass
    assert store.get_entry(ws, str(row["id"]))["outdated"]


def test_catalog_benchmark_uses_prepared_workspace(system):
    from classifier.benchmark import measure

    ws, emb, _, _, _ = system
    report = measure(ws, emb.model, [("probe", emb.embed("investor"))], k=2, effort=40)
    assert report["records"] == 2
    assert report["mean_recall_at_k"] == 1
    assert report["queries"][0]["plan"]
    with pytest.raises(ValueError):
        measure(ws, emb.model, [], k=0)


def test_seed_is_repeatable_and_preserves_reviews_and_edits(system):
    from classifier.seed import seed

    ws, emb, _, _, _ = system
    result = seed(ws)
    assert result["catalog_added"] == 114
    assert result["entries_added"] == 562
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM pt_entries WHERE workspace_id=%s AND external_id='demo-001'",
            (ws,),
        ).fetchone()
        matter = conn.execute(
            "SELECT id FROM matters WHERE workspace_id=%s AND client_name='Acme Ventures' AND matter_name='Series A financing'",
            (ws,),
        ).fetchone()
    decide(ws, row, emb, "accept", str(matter["id"]))
    repeat = seed(ws)
    assert repeat["catalog_added"] == 0 and repeat["entries_added"] == 0
    assert store.get_entry(ws, str(row["id"]))["accepted_id"] == str(matter["id"])
    current = store.get_entry(ws, str(row["id"]))
    changed = TimeEntry.model_validate(
        {**current["input"], "Narrative": "A locally edited narrative"}
    )
    store.edit_entry(
        ws,
        str(row["id"]),
        Edit(revision=current["revision"], input=changed, reset_decision=True),
    )
    assert seed(ws)["entries_added"] == 0
    assert (
        store.get_entry(ws, str(row["id"]))["input"]["Narrative"] == changed.Narrative
    )


def test_stale_inflight_scoring_cannot_publish_after_feedback(system, monkeypatch):
    ws, emb, engine, ids, _ = system
    probe = source(ws, entry("Investor questions", ident="probe"))
    other = source(ws, entry("Investor vesting", ident="source"))
    original = engine.classify

    def classify_then_feedback(*args, **kwargs):
        result = original(*args, **kwargs)
        decide(ws, other, emb, "accept", ids["a"])
        return result

    monkeypatch.setattr(engine, "classify", classify_then_feedback)
    engine.process_entry(ws, probe)
    current = store.get_entry(ws, str(probe["id"]))
    assert current["state"] == "waiting" and current["run_id"] is None


def test_example_presence_without_similarity_adds_no_score():
    from classifier.scoring import feature_values

    matter = {"client_name": "Acme", "matter_id": "finance", "metadata": {}}
    probe = entry().record()
    without = feature_values(probe, matter, 0.5, [])
    with_zero = feature_values(probe, matter, 0.5, [{"similarity": 0}])
    assert with_zero["has_example_match"] == 1
    assert score(without, demo_scorer()) == score(with_zero, demo_scorer())
    with_similar = feature_values(probe, matter, 0.5, [{"similarity": 0.8}])
    assert score(with_similar, demo_scorer())[0] - score(without, demo_scorer())[
        0
    ] == pytest.approx(0.8 * demo_scorer()["weights"]["example_similarity"])


@pytest.mark.parametrize("version", ["random-42-v1", "random-42-v2"])
def test_scorer_upgrade_queues_open_entries_preserves_decisions_and_history(
    system, version
):
    from psycopg.types.json import Jsonb

    ws, emb, engine, ids, _ = system
    open_row = source(ws, entry(ident="open"))
    approved = source(ws, entry(ident="approved"))
    engine.process_entry(ws, open_row)
    decide(ws, approved, emb, "accept", ids["a"])
    old = demo_scorer()
    old["version"] = version
    old["features"].append("timekeeper_involvement")
    old["weights"]["timekeeper_involvement"] = 0.25416097453934494
    if version == "random-42-v1":
        old["features"].insert(2, "has_example_match")
        old["weights"]["has_example_match"] = 0.22367474842702245
    with connect() as conn:
        conn.execute("UPDATE pt_workspaces SET scorer=%s WHERE id=%s", (Jsonb(old), ws))
    saved = engine.classify(ws, entry().record())
    with connect() as conn:
        store.workspace(conn, ws)
        actual = conn.execute(
            "SELECT * FROM pt_workspaces WHERE id=%s", (ws,)
        ).fetchone()
        assert actual["scorer"] == demo_scorer()
        assert (
            conn.execute(
                "SELECT snapshot FROM pt_runs WHERE workspace_id=%s AND id=%s",
                (ws, saved["id"]),
            ).fetchone()["snapshot"]
            == saved
        )
        store.workspace(conn, ws)
        assert (
            conn.execute(
                "SELECT generation FROM pt_workspaces WHERE id=%s", (ws,)
            ).fetchone()["generation"]
            == actual["generation"]
        )
    assert store.get_entry(ws, str(open_row["id"]))["state"] == "waiting"
    assert store.get_entry(ws, str(approved["id"]))["accepted_id"] == ids["a"]


def test_prior_timekeeper_involvement_cannot_boost_a_matter():
    from classifier.scoring import FEATURES, feature_values

    probe = entry(user="sam").record()
    matter = {"client_name": "Acme", "matter_id": "finance", "metadata": {}}
    without = feature_values(probe, matter, 0.5, [])
    matter["metadata"]["InvolvedTimekeepers"] = ["sam", "alex"]
    involved = feature_values(probe, matter, 0.5, [])
    assert involved == without
    assert "timekeeper_involvement" not in FEATURES
    assert "timekeeper_involvement" not in demo_scorer()["weights"]
    assert score(involved, demo_scorer()) == score(without, demo_scorer())
    # Preserve the remaining weights; removal must not boost other features.
    assert demo_scorer()["weights"]["catalog_similarity"] == pytest.approx(
        0.20162455608114557
    )
    assert demo_scorer()["weights"]["example_similarity"] == pytest.approx(
        0.11433294016332621
    )
