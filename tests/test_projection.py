"""Projection fidelity, frozen axes, feedback relabeling and historical evidence."""

from uuid import uuid4

import numpy as np
import pytest

from classifier import store
from classifier.database import connect
from classifier.embeddings import INSTRUCTION
from classifier.projection import cache_key, graph, project_vectors
from tests.test_python_classifier import decide, entry, source, system  # noqa: F401


def test_fixed_basis_and_full_dimension_distance():
    base = np.eye(4, dtype=np.float32)[:3]
    extra = np.asarray([[0, 0, 0, 1]], dtype=np.float32)
    old, variance = project_vectors(base, base)
    new, _ = project_vectors(base, np.concatenate([base, extra]))
    np.testing.assert_allclose(old, new[:3], atol=1e-12)
    assert variance == pytest.approx(1)
    assert np.isfinite(new).all()


def test_degenerate_projection():
    v = [[1.0, 0, 0], [1.0, 0, 0]]
    coords, variance = project_vectors(v, v)
    assert np.isfinite(coords).all() and variance == 0
    np.testing.assert_allclose(coords, 0)


def seed_cache(emb, text):
    with connect() as conn:
        conn.execute(
            "INSERT INTO pt_embedding_cache VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (emb.model, cache_key(emb.model, text), emb.embed(text)),
        )


def test_graph_feedback_relabels_without_moving_vectors(system):
    ws, emb, engine, ids, _ = system
    e = entry("Investor rights and vesting", ident="source")
    row = source(ws, e)
    probe = entry("Discussed investor vesting schedules", ident="probe").record()
    seed_cache(emb, INSTRUCTION + probe["Narrative"])
    seed_cache(emb, e.Narrative)
    before = engine.classify(ws, probe)
    initial = graph(ws, before["id"])
    assert not any(n["kind"] == "example" for n in initial["nodes"])
    decide(ws, row, emb, "accept", ids["a"])
    accepted = engine.comparison(ws, before["id"])["after"]
    calls = emb.calls
    added = graph(ws, before["id"], accepted["id"])
    assert emb.calls == calls  # Viewing a graph does not run inference.
    old_query = next(n for n in initial["nodes"] if n["kind"] == "query")
    query = next(n for n in added["nodes"] if n["kind"] == "query")
    assert query["x"] == pytest.approx(old_query["x"])
    assert query["y"] == pytest.approx(old_query["y"])
    assert query["cosine_distance"] == pytest.approx(0, abs=1e-7)
    example = next(n for n in added["nodes"] if n["kind"] == "example")
    assert example["before"] is None and example["after"]["matter_id"] == ids["a"]
    q = emb.embed(INSTRUCTION + probe["Narrative"])
    v = emb.embed(e.Narrative)
    assert example["cosine_distance"] == pytest.approx(1 - float(q @ v), abs=1e-6)
    decide(ws, row, emb, "accept", ids["b"])
    corrected = engine.comparison(ws, accepted["id"])["after"]
    relabeled = graph(ws, accepted["id"], corrected["id"])
    node = next(n for n in relabeled["nodes"] if n["kind"] == "example")
    assert (
        node["before"]["matter_id"] == ids["a"]
        and node["after"]["matter_id"] == ids["b"]
    )
    frozen = graph(ws, accepted["id"])
    frozen_node = next(n for n in frozen["nodes"] if n["kind"] == "example")
    assert node["x"] == pytest.approx(frozen_node["x"])
    assert node["y"] == pytest.approx(frozen_node["y"])
    decide(ws, row, emb, "undo")
    undone = engine.comparison(ws, accepted["id"])["after"]
    historical = graph(ws, accepted["id"], undone["id"])
    n = next(n for n in historical["nodes"] if n["kind"] == "example")
    assert n["before"]["matter_id"] == ids["a"] and n["after"] is None
    with pytest.raises(store.Missing):
        graph("another-workspace", before["id"])
    other = engine.classify(ws, entry("Different narrative").record())
    with pytest.raises(store.Conflict):
        graph(ws, before["id"], other["id"])


def test_graph_requires_saved_query_vector(system):
    ws, _, engine, _, _ = system
    before = engine.classify(ws, entry("Uncached " + uuid4().hex).record())
    with pytest.raises(store.NotReady, match="query vector"):
        graph(ws, before["id"])
