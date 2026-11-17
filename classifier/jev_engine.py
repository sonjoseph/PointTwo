"""Text decisions over retrieved evidence; network inference runs outside DB locks."""

import time
from typing import ClassVar
from uuid import UUID

from psycopg.types.json import Jsonb

from . import store
from .database import connect
from .engine import Engine
from .jev import MODEL, NONE, PROMPT_VERSION, interpret, request, shortlist
from .scoring import FEATURES, normalize

GATES = {"minimum_probability": 0.7, "minimum_margin": 0.1, "minimum_support": 0.7}


class DecisionUnavailable(RuntimeError):
    pass


class JevEngine(Engine):
    configuration: ClassVar[dict] = {
        "kind": "jev",
        "model": MODEL,
        "prompt": PROMPT_VERSION,
        "shortlist_size": 10,
        "policy": GATES,
    }

    def __init__(self, embedder, client):
        super().__init__(embedder)
        self.client = client

    def classify(self, ws, entry, entry_id=None, persist=True, exact=False):
        start = time.perf_counter()
        snapshot = super().classify(ws, entry, entry_id, persist=False, exact=exact)
        candidates = shortlist(snapshot, self.configuration["shortlist_size"])
        # The debugger must show exactly the examples supplied to the model.
        for candidate in candidates:
            seen = {normalize(entry["Narrative"])}
            examples = []
            for example in candidate["examples"]:
                normalized = normalize(example["narrative"])
                if normalized not in seen:
                    examples.append(example)
                    seen.add(normalized)
            candidate["examples"] = examples[:3]
        snapshot["baseline_scorer"] = snapshot["scorer"]
        snapshot["scorer"] = {
            **self.configuration,
            "version": MODEL + "/" + PROMPT_VERSION,
            "features": list(FEATURES),
            "weights": {},
            "bias": 0,
            "trained": False,
            "calibrated": False,
        }
        request_ms = 0
        if candidates:
            payload, mapping = request(snapshot, candidates)
            try:
                raw, request_ms = self.client.decide(payload)
                judged = interpret(raw, mapping, **GATES)
            except (ValueError, KeyError, TypeError, RuntimeError):
                raise DecisionUnavailable(
                    "Jev could not complete a valid decision. Retry the entry; no fallback assignment was made."
                ) from None
            by_label = {(c["ClientId"], c["MatterId"]): c for c in candidates}
            options = {c["CandidateId"]: k for k, c in mapping.items()}
            candidates = [by_label[tuple(label)] for label in judged["labels"]]
            for candidate in candidates:
                candidate["baseline_score"] = candidate["Score"]
                candidate["baseline_contributions"] = candidate.pop("contributions")
                candidate["contributions"] = {}
                candidate["Score"] = judged["probabilities"][
                    options[candidate["CandidateId"]]
                ]
                candidate["Rationale"] = (
                    f"Jev option probability {candidate['Score']:.3f}; "
                    f"catalog text and {len(candidate['examples'])} confirmed example(s) supplied. "
                    + (
                        "Reviewer assignment takes precedence. "
                        if candidate["decision"] == "accept"
                        else ""
                    )
                    + (
                        "Review required: " + "; ".join(judged["reasons"])
                        if judged["reasons"]
                        else "Draft suggestion; human approval required."
                    )
                )
            snapshot.update(
                low_confidence=judged["abstained"],
                reasons=judged["reasons"]
                or [
                    "Explicit reviewer assignment"
                    if judged["explicit_override"]
                    else "Draft suggestion; human approval required"
                ],
                margin=judged["margin"],
                decision={
                    "kind": "jev",
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "insufficient_evidence_probability": judged["probabilities"][NONE],
                    "sufficient_evidence": judged["sufficient_evidence"],
                    "selected_option": judged["selected_option"],
                    "selected_label": (
                        "Insufficient evidence"
                        if judged["selected_option"] == NONE
                        else mapping[judged["selected_option"]]["ClientName"]
                        + " / "
                        + mapping[judged["selected_option"]]["MatterName"]
                    ),
                    "provider_confidence": raw["answers"]["assignment"].get(
                        "confidence"
                    ),
                    "usage": raw.get("usage", {}),
                    "explicit_override": judged["explicit_override"],
                    "thresholds": GATES,
                    "thresholds_validated": False,
                },
                decision_request=payload,
                decision_response=raw,
            )
        else:
            snapshot.update(
                low_confidence=True,
                reasons=[
                    "No eligible candidates; search the catalog or review rejected suggestions"
                ],
                margin=0,
                decision={"kind": "jev", "model": MODEL, "not_called": True},
            )
        snapshot["candidates"] = candidates
        snapshot["suggestions"] = [
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
            for c in candidates[:5]
        ]
        snapshot["retrieval"]["shortlist_count"] = len(candidates)
        snapshot["retrieval"]["settings"]["shortlist_size"] = self.configuration[
            "shortlist_size"
        ]
        snapshot["timing_ms"].update(
            decision=request_ms, total=(time.perf_counter() - start) * 1000
        )
        # Feedback/imports can change while the remote model is running. Never publish
        # that response as current, even for the synchronous debugger/API path.
        with connect() as conn, conn.transaction():
            store.lock(conn, ws)
            state = conn.execute(
                "SELECT generation FROM pt_workspaces WHERE id=%s", (ws,)
            ).fetchone()
            if state["generation"] != snapshot["generation"]:
                raise store.NotReady(
                    "Evidence changed during Jev inference; retry with current evidence"
                )
            if entry_id:
                source = conn.execute(
                    "SELECT input FROM pt_entries WHERE workspace_id=%s AND id=%s",
                    (ws, UUID(entry_id)),
                ).fetchone()
                if source is None or source["input"] != entry:
                    raise store.NotReady(
                        "Entry changed during Jev inference; reopen and retry"
                    )
            if persist:
                conn.execute(
                    "INSERT INTO pt_runs(workspace_id,id,entry_id,snapshot) VALUES(%s,%s,%s,%s)",
                    (
                        ws,
                        UUID(snapshot["id"]),
                        UUID(entry_id) if entry_id else None,
                        Jsonb(snapshot),
                    ),
                )
        return snapshot
