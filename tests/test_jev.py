import json

import httpx
import pytest

from classifier.compare_jev import measure, metrics, retrieval_metrics
from classifier.jev import MODEL, NONE, JevClient, interpret, request, shortlist


def candidate(name, similarity, *, lexical=0, decision=None):
    return {
        "CandidateId": name,
        "ClientId": name,
        "MatterId": "financing",
        "ClientName": name,
        "MatterName": "Financing",
        "Status": "Open",
        "decision": decision,
        "Score": 1 - similarity,
        "catalog_text": f"{name}: financing documents",
        "examples": [],
        "features": {
            "catalog_similarity": similarity,
            "example_similarity": 0,
            "client_name_match": lexical,
            "matter_reference_match": 0,
        },
    }


def snapshot():
    return {
        "input": {
            "Narrative": "Drafted documents for Acme",
            "EntryId": "probe",
            "ClientName": "WRONG IMPORTED CLIENT",
            "UserId": "private-user-id",
        },
        "candidates": [candidate("Acme", 0.8), candidate("Birch", 0.6)],
    }


def response(mapping, *, support=0.9):
    keys = list(mapping)
    return {
        "model": MODEL,
        "answers": {
            "assignment": {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {keys[0]: 0.85, keys[1]: 0.1, NONE: 0.05},
            },
            "sufficient_evidence": {"type": "noul", "noul": support},
        },
    }


def test_shortlist_does_not_use_linear_weights_or_expected_labels():
    s = snapshot()
    s["candidates"].append(candidate("Explicit", 0.1, lexical=1))
    s["candidates"].append(candidate("Rejected", 1, decision="reject"))
    original = shortlist(s, 2)
    for c in s["candidates"]:
        c["Score"] *= -100
    s["expected"] = ["Rejected", "financing"]
    assert [c["ClientId"] for c in original] == ["Explicit", "Acme"]
    assert shortlist(s, 2) == original


def test_request_uses_text_not_imported_labels_scores_or_gold():
    s = snapshot()
    s["expected"] = ["GOLD ONLY", "secret"]
    s["candidates"][0]["examples"] = [
        {"narrative": s["input"]["Narrative"]},
        {"narrative": "Negotiated investor rights"},
        {"narrative": "NEGOTIATED  investor rights"},
    ]
    payload, mapping = request(s, shortlist(s))
    serialized = json.dumps(payload)
    assert "WRONG IMPORTED CLIENT" not in serialized
    assert "private-user-id" not in serialized
    assert "GOLD ONLY" not in serialized
    assert '"Score"' not in serialized
    acme_key = next(k for k, c in mapping.items() if c["ClientId"] == "Acme")
    assert payload["questions"]["assignment"]["criteria"][acme_key][
        "confirmed_narratives_for_this_matter"
    ] == ["Negotiated investor rights"]
    s["candidates"].reverse()
    assert request(s, shortlist(s))[0] == payload


def test_decision_ranks_shared_candidates_and_uses_support_gate():
    _, mapping = request(snapshot(), shortlist(snapshot()))
    raw = response(mapping)
    result = interpret(raw, mapping)
    assert result["abstained"] is False
    raw["answers"]["sufficient_evidence"]["noul"] = 0.2
    assert interpret(raw, mapping)["abstained"] is True
    keys = list(mapping)
    raw["answers"]["assignment"].update(
        choice=NONE, probabilities={keys[0]: 0.1, keys[1]: 0.1, NONE: 0.8}
    )
    assert "Insufficient evidence selected" in interpret(raw, mapping)["reasons"]


@pytest.mark.parametrize(
    "fault", ["unknown", "missing", "nan", "sum", "model", "choice"]
)
def test_malformed_decisions_are_rejected(fault):
    _, mapping = request(snapshot(), shortlist(snapshot()))
    raw = response(mapping)
    answer = raw["answers"]["assignment"]
    if fault == "unknown":
        answer["probabilities"]["invented"] = 0
    if fault == "missing":
        del answer["probabilities"][NONE]
    if fault == "nan":
        answer["probabilities"][NONE] = float("nan")
    if fault == "sum":
        answer["probabilities"][NONE] = 0.9
    if fault == "model":
        raw["model"] = "jev-latest"
    if fault == "choice":
        answer["choice"] = NONE
    with pytest.raises(ValueError):
        interpret(raw, mapping)


def test_explicit_accept_is_preserved_and_probability_not_rewritten():
    _, mapping = request(snapshot(), shortlist(snapshot()))
    raw = response(mapping, support=0)
    accepted = list(mapping)[1]
    mapping[accepted]["decision"] = "accept"
    result = interpret(raw, mapping)
    assert result["explicit_override"] and not result["abstained"]
    assert result["probabilities"][accepted] == 0.1
    assert result["labels"][0][0] == mapping[accepted]["ClientId"]


def test_missing_retrieval_label_is_not_inserted_and_errors_count():
    case = {"expected": ["Absent", "financing"], "should_abstain": False}
    judged = measure([["Acme", "financing"]], False, case, 10)
    result = metrics([{**case, "jev": judged}], "jev")
    assert result["top1_accuracy"] == 0 and result["incorrect_nonabstained"] == 1
    failed = measure([], True, case, None, error="RuntimeError")
    result = metrics([{**case, "jev": failed}], "jev")
    assert result["errors"] == 1 and result["top1_accuracy"] == 0
    assert result["p95_ms"] is None
    assert metrics([{**case, "jev": None}], "jev") is None


def test_retrieval_report_separates_pruning_from_cosine_ranking():
    rows = [
        {
            "expected": ["Acme", "financing"],
            "full_candidate_covered": True,
            "shortlist_covered": False,
            "catalog_cosine_order": [["Birch", "financing"]],
        },
        {
            "expected": ["Birch", "financing"],
            "full_candidate_covered": True,
            "shortlist_covered": True,
            "catalog_cosine_order": [["Acme", "financing"], ["Birch", "financing"]],
        },
        {"expected": None},
    ]
    assert retrieval_metrics(rows) == {
        "answerable": 2,
        "full_candidate_coverage": 1,
        "shortlist_coverage": 0.5,
        "catalog_cosine_top1": 0,
        "catalog_cosine_top5": 0.5,
    }


def test_transport_protocol_and_redacted_failure():
    payload, mapping = request(snapshot(), shortlist(snapshot()))

    def handler(req):
        assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
        assert req.headers["Authorization"] == "Bearer test-key"
        assert json.loads(req.content) == payload
        return httpx.Response(200, json=response(mapping))

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = JevClient("test-key", transport)
        raw, elapsed = client.decide(payload)
        assert interpret(raw, mapping)["abstained"] is False and elapsed >= 0
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, text="SECRET"))
    ) as transport:
        with pytest.raises(RuntimeError, match="HTTP 401") as error:
            JevClient("test-key", transport).decide(payload)
        assert "SECRET" not in str(error.value)


def test_live_requests_require_credentials(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevClient()
