"""Pure, reproducible linear scoring, independent of retrieval and review decisions."""

import re
import unicodedata
import numpy as np

FEATURES = (
    "catalog_similarity",
    "example_similarity",
    "client_name_match",
    "matter_reference_match",
    "timekeeper_involvement",
)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def contains(text, phrase):
    return bool(phrase.strip()) and bool(
        re.search(
            r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", normalize(text)
        )
    )


def demo_scorer():
    # Preserve the original five coefficients; remove only the availability bonus.
    values = np.random.default_rng(42).uniform(0, 1, 6)
    values /= values.sum()
    values = np.delete(values, 2)
    return {
        "version": "random-42-v2",
        "features": list(FEATURES),
        "weights": dict(zip(FEATURES, values.tolist())),
        "bias": 0.0,
        "trained": False,
        "calibrated": False,
        "policy": {
            "version": "review-v1",
            "minimum_catalog_similarity": 0.45,
            "minimum_margin": 0.04,
            "thresholds_validated": False,
        },
    }


def feature_values(entry, matter, catalog_similarity, examples):
    values = dict(
        zip(
            FEATURES,
            [
                max(0, min(1, catalog_similarity)),
                max([max(0, min(1, e["similarity"])) for e in examples], default=0),
                int(contains(entry["Narrative"], matter["client_name"])),
                int(contains(entry["Narrative"], matter["matter_id"])),
                int(
                    entry["UserId"] in matter["metadata"].get("InvolvedTimekeepers", [])
                ),
            ],
        )
    )

    # Availability is diagnostic/context for review rules, not a scoring feature.
    values["has_example_match"] = int(bool(examples))
    return values


def score(features, scorer):
    contributions = {f: float(features[f] * scorer["weights"][f]) for f in FEATURES}
    return float(scorer["bias"] + sum(contributions.values())), contributions


def review_policy(candidates, scorer):
    """Conservative uncalibrated demo policy; suggestions always require review."""
    eligible = [c for c in candidates if c["decision"] != "reject"]
    if not eligible:
        return True, ["No eligible candidates"], 0.0
    top = eligible[0]
    margin = top["Score"] - eligible[1]["Score"] if len(eligible) > 1 else top["Score"]
    if top["decision"] == "accept":
        return False, ["Explicit reviewer assignment; score unchanged"], margin
    reasons = []
    p = scorer["policy"]
    if margin < p["minimum_margin"]:
        reasons.append("Leading candidates have similar scores")
    if (
        top["features"]["catalog_similarity"] < p["minimum_catalog_similarity"]
        and not top["features"]["has_example_match"]
    ):
        reasons.append("Weak semantic evidence")
    # A name or label learned from context is required before marking a draft ready.
    if not (
        top["features"]["client_name_match"]
        or top["features"]["matter_reference_match"]
        or top["features"]["has_example_match"]
    ):
        reasons.append("No explicit client/matter reference or confirmed example")
    if top["Status"] == "Closed":
        reasons.append("Closed matter needs explicit review")
    return (
        bool(reasons),
        reasons or ["Draft suggestion; human approval required"],
        float(margin),
    )
