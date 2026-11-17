"""Fit a candidate scorer from point-in-time run snapshots; never auto-deploy it.

Schedule this command externally for weekly production experiments. The local demo
keeps its persisted random weights. Artifacts contain metrics and coefficients,
not private narratives or credentials.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from .database import connect
from .scoring import FEATURES, normalize


def fit(train, validation, test):
    def arrays(groups):
        x = []
        y = []
        weights = []
        for group in groups:
            for c in group["candidates"]:
                x.append([c["features"][f] for f in FEATURES])
                y.append(c["CandidateId"] == group["label"])
                weights.append(1 / len(group["candidates"]))
        return np.asarray(x), np.asarray(y), np.asarray(weights)

    x, y, w = arrays(train)
    if len(set(y)) != 2:
        raise ValueError("Training needs both correct and incorrect candidates")
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(x, y, sample_weight=w)

    def accuracy(groups, learned):
        correct = 0
        for g in groups:
            if not g["candidates"]:
                continue
            values = (
                model.decision_function(
                    np.asarray(
                        [[c["features"][f] for f in FEATURES] for c in g["candidates"]]
                    )
                )
                if learned
                else [c["Score"] for c in g["candidates"]]
            )
            chosen = g["candidates"][int(np.argmax(values))]["CandidateId"]
            correct += chosen == g["label"]
        return correct / len(groups) if groups else None

    return {
        "weights": dict(zip(FEATURES, model.coef_[0].tolist())),
        "bias": float(model.intercept_[0]),
        "features": list(FEATURES),
        "trained": True,
        "calibrated": False,
        "metrics": {
            name: {
                "entries": len(groups),
                "baseline_top1": accuracy(groups, False),
                "candidate_top1": accuracy(groups, True),
            }
            for name, groups in [("validation", validation), ("test", test)]
        },
    }


def run(workspace, output):
    with connect() as conn:
        # Train against latest valid accepted labels, using a snapshot from before
        # any feedback on that source entry. A later corrected label supersedes the old label.
        rows = conn.execute(
            """SELECT e.id,e.input,e.accepted_id,r.snapshot,r.created_at
            FROM pt_entries e JOIN LATERAL (
                SELECT snapshot,created_at FROM pt_runs r WHERE r.workspace_id=e.workspace_id AND r.entry_id=e.id
                AND r.created_at < (SELECT min(d.created_at) FROM pt_decisions d WHERE d.workspace_id=e.workspace_id AND d.entry_id=e.id)
                ORDER BY r.created_at LIMIT 1
             ) r ON true WHERE e.workspace_id=%s AND e.accepted_id IS NOT NULL
            AND EXISTS(SELECT 1 FROM pt_decisions d JOIN matters m ON m.workspace_id=d.workspace_id AND m.id=d.matter_id
                WHERE d.workspace_id=e.workspace_id AND d.entry_id=e.id AND d.entry_revision=e.source_revision
                AND d.action='accept' AND d.matter_id=e.accepted_id AND d.matter_revision=m.revision)
            ORDER BY r.created_at,e.id""",
            (workspace,),
        ).fetchall()
    groups = []
    seen = set()
    skipped = Counter()
    for row in rows:
        s = row["snapshot"]
        key = normalize(s["input"]["Narrative"])
        if key in seen:
            skipped["duplicate"] += 1
            continue
        if s["input"] != row["input"]:
            skipped["source_edited"] += 1
            continue
        if any(
            normalize(example["narrative"]) == key
            for c in s["candidates"]
            for example in c["examples"]
        ):
            skipped["self_or_duplicate_evidence"] += 1
            continue
        if any(c["decision"] for c in s["candidates"]):
            skipped["explicit_decision_leakage"] += 1
            continue
        seen.add(key)
        groups.append(
            {
                "candidates": s["candidates"],
                "label": str(row["accepted_id"]),
                "scorer_version": s["scorer"]["version"],
                "model": s["model"],
                "id": str(row["id"]),
            }
        )
    if len(groups) < 100:
        raise ValueError(
            f"Need at least 100 eligible reviewed entries; found {len(groups)}. Demo weights remain active."
        )
    if (
        len({g["model"] for g in groups}) != 1
        or len({g["scorer_version"] for g in groups}) != 1
    ):
        raise ValueError(
            "Select a compatible model/scorer cohort before comparing releases"
        )
    a = int(0.6 * len(groups))
    b = int(0.8 * len(groups))
    artifact = fit(groups[:a], groups[a:b], groups[b:])
    artifact.update(
        {
            "version": "candidate-linear-v1",
            "entry_count": len(groups),
            "skipped": dict(skipped),
            "status": "candidate_only",
            "deployment": "Requires calibration, abstention and subgroup checks, and a versioned release; not activated.",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "metrics": artifact["metrics"],
                "status": "candidate_only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="local-review")
    p.add_argument(
        "--output", type=Path, default=Path(".pointtwo/training/candidate.json")
    )
    args = p.parse_args()
    try:
        run(args.workspace, args.output)
    except ValueError as e:
        p.exit(2, str(e) + "\n")
