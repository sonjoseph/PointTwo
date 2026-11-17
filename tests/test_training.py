import pytest
from classifier.scoring import FEATURES
from classifier.train import fit


def test_trained_weights_improve_on_inverted_baseline():
    groups = []
    for i in range(60):
        candidates = [
            {
                "CandidateId": name,
                "features": dict.fromkeys(FEATURES, value),
                "Score": 1 - value,
            }
            for name, value in [("correct", 0.9), ("wrong", 0.1)]
        ]
        groups.append({"label": "correct", "candidates": candidates})
    result = fit(groups[:40], groups[40:50], groups[50:])
    assert result["metrics"]["test"]["baseline_top1"] == 0
    assert result["metrics"]["test"]["candidate_top1"] == 1
    assert not result["calibrated"]
    assert set(result["weights"]) == set(FEATURES)


def test_training_requires_both_labels():
    with pytest.raises(ValueError):
        fit(
            [
                {
                    "label": "correct",
                    "candidates": [
                        {
                            "CandidateId": "correct",
                            "features": dict.fromkeys(FEATURES, 1),
                            "Score": 1,
                        }
                    ],
                }
            ],
            [],
            [],
        )
