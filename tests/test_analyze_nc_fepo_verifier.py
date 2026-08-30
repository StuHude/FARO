from tools.analyze_nc_fepo_verifier import select_threshold, summarize


def test_threshold_separates_target_and_null():
    rows = [
        {"target_exists": True, "target_margin": 2.0, "source": "p"},
        {"target_exists": True, "target_margin": 1.0, "source": "p"},
        {"target_exists": False, "target_margin": -1.0, "source": "n"},
        {"target_exists": False, "target_margin": -2.0, "source": "n"},
    ]
    threshold = select_threshold(rows)
    assert -1.0 < threshold < 1.0
    assert summarize(rows, threshold)["balanced_accuracy"] == 1.0
