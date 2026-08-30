from tools.analyze_text_calibration import summarize


def test_paired_summary_counts_discordant_rows():
    baseline = {
        "a": {"abstention": False, "target_abstention": False},
        "b": {"abstention": True, "target_abstention": False},
        "c": {"abstention": True, "target_abstention": True},
    }
    candidate = {
        "a": {"abstention": False, "target_abstention": False},
        "b": {"abstention": False, "target_abstention": False},
        "c": {"abstention": False, "target_abstention": True},
    }

    import random

    result = summarize(
        ["a", "b", "c"],
        baseline,
        candidate,
        bootstrap_samples=100,
        rng=random.Random(1),
    )
    assert result["baseline_accuracy"] == 2 / 3
    assert result["candidate_accuracy"] == 2 / 3
    assert result["paired_delta"] == 0.0
    assert result["discordant"] == {
        "baseline_only_correct": 1,
        "candidate_only_correct": 1,
    }
