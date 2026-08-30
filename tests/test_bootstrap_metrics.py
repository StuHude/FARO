from tools.bootstrap_metrics import bootstrap_mean, bootstrap_paired_delta, summarize_records


def test_bootstrap_is_deterministic_and_bounded():
    result = bootstrap_mean([0.1, 0.2, 0.3, 0.4], repeats=200, seed=7)
    assert result == bootstrap_mean([0.1, 0.2, 0.3, 0.4], repeats=200, seed=7)
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_paired_delta_and_slice_summary():
    result = bootstrap_paired_delta([0.1, 0.2], [0.2, 0.4], repeats=100, seed=2)
    assert abs(result["mean"] - 0.15) < 1e-9
    grouped = summarize_records(
        [{"slice": "a", "score": 0.2}, {"slice": "b", "score": 0.4}],
        metric="score",
        group_key="slice",
        repeats=50,
        seed=3,
    )
    assert set(grouped) == {"a", "b"}
