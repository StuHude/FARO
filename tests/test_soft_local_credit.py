from projects.pixvl_idea3.failure_evidence import soft_local_scale


def test_soft_local_scale_is_weighted_and_centered():
    routing = {
        "buckets": {
            "semantic": {"ce_scale": 1.0},
            "relation": {"ce_scale": 1.4},
            "geometry": {"ce_scale": 0.8},
        },
        "soft_credit_gain": 1.0,
    }
    value = soft_local_scale(
        routing,
        {"semantic": 0.25, "relation": 0.5, "geometry": 0.25},
        "ce_scale",
        fallback_bucket="geometry",
    )
    assert abs(value - 1.15) < 1e-6


def test_soft_local_scale_falls_back_without_evidence():
    routing = {"buckets": {"geometry": {"rl_scale": 1.2}}}
    assert soft_local_scale(
        routing, {}, "rl_scale", fallback_bucket="geometry"
    ) == 1.2
