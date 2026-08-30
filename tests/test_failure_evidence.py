from projects.pixvl_idea3.failure_evidence import (
    CAPABILITIES,
    failure_evidence,
    local_scope,
    mix_local_losses,
    predicted_only_evidence_route,
    soft_route,
)


def test_evidence_is_bounded_and_routes_softly():
    deficits = failure_evidence(
        semantic_coverage=0.25,
        unsupported_claim_rate=0.8,
        target_confuser_margin=0.1,
        boundary_iou=0.2,
        area_error=1.7,
        no_target_error=0.4,
    )
    assert set(deficits) == set(CAPABILITIES)
    assert all(0.0 <= value <= 1.0 for value in deficits.values())

    route = soft_route(deficits, temperature=0.3, min_failure=0.2)
    weights = route["weights"]
    assert set(weights) == set(CAPABILITIES)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert route["failure"] is True


def test_clean_samples_preserve_the_policy_objective_only():
    route = soft_route({name: 0.01 for name in CAPABILITIES}, min_failure=0.2)
    assert mix_local_losses(
        {name: 10.0 for name in CAPABILITIES}, route, preservation_loss=0.3
    ) == 0.3


def test_scopes_are_task_matched():
    assert local_scope("semantic", "maskcap") == "semantic_text"
    assert local_scope("relation", "refseg") == "relation_and_mask"
    assert local_scope("relation", "maskcap") == "semantic_text"
    assert local_scope("geometry", "refseg") == "mask"


def test_predicted_only_adapter_does_not_require_targets():
    route = predicted_only_evidence_route(
        prompt_text="segment the person on the left",
        predicted_text="<|mt_start|> <|mt_0001|> <|mt_end|>",
        task="refseg",
        answer_confidence=0.9,
    )
    assert route["predicted_only"] is True
    assert route["failure_route_reasons"] == ["predicted_only_evidence"]
    assert abs(sum(route["route_weights"].values()) - 1.0) < 1e-6
