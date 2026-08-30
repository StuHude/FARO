from __future__ import annotations

from projects.pixvl_idea3.failure_evidence import (
    CAPABILITIES,
    predicted_only_evidence,
    predicted_only_evidence_route,
)


def test_predicted_only_route_has_no_target_fields_and_normalizes_weights():
    route = predicted_only_evidence_route(
        prompt_text="segment the red car left of the bus",
        predicted_text="<seg_12><seg_13>",
        task="refseg",
    )

    assert route["predicted_only"] is True
    assert route["failure_route_reasons"] == ["predicted_only_evidence"]
    assert set(route["route_weights"]) == set(CAPABILITIES)
    assert abs(sum(route["route_weights"].values()) - 1.0) < 1e-6
    assert "mask_binary" not in route
    assert "gt_codes" not in route
    assert "caption" not in route


def test_caption_probe_uses_prompt_alignment_and_relation_signal():
    evidence = predicted_only_evidence(
        prompt_text="describe the red car left of the bus",
        predicted_text="red car left of bus",
        task="maskcap",
        answer_confidence=1.0,
    )

    assert evidence["semantic"] < 0.5
    assert evidence["relation"] < 0.5
    assert evidence["geometry"] == 0.0


def test_empty_rollout_is_a_failure_without_crashing():
    route = predicted_only_evidence_route(
        prompt_text="segment the object",
        predicted_text="",
        task="refseg",
        answer_confidence=0.2,
    )

    assert route["failure"] is True
    assert route["max_deficit"] >= 0.25


def test_refseg_relation_and_no_target_use_predicted_prompt_evidence():
    relation = predicted_only_evidence_route(
        prompt_text="segment the object to the left of the bus",
        predicted_text="<|mt_start|><|mt_0001|><|mt_0256|><|mt_end|>",
        task="refseg",
        answer_confidence=0.2,
    )
    assert relation["failure_route"] == "relation"
    no_target = predicted_only_evidence_route(
        prompt_text="segment the nonexistent unicorn that is not present",
        predicted_text="<|mt_start|><|mt_0001|><|mt_0256|><|mt_end|>",
        task="refseg",
        answer_confidence=0.9,
    )
    assert no_target["failure_route"] == "semantic"
