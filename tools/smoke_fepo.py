#!/usr/bin/env python3
"""Deterministic FEPO contract smoke; no model, CUDA, or dataset required."""

from projects.pixvl_idea3.failure_evidence import (
    CAPABILITIES,
    failure_evidence,
    local_scope,
    mix_local_losses,
    soft_route,
)


def main() -> None:
    deficits = failure_evidence(
        semantic_coverage=0.4,
        unsupported_claim_rate=0.1,
        target_confuser_margin=0.2,
        boundary_iou=0.5,
        area_error=0.2,
    )
    route = soft_route(deficits, temperature=0.25, min_failure=0.2)
    weights = route["weights"]
    assert set(weights) == set(CAPABILITIES)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert route["failure"] is True
    assert local_scope("semantic", "maskcap") == "semantic_text"
    assert local_scope("relation", "refseg") == "relation_and_mask"
    assert local_scope("geometry", "refseg") == "mask"
    correction = mix_local_losses(
        {name: 1.0 for name in CAPABILITIES}, route, preservation_loss=0.25
    )
    assert 1.0 < correction < 1.3
    clean_route = soft_route({name: 0.01 for name in CAPABILITIES}, min_failure=0.2)
    assert mix_local_losses({name: 10.0 for name in CAPABILITIES}, clean_route) == 0.0
    print("FEPO_SMOKE_OK", route)


if __name__ == "__main__":
    main()
