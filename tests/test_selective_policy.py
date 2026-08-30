import pytest
import torch

from projects.pixvl_idea3.selective_policy import (
    anchor_relative_advantages,
    project_conflicting_gradient,
    selective_outcome_loss_scales,
)


CFG = {
    "selective_outcome_loss_scales": {
        "positive": {"ce": 0.5},
        "negative": {"ce": 2.0, "kl": 4.0},
    }
}


def test_selective_scales_follow_target_existence():
    positive = selective_outcome_loss_scales(
        CFG, {"task": "refseg", "meta": {"no_target": False}}
    )
    negative = selective_outcome_loss_scales(
        CFG, {"task": "refseg", "meta": {"no_target": True}}
    )
    assert positive == {"ce": 0.5, "rl": 1.0, "opd": 1.0, "kl": 1.0}
    assert negative == {"ce": 2.0, "rl": 1.0, "opd": 1.0, "kl": 4.0}


def test_non_refseg_is_unchanged_and_negative_scales_are_rejected():
    assert selective_outcome_loss_scales(CFG, {"task": "maskcap"}) == {
        "ce": 1.0,
        "rl": 1.0,
        "opd": 1.0,
        "kl": 1.0,
    }
    with pytest.raises(ValueError):
        selective_outcome_loss_scales(
            {"selective_outcome_loss_scales": {"negative": {"ce": -1}}},
            {"task": "refseg", "meta": {"no_target": True}},
        )


def test_conflicting_objective_gradient_is_projected_to_constraint_boundary():
    projected, stats = project_conflicting_gradient(
        [torch.tensor([-2.0, 1.0])], [torch.tensor([1.0, 0.0])]
    )
    assert stats["active"] is True
    assert torch.allclose(projected[0], torch.tensor([0.0, 1.0]))
    assert float((projected[0] * torch.tensor([1.0, 0.0])).sum()) >= 0.0


def test_aligned_objective_gradient_is_unchanged():
    objective = [torch.tensor([2.0, 1.0])]
    projected, stats = project_conflicting_gradient(
        objective, [torch.tensor([1.0, 0.0])]
    )
    assert stats["active"] is False
    assert torch.equal(projected[0], objective[0])


def test_anchor_relative_advantage_keeps_only_improvements():
    advantages = anchor_relative_advantages([0.4, 0.6, 0.9], 0.6)
    assert torch.allclose(advantages, torch.tensor([0.0, 0.0, 0.3]))
