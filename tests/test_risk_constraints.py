import torch

from projects.pixvl_idea3.risk_constraints import outcome_constraint, update_dual


def test_constraint_only_uses_no_target_refseg_rows():
    samples = [
        {"task": "refseg", "meta": {"no_target": True}},
        {"task": "refseg", "meta": {"no_target": False}},
        {"task": "maskcap", "meta": {"no_target": True}},
    ]
    losses = torch.tensor([2.0, 9.0, 7.0], requires_grad=True)
    violation, null_loss, mask = outcome_constraint(losses, samples, budget=1.0)
    assert mask.tolist() == [True, False, False]
    assert null_loss.item() == 2.0
    assert violation.item() == 1.0
    violation.backward()
    assert losses.grad.tolist() == [1.0, 0.0, 0.0]


def test_dual_update_is_projected():
    assert update_dual(0.2, 0.5, learning_rate=1.0, maximum=0.4) == 0.4
    assert update_dual(0.2, -1.0, learning_rate=1.0, maximum=10.0) == 0.0
