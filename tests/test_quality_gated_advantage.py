import torch

from projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer import quality_gated_advantages


def test_quality_gate_suppresses_uniformly_failed_group():
    advantages = quality_gated_advantages(
        [0.10, 0.20, 0.30, 0.40], threshold=0.5, temperature=0.05
    )
    assert torch.equal(advantages, torch.zeros(4))


def test_quality_gate_keeps_successful_rollout_direction():
    advantages = quality_gated_advantages(
        [0.10, 0.20, 0.55, 0.80], threshold=0.5, temperature=0.05
    )
    assert advantages[0] == 0
    assert advantages[1] == 0
    assert advantages[2] > 0
    assert advantages[3] > advantages[2]


def test_quality_gate_singleton_has_no_relative_signal():
    advantages = quality_gated_advantages([0.9], threshold=0.5)
    assert torch.equal(advantages, torch.zeros(1))
