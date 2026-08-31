from __future__ import annotations

import runpy
from pathlib import Path

import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    clipped_scope_policy_loss,
    predicted_evidence_scope_masks,
)
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE,
    UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE,
    validate_tail_gppo_config,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"


def _load(stage: str, monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    return runpy.run_path(
        str(ROOT / "Sa2VA/projects/samtok_selective/configs" / f"{stage}.py")
    )["config"]


def test_pes_configs_and_negative_control_are_fixed(monkeypatch, tmp_path):
    config = _load(UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE, monkeypatch, tmp_path)
    validate_tail_gppo_config(config)
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert config["tail_gppo"]["pes_evidence_shuffle"] is False
    assert config["tail_gppo"]["pes_confident_margin"] == 1.0
    assert "pes_confident_mass" not in config["tail_gppo"]
    assert "pes_ambiguous_mass" not in config["tail_gppo"]
    shuffled = _load(UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE, monkeypatch, tmp_path)
    validate_tail_gppo_config(shuffled)
    assert shuffled["tail_gppo"]["pes_evidence_shuffle"] is True
    assert shuffled["tail_gppo"]["pes_evidence_shuffle_seed"] == 1907


def test_pes_scope_and_loss_are_detached_and_finite():
    entropy = torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8], [2.0, 2.0, 2.0], [0.2, 0.3, 0.4]])
    mass = torch.tensor([[0.9, 0.9, 0.9], [0.3, 0.3, 0.3], [0.05, 0.05, 0.05], [0.8, 0.8, 0.8]])
    sampled = [[0, 1, 2], [0, 1, 3], [0, 4, 2], [5, 1, 2]]
    scope, states = predicted_evidence_scope_masks(
        entropy,
        mass,
        sampled,
        [0, 1, 2],
        native_margins=torch.tensor(
            [[1.2, 1.2, 1.2], [0.4, 0.4, 0.4], [0.1, 0.1, 0.1], [1.1, 1.1, 1.1]]
        ),
    )
    assert states.tolist() == [0, 1, 2, 0]
    assert not scope.requires_grad
    margin = torch.tensor([[1.2, 1.2, 1.2], [0.4, 0.4, 0.4], [0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])
    margin_scope, margin_states = predicted_evidence_scope_masks(
        entropy, mass, sampled, [0, 1, 2], native_margins=margin,
        confident_margin=1.0, ambiguous_margin=0.25,
    )
    assert margin_states.tolist() == [0, 1, 2, 0]
    assert margin_scope.shape == scope.shape
    current = torch.zeros_like(scope, requires_grad=True)
    loss, ratio, _ = clipped_scope_policy_loss(
        current, torch.zeros_like(scope), torch.tensor([1.0, -1.0, 0.5, 0.2]), scope, 0.2
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(ratio).all()


def test_pes_scope_loss_is_token_local_and_advantage_detached():
    current = torch.tensor(
        [[0.20, -0.10, 0.30], [0.05, 0.40, -0.20]],
        requires_grad=True,
    )
    behavior = (current.detach() - 0.05)
    advantages = torch.tensor([2.0, -1.0], requires_grad=True)
    action_mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

    loss, ratio, clip_fraction = clipped_scope_policy_loss(
        current,
        behavior,
        advantages,
        action_mask,
        0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(ratio).all()
    assert torch.isfinite(clip_fraction)
    assert advantages.grad is None
    assert current.grad is not None
    assert current.grad[0, 1:].abs().sum() == 0
    assert current.grad[1, 0].abs() == 0
    assert current.grad[0, 0].abs() > 0
    assert current.grad[1, 1:].abs().sum() > 0


def test_pes_shuffle_changes_only_state_assignment():
    entropy = torch.full((4, 3), 0.1)
    mass = torch.full_like(entropy, 0.5)
    margin = torch.full_like(entropy, 1.2)
    sampled = [[0, 1, 9], [8, 1, 2], [0, 7, 2], [0, 1, 2]]
    kwargs = dict(
        controlled_entropies=entropy,
        top_support_masses=mass,
        sampled_codes=sampled,
        native_codes=[0, 1, 2],
        native_margins=margin,
    )
    normal_scope, normal_states = predicted_evidence_scope_masks(**kwargs)
    shuffled_scope, shuffled_states = predicted_evidence_scope_masks(
        **kwargs, shuffle_seed=1907
    )
    generator = torch.Generator()
    generator.manual_seed(1907)
    expected_states = normal_states[torch.randperm(4, generator=generator)]
    assert torch.equal(shuffled_states, expected_states)
    assert shuffled_scope.shape == normal_scope.shape
    assert shuffled_scope.requires_grad is False


def test_pes_rollout_margin_is_sampled_action_aware():
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    # Guard the registered evidence semantics against a regression to the
    # native top-1/top-2 proxy, which cannot identify the sampled action.
    assert "sampled_logit = candidate_logits.gather" in source
    assert "native_logit - sampled_logit" in source
    assert "top_two[:, 0] - top_two[:, 1]" not in source
