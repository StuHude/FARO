from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("apes_probe", ROOT / "tools/apes_probe.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
probability_gap_scope_masks = MODULE.probability_gap_scope_masks


def test_probability_gap_scope_contract_and_detach():
    entropy = torch.tensor([[0.1, 0.2], [0.6, 0.7], [2.0, 2.0]])
    native = torch.tensor([[0.8, 0.8], [0.7, 0.7], [0.6, 0.6]], requires_grad=True)
    sampled = torch.tensor([[0.75, 0.75], [0.50, 0.50], [0.0, 0.0]], requires_grad=True)
    scope, states, gap = probability_gap_scope_masks(
        entropy, native, sampled, [[0, 4], [4, 5], [0, 1]], [0, 1]
    )
    assert states.tolist() == [0, 1, 2]
    assert scope.tolist() == [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    assert torch.allclose(gap, native.detach() - sampled.detach())
    assert not scope.requires_grad and not states.requires_grad and not gap.requires_grad


def test_probability_gap_shuffle_is_seeded_state_only():
    entropy = torch.full((4, 3), 0.1)
    native = torch.full((4, 3), 0.8)
    sampled = torch.full((4, 3), 0.75)
    codes = [[0, 1, 9], [8, 1, 2], [0, 7, 2], [0, 1, 2]]
    kwargs = dict(entropies=entropy, native_probabilities=native, sampled_probabilities=sampled,
                  sampled_codes=codes, native_codes=[0, 1, 2])
    _, normal, _ = probability_gap_scope_masks(**kwargs)
    _, shuffled, _ = probability_gap_scope_masks(**kwargs, shuffle_seed=1907)
    generator = torch.Generator()
    generator.manual_seed(1907)
    assert torch.equal(shuffled, normal[torch.randperm(4, generator=generator)])
