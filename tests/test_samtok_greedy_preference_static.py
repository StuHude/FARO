from __future__ import annotations

import runpy
from pathlib import Path

import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    grammar_sequence_from_codes,
    greedy_crossing_preference_loss,
)
from projects.samtok_selective.greedy_preference_gr_cppo_contract import (
    STAGE,
    TWENTY_STEP_STAGE,
    validate_greedy_preference_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_greedy_preference_configs_are_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for stage, steps in ((STAGE, 1), (TWENTY_STEP_STAGE, 20)):
        config = runpy.run_path(str(CONFIG_ROOT / f"{stage}.py"))["config"]
        validate_greedy_preference_gr_cppo_config(config)
        method = config["greedy_preference_entropy_gr_cppo"]
        assert config["optimizer"]["max_steps"] == steps
        assert method["positive_reward"] == "plain_ciou"
        assert method["minimum_improvement"] == 1e-4
        assert method["native_scoring_temperature"] == 1.0
        assert method["max_epoch0_ratio_deviation"] == 0.01
        assert "tail_gppo" not in config
        assert "improvement_entropy_gr_cppo" not in config


def test_greedy_sequence_and_preference_loss_have_the_registered_direction():
    grammar = (10, [[100, 101], [200, 201]], 11)
    sequence = grammar_sequence_from_codes([1, 3], grammar, torch.device("cpu"))
    assert sequence.tolist() == [[10, 101, 201, 11]]

    best = torch.tensor([-1.5], requires_grad=True)
    greedy = torch.tensor([-1.0], requires_grad=True)
    old_best = torch.tensor([-1.5])
    old_odds = torch.tensor([-0.5])
    loss, shift, ratio = greedy_crossing_preference_loss(
        best, greedy, old_best, old_odds, True
    )
    loss.backward()
    assert shift.item() == 0.0
    assert ratio.item() == 1.0
    assert best.grad.item() < 0.0
    assert greedy.grad.item() > 0.0
