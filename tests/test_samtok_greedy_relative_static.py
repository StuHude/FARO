from __future__ import annotations

import runpy
from pathlib import Path

import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import greedy_relative_advantages
from projects.samtok_selective.greedy_relative_gr_cppo_contract import (
    STAGE,
    TWENTY_STEP_STAGE,
    validate_greedy_relative_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_greedy_relative_configs_are_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for stage, steps in ((STAGE, 1), (TWENTY_STEP_STAGE, 20)):
        config = runpy.run_path(str(CONFIG_ROOT / f"{stage}.py"))["config"]
        validate_greedy_relative_gr_cppo_config(config)
        method = config["greedy_relative_entropy_gr_cppo"]
        assert config["optimizer"]["max_steps"] == steps
        assert method["advantage"] == "greedy_reward_delta_mean_abs_normalized"
        assert method["advantage_epsilon"] == 1e-6
        assert method["positive_reward"] == "plain_ciou"
        assert "greedy_preference_entropy_gr_cppo" not in config


def test_greedy_relative_advantages_keep_both_signs_and_unit_mean_magnitude():
    advantages = greedy_relative_advantages(
        torch.tensor([0.9, 0.7, 0.5, 0.3]), torch.tensor(0.6)
    )
    assert bool((advantages > 0).any())
    assert bool((advantages < 0).any())
    assert torch.allclose(advantages.abs().mean(), torch.tensor(1.0))
