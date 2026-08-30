from __future__ import annotations

import runpy
from pathlib import Path

import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    sign_balanced_greedy_advantages,
)
from projects.samtok_selective.sign_balanced_gr_cppo_contract import (
    STAGE,
    TWENTY_STEP_STAGE,
    validate_sign_balanced_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_sign_balanced_configs_are_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for stage, steps in ((STAGE, 1), (TWENTY_STEP_STAGE, 20)):
        config = runpy.run_path(str(CONFIG_ROOT / f"{stage}.py"))["config"]
        validate_sign_balanced_gr_cppo_config(config)
        method = config["sign_balanced_entropy_gr_cppo"]
        assert config["optimizer"]["max_steps"] == steps
        assert method["advantage"] == "greedy_delta_equal_sign_l1_mass"
        assert method["advantage_epsilon"] == 1e-6


def test_sign_balanced_advantages_equalize_l1_mass_and_total_scale():
    advantages = sign_balanced_greedy_advantages(
        torch.tensor([0.9, 0.61, 0.5, 0.3]), torch.tensor(0.6)
    )
    assert torch.allclose(advantages[advantages > 0].sum(), torch.tensor(2.0))
    assert torch.allclose(-advantages[advantages < 0].sum(), torch.tensor(2.0))
    assert torch.allclose(advantages.abs().mean(), torch.tensor(1.0))
