from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    action_budget_native_rank_local_geometry_advantages,
)
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE,
    validate_tail_gppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_ab_fepo_config_and_wrapper_are_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(
        str(
            CONFIG_ROOT
            / f"{UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE}.py"
        )
    )["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["rollouts_per_prompt"] == 4
    assert method["action_budget"] == 2
    assert method["action_budget_excess_penalty"] == 0.10
    text = (
        FARO_ROOT
        / "scripts/submit_samtok_tb_gppo_action_budget_native_rank_local.sh"
    ).read_text(encoding="utf-8")
    assert "rows >= 5000" in text
    assert "dna-fepo-action-budget-native-rank-local-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text


def test_ab_fepo_penalizes_only_over_budget_joint_improvements():
    raw = torch.tensor(
        [[0.80, 0.80], [0.79, 0.79], [0.78, 0.78], [0.80, 0.70]]
    )
    native = torch.tensor([0.70, 0.70])
    sampled = [
        [1, 1, 0, 0],  # two changes, no excess penalty
        [1, 1, 1, 0],  # three changes, discounted
        [1, 1, 1, 1],  # four changes, discounted more
        [1, 0, 0, 0],  # mixed geometry, no credit
    ]
    greedy = [0, 0, 0, 0]
    advantages = action_budget_native_rank_local_geometry_advantages(
        raw, native, sampled, greedy
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > advantages[1] > advantages[2] > 0
    assert advantages[3] == 0
    with pytest.raises(ValueError, match="action_budget"):
        action_budget_native_rank_local_geometry_advantages(
            raw, native, sampled, greedy, action_budget=5
        )
