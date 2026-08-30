from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    calibrated_rollout_uncertainty,
    uncertainty_calibrated_native_rank_local_geometry_advantages,
)
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE,
    validate_tail_gppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_r25_config_and_submit_are_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(
        str(CONFIG_ROOT / f"{UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE}.py")
    )["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "uncertainty_calibrated_native_rank_local"
    assert method["uncertainty_confidence_floor"] == 0.25
    submit = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_uncertainty_native_rank_local.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in submit
    assert "dna-fepo-uncertainty-native-rank-local-10step-2g" in submit
    assert "submit_samtok_tb_gppo.sh" in submit


def test_r25_uncertainty_credit_is_positive_only_and_confidence_weighted():
    entropy = torch.tensor([[0.1, 0.1], [1.0, 1.0], [0.2, 0.2], [0.2, 0.2]])
    mass = torch.tensor([[0.95, 0.95], [0.50, 0.50], [0.90, 0.90], [0.90, 0.90]])
    uncertainty = calibrated_rollout_uncertainty(entropy, mass, support_size=8)
    assert uncertainty.shape == (4,)
    assert torch.isfinite(uncertainty).all()
    assert uncertainty[1] > uncertainty[0]

    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1], [1, 1], [0, 0], [0, 0]]
    greedy = [0, 0]
    advantages = uncertainty_calibrated_native_rank_local_geometry_advantages(
        raw, native, sampled, greedy, uncertainty
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > advantages[1]
    assert advantages[2] == 0
    assert advantages[3] == 0
    with pytest.raises(ValueError, match="confidence_floor"):
        uncertainty_calibrated_native_rank_local_geometry_advantages(
            raw, native, sampled, greedy, uncertainty, confidence_floor=0.0
        )
