from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "Sa2VA/projects/samtok_selective/configs/"
    "fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu.py"
)


def test_ba_config_keeps_fixed_samtok_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    cfg = runpy.run_path(str(CONFIG))["config"]
    validate_tail_gppo_config(cfg)
    assert cfg["stage"] == UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE
    assert cfg["data"]["expected_rows"] == 5120
    assert cfg["optimizer"]["max_steps"] == 10
    paired = cfg["tail_gppo"]["paired_view_geometry"]
    assert paired["aggregation"] == "boundary_bottleneck_min"
    assert paired["mode"] == "gt_verified_boundary_bottleneck_paired_view_reward"
    assert paired["uses_pixvl_teacher"] is False


def test_ba_wrapper_preserves_submission_guards():
    source = (ROOT / "scripts/submit_samtok_tb_gppo_boundary_bottleneck_paired_view.sh").read_text()
    assert "rows >= 5000" in source
    assert "dna-fepo-boundary-bottleneck-paired-view-10step-2g" in source
    assert "submit_samtok_tb_gppo.sh" in source
    monitor = (ROOT / "scripts/monitor_fepo_late_screens.sh").read_text()
    assert "boundary_bottleneck_paired_view" in monitor


def test_ba_trainer_uses_minimum_of_two_verified_credits():
    source = (ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text()
    assert "def boundary_bottleneck_paired_view_geometry_advantages" in source
    assert "torch.minimum(clean.clamp_min(0.0), augmented.clamp_min(0.0))" in source
    assert "boundary_bottleneck_paired_view_enabled" in source
