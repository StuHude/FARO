from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_PAIRED_VIEW_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu.py"


def test_paired_view_contract_and_submission_guards(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    cfg = runpy.run_path(str(CONFIG))["config"]
    validate_tail_gppo_config(cfg)
    assert cfg["stage"] == UNIFIED_PAIRED_VIEW_STAGE
    assert cfg["data"]["expected_rows"] == 5120
    assert cfg["optimizer"]["max_steps"] == 10
    paired = cfg["tail_gppo"]["paired_view_geometry"]
    assert paired["aggregation"] == "geometric_mean"
    assert paired["brightness"] == 1.03
    assert paired["contrast"] == 0.97
    assert paired["uses_pixvl_teacher"] is False
    submit = (ROOT / "scripts/submit_samtok_tb_gppo_paired_view.sh").read_text()
    assert "rows >= 5000" in submit
    assert "dna-fepo-paired-view-10step-2g" in submit
    assert "submit_samtok_tb_gppo.sh" in submit


def test_paired_view_implementation_is_gt_verified_and_localized():
    trainer = (ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text()
    assert trainer.count("def paired_view_native_rank_local_geometry_advantages") == 1
    assert "build_target_preserving_view_samples" in trainer
    assert "augmented_geometry" in trainer
    assert "augmented_native_codes" in trainer
    assert "paired_view_reward_correlation" in trainer
    assert "paired_view_joint_positive_fraction" in trainer
    assert "native_reference_midrank_first_divergence" in trainer
