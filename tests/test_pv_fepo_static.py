from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_PAIRED_VIEW_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu.py"


def test_pv_config_is_fixed_and_samtok_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    anchor = ROOT / "outputs/samtok_selective/continued_sft_to500/adapter"
    monkeypatch.setenv("SAMTOK_STANDALONE_ADAPTER", str(anchor))
    config = runpy.run_path(str(CONFIG))["config"]
    validate_tail_gppo_config(config, ROOT)
    assert config["stage"] == UNIFIED_PAIRED_VIEW_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert config["tail_gppo"]["rollouts_per_prompt"] == 4
    assert config["tail_gppo"]["paired_view_geometry"]["aggregation"] == "geometric_mean"


def test_pv_trainer_uses_gt_verified_dual_view_credit_without_teacher():
    source = (ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "paired_view_native_rank_local_geometry_advantages" in source
    assert "build_target_preserving_view_samples" in source
    assert "augmented_geometry" in source
    assert "same_row_ground_truth_mask_geometry" in source


def test_pv_wrapper_has_resource_and_data_guards():
    source = (ROOT / "scripts/submit_samtok_tb_gppo_paired_view.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in source
    assert "dna-fepo-paired-view-10step-2g" in source
    assert "submit_samtok_tb_gppo.sh" in source
