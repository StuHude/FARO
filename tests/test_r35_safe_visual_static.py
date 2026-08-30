from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_SAFE_VISUAL_INTERFACE_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu.py"


def test_r35_safe_visual_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.setenv(
        "SAMTOK_STANDALONE_ADAPTER",
        str(ROOT / "outputs/samtok_selective/continued_sft_to500/adapter"),
    )
    config = runpy.run_path(str(CONFIG))["config"]
    validate_tail_gppo_config(config, ROOT)
    assert config["stage"] == UNIFIED_SAFE_VISUAL_INTERFACE_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert config["model"]["adapter_mode"] == "frozen_anchor_plus_visual_projector"
    assert config["tail_gppo"]["null_ce_weight"] == 2.0
    assert config["tail_gppo"]["margin_weight"] == 1.0


def test_r35_submission_guards_tags_and_budget():
    source = (ROOT / "scripts/submit_samtok_tb_gppo_safe_visual_interface.sh").read_text()
    assert "rows >= 5000" in source
    assert "dna-fepo-safe-visual-interface-10step-2g" in source
    assert "submit_samtok_tb_gppo.sh" in source


def test_r35_is_registered_in_probe_and_late_monitor():
    probe = (ROOT / "tools/run_fepo_candidate_probe.py").read_text()
    monitor = (ROOT / "scripts/monitor_fepo_late_screens.sh").read_text()
    assert '"R35": "projects.samtok_selective.configs.' in probe
    assert '"R35": "scripts/submit_samtok_tb_gppo_safe_visual_interface.sh"' in probe
    assert "safe_visual_interface" in monitor


def test_r35_trigger_waits_for_both_closed_pv_comparisons():
    source = (ROOT / "scripts/submit_r35_after_pv_decision.sh").read_text()
    assert "paired_view_vs_r18_bootstrap20k.json" in source
    assert "paired_view_vs_matched_sft_bootstrap20k.json" in source
    assert "paired_view_vs_r18_slices20k.json" in source
    assert 'int(slice_report.get("num_paired", 0)) != 512' in source
    assert 'not bool(slice_report.get("slice_gate", False))' in source
    assert "ci_corrected_promotion_gate" in source
    assert "num_paired" in source
    assert "bootstrap_repeats" in source
    assert "--namespace=ailab-dnacoding" in source
    assert "pv_training_gate.json" in source
    assert "closed_training_gate" in source
    assert "r18_matched_sft_vs_r18_bootstrap20k.json" in source
    assert "control_plane_unavailable" in source
    assert "rjob list --namespace=ailab-dnacoding >/dev/null 2>&1" in source


def test_finalizer_does_not_treat_missing_pv_as_complete():
    source = (ROOT / "scripts/monitor_finalize_matched_sft_pv.sh").read_text()
    assert 'PV_R18="$FARO_ROOT/evals/paired_view_vs_r18_bootstrap20k.json"' in source
    assert 'PV_SFT="$FARO_ROOT/evals/paired_view_vs_matched_sft_bootstrap20k.json"' in source
    assert 'PV_DECISION="$FARO_ROOT/evals/pv_training_gate.json"' in source
    assert 'SFT_R18="$FARO_ROOT/evals/r18_matched_sft_vs_r18_bootstrap20k.json"' in source
    assert 'while [[ ! -s "$SFT_R18" || ( ! -s "$PV_R18" && ! -s "$PV_DECISION" ) ]]; do' in source
    assert 'if [[ -s "$SFT_R18" && ( -s "$PV_R18" || -s "$PV_DECISION" ) ]]; then' in source
    helper = (ROOT / "scripts/finalize_matched_sft_pv.sh").read_text()
    assert 'echo "waiting for $PV"' in helper
    assert "exit 3" in helper
