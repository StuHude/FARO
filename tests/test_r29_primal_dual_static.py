from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_primal_dual_null_risk_10step_2gpu.py"


def test_r29_contract_and_submission_guards(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    cfg = runpy.run_path(str(CONFIG))["config"]
    validate_tail_gppo_config(cfg)
    method = cfg["tail_gppo"]
    assert cfg["stage"] == UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE
    assert cfg["data"]["expected_rows"] == 5120
    assert cfg["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "primal_dual_null_risk_native_rank_local"
    assert method["primal_dual_risk"] == "lower10_current_minus_anchor_margin_excess"
    assert method["primal_dual_lambda_init"] == 1.0
    assert method["primal_dual_eta"] == 0.20
    assert method["primal_dual_lambda_cap"] == 4.0
    submit = (ROOT / "scripts/submit_samtok_tb_gppo_primal_dual_null_risk.sh").read_text()
    assert "rows >= 5000" in submit
    assert "dna-fepo-primal-dual-null-risk-10step-2g" in submit
    assert "submit_samtok_tb_gppo.sh" in submit


def test_r29_dual_update_sequence_is_bounded_and_inactive_when_safe():
    anchor_q10, slack = 0.80, 0.05
    observed = [0.72, 0.73, 0.76, 0.78, 0.74, 0.76, 0.71, 0.77]
    lam = 1.0
    history = []
    for q10 in observed:
        excess = max(0.0, anchor_q10 - slack - q10) / slack
        lam = min(4.0, lam + 0.20 * excess)
        history.append(round(lam, 6))
    assert history == [1.12, 1.20, 1.20, 1.20, 1.24, 1.24, 1.40, 1.40]
    safe = 1.0
    for _ in range(8):
        excess = max(0.0, anchor_q10 - slack - 0.80) / slack
        safe = min(4.0, safe + 0.20 * excess)
    assert safe == 1.0
