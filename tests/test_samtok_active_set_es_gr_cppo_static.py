from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.active_set_gr_cppo_contract import (
    METHOD,
    STAGE,
    TWENTY_STEP_STAGE,
    active_set_flags,
    derive_active_set_budgets,
    validate_active_set_gr_cppo_config,
)
from projects.samtok_selective.config import REPO_ROOT


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = FARO_ROOT / (
    "Sa2VA/projects/samtok_selective/configs/"
    "fepo_active_set_es_gr_cppo_one_step_2gpu.py"
)


def test_active_set_one_step_config_is_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(str(CONFIG))["config"]
    validate_active_set_gr_cppo_config(config, REPO_ROOT)
    method = config["active_set_entropy_gr_cppo"]
    assert config["stage"] == STAGE
    assert config["optimizer"]["max_steps"] == 1
    assert method["method"] == METHOD
    assert method["selective_risk_mode"] == "fixed_training_sentinel_active_set"
    assert method["sentinel_source"] == "sorted_training_no_target_ids"
    assert method["sentinel_rows_total"] == 8
    assert method["anchor_budget_source"] == "frozen_initialization_pre_update"
    assert method["holdout_access"] is False
    assert method["positive_reward"] == "plain_ciou"
    assert method["exploration"] == "per_prefix_topm_collision_support"


def test_active_set_20step_config_is_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG.with_name("fepo_active_set_es_gr_cppo_20step_2gpu.py")
    config = runpy.run_path(str(path))["config"]
    validate_active_set_gr_cppo_config(config, REPO_ROOT)
    assert config["stage"] == TWENTY_STEP_STAGE
    assert config["optimizer"]["max_steps"] == 20
    assert config["checkpoint"]["adapter_init"].endswith(
        "continued_sft_to500/adapter"
    )


def test_active_set_budget_and_flags_are_bounded():
    ce_budget, margin_budget = derive_active_set_budgets(
        0.01,
        3.0,
        null_ce_relative_slack=0.1,
        null_ce_absolute_slack=0.001,
        margin_slack=0.25,
    )
    assert ce_budget == pytest.approx(0.012)
    assert margin_budget == pytest.approx(2.75)
    assert active_set_flags(
        0.011, 2.8, null_ce_budget=ce_budget, margin_budget=margin_budget
    ) == (False, False)
    assert active_set_flags(
        0.013, 2.7, null_ce_budget=ce_budget, margin_budget=margin_budget
    ) == (True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor_null_ce": -1.0, "anchor_margin_min": 1.0},
        {"anchor_null_ce": 0.1, "anchor_margin_min": float("nan")},
    ],
)
def test_active_set_rejects_invalid_budget_inputs(kwargs):
    with pytest.raises(ValueError):
        derive_active_set_budgets(
            **kwargs,
            null_ce_relative_slack=0.1,
            null_ce_absolute_slack=1e-6,
            margin_slack=0.25,
        )


def test_active_set_scripts_lock_dnacoding_and_positive_tags():
    one_step = (
        FARO_ROOT / "scripts/submit_samtok_active_set_es_gr_cppo_one_step.sh"
    ).read_text(encoding="utf-8")
    assert "JOB_NAME must start with dna-" in one_step
    assert "--namespace=ailab-dnacoding" in one_step
    assert "--gpu=2" in one_step
    assert "--nproc_per_node=2" in one_step
    assert '--positive-tags="$POSITIVE_TAGS"' in one_step
    assert "TAGS_FILE=$FARO_ROOT/rjob_tags.txt" in one_step
    assert "rjob_tags_16gpu_partition.txt" not in one_step
    assert "active_set_gr_cppo_contract" in one_step
    assert "projects.pixvl_" not in one_step.lower()
    wrapper = (
        FARO_ROOT / "scripts/submit_samtok_active_set_es_gr_cppo_20step.sh"
    ).read_text(encoding="utf-8")
    assert "fepo_active_set_es_gr_cppo_20step_2gpu.py" in wrapper
    assert "dna-samtok-fepo-active-set-es-20step-2g" in wrapper
