from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.config import REPO_ROOT
from projects.samtok_selective.entropy_gr_cppo_contract import (
    METHOD,
    STAGE,
    SUPPORT_SIZE,
    TARGET_EFFECTIVE_SUPPORT,
    TWENTY_STEP_STAGE,
    validate_entropy_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    FARO_ROOT
    / "Sa2VA/projects/samtok_selective/configs/fepo_entropy_gr_cppo_one_step_2gpu.py"
)


def test_es_gr_cppo_config_is_preregistered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(str(CONFIG))["config"]
    validate_entropy_gr_cppo_config(config, REPO_ROOT)
    method = config["entropy_gr_cppo"]
    assert config["stage"] == STAGE
    assert config["data"]["pairs_per_device_batch"] == 4
    assert method["method"] == METHOD
    assert method["support_size"] == SUPPORT_SIZE == 8
    assert method["target_effective_support"] == TARGET_EFFECTIVE_SUPPORT == 4.0
    assert method["exploration"] == "per_prefix_topm_collision_support"
    assert method["rescore_policy"] == "frozen_old_support_and_temperature"
    assert method["min_multitrajectory_groups"] == 6
    assert method["min_nonconstant_reward_groups"] == 2
    assert method["min_improved_over_greedy_rollouts"] == 1


def test_es_gr_cppo_20step_config_is_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG.with_name("fepo_entropy_gr_cppo_20step_2gpu.py")
    config = runpy.run_path(str(path))["config"]
    validate_entropy_gr_cppo_config(config, REPO_ROOT)
    assert config["stage"] == TWENTY_STEP_STAGE
    assert config["optimizer"]["max_steps"] == 20
    assert config["checkpoint"]["adapter_init"].endswith(
        "continued_sft_to500/adapter"
    )


def test_es_submit_script_locks_dnacoding_positive_tags():
    script = (FARO_ROOT / "scripts/submit_samtok_es_gr_cppo_one_step.sh").read_text(
        encoding="utf-8"
    )
    assert "JOB_NAME must start with dna-" in script
    assert "--namespace=ailab-dnacoding" in script
    assert "--gpu=2" in script
    assert "--nproc_per_node=2" in script
    assert '--positive-tags="$POSITIVE_TAGS"' in script
    assert "rjob_tags.txt" in script
    assert "continued_sft_to500/adapter" in script
    assert "entropy_gr_cppo_contract" in script
    assert "rows >= 5000" in script
    assert "One-step training jobs are disabled" in script
    assert "projects.pixvl_" not in script.lower()
    wrapper = (
        FARO_ROOT / "scripts/submit_samtok_es_gr_cppo_20step.sh"
    ).read_text(encoding="utf-8")
    assert "fepo_entropy_gr_cppo_20step_2gpu.py" in wrapper
    assert "dna-samtok-fepo-es-gr-cppo-20step-2g" in wrapper
