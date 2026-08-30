from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.ampcpo_contract import validate_continued_adapter
from projects.samtok_selective.config import REPO_ROOT
from projects.samtok_selective.gr_cppo_contract import (
    FROZEN_ANCHOR_MODEL_SHA256,
    FROZEN_ANCHOR_TOTAL_STEPS,
    METHOD,
    POLICY_EPOCHS,
    ROLLOUTS_PER_PROMPT,
    STAGES,
    expected_frozen_anchor,
    validate_frozen_anchor,
    validate_gr_cppo_config,
)
from projects.samtok_selective.manifests import assert_training_source_clean


FARO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = FARO_ROOT / "Sa2VA" / "projects" / "samtok_selective"
CONFIG_ROOT = PACKAGE_ROOT / "configs"


@pytest.mark.parametrize(
    ("filename", "stage", "steps"),
    (
        ("fepo_gr_cppo_one_step_2gpu.py", "fepo_gr_cppo_one_step_2gpu", 1),
        ("fepo_gr_cppo_20step_2gpu.py", "fepo_gr_cppo_20step_2gpu", 20),
    ),
)
def test_gr_cppo_configs_are_locked_to_registered_stages(
    monkeypatch, tmp_path, filename, stage, steps
):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(str(CONFIG_ROOT / filename))["config"]
    validate_gr_cppo_config(config, REPO_ROOT)

    assert config["stage"] == stage
    assert config["optimizer"]["max_steps"] == STAGES[stage] == steps
    assert config["optimizer"]["grad_accum_steps"] == 1
    assert config["runtime"]["expected_world_size"] == 2
    assert Path(config["checkpoint"]["output_dir"]).resolve() == (
        REPO_ROOT / "outputs" / "samtok_selective" / stage
    )
    method = config["gr_cppo"]
    assert method["method"] == METHOD
    assert method["rollouts_per_prompt"] == ROLLOUTS_PER_PROMPT == 4
    assert method["policy_epochs"] == POLICY_EPOCHS == 2
    assert method["multimodal_batching"] == "processor_reencode_one_image_per_rollout"
    assert method["behavior_logprob"] == "detached_rollout_policy"
    assert method["ppo_action_logprob_scope"] == "sampled_depth_specific_code_tokens_only"
    assert method["forced_boundary_probability"] == 1.0
    assert method["advantage"] == "group_standardized"
    assert method["negative_objective"] == "canonical_no_target_ce"
    assert method["margin_constraint"] == "first_null_token_vs_mask_start_hinge"
    assert method["require_nonconstant_rewards"] is True
    assert method["require_epoch2_ratio_change"] is True
    assert "continued_sft_to500/adapter" in config["checkpoint"]["adapter_init"]
    assert FROZEN_ANCHOR_TOTAL_STEPS == 500
    assert FROZEN_ANCHOR_MODEL_SHA256.startswith("7b409c9f")


def _make_adapter(path: Path, *, status: str = "finished") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path.parent / "metrics.json").write_text(
        json.dumps(
            {"stage": "continued_sft", "status": status, "steps_completed": 100}
        ),
        encoding="utf-8",
    )
    (path.parent / "provenance_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "base_checkpoint": "/checkpoints/SAMTok/base"}
        ),
        encoding="utf-8",
    )
    return path


def test_gr_cppo_requires_finished_provenanced_parent_adapter(tmp_path):
    adapter = _make_adapter(tmp_path / "continued_sft" / "adapter")
    identity = validate_continued_adapter(
        adapter, repo_root=tmp_path, expected_path=adapter
    )
    assert identity["parent_stage"] == "continued_sft"
    assert identity["parent_steps_completed"] == 100
    assert len(identity["adapter_model_sha256"]) == 64

    unfinished = _make_adapter(
        tmp_path / "continued_sft_running" / "adapter", status="running"
    )
    with pytest.raises(ValueError, match="status=finished"):
        validate_continued_adapter(
            unfinished, repo_root=tmp_path, expected_path=unfinished
        )


def test_gr_cppo_frozen_total500_anchor_identity():
    anchor = expected_frozen_anchor(REPO_ROOT)
    identity = validate_frozen_anchor(anchor, repo_root=REPO_ROOT, hash_model=False)
    assert identity["anchor_stage"] == "continued_sft_to500"
    assert identity["registered_total_steps"] == 500
    assert identity["adapter_config_sha256"] == (
        "862495c04a30965280d7ce18f199297f698e6403bfeda522feb8ab449cb66afa"
    )
    assert identity["adapter_model_sha256"] is None


def test_gr_cppo_source_is_standalone():
    assert_training_source_clean(PACKAGE_ROOT)
    trainer = (PACKAGE_ROOT / "fepo_gr_cppo_trainer.py").read_text(encoding="utf-8")
    assert "projects.pixvl_" not in trainer.lower()
    assert "counterfactual" not in trainer.lower()
    assert '"sampled_depth_specific_code_tokens_only"' in trainer
    assert '"forced_boundary_probability": 1.0' in trainer


def test_gr_cppo_submit_scripts_lock_dnacoding_resources_and_tags():
    common_path = FARO_ROOT / "scripts" / "submit_samtok_gr_cppo.sh"
    common = common_path.read_text(encoding="utf-8")
    assert "JOB_NAME must start with dna-" in common
    assert "--namespace=ailab-dnacoding" in common
    assert "--gpu=2" in common
    assert "--nproc_per_node=2" in common
    assert '--positive-tags="$POSITIVE_TAGS"' in common
    assert "rjob_tags.txt" in common
    assert "continued_sft_to500/adapter" in common
    assert "gr_cppo_contract" in common
    assert "--skip-model-hash" in common
    assert "fepo_gr_cppo_trainer" in common
    assert "projects.pixvl_" not in common.lower()

    wrappers = {
        "submit_samtok_gr_cppo_one_step.sh": (
            "fepo_gr_cppo_one_step_2gpu.py",
            "dna-samtok-fepo-gr-cppo-one-step-2g",
        ),
        "submit_samtok_gr_cppo_20step.sh": (
            "fepo_gr_cppo_20step_2gpu.py",
            "dna-samtok-fepo-gr-cppo-20step-2g",
        ),
    }
    for filename, (config, job_name) in wrappers.items():
        text = (FARO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert config in text
        assert job_name in text
        assert 'exec bash "$SCRIPT_DIR/submit_samtok_gr_cppo.sh"' in text
