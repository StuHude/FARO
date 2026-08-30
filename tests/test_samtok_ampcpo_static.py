from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.ampcpo_contract import (
    EXPECTED_STEPS,
    STAGE,
    validate_ampcpo_config,
    validate_continued_adapter,
)
from projects.samtok_selective.config import REPO_ROOT
from projects.samtok_selective.manifests import assert_training_source_clean


FARO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = FARO_ROOT / "Sa2VA" / "projects" / "samtok_selective"


def test_ampcpo_config_is_locked_to_registered_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config_path = PACKAGE_ROOT / "configs" / "fepo_ampcpo_smoke_2gpu.py"
    config = runpy.run_path(str(config_path))["config"]
    validate_ampcpo_config(config, REPO_ROOT)
    assert config["stage"] == STAGE
    assert config["optimizer"]["max_steps"] == EXPECTED_STEPS == 20
    assert config["runtime"]["expected_world_size"] == 2
    assert config["ampcpo"]["positive_reward"] == "plain_ciou"
    assert config["ampcpo"]["negative_objective"] == "canonical_no_target_ce"
    assert config["ampcpo"]["policy_surrogate"] == "teacher_forced_greedy_mask_action_cppo"
    assert config["ampcpo"]["margin_constraint"] == "first_null_token_vs_mask_start_hinge"


def _make_adapter(path: Path, status: str = "finished") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path.parent / "metrics.json").write_text(
        json.dumps({"stage": "continued_sft", "status": status, "steps_completed": 100}),
        encoding="utf-8",
    )
    (path.parent / "provenance_manifest.json").write_text(
        json.dumps({"schema_version": 1, "base_checkpoint": "/checkpoints/SAMTok/base"}),
        encoding="utf-8",
    )
    return path


def test_ampcpo_parent_adapter_requires_finished_provenance(tmp_path):
    adapter = _make_adapter(tmp_path / "continued_sft" / "adapter")
    identity = validate_continued_adapter(
        adapter, repo_root=tmp_path, expected_path=adapter
    )
    assert identity["parent_stage"] == "continued_sft"
    assert identity["parent_steps_completed"] == 100
    assert len(identity["adapter_model_sha256"]) == 64

    unfinished = _make_adapter(tmp_path / "unfinished" / "adapter", status="running")
    with pytest.raises(ValueError, match="status=finished"):
        validate_continued_adapter(
            unfinished, repo_root=tmp_path, expected_path=unfinished
        )


def test_ampcpo_source_and_submit_contracts_are_standalone():
    assert_training_source_clean(PACKAGE_ROOT)
    trainer = (PACKAGE_ROOT / "fepo_ampcpo_trainer.py").read_text(encoding="utf-8")
    submit = (FARO_ROOT / "scripts" / "submit_samtok_ampcpo_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "projects.pixvl_" not in trainer.lower()
    assert "projects.pixvl_" not in submit.lower()
    assert "--namespace=ailab-dnacoding" in submit
    assert "--positive-tags=\"$POSITIVE_TAGS\"" in submit
    assert "rjob_tags.txt" in submit
    assert "--nproc_per_node=2" in submit
    assert "continued_sft/adapter" in submit
