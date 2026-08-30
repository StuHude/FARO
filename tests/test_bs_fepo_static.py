from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_geometry import select_registered_ids
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_BOUNDARY_STRATIFIED_STAGE,
    validate_tail_gppo_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bs_config_is_fixed_and_samtok_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(
        str(
            ROOT
            / "Sa2VA/projects/samtok_selective/configs/"
            f"{UNIFIED_BOUNDARY_STRATIFIED_STAGE}.py"
        )
    )["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["boundary_stratified_schedule"] is True
    assert method["boundary_sampling_mix"] == {
        "ordinary": 0.50,
        "thin": 0.25,
        "boundary_hard": 0.25,
    }
    wrapper = (ROOT / "scripts/submit_samtok_tb_gppo_boundary_stratified_native_rank_local.sh").read_text()
    assert "rows >= 5000" in wrapper
    assert "dna-fepo-boundary-stratified-native-rank-local-10step-2g" in wrapper


def test_bs_schedule_is_disjoint_and_50_25_25():
    records = {}
    for i in range(420):
        if i < 120:
            flags = {"boundary_hard": True, "thin": False, "hard_geometry": True}
        elif i < 240:
            flags = {"boundary_hard": False, "thin": True, "hard_geometry": True}
        else:
            flags = {"boundary_hard": False, "thin": False, "hard_geometry": False}
        records[f"id-{i:03d}"] = flags
    schedule = select_registered_ids(
        {"records": records}, boundary_stratified=True, schedule_per_stratum=80
    )
    assert len(schedule["batches"]) == 80
    for batch in schedule["batches"]:
        assert sum(records[x]["boundary_hard"] for x in batch) == 1
        assert sum(records[x]["thin"] for x in batch) == 1
        assert sum(not records[x]["hard_geometry"] for x in batch) == 2
    ids = schedule["schedule_pair_ids"]
    assert len(ids) == len(set(ids))


def test_bs_schedule_handles_fully_overlapping_real_geometry():
    records = {}
    for i in range(2560):
        overlap = i < 640
        records[f"id-{i:04d}"] = {
            "thin": overlap,
            "boundary_hard": overlap,
            "hard_geometry": overlap,
        }
    schedule = select_registered_ids(
        {"records": records}, boundary_stratified=True, schedule_per_stratum=80
    )
    assert len(schedule["batches"]) == 80
    assignment = schedule["boundary_stratum_assignment"]
    for batch in schedule["batches"]:
        assert [assignment[x] for x in batch].count("thin") == 1
        assert [assignment[x] for x in batch].count("boundary_hard") == 1
        assert [assignment[x] for x in batch].count("ordinary") == 2
