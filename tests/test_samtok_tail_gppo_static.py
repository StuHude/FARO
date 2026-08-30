from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.tail_gppo_contract import (
    ARMS,
    METHOD,
    UNIFIED_STAGE,
    UNIFIED_DEPTH_LOCAL_STAGE,
    UNIFIED_NATIVE_RANK_PARETO_STAGE,
    UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE,
    UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE,
    UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE,
    UNIFIED_NATIVE_RANK_LOCAL_STAGE,
    UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE,
    UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE,
    UNIFIED_ANCHOR_KL_STAGE,
    UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE,
    UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE,
    UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE,
    validate_tail_gppo_config,
)
from projects.samtok_selective.fepo_gr_cppo_trainer import (
    asymmetric_signed_native_relative_depth_local_advantages,
    depth_local_geometry_advantages,
    shuffled_depth_local_geometry_advantages,
    scale_stratified_native_rank_local_geometry_advantages,
    bidirectional_coarse_fine_native_geometry_advantages,
    anchor_categorical_kl,
    confidence_gated_native_rank_local_geometry_advantages,
    margin_calibrated_native_rank_local_geometry_advantages,
    soft_native_dominance_depth_local_geometry_advantages,
)
from projects.samtok_selective.tail_geometry import select_registered_ids
import torch


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_all_one_step_arms_share_registered_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for arm in sorted(ARMS):
        path = CONFIG_ROOT / f"fepo_tb_gppo_{arm}_one_step_2gpu.py"
        config = runpy.run_path(str(path))["config"]
        validate_tail_gppo_config(config)
        method = config["tail_gppo"]
        assert method["method"] == METHOD
        assert method["arm"] == arm
        assert method["fifo_capacity"] == 16
        assert method["boundary_width"] == 2
        assert method["sentinel_rows_total"] == 32
        assert config["optimizer"]["max_steps"] == 1


def test_submit_script_uses_only_tagged_dnacoding_samtok():
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo.sh").read_text(
        encoding="utf-8"
    )
    assert "--namespace=ailab-dnacoding" in text
    assert '--positive-tags="$POSITIVE_TAGS"' in text
    assert "JOB_NAME must start with dna-" in text
    assert "--gpu=2" in text
    assert "--nproc_per_node=2" in text
    assert "TAGS_FILE=$FARO_ROOT/rjob_tags.txt" in text
    assert "rjob_tags_16gpu_partition.txt" not in text
    assert "continued_sft_to500/adapter" in text
    assert "tail_gppo_contract" in text
    assert "rows >= 5000" in text
    assert "one-step budgets are disabled" in text
    assert "projects.pixvl_" not in text.lower()


def test_tail_risk_gate_uses_post_optimizer_final_measurement():
    text = (
        FARO_ROOT
        / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    optimizer_step = text.index("                optimizer.step()")
    final_measurement = text.index("    final_tail_margin_delta_q10: float | None")
    gate = text.index("    tail_risk_gate_passed = True", final_measurement)
    assert optimizer_step < final_measurement < gate
    gate_text = text[gate : text.index("    gate_passed = (", gate)]
    assert "final_tail_margin_delta_q10" in gate_text
    assert "final_tail_margin_violation_rate" in gate_text
    assert "tail_mean_violation_rate < 0.05" not in gate_text


def test_unified_sentinel_stage_uses_registered_single_buffer(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / "fepo_tb_gppo_tail_balanced_unified_sentinel_10step_2gpu.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_STAGE
    assert config["optimizer"]["max_steps"] == 10
    assert config["data"]["expected_rows"] >= 5000
    assert config["data"]["expected_no_target_rows"] >= 2500
    assert method["unified_sentinel"] is True
    assert method["sentinel_rows_total"] == 32
    assert method["sentinel_source"] == "registered_tail_no_target_ids"
    assert method["holdout_access"] is False


def test_unified_sentinel_has_one_fixed_shape_measurement_path():
    text = (FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text(
        encoding="utf-8"
    )
    start = text.index("if unified_sentinel_enabled:")
    block = text[start : text.index("elif active_set_enabled:", start)]
    assert "global_stats = accelerator.gather(local_stats)" in block
    assert "evaluate_null_margins" not in block
    assert "global_selected = int(" in block


def test_native_rank_pareto_geometry_screen_is_strictly_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / "fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_NATIVE_RANK_PARETO_STAGE
    assert config["data"]["expected_rows"] >= 5000
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "native_anchored_rank_pareto"
    assert method["rank_pareto_tie_policy"] == "native_reference_midrank_geomean"


def test_native_rank_pareto_submit_wrapper_keeps_data_and_job_guards():
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_native_rank_pareto.sh").read_text(
        encoding="utf-8"
    )
    assert "fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu.py" in text
    assert "egfepo_train_5120.jsonl" in text
    assert "rows >= 5000" in text
    assert "dna-fepo-native-rank-pareto-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text


def test_native_rank_pareto_credit_is_detached_advantage_only():
    text = (
        FARO_ROOT
        / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    start = text.index("def native_anchored_rank_pareto_advantages(")
    block = text[start : start + 1800]
    assert "explicit reference point" in block
    assert "torch.sqrt" in block
    assert "raw_geometry" in block
    assert "native_geometry" in block


def test_native_rank_pareto_computes_clean_native_reference_geometry():
    text = (
        FARO_ROOT
        / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    start = text.index("if (\n                    boundary_credit_enabled")
    block = text[start : text.index("native_greedy_codes", start)]
    assert "native_rank_pareto_enabled" in block


def test_depth_local_geometry_config_and_wrapper_are_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / "fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_DEPTH_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "depth_local_geometry"
    assert method["depth_local_credit_policy"] == "earliest_divergence_rarity_geomean"
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_depth_local.sh").read_text(
        encoding="utf-8"
    )
    assert "rjob_tags.txt" in (FARO_ROOT / "scripts/submit_samtok_tb_gppo.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-depth-local-geometry-10step-2g" in text


def test_depth_local_credit_requires_joint_gain_and_localizes_first_difference():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1], [1, 1], [0, 0], [1, 0]]
    greedy = [0, 0]
    advantages = depth_local_geometry_advantages(
        raw, native, sampled, greedy, minimum_improvement=1e-4
    )
    assert advantages.shape == (4,)
    assert torch.isfinite(advantages).all()
    assert advantages[0] > 0
    assert advantages[1] > 0
    assert advantages[2] == 0
    assert advantages[3] == 0


def test_shuffled_depth_local_control_is_registered_and_uses_full_training_budget(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "depth_local_geometry_shuffled"
    assert method["depth_local_credit_policy"] == "cyclic_depth_shuffle_rarity_geomean"
    assert method["depth_local_shuffle_seed"] == 20260827
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_depth_local_shuffle.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-depth-local-shuffle-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text


def test_shuffled_depth_local_preserves_active_joint_gain_and_rarity_set():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1, 2, 3], [1, 1, 2, 3], [0, 0, 2, 3], [1, 0, 2, 3]]
    greedy = [0, 0, 2, 3]
    base = depth_local_geometry_advantages(
        raw, native, sampled, greedy, minimum_improvement=1e-4
    )
    shuffled = shuffled_depth_local_geometry_advantages(
        raw, native, sampled, greedy, minimum_improvement=1e-4, shuffle_seed=20260827
    )
    assert torch.isfinite(shuffled).all()
    assert torch.equal(base > 0, shuffled > 0)
    assert torch.equal(base == 0, shuffled == 0)
    assert not torch.allclose(base, shuffled)
    assert torch.allclose(
        shuffled[shuffled > 0].mean(), torch.tensor(1.0), atol=1e-6
    )


def test_signed_native_relative_depth_local_r20_preserves_regressions_and_beta_lock(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "asymmetric_signed_native_depth_local"
    assert method["depth_local_credit_policy"] == "signed_native_relative_asymmetric"
    assert method["depth_local_beta"] == 0.25
    assert method["positive_only_credit"] is not True
    text = (
        FARO_ROOT
        / "scripts/submit_samtok_tb_gppo_signed_native_depth_local_beta025.sh"
    ).read_text(encoding="utf-8")
    assert "egfepo_train_5120.jsonl" in text
    assert "rows >= 5000" in text
    assert "dna-fepo-signed-native-depth-local-beta025-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text
    trainer = (
        FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    assert "signed_native_depth_local_enabled" in trainer
    assert "depth_local_beta" in trainer
    assert "asymmetric_signed_native_relative_depth_local_advantages" in trainer


def test_native_rank_local_r21_config_and_credit_are_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_NATIVE_RANK_LOCAL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_NATIVE_RANK_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "native_anchored_rank_local"
    assert method["depth_local_credit_policy"] == "native_reference_midrank_first_divergence"
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_native_rank_local.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-native-rank-local-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text
    trainer = (FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "native_anchored_rank_local_geometry_advantages" in trainer
    assert "native_rank_local_enabled" in trainer


def test_scale_stratified_native_rank_local_r22_is_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "scale_stratified_native_rank_local"
    assert method["positive_reward"] == "fifo16_rank_ciou_boundary_iou"
    assert method["area_stratified_schedule"] is True
    assert method["area_rank_weights"] == {
        "small": [0.35, 0.65],
        "medium": [0.50, 0.50],
        "large": [0.65, 0.35],
    }
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_scale_stratified_native_rank_local.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-scale-stratified-native-rank-local-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text


def test_scale_stratified_credit_locks_weights_and_localizes_joint_gain():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1], [1, 1], [0, 0], [1, 0]]
    greedy = [0, 0]
    advantages = scale_stratified_native_rank_local_geometry_advantages(
        raw, native, sampled, greedy, "small"
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > 0
    assert advantages[1] > 0
    assert advantages[2] == 0
    assert advantages[3] == 0
    with pytest.raises(ValueError, match="fixed by the registered area stratum"):
        scale_stratified_native_rank_local_geometry_advantages(
            raw, native, sampled, greedy, "small", axis_weights=(0.5, 0.5)
        )


def test_legacy_registered_schedule_does_not_require_area_metadata():
    records = {
        f"pair-{index:03d}": {"hard_geometry": index % 2 == 0}
        for index in range(240)
    }
    schedule = select_registered_ids(
        {"records": records}, schedule_per_stratum=80, area_stratified=False
    )
    assert len(schedule["schedule_pair_ids"]) == 160
    assert "area_stratified" not in schedule
    assert "area_strata" not in schedule


def test_confidence_gated_r27_is_registered_and_uses_fixed_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "confidence_gated_native_rank_local"
    assert method["uncertainty_source"] == "calibrated_entropy_plus_missing_top_support_mass"
    assert method["confidence_threshold"] == 0.60
    assert method["confidence_floor"] == 0.25
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_confidence_gated_native_rank_local.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-confidence-gated-native-rank-local-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text


def test_confidence_gate_suppresses_low_confidence_credit():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1], [1, 1], [0, 0], [1, 0]]
    greedy = [0, 0]
    uncertainty = torch.tensor([0.1, 0.9, 0.1, 0.1])
    advantages = confidence_gated_native_rank_local_geometry_advantages(
        raw, native, sampled, greedy, uncertainty, confidence_threshold=0.60
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > 0
    assert advantages[1] == 0
    assert advantages[2] == 0
    assert advantages[3] == 0
    with pytest.raises(ValueError, match="cannot exceed"):
        confidence_gated_native_rank_local_geometry_advantages(
            raw, native, sampled, greedy, uncertainty,
            confidence_threshold=0.5, confidence_floor=0.6,
        )


def test_margin_calibrated_r28_is_registered_and_rewards_joint_margin(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "margin_calibrated_native_rank_local"
    assert method["margin_power"] == 0.5
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_margin_calibrated_native_rank_local.sh").read_text(encoding="utf-8")
    assert "rows >= 5000" in text
    assert "dna-fepo-margin-calibrated-native-rank-local-10step-2g" in text
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    advantages = margin_calibrated_native_rank_local_geometry_advantages(
        raw, native, [[0, 1], [1, 1], [0, 0], [1, 0]], [0, 0]
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > advantages[1] > 0
    assert advantages[2] == 0 and advantages[3] == 0


def test_soft_native_dominance_r34_is_registered_and_continuous(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "soft_native_dominance_depth_local"
    assert method["soft_dominance_temperature"] == 0.02
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_soft_native_dominance.sh").read_text(encoding="utf-8")
    assert "rows >= 5000" in text
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    advantages = soft_native_dominance_depth_local_geometry_advantages(
        raw, native, [[0, 1], [1, 1], [0, 0], [1, 0]], [0, 0]
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > advantages[1] > 0
    assert advantages[2] == 0 and advantages[3] == 0


def test_bidirectional_coarse_fine_r23_is_registered_and_uses_both_ends(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE
    assert config["seed"] == 17
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "bidirectional_coarse_fine_native_geometry"
    assert method["depth_local_credit_policy"] == "native_reference_bidirectional_coarse_fine"
    assert method["coarse_depth_weight"] == 0.5
    assert method["fine_depth_weight"] == 0.5
    assert config["seed"] == 17
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_bidirectional_coarse_fine.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-bidirectional-coarse-fine-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text
    trainer = (FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "bidirectional_coarse_fine_native_geometry_advantages" in trainer
    assert "bidirectional_coarse_fine_enabled" in trainer


def test_bidirectional_credit_requires_joint_gain_and_is_finite():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1, 2, 3], [0, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]]
    greedy = [0, 0, 0, 0]
    advantages = bidirectional_coarse_fine_native_geometry_advantages(
        raw, native, sampled, greedy
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0] > 0
    assert advantages[1] > 0
    assert advantages[2] == 0
    assert advantages[3] == 0
    with pytest.raises(ValueError, match="coarse/fine weights must sum"):
        bidirectional_coarse_fine_native_geometry_advantages(
            raw, native, sampled, greedy, coarse_weight=0.7, fine_weight=0.7
        )


def test_anchor_kl_r24_is_registered_and_detached():
    import torch

    current = [torch.tensor([1.0, 0.0], requires_grad=True)]
    anchor = [torch.tensor([1.0, 0.0])]
    value = anchor_categorical_kl(current, anchor)
    assert torch.isfinite(value)
    assert float(value.item()) == 0.0
    value.backward()
    assert current[0].grad is not None
    assert not anchor[0].requires_grad


def test_anchor_kl_r24_config_and_submit_are_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_ANCHOR_KL_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_ANCHOR_KL_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["anchor_kl_enabled"] is True
    assert method["anchor_buffer_rows"] == 64
    assert method["anchor_kl_epsilon"] == 0.02
    assert method["anchor_kl_lambda"] == 0.5
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_anchor_kl.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-anchor-kl-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text
    trainer = (FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "global_anchor_kl_active_observations > 0" in trainer


def test_signed_native_relative_depth_local_only_penalizes_joint_regressions():
    raw = torch.tensor(
        [
            [0.80, 0.80],  # positive on both axes
            [0.60, 0.90],  # mixed trade-off: neutral
            [0.60, 0.60],  # regression on both axes
            [0.70, 0.70],  # native codes: zero credit
        ]
    )
    native = torch.tensor([0.70, 0.70])
    sampled = [[1, 0], [1, 1], [2, 2], [0, 0]]
    greedy = [0, 0]
    advantages = asymmetric_signed_native_relative_depth_local_advantages(
        raw, native, sampled, greedy, beta=0.25
    )
    assert advantages.shape == (4,)
    assert torch.isfinite(advantages).all()
    assert advantages[0] > 0
    assert advantages[1] == 0
    assert advantages[2] < 0
    assert advantages[3] == 0
    assert torch.allclose(advantages[advantages != 0].abs().mean(), torch.tensor(1.0), atol=1e-6)
    with pytest.raises(ValueError, match="beta=0.25"):
        asymmetric_signed_native_relative_depth_local_advantages(
            raw, native, sampled, greedy, beta=0.5
        )


def test_rarity_free_depth_local_stage_is_registered_and_locked(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG_ROOT / f"{UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE}.py"
    config = runpy.run_path(str(path))["config"]
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    assert config["stage"] == UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert method["pareto_credit_mode"] == "depth_local_geometry_rarity_free"
    assert method["depth_local_credit_policy"] == "earliest_divergence_geomean_no_rarity"
    assert method["depth_local_rarity_weight"] == 0.0
    text = (FARO_ROOT / "scripts/submit_samtok_tb_gppo_depth_local_rarity_free.sh").read_text(
        encoding="utf-8"
    )
    assert "rows >= 5000" in text
    assert "dna-fepo-depth-local-rarity-free-10step-2g" in text
    assert "submit_samtok_tb_gppo.sh" in text
    trainer = (
        FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    assert '"depth_local_geometry_rarity_free"' in trainer
    assert "depth_local_rarity_free_enabled" in trainer
    assert "rarity_weight=(" in trainer


def test_rarity_free_credit_reuses_local_geometry_without_frequency_bonus():
    raw = torch.tensor([[0.80, 0.80], [0.75, 0.75], [0.80, 0.70], [0.70, 0.80]])
    native = torch.tensor([0.70, 0.70])
    sampled = [[0, 1], [1, 1], [0, 0], [1, 0]]
    greedy = [0, 0]
    rarity_free = depth_local_geometry_advantages(
        raw,
        native,
        sampled,
        greedy,
        minimum_improvement=1e-4,
        depth_decay=0.85,
        rarity_weight=0.0,
    )
    assert torch.isfinite(rarity_free).all()
    assert torch.equal(rarity_free > 0, torch.tensor([True, True, False, False]))
    assert rarity_free[0].item() == pytest.approx(1.2597, rel=2e-3)
    assert rarity_free[1].item() == pytest.approx(0.7403, rel=2e-3)


def test_pareto_prefix_replay_selects_jointly_feasible_candidate():
    text = (
        FARO_ROOT
        / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    start = text.index("if verified_replay_enabled or verified_prefix_replay_enabled or pareto_prefix_replay_enabled:")
    block = text[start : text.index("preference_best_sequence = None", start)]
    assert "pareto_valid = (ciou_gain > minimum_improvement)" in block
    assert "boundary_gain > minimum_improvement" in block
    assert "geometry_score = torch.sqrt" in block
    assert "masked_fill(~pareto_valid" in block
