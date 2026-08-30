from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, validate_config
from .entropy_gr_cppo_contract import (
    CALIBRATION_ITERATIONS,
    POLICY_EPOCHS,
    ROLLOUTS_PER_PROMPT,
    SUPPORT_SIZE,
    TARGET_EFFECTIVE_SUPPORT,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from .gr_cppo_contract import expected_frozen_anchor, validate_frozen_anchor
from .tail_geometry import (
    BOUNDARY_WIDTH,
    FIFO_CAPACITY,
    FIFO_INIT_SIZE,
    SENTINEL_SIZE,
    SHUFFLE_SEED,
)


METHOD = "standalone_samtok_tail_balanced_geometry_gppo"
ARMS = {"plain_rank", "tail_balanced", "shuffled_labels"}
STAGES = {
    f"fepo_tb_gppo_{arm}_{suffix}_2gpu": steps
    for arm in ARMS
    for suffix, steps in (("one_step", 1), ("20step", 20))
}
STAGES["fepo_tb_gppo_tail_balanced_active_set_10step_2gpu"] = 10
UNIFIED_STAGE = "fepo_tb_gppo_tail_balanced_unified_sentinel_10step_2gpu"
STAGES[UNIFIED_STAGE] = 10
UNIFIED_PLAIN_STAGE = "fepo_tb_gppo_plain_rank_unified_sentinel_10step_2gpu"
STAGES[UNIFIED_PLAIN_STAGE] = 10
UNIFIED_IMPROVEMENT_STAGE = "fepo_tb_gppo_plain_rank_unified_improvement_10step_2gpu"
STAGES[UNIFIED_IMPROVEMENT_STAGE] = 10
UNIFIED_PREFIX_STAGE = "fepo_tb_gppo_plain_rank_unified_prefix_credit_10step_2gpu"
STAGES[UNIFIED_PREFIX_STAGE] = 10
UNIFIED_PARETO_STAGE = "fepo_tb_gppo_plain_rank_unified_pareto_geometry_10step_2gpu"
STAGES[UNIFIED_PARETO_STAGE] = 10
UNIFIED_REPLAY_STAGE = "fepo_tb_gppo_plain_rank_unified_verified_replay_10step_2gpu"
STAGES[UNIFIED_REPLAY_STAGE] = 10
UNIFIED_PREFIX_REPLAY_STAGE = "fepo_tb_gppo_plain_rank_unified_verified_prefix_replay_10step_2gpu"
STAGES[UNIFIED_PREFIX_REPLAY_STAGE] = 10
UNIFIED_PARETO_PREFIX_REPLAY_STAGE = "fepo_tb_gppo_plain_rank_unified_pareto_prefix_replay_10step_2gpu"
STAGES[UNIFIED_PARETO_PREFIX_REPLAY_STAGE] = 10
UNIFIED_RANK_PARETO_STAGE = "fepo_tb_gppo_plain_rank_unified_rank_pareto_geometry_10step_2gpu"
STAGES[UNIFIED_RANK_PARETO_STAGE] = 10
UNIFIED_NATIVE_RANK_PARETO_STAGE = "fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu"
STAGES[UNIFIED_NATIVE_RANK_PARETO_STAGE] = 10
UNIFIED_DEPTH_LOCAL_STAGE = "fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu"
STAGES[UNIFIED_DEPTH_LOCAL_STAGE] = 10
UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_depth_local_shuffle_10step_2gpu"
)
STAGES[UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE] = 10
UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu"
)
STAGES[UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE] = 10
UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_10step_2gpu"
)
STAGES[UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_STAGE] = 10
UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_seed17_100step_2gpu"
)
STAGES[UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE] = 100
UNIFIED_UNIFORM_JOINT_STAGE = "fepo_tb_gppo_plain_rank_unified_uniform_joint_geometry_10step_2gpu"
STAGES[UNIFIED_UNIFORM_JOINT_STAGE] = 10
UNIFIED_DEPTH_LOCAL_EVIDENCE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_depth_local_evidence_10step_2gpu"
)
STAGES[UNIFIED_DEPTH_LOCAL_EVIDENCE_STAGE] = 10
UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_signed_native_depth_local_beta025_10step_2gpu"
)
STAGES[UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE] = 10
UNIFIED_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_scale_stratified_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_bidirectional_coarse_fine_10step_2gpu"
)
STAGES[UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE] = 10
UNIFIED_ANCHOR_KL_STAGE = "fepo_tb_gppo_plain_rank_unified_anchor_kl_10step_2gpu"
STAGES[UNIFIED_ANCHOR_KL_STAGE] = 10
UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_uncertainty_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_action_budget_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu"
)
STAGES[UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE] = 10
UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_shuffled_10step_2gpu"
)
STAGES[UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE] = 10
UNIFIED_BOUNDARY_STRATIFIED_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_BOUNDARY_STRATIFIED_STAGE] = 10
UNIFIED_CONSERVATIVE_NULL_TAIL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_conservative_null_tail_10step_2gpu"
)
STAGES[UNIFIED_CONSERVATIVE_NULL_TAIL_STAGE] = 10
UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_confidence_gated_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_margin_calibrated_native_rank_local_10step_2gpu"
)
STAGES[UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE] = 10
UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_primal_dual_null_risk_10step_2gpu"
)
STAGES[UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE] = 10
UNIFIED_GROUNDED_INTERFACE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_grounded_interface_10step_2gpu"
)
STAGES[UNIFIED_GROUNDED_INTERFACE_STAGE] = 10
UNIFIED_SAFE_VISUAL_INTERFACE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu"
)
STAGES[UNIFIED_SAFE_VISUAL_INTERFACE_STAGE] = 10
UNIFIED_PAIRED_VIEW_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu"
)
STAGES[UNIFIED_PAIRED_VIEW_STAGE] = 10
UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu"
)
STAGES[UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE] = 10
UNIFIED_NATIVE_RANK_SIGNED_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_native_rank_signed_depth_local_10step_2gpu"
)
STAGES[UNIFIED_NATIVE_RANK_SIGNED_STAGE] = 10
UNIFIED_NATIVE_RANK_SIGNED_20_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_native_rank_signed_depth_local_20step_2gpu"
)
STAGES[UNIFIED_NATIVE_RANK_SIGNED_20_STAGE] = 20
UNIFIED_NATIVE_RANK_SIGNED_20_SEED18_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_native_rank_signed_depth_local_20step_seed18_2gpu"
)
STAGES[UNIFIED_NATIVE_RANK_SIGNED_20_SEED18_STAGE] = 20
UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE = (
    "fepo_tb_gppo_plain_rank_unified_soft_native_dominance_depth_local_10step_2gpu"
)
STAGES[UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE] = 10


def validate_tail_gppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported TB-GPPO stage: {stage}")
    if int(config["optimizer"]["max_steps"]) != STAGES[stage]:
        raise ValueError(f"{stage} must run exactly {STAGES[stage]} outer steps")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("TB-GPPO requires exactly two processes")
    if int(config["data"]["pairs_per_device_batch"]) != 4:
        raise ValueError("TB-GPPO requires four pairs per process")
    expected_anchor = expected_frozen_anchor(repo_root)
    if Path(config["checkpoint"].get("adapter_init") or "").resolve() != expected_anchor:
        raise ValueError(f"TB-GPPO must initialize from {expected_anchor}")
    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"TB-GPPO output must be {expected_output}")

    method = config.get("tail_gppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"TB-GPPO method must be {METHOD}")
    arm = method.get("arm")
    if arm not in ARMS or f"fepo_tb_gppo_{arm}_" not in stage:
        raise ValueError("TB-GPPO arm and stage do not match")
    unified = stage in {
        UNIFIED_STAGE,
        UNIFIED_PLAIN_STAGE,
        UNIFIED_IMPROVEMENT_STAGE,
        UNIFIED_PREFIX_STAGE,
        UNIFIED_PARETO_STAGE,
        UNIFIED_REPLAY_STAGE,
        UNIFIED_PREFIX_REPLAY_STAGE,
        UNIFIED_PARETO_PREFIX_REPLAY_STAGE,
        UNIFIED_RANK_PARETO_STAGE,
        UNIFIED_NATIVE_RANK_PARETO_STAGE,
        UNIFIED_DEPTH_LOCAL_STAGE,
        UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE,
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE,
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_STAGE,
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE,
        UNIFIED_UNIFORM_JOINT_STAGE,
        UNIFIED_DEPTH_LOCAL_EVIDENCE_STAGE,
        UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE,
        UNIFIED_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE,
        UNIFIED_ANCHOR_KL_STAGE,
        UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE,
        UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE,
        UNIFIED_BOUNDARY_STRATIFIED_STAGE,
        UNIFIED_CONSERVATIVE_NULL_TAIL_STAGE,
        UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE,
        UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE,
        UNIFIED_GROUNDED_INTERFACE_STAGE,
        UNIFIED_SAFE_VISUAL_INTERFACE_STAGE,
        UNIFIED_PAIRED_VIEW_STAGE,
        UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE,
        UNIFIED_NATIVE_RANK_SIGNED_STAGE,
        UNIFIED_NATIVE_RANK_SIGNED_20_STAGE,
        UNIFIED_NATIVE_RANK_SIGNED_20_SEED18_STAGE,
        UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE,
    }
    if unified and method.get("unified_sentinel") is not True:
        raise ValueError("Unified TB-GPPO stage requires unified_sentinel=true")
    if not unified and method.get("unified_sentinel") is True:
        raise ValueError("unified_sentinel is only valid for the registered unified stage")
    exact = {
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "policy_epochs": POLICY_EPOCHS,
        "support_size": SUPPORT_SIZE,
        "calibration_iterations": CALIBRATION_ITERATIONS,
        "boundary_width": BOUNDARY_WIDTH,
        "fifo_capacity": FIFO_CAPACITY,
        "fifo_init_rows": FIFO_INIT_SIZE,
        "sentinel_rows_total": SENTINEL_SIZE,
        "hard_label_shuffle_seed": SHUFFLE_SEED,
    }
    for key, expected in exact.items():
        if int(method.get(key, -1)) != expected:
            raise ValueError(f"TB-GPPO requires {key}={expected}")
    floats = {
        "target_effective_support": TARGET_EFFECTIVE_SUPPORT,
        "temperature_min": TEMPERATURE_MIN,
        "temperature_max": TEMPERATURE_MAX,
        "ordinary_ciou_weight": 0.6,
        "ordinary_boundary_weight": 0.4,
        "hard_ciou_weight": 0.4,
        "hard_boundary_weight": 0.6,
        "plain_ciou_weight": 0.5,
        "plain_boundary_weight": 0.5,
        "sentinel_degradation_budget": 0.05,
        "sentinel_tail_quantile": 0.10,
        "sentinel_tail_weight": 0.25,
    }
    if stage == UNIFIED_CONSERVATIVE_NULL_TAIL_STAGE:
        floats.update({"sentinel_tail_quantile": 0.05, "sentinel_tail_weight": 0.50})
    for key, expected in floats.items():
        value = float(method.get(key, float("nan")))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"TB-GPPO requires {key}={expected}")
    required = {
        "exploration": "per_prefix_topm_collision_support",
        "rescore_policy": "frozen_old_support_and_temperature",
        "positive_reward": "fifo16_rank_ciou_boundary_iou",
        "advantage": "group_standardized",
        "negative_objective": "canonical_no_target_ce",
        "margin_constraint": "first_null_token_vs_mask_start_hinge",
        "sentinel_risk": "lower10_current_minus_anchor_margin",
        "schedule": "registered_same_ids_half_hard_half_ordinary",
    }
    if method.get("positive_only_credit") is True:
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "positive_greedy_improvement_mean_normalized"
    if method.get("prefix_credit_mode") == "hierarchical_geometry_prefix":
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "hierarchical_prefix_improvement_mean_normalized"
    if method.get("pareto_credit_mode") == "pareto_geometry_improvement":
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "pareto_geometry_improvement_geomean_normalized"
    if method.get("verified_replay_mode") == "best_sampled_cIoU_replay":
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "group_standardized"
    if method.get("verified_replay_mode") == "best_sampled_prefix_replay":
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "group_standardized"
    if method.get("verified_replay_mode") == "best_sampled_pareto_prefix_replay":
        required["positive_reward"] = "raw_ciou"
        required["advantage"] = "group_standardized"
    if method.get("pareto_credit_mode") == "asymmetric_signed_native_depth_local":
        required["advantage"] = "signed_native_relative_depth_local_mean_abs_normalized"
    if method.get("pareto_credit_mode") == "scale_stratified_native_rank_local":
        required["advantage"] = "area_stratum_native_rank_local_mean_normalized"
        required["schedule"] = "registered_area_stratified_hard_three_scale"
    if method.get("pareto_credit_mode") == "bidirectional_coarse_fine_native_geometry":
        required["advantage"] = "native_reference_bidirectional_coarse_fine_mean_normalized"
    if method.get("pareto_credit_mode") == "uncertainty_calibrated_native_rank_local":
        required["advantage"] = "native_reference_uncertainty_calibrated_mean_normalized"
    if method.get("pareto_credit_mode") == "action_budget_native_rank_local":
        required["advantage"] = "native_reference_action_budget_mean_normalized"
    if method.get("pareto_credit_mode") == "boundary_stratified_native_rank_local":
        required["advantage"] = "native_reference_boundary_stratified_mean_normalized"
        required["schedule"] = "registered_boundary_stratified_50_25_25"
    if method.get("pareto_credit_mode") == "confidence_gated_native_rank_local":
        required["advantage"] = "native_reference_confidence_gated_mean_normalized"
    if method.get("pareto_credit_mode") == "margin_calibrated_native_rank_local":
        required["advantage"] = "native_reference_margin_calibrated_mean_normalized"
    if method.get("pareto_credit_mode") == "primal_dual_null_risk_native_rank_local":
        required["advantage"] = "native_reference_primal_dual_null_risk_mean_normalized"
    if method.get("pareto_credit_mode") == "soft_native_dominance_depth_local":
        required["advantage"] = "native_reference_soft_dominance_mean_normalized"
    if method.get("pareto_credit_mode") == "predicted_evidence_scope":
        required["advantage"] = "native_reference_predicted_evidence_scope_mean_normalized"
    for key, expected in required.items():
        if method.get(key) != expected:
            raise ValueError(f"TB-GPPO requires {key}={expected}")
    if unified:
        unified_required = {
            "selective_risk_mode": "fixed_training_sentinel_active_set",
            "sentinel_source": "registered_tail_no_target_ids",
            "anchor_budget_source": "frozen_initialization_pre_update",
            "holdout_access": False,
        }
        for key, expected in unified_required.items():
            if method.get(key) != expected:
                raise ValueError(f"Unified TB-GPPO requires {key}={expected}")
        if int(method.get("sentinel_rows_total", 0)) != SENTINEL_SIZE:
            raise ValueError("Unified TB-GPPO must use the shared 32-row sentinel")
        for key in ("null_ce_relative_slack", "null_ce_absolute_slack", "margin_slack"):
            value = float(method.get(key, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Unified TB-GPPO {key} must be finite and nonnegative")
    if stage == UNIFIED_IMPROVEMENT_STAGE:
        if method.get("arm") != "plain_rank" or method.get("positive_only_credit") is not True:
            raise ValueError("Unified improvement stage requires plain_rank positive-only credit")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Unified improvement stage must use raw cIoU reward")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Unified improvement requires {key}={expected}")
    if stage == UNIFIED_PREFIX_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Unified prefix stage requires plain_rank")
        if method.get("positive_only_credit") is not True:
            raise ValueError("Unified prefix stage requires positive-only credit")
        if method.get("prefix_credit_mode") != "hierarchical_geometry_prefix":
            raise ValueError("Unified prefix stage requires hierarchical prefix credit")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Unified prefix stage must use raw cIoU reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("prefix_depth_decay", 0.85),
            ("prefix_novelty_weight", 0.5),
        ):
            if not math.isclose(
                float(method.get(key, float("nan"))),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Unified prefix requires {key}={expected}")
    if stage == UNIFIED_PARETO_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Unified Pareto stage requires plain_rank")
        if method.get("positive_only_credit") is not True:
            raise ValueError("Unified Pareto stage requires positive-only credit")
        if method.get("pareto_credit_mode") != "pareto_geometry_improvement":
            raise ValueError("Unified Pareto stage requires Pareto geometry credit")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Unified Pareto stage must use raw cIoU reward")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6)):
            if not math.isclose(
                float(method.get(key, float("nan"))),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Unified Pareto requires {key}={expected}")
    if stage == UNIFIED_REPLAY_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Unified replay stage requires plain_rank")
        if method.get("verified_replay_mode") != "best_sampled_cIoU_replay":
            raise ValueError("Unified replay stage requires verified replay mode")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Unified replay stage must use raw cIoU reward")
        weight = float(method.get("verified_replay_weight", float("nan")))
        if not math.isclose(weight, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unified replay requires verified_replay_weight=0.05")
        threshold = float(method.get("minimum_improvement", float("nan")))
        if not math.isclose(threshold, 1e-4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unified replay requires minimum_improvement=1e-4")
    if stage == UNIFIED_PREFIX_REPLAY_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Unified prefix replay stage requires plain_rank")
        if method.get("verified_replay_mode") != "best_sampled_prefix_replay":
            raise ValueError("Unified prefix replay requires prefix replay mode")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Unified prefix replay must use raw cIoU reward")
        weight = float(method.get("verified_replay_weight", float("nan")))
        if not math.isclose(weight, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unified prefix replay requires verified_replay_weight=0.05")
        threshold = float(method.get("minimum_improvement", float("nan")))
        if not math.isclose(threshold, 1e-4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Unified prefix replay requires minimum_improvement=1e-4")
    if stage == UNIFIED_PARETO_PREFIX_REPLAY_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Pareto prefix replay stage requires plain_rank")
        if method.get("verified_replay_mode") != "best_sampled_pareto_prefix_replay":
            raise ValueError("Pareto prefix replay requires registered mode")
        if method.get("tail_reward_mode") != "raw_ciou":
            raise ValueError("Pareto prefix replay must use raw cIoU reward")
        if not math.isclose(float(method.get("verified_replay_weight", float("nan"))), 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Pareto prefix replay requires verified_replay_weight=0.05")
        if not math.isclose(float(method.get("minimum_improvement", float("nan"))), 1e-4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Pareto prefix replay requires minimum_improvement=1e-4")
    if stage == UNIFIED_RANK_PARETO_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Rank-Pareto stage requires plain_rank")
        if method.get("pareto_credit_mode") != "rank_pareto_geometry":
            raise ValueError("Rank-Pareto stage requires rank Pareto geometry credit")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Rank-Pareto keeps the registered geometry reward")
        if method.get("rank_pareto_tie_policy") != "midrank_geomean_group_standardized":
            raise ValueError("Rank-Pareto tie policy is not registered")
    if stage == UNIFIED_NATIVE_RANK_PARETO_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Native-anchored rank-Pareto stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_pareto":
            raise ValueError("Native stage requires native-anchored rank-Pareto credit")
        if method.get("rank_pareto_tie_policy") != "native_reference_midrank_geomean":
            raise ValueError("Native stage requires native-reference midrank policy")
        if not math.isclose(
            float(method.get("minimum_improvement", float("nan"))),
            1e-4,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Native stage requires minimum_improvement=1e-4")
    if stage == UNIFIED_DEPTH_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Depth-local geometry stage requires plain_rank")
        if method.get("pareto_credit_mode") != "depth_local_geometry":
            raise ValueError("Depth-local stage requires depth-local geometry credit")
        if method.get("depth_local_credit_policy") != "earliest_divergence_rarity_geomean":
            raise ValueError("Depth-local stage requires the registered credit policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Depth-local stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("depth_local_rarity_weight", 0.5),
        ):
            if not math.isclose(
                float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Depth-local stage requires {key}={expected}")
    if stage == UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Shuffled depth-local stage requires plain_rank")
        if method.get("pareto_credit_mode") != "depth_local_geometry_shuffled":
            raise ValueError("Shuffled stage requires shuffled depth-local geometry credit")
        if method.get("depth_local_credit_policy") != "cyclic_depth_shuffle_rarity_geomean":
            raise ValueError("Shuffled stage requires the registered cyclic credit policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Shuffled depth-local stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("depth_local_rarity_weight", 0.5),
            ("depth_local_shuffle_seed", 20260827),
        ):
            value = float(method.get(key, float("nan")))
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Shuffled depth-local stage requires {key}={expected}")
    if stage in {
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_STAGE,
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_STAGE,
        UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE,
        UNIFIED_DEPTH_LOCAL_EVIDENCE_STAGE,
    }:
        if method.get("arm") != "plain_rank":
            raise ValueError("Rarity-free depth-local stage requires plain_rank")
        if method.get("pareto_credit_mode") != "depth_local_geometry_rarity_free":
            raise ValueError("Rarity-free stage requires rarity-free depth-local credit")
        if method.get("depth_local_credit_policy") != "earliest_divergence_geomean_no_rarity":
            raise ValueError("Rarity-free stage requires the registered no-rarity policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Rarity-free depth-local stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("depth_local_rarity_weight", 0.0),
        ):
            if not math.isclose(
                float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Rarity-free depth-local stage requires {key}={expected}")
        if stage == UNIFIED_DEPTH_LOCAL_EVIDENCE_STAGE:
            evidence_gate = method.get("evidence_gate")
            if not isinstance(evidence_gate, dict) or evidence_gate.get("mode") != "view_drop":
                raise ValueError("Evidence stage requires the registered view-drop gate")
            if float(evidence_gate.get("scale", -1.0)) != 0.25:
                    raise ValueError("Evidence stage requires evidence_gate.scale=0.25")
            if float(method.get("temperature", float("nan"))) != 1.0:
                raise ValueError("Evidence stage requires temperature=1.0")
    if stage == UNIFIED_UNIFORM_JOINT_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Uniform joint stage requires plain_rank")
        if method.get("pareto_credit_mode") != "depth_local_geometry_rarity_free":
            raise ValueError("Uniform joint stage requires rarity-free local credit")
        if method.get("depth_local_credit_policy") != "uniform_joint_geometry_no_rarity":
            raise ValueError("Uniform joint stage requires uniform joint policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Uniform joint stage keeps the registered geometry reward")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6), ("depth_local_decay", 1.0), ("depth_local_rarity_weight", 0.0)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Uniform joint stage requires {key}={expected}")
    if stage == UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Signed native-relative stage requires plain_rank")
        if method.get("pareto_credit_mode") != "asymmetric_signed_native_depth_local":
            raise ValueError("Signed native-relative stage requires registered signed credit")
        if method.get("depth_local_credit_policy") != "signed_native_relative_asymmetric":
            raise ValueError("Signed native-relative stage requires the registered signed policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Signed native-relative stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("depth_local_beta", 0.25),
        ):
            if not math.isclose(
                float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Signed native-relative stage requires {key}={expected}")
        if method.get("positive_only_credit") is True:
            raise ValueError("Signed native-relative stage must preserve negative credit")
    if stage == UNIFIED_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Native rank-local stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_local":
            raise ValueError("Native rank-local stage requires native rank-local credit")
        if method.get("depth_local_credit_policy") != "native_reference_midrank_first_divergence":
            raise ValueError("Native rank-local stage requires first-divergence policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Native rank-local stage keeps the registered geometry reward")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6), ("depth_local_decay", 0.85)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Native rank-local stage requires {key}={expected}")
    if stage == UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Scale-stratified stage requires plain_rank")
        if method.get("pareto_credit_mode") != "scale_stratified_native_rank_local":
            raise ValueError("Scale-stratified stage requires area-stratified native credit")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Scale-stratified stage keeps the registered geometry reward")
        if method.get("depth_local_credit_policy") != "native_reference_area_stratum_rank_first_divergence":
            raise ValueError("Scale-stratified stage requires area-stratum rank policy")
        if method.get("area_stratified_schedule") is not True:
            raise ValueError("Scale-stratified stage requires area-stratified schedule")
        if method.get("area_rank_weights") != {
            "small": [0.35, 0.65],
            "medium": [0.50, 0.50],
            "large": [0.65, 0.35],
        }:
            raise ValueError("Scale-stratified stage requires fixed area rank weights")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6), ("depth_local_decay", 0.85)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Scale-stratified stage requires {key}={expected}")
    if stage == UNIFIED_BOUNDARY_STRATIFIED_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Boundary-stratified stage requires plain_rank")
        if method.get("pareto_credit_mode") != "boundary_stratified_native_rank_local":
            raise ValueError("Boundary-stratified stage requires boundary-stratified credit")
        if method.get("depth_local_credit_policy") != "native_reference_boundary_stratum_rank_first_divergence":
            raise ValueError("Boundary-stratified stage requires fixed boundary-stratum policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Boundary-stratified stage keeps the registered geometry reward")
        if method.get("boundary_stratified_schedule") is not True:
            raise ValueError("Boundary-stratified stage requires boundary-stratified schedule")
        if method.get("boundary_sampling_mix") != {
            "ordinary": 0.50,
            "thin": 0.25,
            "boundary_hard": 0.25,
        }:
            raise ValueError("Boundary-stratified stage requires fixed 50/25/25 sampling mix")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6), ("depth_local_decay", 0.85)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Boundary-stratified stage requires {key}={expected}")
    if stage == UNIFIED_BIDIRECTIONAL_COARSE_FINE_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Bidirectional coarse/fine stage requires plain_rank")
        if method.get("pareto_credit_mode") != "bidirectional_coarse_fine_native_geometry":
            raise ValueError("Bidirectional stage requires coarse/fine native geometry credit")
        if method.get("depth_local_credit_policy") != "native_reference_bidirectional_coarse_fine":
            raise ValueError("Bidirectional stage requires registered coarse/fine policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Bidirectional stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("coarse_depth_weight", 0.5),
            ("fine_depth_weight", 0.5),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Bidirectional stage requires {key}={expected}")
    if stage == UNIFIED_ANCHOR_KL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Anchor-KL stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_local":
            raise ValueError("Anchor-KL stage preserves native rank-local geometry credit")
        if method.get("depth_local_credit_policy") != "native_reference_midrank_first_divergence":
            raise ValueError("Anchor-KL stage preserves first-divergence native credit")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Anchor-KL stage requires {key}={expected}")
        if int(method.get("anchor_buffer_rows", -1)) != 64:
            raise ValueError("Anchor-KL stage requires anchor_buffer_rows=64")
        if method.get("anchor_kl_enabled") is not True:
            raise ValueError("Anchor-KL stage requires anchor_kl_enabled=true")
        for key, expected in (("anchor_kl_epsilon", 0.02), ("anchor_kl_lambda", 0.5)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Anchor-KL stage requires {key}={expected}")
    if stage == UNIFIED_UNCERTAINTY_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Uncertainty native rank-local stage requires plain_rank")
        if method.get("pareto_credit_mode") != "uncertainty_calibrated_native_rank_local":
            raise ValueError("Uncertainty stage requires calibrated native rank-local credit")
        if method.get("depth_local_credit_policy") != "native_reference_uncertainty_calibrated_first_divergence":
            raise ValueError("Uncertainty stage requires the registered uncertainty policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Uncertainty stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("uncertainty_confidence_floor", 0.25),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Uncertainty stage requires {key}={expected}")
        if method.get("uncertainty_source") != "calibrated_entropy_plus_missing_top_support_mass":
            raise ValueError("Uncertainty stage requires the registered calibration source")
    if stage == UNIFIED_ACTION_BUDGET_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Action-budget stage requires plain_rank")
        if method.get("pareto_credit_mode") != "action_budget_native_rank_local":
            raise ValueError("Action-budget stage requires native action-budget credit")
        if method.get("depth_local_credit_policy") != "native_reference_action_budget_first_divergence":
            raise ValueError("Action-budget stage requires the registered first-divergence policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Action-budget stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("action_budget", 2),
            ("action_budget_excess_penalty", 0.10),
        ):
            actual = method.get(key, float("nan"))
            if isinstance(expected, int):
                if int(actual) != expected:
                    raise ValueError(f"Action-budget stage requires {key}={expected}")
            elif not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Action-budget stage requires {key}={expected}")
    if stage in {UNIFIED_PREDICTED_EVIDENCE_SCOPE_STAGE, UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE}:
        if method.get("arm") != "plain_rank":
            raise ValueError("PES stage requires plain_rank")
        if method.get("pareto_credit_mode") != "predicted_evidence_scope":
            raise ValueError("PES stage requires predicted evidence scope credit")
        if method.get("depth_local_credit_policy") != "native_reference_predicted_evidence_scope_first_divergence":
            raise ValueError("PES stage requires registered first-divergence evidence scope")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("PES stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("pes_confident_entropy", 0.35),
            ("pes_ambiguous_entropy", 0.70),
            ("pes_confident_margin", 1.0),
            ("pes_ambiguous_margin", 0.25),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"PES stage requires {key}={expected}")
        shuffled = stage == UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE
        if bool(method.get("pes_evidence_shuffle", False)) is not shuffled:
            raise ValueError("PES shuffled stage requires the registered shuffled-evidence control")
        if shuffled and int(method.get("pes_evidence_shuffle_seed", -1)) != 1907:
            raise ValueError("PES shuffled stage requires seed 1907")
    if stage == UNIFIED_CONSERVATIVE_NULL_TAIL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Conservative-null stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_local":
            raise ValueError("Conservative-null stage preserves native rank-local geometry")
        if method.get("depth_local_credit_policy") != "native_reference_midrank_first_divergence":
            raise ValueError("Conservative-null stage preserves first-divergence credit")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("sentinel_tail_quantile", 0.05),
            ("sentinel_tail_weight", 0.50),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Conservative-null stage requires {key}={expected}")
    if stage == UNIFIED_CONFIDENCE_GATED_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Confidence-gated stage requires plain_rank")
        if method.get("pareto_credit_mode") != "confidence_gated_native_rank_local":
            raise ValueError("Confidence-gated stage requires gated native rank-local credit")
        if method.get("depth_local_credit_policy") != "native_reference_confidence_gate_first_divergence":
            raise ValueError("Confidence-gated stage requires first-divergence confidence policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Confidence-gated stage keeps the registered geometry reward")
        if method.get("uncertainty_source") != "calibrated_entropy_plus_missing_top_support_mass":
            raise ValueError("Confidence-gated stage requires the registered calibration source")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("confidence_threshold", 0.60),
            ("confidence_floor", 0.25),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Confidence-gated stage requires {key}={expected}")
    if stage == UNIFIED_MARGIN_CALIBRATED_NATIVE_RANK_LOCAL_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Margin-calibrated stage requires plain_rank")
        if method.get("pareto_credit_mode") != "margin_calibrated_native_rank_local":
            raise ValueError("Margin-calibrated stage requires registered margin credit")
        if method.get("depth_local_credit_policy") != "native_reference_joint_margin_first_divergence":
            raise ValueError("Margin-calibrated stage requires first-divergence margin policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Margin-calibrated stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("margin_power", 0.5),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Margin-calibrated stage requires {key}={expected}")
    if stage == UNIFIED_PRIMAL_DUAL_NULL_RISK_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Primal-dual stage requires plain_rank")
        if method.get("pareto_credit_mode") != "primal_dual_null_risk_native_rank_local":
            raise ValueError("Primal-dual stage requires native geometry credit")
        if method.get("depth_local_credit_policy") != "native_reference_primal_dual_first_divergence":
            raise ValueError("Primal-dual stage requires first-divergence policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Primal-dual stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("primal_dual_lambda_init", 1.0),
            ("primal_dual_eta", 0.20),
            ("primal_dual_lambda_cap", 4.0),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Primal-dual stage requires {key}={expected}")
        if method.get("primal_dual_risk") != "lower10_current_minus_anchor_margin_excess":
            raise ValueError("Primal-dual stage requires lower-tail margin excess")
    if stage in {UNIFIED_GROUNDED_INTERFACE_STAGE, UNIFIED_SAFE_VISUAL_INTERFACE_STAGE}:
        if method.get("arm") != "plain_rank":
            raise ValueError("Grounded-interface stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_local":
            raise ValueError("Grounded-interface stage preserves native rank-local credit")
        if method.get("depth_local_credit_policy") != "native_reference_midrank_first_divergence":
            raise ValueError("Grounded-interface stage requires native first-divergence credit")
        grounded = method.get("grounded_interface")
        if not isinstance(grounded, dict):
            raise ValueError("Grounded-interface stage requires grounded_interface settings")
        expected = {
            "mode": "supervised_dual_view_mask_code_ce",
            "view": "same_row_photometric_target_preserving",
            "target_source": "same_row_ground_truth_mask_codes",
            "uses_pixvl_teacher": False,
            "uses_opd": False,
            "uses_ema": False,
            "uses_counterfactual": False,
            "lambda_sup": 0.10,
            "brightness": 1.03,
            "contrast": 0.97,
            "visual_gradient_threshold": 1e-8,
            "visual_gradient_min_fraction": 0.80,
            "min_positive_mask_rate": 0.95,
        }
        for key, value in expected.items():
            actual = grounded.get(key)
            if isinstance(value, float):
                if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"Grounded-interface requires {key}={value}")
            elif actual != value:
                raise ValueError(f"Grounded-interface requires {key}={value!r}")
        if stage == UNIFIED_SAFE_VISUAL_INTERFACE_STAGE:
            if not math.isclose(float(method.get("null_ce_weight", float("nan"))), 2.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Safe visual interface requires null_ce_weight=2.0")
            if not math.isclose(float(method.get("margin_weight", float("nan"))), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Safe visual interface requires margin_weight=1.0")
    if stage in {
        UNIFIED_PAIRED_VIEW_STAGE,
        UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE,
    }:
        if method.get("arm") != "plain_rank":
            raise ValueError("Paired-view stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_anchored_rank_local":
            raise ValueError("Paired-view stage preserves native rank-local credit")
        if method.get("depth_local_credit_policy") != "native_reference_midrank_first_divergence":
            raise ValueError("Paired-view stage requires native first-divergence credit")
        paired = method.get("paired_view_geometry")
        if not isinstance(paired, dict):
            raise ValueError("Paired-view stage requires paired_view_geometry settings")
        expected = {
            "mode": (
                "gt_verified_boundary_bottleneck_paired_view_reward"
                if stage == UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE
                else "gt_verified_paired_view_reward"
            ),
            "view": "same_row_photometric_target_preserving",
            "target_source": "same_row_ground_truth_mask_geometry",
            "aggregation": (
                "boundary_bottleneck_min"
                if stage == UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE
                else "geometric_mean"
            ),
            "brightness": 1.03,
            "contrast": 0.97,
            "minimum_improvement": 1e-4,
            "depth_decay": 0.85,
            "uses_pixvl_teacher": False,
            "uses_opd": False,
            "uses_ema": False,
            "uses_counterfactual": False,
        }
        for key, value in expected.items():
            actual = paired.get(key)
            if isinstance(value, float):
                if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"Paired-view requires {key}={value}")
            elif actual != value:
                raise ValueError(f"Paired-view requires {key}={value!r}")
    if stage in {UNIFIED_NATIVE_RANK_SIGNED_STAGE, UNIFIED_NATIVE_RANK_SIGNED_20_STAGE}:
        if method.get("arm") != "plain_rank":
            raise ValueError("Native signed rank stage requires plain_rank")
        if method.get("pareto_credit_mode") != "native_rank_signed_depth_local":
            raise ValueError("Native signed rank stage requires signed rank credit")
        if method.get("depth_local_credit_policy") != "native_reference_signed_rank_first_divergence":
            raise ValueError("Native signed rank stage requires first-divergence credit")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Native signed rank stage keeps the registered geometry reward")
        for key, expected in (("minimum_improvement", 1e-4), ("advantage_epsilon", 1e-6), ("depth_local_decay", 0.85)):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Native signed rank stage requires {key}={expected}")
    if stage == UNIFIED_SOFT_NATIVE_DOMINANCE_STAGE:
        if method.get("arm") != "plain_rank":
            raise ValueError("Soft native dominance stage requires plain_rank")
        if method.get("pareto_credit_mode") != "soft_native_dominance_depth_local":
            raise ValueError("Soft native dominance stage requires soft dominance credit")
        if method.get("depth_local_credit_policy") != "native_reference_soft_dominance_first_divergence":
            raise ValueError("Soft native dominance stage requires first-divergence policy")
        if method.get("positive_reward") != "fifo16_rank_ciou_boundary_iou":
            raise ValueError("Soft native dominance stage keeps the registered geometry reward")
        for key, expected in (
            ("minimum_improvement", 1e-4),
            ("advantage_epsilon", 1e-6),
            ("depth_local_decay", 0.85),
            ("soft_dominance_temperature", 0.02),
        ):
            if not math.isclose(float(method.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Soft native dominance stage requires {key}={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args()
    identity = validate_frozen_anchor(
        args.adapter,
        repo_root=args.repo_root,
        hash_model=not args.skip_model_hash,
    )
    print(json.dumps({"status": "ok", "initialization": identity}, sort_keys=True))


if __name__ == "__main__":
    main()
