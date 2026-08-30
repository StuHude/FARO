"""R22: native-relative geometry credit conditioned on training-only area strata."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE,
)


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(
        UNIFIED_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_STAGE
    )
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
config["tail_gppo"].update(
    {
        "unified_sentinel": True,
        "selective_risk_mode": "fixed_training_sentinel_active_set",
        "sentinel_source": "registered_tail_no_target_ids",
        "anchor_budget_source": "frozen_initialization_pre_update",
        "null_ce_relative_slack": 0.05,
        "null_ce_absolute_slack": 0.02,
        "margin_slack": 0.05,
        "holdout_access": False,
        "area_stratified_schedule": True,
        "schedule": "registered_area_stratified_hard_three_scale",
        "advantage": "area_stratum_native_rank_local_mean_normalized",
        "pareto_credit_mode": "scale_stratified_native_rank_local",
        "depth_local_credit_policy": "native_reference_area_stratum_rank_first_divergence",
        "depth_local_decay": 0.85,
        "area_rank_weights": {
            "small": [0.35, 0.65],
            "medium": [0.50, 0.50],
            "large": [0.65, 0.35],
        },
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
