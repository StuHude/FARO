"""R15: deterministic shuffled-depth control for R14 geometry credit."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE,
)


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(
        UNIFIED_DEPTH_LOCAL_SHUFFLE_STAGE
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
        "pareto_credit_mode": "depth_local_geometry_shuffled",
        "depth_local_credit_policy": "cyclic_depth_shuffle_rarity_geomean",
        "depth_local_decay": 0.85,
        "depth_local_rarity_weight": 0.5,
        "depth_local_shuffle_seed": 20260827,
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
