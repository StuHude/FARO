"""Matched 100-step confirmation of the promoted R18 SAMTok policy."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE,
)


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE
config["seed"] = 17
config["data"]["jsonl"] = str(Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl"))
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_DEPTH_LOCAL_RARITY_FREE_SEED17_100_STAGE)
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
        "pareto_credit_mode": "depth_local_geometry_rarity_free",
        "depth_local_credit_policy": "earliest_divergence_geomean_no_rarity",
        "depth_local_decay": 0.85,
        "depth_local_rarity_weight": 0.0,
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
