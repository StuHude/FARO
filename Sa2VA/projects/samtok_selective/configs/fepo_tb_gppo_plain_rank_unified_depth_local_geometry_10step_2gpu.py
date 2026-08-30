"""R14: earliest-divergence, depth-local geometry credit for SAMTok.

The decoded-mask geometry reward is unchanged.  A positive rollout is weighted
by the first SAMTok code depth that differs from native greedy, with a small
prefix-rarity bonus, so credit is localized instead of copied uniformly over
the complete sequence-level action.
"""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import UNIFIED_DEPTH_LOCAL_STAGE


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_DEPTH_LOCAL_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_DEPTH_LOCAL_STAGE)
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
        "pareto_credit_mode": "depth_local_geometry",
        "depth_local_credit_policy": "earliest_divergence_rarity_geomean",
        "depth_local_decay": 0.85,
        "depth_local_rarity_weight": 0.5,
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
