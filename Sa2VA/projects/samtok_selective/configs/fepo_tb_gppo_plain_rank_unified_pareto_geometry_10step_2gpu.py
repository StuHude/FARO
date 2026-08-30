"""R7: Pareto improvement credit for mask geometry.

The policy receives credit only when a sampled mask improves both cIoU and
boundary IoU over the native greedy mask.  The two gains are combined by a
geometric mean while the shared feasibility sentinel remains unchanged.
"""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import UNIFIED_PARETO_STAGE


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_PARETO_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_PARETO_STAGE)
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
        "positive_only_credit": True,
        "tail_reward_mode": "raw_ciou",
        "positive_reward": "raw_ciou",
        "advantage": "pareto_geometry_improvement_geomean_normalized",
        "pareto_credit_mode": "pareto_geometry_improvement",
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
