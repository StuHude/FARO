import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config


STAGE = "fepo_tb_gppo_tail_balanced_active_set_10step_2gpu"
config = copy.deepcopy(build_tail_gppo_config("tail_balanced", 20))
config["stage"] = STAGE
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
method = config["tail_gppo"]
method.update(
    {
        "selective_risk_mode": "fixed_training_sentinel_active_set",
        "sentinel_source": "sorted_training_no_target_ids",
        "anchor_budget_source": "frozen_initialization_pre_update",
        "null_ce_relative_slack": 0.05,
        "null_ce_absolute_slack": 0.02,
        "margin_slack": 0.05,
        "holdout_access": False,
    }
)
