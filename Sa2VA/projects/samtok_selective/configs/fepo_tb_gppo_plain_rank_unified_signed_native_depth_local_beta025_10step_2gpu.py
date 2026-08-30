"""R20: signed, asymmetric native-relative depth-local SAMTok credit.

The geometry reward and frozen SAMTok initialization stay unchanged.  R20
keeps both positive and negative native-relative gains, weights cIoU 0.75 and
boundary IoU 0.25, and assigns the signed signal to the first changed code
depth.  The beta value is contract-locked; this file is not a sweep config.
"""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE,
)


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["stage"] = UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(
        UNIFIED_SIGNED_NATIVE_DEPTH_LOCAL_STAGE
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
        "pareto_credit_mode": "asymmetric_signed_native_depth_local",
        "depth_local_credit_policy": "signed_native_relative_asymmetric",
        "advantage": "signed_native_relative_depth_local_mean_abs_normalized",
        "positive_only_credit": False,
        "depth_local_decay": 0.85,
        "depth_local_beta": 0.25,
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
    }
)
