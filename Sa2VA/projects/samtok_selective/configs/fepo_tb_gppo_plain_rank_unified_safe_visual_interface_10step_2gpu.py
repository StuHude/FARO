"""R35: visual-interface FEPO with fixed abstention protection.

This is an isolated follow-up to R30.  Only the visual merger/deepstack LoRA
is trainable; the geometry rollout and native-relative credit stay unchanged.
The registered change is a stronger fixed null CE and first-action margin
constraint, motivated by the R30 sentinel-margin failure.
"""

import copy
from pathlib import Path

from projects.samtok_selective.configs.tail_gppo_common import build_tail_gppo_config
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_SAFE_VISUAL_INTERFACE_STAGE,
)


config = copy.deepcopy(build_tail_gppo_config("plain_rank", 10))
config["seed"] = 17
config["stage"] = UNIFIED_SAFE_VISUAL_INTERFACE_STAGE
config["data"]["jsonl"] = str(
    Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl")
)
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_SAFE_VISUAL_INTERFACE_STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
config["model"]["adapter_mode"] = "frozen_anchor_plus_visual_projector"
config["lora"].update({"visual_r": 16, "visual_alpha": 32, "visual_dropout": 0.0})
config["representation"] = {
    "anchor_adapter": "frozen",
    "trainable_adapter": "visual",
    "target_scope": "visual.merger_and_deepstack_mergers_only",
    "expected_target_linears": 8,
    "preupdate_equivalence_tolerance": 1e-5,
}
config["tail_gppo"].update(
    {
        "unified_sentinel": True,
        "selective_risk_mode": "fixed_training_sentinel_active_set",
        "sentinel_source": "registered_tail_no_target_ids",
        "anchor_budget_source": "frozen_initialization_pre_update",
        "null_ce_relative_slack": 0.05,
        "null_ce_absolute_slack": 0.02,
        "margin_slack": 0.05,
        "null_ce_weight": 2.0,
        "margin_weight": 1.0,
        "holdout_access": False,
        "pareto_credit_mode": "native_anchored_rank_local",
        "depth_local_credit_policy": "native_reference_midrank_first_divergence",
        "depth_local_decay": 0.85,
        "minimum_improvement": 1e-4,
        "advantage_epsilon": 1e-6,
        "grounded_interface": {
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
        },
    }
)
