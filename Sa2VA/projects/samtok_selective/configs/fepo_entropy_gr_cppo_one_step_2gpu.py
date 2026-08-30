import os

from projects.samtok_selective.config import REPO_ROOT, build_config
from projects.samtok_selective.entropy_gr_cppo_contract import (
    METHOD,
    STAGE,
    SUPPORT_SIZE,
    TARGET_EFFECTIVE_SUPPORT,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    CALIBRATION_ITERATIONS,
)
from projects.samtok_selective.gr_cppo_contract import expected_frozen_anchor


adapter = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER",
    str(expected_frozen_anchor(REPO_ROOT)),
)
config = build_config(continue_from=adapter, stage=STAGE)
config["optimizer"].update(
    {"lr": 5e-7, "warmup_ratio": 0.0, "max_steps": 1, "grad_accum_steps": 1}
)
config["data"]["pairs_per_device_batch"] = 4
config["checkpoint"].update({"save_every": 0, "adapter_init": adapter})
config["entropy_gr_cppo"] = {
    "method": METHOD,
    "rollouts_per_prompt": 4,
    "policy_epochs": 2,
    "rollout_grammar": "mask_start_code_by_depth_mask_end",
    "multimodal_batching": "processor_reencode_one_image_per_rollout",
    "behavior_logprob": "detached_rollout_policy",
    "ppo_action_logprob_scope": "sampled_depth_specific_code_tokens_only",
    "forced_boundary_probability": 1.0,
    "exploration": "per_prefix_topm_collision_support",
    "support_size": SUPPORT_SIZE,
    "target_effective_support": TARGET_EFFECTIVE_SUPPORT,
    "temperature_min": TEMPERATURE_MIN,
    "temperature_max": TEMPERATURE_MAX,
    "calibration_iterations": CALIBRATION_ITERATIONS,
    "temperature_selection_data": "training_logits_only_no_holdout_tuning",
    "rescore_policy": "frozen_old_support_and_temperature",
    "effective_support_tolerance": 0.05,
    "min_target_support_reached_fraction": 1.0,
    "min_multitrajectory_groups": 6,
    "min_nonconstant_reward_groups": 2,
    "min_improved_over_greedy_rollouts": 1,
    "min_positive_policy_grad_norm": 1e-12,
    "advantage": "group_standardized",
    "positive_reward": "plain_ciou",
    "negative_objective": "canonical_no_target_ce",
    "margin_constraint": "first_null_token_vs_mask_start_hinge",
    "clip_epsilon": 0.2,
    "policy_weight": 1.0,
    "null_ce_weight": 1.0,
    "margin_weight": 0.25,
    "margin_target": 0.0,
    "require_nonconstant_rewards": True,
    "reward_std_epsilon": 1e-6,
    "require_epoch2_ratio_change": True,
    "min_epoch2_ratio_abs_deviation": 1e-8,
}
