import os

from projects.samtok_selective.config import REPO_ROOT, build_config
from projects.samtok_selective.gr_cppo_contract import METHOD, expected_frozen_anchor


stage = "fepo_gr_cppo_one_step_2gpu"
adapter = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER",
    str(expected_frozen_anchor(REPO_ROOT)),
)
config = build_config(continue_from=adapter, stage=stage)
config["optimizer"].update(
    {"lr": 5e-7, "warmup_ratio": 0.0, "max_steps": 1, "grad_accum_steps": 1}
)
config["checkpoint"].update({"save_every": 0, "adapter_init": adapter})
config["gr_cppo"] = {
    "method": METHOD,
    "rollouts_per_prompt": 4,
    "policy_epochs": 2,
    "rollout_grammar": "mask_start_code_by_depth_mask_end",
    "multimodal_batching": "processor_reencode_one_image_per_rollout",
    "behavior_logprob": "detached_rollout_policy",
    "ppo_action_logprob_scope": "sampled_depth_specific_code_tokens_only",
    "forced_boundary_probability": 1.0,
    "advantage": "group_standardized",
    "positive_reward": "plain_ciou",
    "negative_objective": "canonical_no_target_ce",
    "margin_constraint": "first_null_token_vs_mask_start_hinge",
    "clip_epsilon": 0.2,
    "temperature": 1.0,
    "policy_weight": 1.0,
    "null_ce_weight": 1.0,
    "margin_weight": 0.25,
    "margin_target": 0.0,
    "require_nonconstant_rewards": True,
    "reward_std_epsilon": 1e-6,
    "require_epoch2_ratio_change": True,
    "min_epoch2_ratio_abs_deviation": 1e-8,
}
