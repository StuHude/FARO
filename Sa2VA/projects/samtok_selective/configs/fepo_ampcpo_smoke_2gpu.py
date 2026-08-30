import os

from projects.samtok_selective.ampcpo_contract import STAGE, expected_continued_adapter
from projects.samtok_selective.config import REPO_ROOT, build_config


adapter = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER",
    str(expected_continued_adapter(REPO_ROOT)),
)
config = build_config(continue_from=adapter, stage=STAGE)
config["optimizer"].update(
    {
        "lr": 5e-7,
        "warmup_ratio": 0.0,
        "max_steps": 20,
        "grad_accum_steps": 1,
    }
)
config["checkpoint"].update({"save_every": 0, "adapter_init": adapter})
config["ampcpo"] = {
    "method": "standalone_samtok_am_cppo",
    "positive_reward": "plain_ciou",
    "policy_surrogate": "teacher_forced_greedy_mask_action_cppo",
    "negative_objective": "canonical_no_target_ce",
    "margin_constraint": "first_null_token_vs_mask_start_hinge",
    "clip_epsilon": 0.2,
    "policy_weight": 1.0,
    "null_ce_weight": 1.0,
    "margin_weight": 0.25,
    "margin_target": 0.0,
}
