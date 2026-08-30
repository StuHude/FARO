from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, validate_config
from .gr_cppo_contract import expected_frozen_anchor, validate_frozen_anchor


METHOD = "standalone_samtok_effective_support_grammar_rollout_cppo"
STAGE = "fepo_es_gr_cppo_one_step_2gpu"
TEN_STEP_STAGE = "fepo_es_gr_cppo_10step_2gpu"
TWENTY_STEP_STAGE = "fepo_es_gr_cppo_20step_2gpu"
STAGES = {STAGE: 1, TEN_STEP_STAGE: 10, TWENTY_STEP_STAGE: 20}
ROLLOUTS_PER_PROMPT = 4
POLICY_EPOCHS = 2
SUPPORT_SIZE = 8
TARGET_EFFECTIVE_SUPPORT = 4.0
TEMPERATURE_MIN = 1.0
TEMPERATURE_MAX = 128.0
CALIBRATION_ITERATIONS = 24


def validate_entropy_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unsupported ES-GR-CPPO stage: {stage}")
    if int(config["optimizer"]["max_steps"]) != STAGES[stage]:
        raise ValueError(f"{stage} must run exactly {STAGES[stage]} outer steps")
    if int(config["optimizer"].get("grad_accum_steps", 0)) != 1:
        raise ValueError("Entropy GR-CPPO requires grad_accum_steps=1")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("Entropy GR-CPPO requires exactly two processes")
    if int(config["data"].get("pairs_per_device_batch", 0)) != 4:
        raise ValueError("ES-GR-CPPO gate requires four pairs per process")
    expected_adapter = expected_frozen_anchor(repo_root)
    if Path(config["checkpoint"].get("adapter_init") or "").resolve() != expected_adapter:
        raise ValueError(f"Entropy GR-CPPO must initialize from {expected_adapter}")
    expected_output = repo_root / "outputs" / "samtok_selective" / str(stage)
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Entropy GR-CPPO output must be {expected_output}")

    method = config.get("entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Entropy GR-CPPO method must be {METHOD}")
    exact = {
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "policy_epochs": POLICY_EPOCHS,
        "calibration_iterations": CALIBRATION_ITERATIONS,
        "support_size": SUPPORT_SIZE,
    }
    for key, expected in exact.items():
        if int(method.get(key, -1)) != expected:
            raise ValueError(f"Entropy GR-CPPO requires {key}={expected}")
    if method.get("exploration") != "per_prefix_topm_collision_support":
        raise ValueError("Entropy GR-CPPO exploration policy is not registered")
    if method.get("temperature_selection_data") != "training_logits_only_no_holdout_tuning":
        raise ValueError("Entropy temperature calibration must not use holdout feedback")
    if method.get("rescore_policy") != "frozen_old_support_and_temperature":
        raise ValueError("PPO rescoring must reuse old support and temperature")
    float_contract = {
        "target_effective_support": TARGET_EFFECTIVE_SUPPORT,
        "temperature_min": TEMPERATURE_MIN,
        "temperature_max": TEMPERATURE_MAX,
    }
    for key, expected in float_contract.items():
        value = float(method.get(key, float("nan")))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Entropy GR-CPPO requires {key}={expected}")
    if method.get("rollout_grammar") != "mask_start_code_by_depth_mask_end":
        raise ValueError("Entropy GR-CPPO rollout grammar is not registered")
    if method.get("ppo_action_logprob_scope") != "sampled_depth_specific_code_tokens_only":
        raise ValueError("Entropy GR-CPPO ratios must score sampled code actions only")
    if float(method.get("forced_boundary_probability", -1.0)) != 1.0:
        raise ValueError("Forced grammar boundaries must have probability one")
    if method.get("advantage") != "group_standardized":
        raise ValueError("Entropy GR-CPPO requires group-standardized advantages")
    if method.get("positive_reward") != "plain_ciou":
        raise ValueError("Entropy GR-CPPO positive reward must be plain_ciou")
    if method.get("negative_objective") != "canonical_no_target_ce":
        raise ValueError("Entropy GR-CPPO requires canonical no-target CE")
    if method.get("margin_constraint") != "first_null_token_vs_mask_start_hinge":
        raise ValueError("Entropy GR-CPPO first-action margin is not registered")
    if method.get("require_nonconstant_rewards") is not True:
        raise ValueError("Entropy GR-CPPO requires a nonconstant reward group")
    if method.get("require_epoch2_ratio_change") is not True:
        raise ValueError("Entropy GR-CPPO requires a changed epoch-2 ratio")
    if not 0.0 < float(method.get("clip_epsilon", -1.0)) < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    for key in (
        "policy_weight",
        "null_ce_weight",
        "margin_weight",
        "reward_std_epsilon",
        "min_epoch2_ratio_abs_deviation",
        "effective_support_tolerance",
        "min_target_support_reached_fraction",
        "min_multitrajectory_groups",
        "min_nonconstant_reward_groups",
        "min_improved_over_greedy_rollouts",
        "min_positive_policy_grad_norm",
    ):
        value = float(method.get(key, float("nan")))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    reached = float(method["min_target_support_reached_fraction"])
    if not 0.0 < reached <= 1.0:
        raise ValueError("min_target_support_reached_fraction must be in (0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args()
    identity = validate_frozen_anchor(
        args.adapter,
        repo_root=args.repo_root,
        hash_model=not args.skip_model_hash,
    )
    print(json.dumps({"status": "ok", "initialization": identity}, sort_keys=True))


if __name__ == "__main__":
    main()
