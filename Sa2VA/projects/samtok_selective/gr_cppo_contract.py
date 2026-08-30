from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .ampcpo_contract import sha256_file
from .config import REPO_ROOT, validate_config
from .evidence_gate import validate_evidence_gate_config


METHOD = "standalone_samtok_grammar_rollout_cppo"
STAGES = {
    "fepo_gr_cppo_one_step_2gpu": 1,
    "fepo_gr_cppo_20step_2gpu": 20,
    "fepo_evidence_gated_one_step_2gpu": 1,
    "fepo_evidence_gated_10step_2gpu": 10,
    "fepo_evidence_gated_20step_2gpu": 20,
}
ROLLOUTS_PER_PROMPT = 4
POLICY_EPOCHS = 2
FROZEN_ANCHOR_STAGE = "continued_sft_to500"
FROZEN_ANCHOR_TOTAL_STEPS = 500
FROZEN_ANCHOR_LOCAL_STEPS = 200
FROZEN_ANCHOR_CONFIG_SHA256 = "862495c04a30965280d7ce18f199297f698e6403bfeda522feb8ab449cb66afa"
FROZEN_ANCHOR_MODEL_SHA256 = "7b409c9f2bc3cf2da61adb9c86270dcda6a1991082d5dca4bb1c8a593ea4dfed"


def expected_frozen_anchor(repo_root: str | Path = REPO_ROOT) -> Path:
    return (
        Path(repo_root).resolve()
        / "outputs"
        / "samtok_selective"
        / FROZEN_ANCHOR_STAGE
        / "adapter"
    )


def validate_frozen_anchor(
    adapter_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    hash_model: bool = True,
) -> dict[str, Any]:
    adapter = Path(adapter_path).resolve()
    expected = expected_frozen_anchor(repo_root)
    if adapter != expected:
        raise ValueError(f"GR-CPPO must use the frozen total-500-step anchor: {expected}")
    config_path = adapter / "adapter_config.json"
    model_path = adapter / "adapter_model.safetensors"
    for artifact in (config_path, model_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Frozen anchor is missing nonempty {artifact.name}: {adapter}")
    config_sha = sha256_file(config_path)
    if config_sha != FROZEN_ANCHOR_CONFIG_SHA256:
        raise ValueError("Frozen total-500-step anchor config SHA256 does not match")
    model_sha = sha256_file(model_path) if hash_model else None
    if model_sha is not None and model_sha != FROZEN_ANCHOR_MODEL_SHA256:
        raise ValueError("Frozen total-500-step anchor model SHA256 does not match")

    run_dir = adapter.parent
    metrics_path = run_dir / "metrics.json"
    provenance_path = run_dir / "provenance_manifest.json"
    if not metrics_path.is_file() or not provenance_path.is_file():
        raise ValueError("Frozen anchor requires metrics.json and provenance_manifest.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if metrics.get("stage") != FROZEN_ANCHOR_STAGE or metrics.get("status") != "finished":
        raise ValueError("Frozen anchor metrics must identify a finished continued_sft_to500 run")
    if metrics.get("steps_completed") != FROZEN_ANCHOR_LOCAL_STEPS:
        raise ValueError("Frozen anchor must record its final 200-step continuation")
    parent = provenance.get("initialization_adapter") or {}
    expected_parent = (
        Path(repo_root).resolve()
        / "outputs"
        / "samtok_selective"
        / "continued_sft_stage2"
        / "adapter"
    )
    if Path(parent.get("path") or "").resolve() != expected_parent:
        raise ValueError("Frozen anchor provenance does not identify continued_sft_stage2")
    return {
        "path": str(adapter),
        "anchor_stage": FROZEN_ANCHOR_STAGE,
        "registered_total_steps": FROZEN_ANCHOR_TOTAL_STEPS,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "metrics_sha256": sha256_file(metrics_path),
        "provenance_sha256": sha256_file(provenance_path),
    }


def validate_gr_cppo_config(config: dict[str, Any], repo_root: str | Path = REPO_ROOT) -> None:
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unsupported GR-CPPO stage: {stage}")
    if int(config["optimizer"]["max_steps"]) != STAGES[stage]:
        raise ValueError(f"{stage} must run exactly {STAGES[stage]} outer steps")
    if int(config["optimizer"].get("grad_accum_steps", 0)) != 1:
        raise ValueError("GR-CPPO requires grad_accum_steps=1")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("GR-CPPO requires exactly two processes")
    expected_adapter = expected_frozen_anchor(repo_root)
    if Path(config["checkpoint"].get("adapter_init") or "").resolve() != expected_adapter:
        raise ValueError(f"GR-CPPO must initialize from {expected_adapter}")
    method = config.get("gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"GR-CPPO method must be {METHOD}")
    expected_output = repo_root / "outputs" / "samtok_selective" / str(stage)
    evidence_gate = method.get("evidence_gate")
    if stage.startswith("fepo_evidence_gated_") and isinstance(evidence_gate, dict):
        expected_output = expected_output.with_name(
            f"{stage}_{evidence_gate.get('mode', 'view_drop')}"
        )
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"GR-CPPO output must be {expected_output}")

    if int(method.get("rollouts_per_prompt", 0)) != ROLLOUTS_PER_PROMPT:
        raise ValueError("GR-CPPO requires K=4 rollouts")
    if int(method.get("policy_epochs", 0)) != POLICY_EPOCHS:
        raise ValueError("GR-CPPO requires policy_epochs=2")
    if method.get("advantage") != "group_standardized":
        raise ValueError("GR-CPPO requires group-standardized advantages")
    if method.get("rollout_grammar") != "mask_start_code_by_depth_mask_end":
        raise ValueError("GR-CPPO rollout grammar is not registered")
    if method.get("ppo_action_logprob_scope") != "sampled_depth_specific_code_tokens_only":
        raise ValueError("GR-CPPO ratios must score sampled code actions only")
    if float(method.get("forced_boundary_probability", -1.0)) != 1.0:
        raise ValueError("Forced mask boundaries must have constrained probability one")
    if method.get("positive_reward") != "plain_ciou":
        raise ValueError("GR-CPPO positive reward must be plain_ciou")
    if method.get("negative_objective") != "canonical_no_target_ce":
        raise ValueError("GR-CPPO negative objective must be canonical_no_target_ce")
    if method.get("margin_constraint") != "first_null_token_vs_mask_start_hinge":
        raise ValueError("GR-CPPO first-action margin is not registered")
    clip = float(method.get("clip_epsilon", -1.0))
    if not 0.0 < clip < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    temperature = float(method.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    for key in ("policy_weight", "null_ce_weight", "margin_weight"):
        value = float(method.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(float(method.get("margin_target", float("nan")))):
        raise ValueError("margin_target must be finite")
    if method.get("require_nonconstant_rewards") is not True:
        raise ValueError("GR-CPPO gate requires nonconstant K=4 rewards")
    if method.get("require_epoch2_ratio_change") is not True:
        raise ValueError("GR-CPPO gate requires a changed post-update epoch-2 ratio")
    for key in ("reward_std_epsilon", "min_epoch2_ratio_abs_deviation"):
        value = float(method.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    evidence_gate = method.get("evidence_gate")
    if evidence_gate is not None:
        validate_evidence_gate_config(evidence_gate)
    if method.get("exploration") == "per_prefix_topm_collision_support":
        for key in (
            "support_size",
            "calibration_iterations",
            "min_nonconstant_reward_groups",
            "min_multitrajectory_groups",
            "min_improved_over_greedy_rollouts",
        ):
            if int(method.get(key, 0)) < 1:
                raise ValueError(f"effective-support exploration requires positive {key}")
        for key in (
            "target_effective_support",
            "temperature_min",
            "temperature_max",
            "effective_support_tolerance",
            "min_target_support_reached_fraction",
        ):
            value = float(method.get(key, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"effective-support exploration requires finite {key}")


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
