from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, validate_config
from .entropy_gr_cppo_contract import (
    METHOD as ES_METHOD,
    STAGE as ES_STAGE,
    TWENTY_STEP_STAGE as ES_TWENTY_STEP_STAGE,
    validate_entropy_gr_cppo_config,
)
from .gr_cppo_contract import expected_frozen_anchor, validate_frozen_anchor


METHOD = "standalone_samtok_effective_support_active_set_grammar_rollout_cppo"
STAGE = "fepo_active_set_es_gr_cppo_one_step_2gpu"
TWENTY_STEP_STAGE = "fepo_active_set_es_gr_cppo_20step_2gpu"
STAGES = {STAGE: 1, TWENTY_STEP_STAGE: 20}


def derive_active_set_budgets(
    anchor_null_ce: float,
    anchor_margin_min: float,
    *,
    null_ce_relative_slack: float,
    null_ce_absolute_slack: float,
    margin_slack: float,
) -> tuple[float, float]:
    values = (
        anchor_null_ce,
        anchor_margin_min,
        null_ce_relative_slack,
        null_ce_absolute_slack,
        margin_slack,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Active-set budget inputs must be finite")
    if anchor_null_ce < 0.0:
        raise ValueError("Anchor null CE must be nonnegative")
    if min(null_ce_relative_slack, null_ce_absolute_slack, margin_slack) < 0.0:
        raise ValueError("Active-set budget slacks must be nonnegative")
    null_ce_budget = (
        anchor_null_ce * (1.0 + null_ce_relative_slack)
        + null_ce_absolute_slack
    )
    margin_budget = anchor_margin_min - margin_slack
    return null_ce_budget, margin_budget


def active_set_flags(
    current_null_ce: float,
    current_margin_min: float,
    *,
    null_ce_budget: float,
    margin_budget: float,
) -> tuple[bool, bool]:
    values = (current_null_ce, current_margin_min, null_ce_budget, margin_budget)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Active-set risk measurements must be finite")
    return current_null_ce > null_ce_budget, current_margin_min < margin_budget


def validate_active_set_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unsupported active-set stage: {stage}")
    if int(config["optimizer"]["max_steps"]) != STAGES[stage]:
        raise ValueError(f"{stage} must run exactly {STAGES[stage]} outer steps")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("Active-set ES-GR-CPPO requires exactly two processes")
    if int(config["data"].get("pairs_per_device_batch", 0)) != 4:
        raise ValueError("Active-set ES gate requires four pairs per process")
    expected_adapter = expected_frozen_anchor(repo_root)
    if Path(config["checkpoint"].get("adapter_init") or "").resolve() != expected_adapter:
        raise ValueError(f"Active-set ES must initialize from {expected_adapter}")
    expected_output = repo_root / "outputs" / "samtok_selective" / str(stage)
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Active-set ES output must be {expected_output}")

    method = config.get("active_set_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Active-set method must be {METHOD}")
    # Reuse the frozen effective-support contract for every rollout and PPO
    # field.  Only the risk constraint and stage/output namespace differ.
    inherited = copy.deepcopy(config)
    inherited["stage"] = (
        ES_STAGE if STAGES[stage] == 1 else ES_TWENTY_STEP_STAGE
    )
    inherited["optimizer"]["max_steps"] = STAGES[stage]
    inherited["checkpoint"]["output_dir"] = str(
        repo_root
        / "outputs"
        / "samtok_selective"
        / inherited["stage"]
    )
    inherited["provenance"]["manifest_path"] = str(
        Path(inherited["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    inherited["entropy_gr_cppo"] = copy.deepcopy(method)
    inherited["entropy_gr_cppo"]["method"] = ES_METHOD
    inherited.pop("active_set_entropy_gr_cppo", None)
    validate_entropy_gr_cppo_config(inherited, repo_root)

    if method.get("selective_risk_mode") != "fixed_training_sentinel_active_set":
        raise ValueError("Active-set selective-risk mode is not registered")
    if method.get("sentinel_source") != "sorted_training_no_target_ids":
        raise ValueError("Active-set sentinel must be sorted training no-target IDs")
    if method.get("anchor_budget_source") != "frozen_initialization_pre_update":
        raise ValueError("Active-set budgets must come from the frozen initialization")
    sentinel_rows = int(method.get("sentinel_rows_total", 0))
    if sentinel_rows != 8 or sentinel_rows % int(config["runtime"]["expected_world_size"]):
        raise ValueError("Active-set requires eight sentinel rows split over two ranks")
    for key in (
        "null_ce_relative_slack",
        "null_ce_absolute_slack",
        "margin_slack",
    ):
        value = float(method.get(key, float("nan")))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if method.get("holdout_access") is not False:
        raise ValueError("Active-set risk control must explicitly forbid holdout access")


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
