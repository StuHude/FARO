from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .gr_cppo_contract import validate_frozen_anchor
from .greedy_relative_gr_cppo_contract import (
    METHOD as RELATIVE_METHOD,
    STAGE as RELATIVE_STAGE,
    TWENTY_STEP_STAGE as RELATIVE_TWENTY_STEP_STAGE,
    validate_greedy_relative_gr_cppo_config,
)


METHOD = "standalone_samtok_sign_balanced_greedy_relative_es_ppo"
STAGE = "fepo_sign_balanced_es_ppo_one_step_2gpu"
TWENTY_STEP_STAGE = "fepo_sign_balanced_es_ppo_20step_2gpu"
STAGES = {STAGE: 1, TWENTY_STEP_STAGE: 20}


def validate_sign_balanced_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported sign-balanced stage: {stage}")
    method = config.get("sign_balanced_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Sign-balanced method must be {METHOD}")
    if method.get("advantage") != "greedy_delta_equal_sign_l1_mass":
        raise ValueError("Sign-balanced advantage is not registered")

    inherited = copy.deepcopy(config)
    inherited["stage"] = (
        RELATIVE_STAGE if STAGES[stage] == 1 else RELATIVE_TWENTY_STEP_STAGE
    )
    inherited["optimizer"]["max_steps"] = STAGES[stage]
    inherited["checkpoint"]["output_dir"] = str(
        repo_root / "outputs" / "samtok_selective" / inherited["stage"]
    )
    inherited["provenance"]["manifest_path"] = str(
        Path(inherited["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    inherited_method = copy.deepcopy(method)
    inherited_method["method"] = RELATIVE_METHOD
    inherited_method["advantage"] = "greedy_reward_delta_mean_abs_normalized"
    inherited["greedy_relative_entropy_gr_cppo"] = inherited_method
    inherited.pop("sign_balanced_entropy_gr_cppo", None)
    validate_greedy_relative_gr_cppo_config(inherited, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Sign-balanced output must be {expected_output}")


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
