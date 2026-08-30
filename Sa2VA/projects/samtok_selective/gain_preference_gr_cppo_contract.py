from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .gr_cppo_contract import validate_frozen_anchor
from .greedy_preference_gr_cppo_contract import (
    METHOD as UNIFORM_METHOD,
    STAGE as UNIFORM_STAGE,
    TWENTY_STEP_STAGE as UNIFORM_TWENTY_STEP_STAGE,
    validate_greedy_preference_gr_cppo_config,
)


METHOD = "standalone_samtok_gain_calibrated_greedy_preference"
STAGE = "fepo_gain_preference_one_step_2gpu"
TWENTY_STEP_STAGE = "fepo_gain_preference_20step_2gpu"
STAGES = {STAGE: 1, TWENTY_STEP_STAGE: 20}


def validate_gain_preference_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported gain-preference stage: {stage}")
    method = config.get("gain_preference_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Gain-preference method must be {METHOD}")
    if method.get("preference_weighting") != "active_ciou_gain_mean_normalized":
        raise ValueError("Gain preference requires mean-normalized active cIoU gain")

    inherited = copy.deepcopy(config)
    inherited["stage"] = (
        UNIFORM_STAGE if STAGES[stage] == 1 else UNIFORM_TWENTY_STEP_STAGE
    )
    inherited["optimizer"]["max_steps"] = STAGES[stage]
    inherited["checkpoint"]["output_dir"] = str(
        repo_root / "outputs" / "samtok_selective" / inherited["stage"]
    )
    inherited["provenance"]["manifest_path"] = str(
        Path(inherited["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    inherited_method = copy.deepcopy(method)
    inherited_method["method"] = UNIFORM_METHOD
    inherited_method.pop("preference_weighting", None)
    inherited["greedy_preference_entropy_gr_cppo"] = inherited_method
    inherited.pop("gain_preference_entropy_gr_cppo", None)
    validate_greedy_preference_gr_cppo_config(inherited, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Gain-preference output must be {expected_output}")


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
