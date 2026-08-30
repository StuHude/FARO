from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .entropy_gr_cppo_contract import (
    METHOD as ES_METHOD,
    STAGE as ES_STAGE,
    TWENTY_STEP_STAGE as ES_TWENTY_STEP_STAGE,
    validate_entropy_gr_cppo_config,
)
from .gr_cppo_contract import validate_frozen_anchor


METHOD = "standalone_samtok_greedy_crossing_online_preference"
STAGE = "fepo_greedy_preference_one_step_2gpu"
TWENTY_STEP_STAGE = "fepo_greedy_preference_20step_2gpu"
STAGES = {STAGE: 1, TWENTY_STEP_STAGE: 20}


def validate_greedy_preference_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported greedy-preference stage: {stage}")
    method = config.get("greedy_preference_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Greedy-preference method must be {METHOD}")
    required = {
        "preference_pair": "best_improving_sample_vs_native_greedy",
        "preference_loss": "softplus_negative_native_log_odds_shift",
        "positive_reward": "plain_ciou",
    }
    for key, expected in required.items():
        if method.get(key) != expected:
            raise ValueError(f"Greedy preference requires {key}={expected!r}")
    floats = {
        "minimum_improvement": 1e-4,
        "native_scoring_temperature": 1.0,
        "max_epoch0_ratio_deviation": 0.01,
    }
    for key, expected in floats.items():
        value = float(method.get(key, float("nan")))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Greedy preference requires {key}={expected}")

    inherited = copy.deepcopy(config)
    inherited["stage"] = ES_STAGE if STAGES[stage] == 1 else ES_TWENTY_STEP_STAGE
    inherited["optimizer"]["max_steps"] = STAGES[stage]
    inherited["checkpoint"]["output_dir"] = str(
        repo_root / "outputs" / "samtok_selective" / inherited["stage"]
    )
    inherited["provenance"]["manifest_path"] = str(
        Path(inherited["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    inherited_method = copy.deepcopy(method)
    inherited_method["method"] = ES_METHOD
    inherited_method.pop("preference_pair", None)
    inherited_method.pop("preference_loss", None)
    inherited_method.pop("minimum_improvement", None)
    inherited_method.pop("native_scoring_temperature", None)
    inherited_method.pop("max_epoch0_ratio_deviation", None)
    inherited["entropy_gr_cppo"] = inherited_method
    inherited.pop("greedy_preference_entropy_gr_cppo", None)
    validate_entropy_gr_cppo_config(inherited, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Greedy-preference output must be {expected_output}")


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
