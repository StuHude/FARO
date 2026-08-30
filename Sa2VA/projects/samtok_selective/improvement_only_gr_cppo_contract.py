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


METHOD = "standalone_samtok_improvement_only_effective_support_ppo"
STAGE = "fepo_improvement_only_es_ppo_one_step_2gpu"
TWENTY_STEP_STAGE = "fepo_improvement_only_es_ppo_20step_2gpu"
STAGES = {STAGE: 1, TWENTY_STEP_STAGE: 20}


def validate_improvement_only_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported improvement-only stage: {stage}")
    method = config.get("improvement_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Improvement-only method must be {METHOD}")
    if method.get("advantage") != "positive_greedy_improvement_mean_normalized":
        raise ValueError("Improvement-only advantage is not registered")
    if method.get("positive_reward") != "plain_ciou":
        raise ValueError("Improvement-only reward must remain plain cIoU")
    floats = {"minimum_improvement": 1e-4, "advantage_epsilon": 1e-6}
    for key, expected in floats.items():
        value = float(method.get(key, float("nan")))
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Improvement-only requires {key}={expected}")

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
    inherited_method["advantage"] = "group_standardized"
    inherited_method.pop("minimum_improvement", None)
    inherited_method.pop("advantage_epsilon", None)
    inherited["entropy_gr_cppo"] = inherited_method
    inherited.pop("improvement_entropy_gr_cppo", None)
    validate_entropy_gr_cppo_config(inherited, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Improvement-only output must be {expected_output}")


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
