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
from .tail_geometry import BOUNDARY_WIDTH


METHOD = "standalone_samtok_boundary_credit_effective_support_gr_cppo"
STAGE = "fepo_boundary_credit_es_gr_cppo_one_step_2gpu"
TEN_STEP_STAGE = "fepo_boundary_evidence_10step_2gpu"
TEN_STEP_NONE_STAGE = "fepo_boundary_none_10step_2gpu"
TWENTY_STEP_STAGE = "fepo_boundary_credit_es_gr_cppo_20step_2gpu"
STAGES = {
    STAGE: 1,
    TEN_STEP_STAGE: 10,
    TEN_STEP_NONE_STAGE: 10,
    TWENTY_STEP_STAGE: 20,
}


def validate_boundary_credit_gr_cppo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported boundary-credit stage: {stage}")
    method = config.get("boundary_entropy_gr_cppo")
    if not isinstance(method, dict) or method.get("method") != METHOD:
        raise ValueError(f"Boundary-credit method must be {METHOD}")
    if method.get("positive_reward") != "raw_half_ciou_half_boundary_iou":
        raise ValueError("Boundary-credit reward is not registered")
    for key in ("ciou_weight", "boundary_iou_weight"):
        if not math.isclose(float(method.get(key, float("nan"))), 0.5, abs_tol=1e-12):
            raise ValueError(f"Boundary-credit requires {key}=0.5")
    if int(method.get("boundary_width", -1)) != BOUNDARY_WIDTH:
        raise ValueError(f"Boundary-credit requires boundary_width={BOUNDARY_WIDTH}")

    inherited = copy.deepcopy(config)
    inherited["stage"] = {
        1: ES_STAGE,
        10: "fepo_es_gr_cppo_10step_2gpu",
        20: ES_TWENTY_STEP_STAGE,
    }[STAGES[stage]]
    inherited["optimizer"]["max_steps"] = STAGES[stage]
    inherited["checkpoint"]["output_dir"] = str(
        repo_root / "outputs" / "samtok_selective" / inherited["stage"]
    )
    inherited["provenance"]["manifest_path"] = str(
        Path(inherited["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    inherited_method = copy.deepcopy(method)
    inherited_method["method"] = ES_METHOD
    inherited_method["positive_reward"] = "plain_ciou"
    for key in ("ciou_weight", "boundary_iou_weight", "boundary_width"):
        inherited_method.pop(key, None)
    inherited["entropy_gr_cppo"] = inherited_method
    inherited.pop("boundary_entropy_gr_cppo", None)
    validate_entropy_gr_cppo_config(inherited, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / stage
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Boundary-credit output must be {expected_output}")


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
