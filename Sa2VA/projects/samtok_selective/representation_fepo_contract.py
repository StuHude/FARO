from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .entropy_gr_cppo_contract import (
    STAGE as ES_ONE_STEP_STAGE,
    TEN_STEP_STAGE as ES_TEN_STEP_STAGE,
    TWENTY_STEP_STAGE as ES_TWENTY_STEP_STAGE,
    validate_entropy_gr_cppo_config,
)


ONE_STEP_STAGE = "fepo_projector_plastic_one_step_2gpu"
TEN_STEP_STAGE = "fepo_projector_plastic_10step_2gpu"
TWENTY_STEP_STAGE = "fepo_projector_plastic_20step_2gpu"
STAGES = {
    ONE_STEP_STAGE: (1, ES_ONE_STEP_STAGE),
    TEN_STEP_STAGE: (10, ES_TEN_STEP_STAGE),
    TWENTY_STEP_STAGE: (20, ES_TWENTY_STEP_STAGE),
}
ADAPTER_MODE = "frozen_anchor_plus_visual_projector"
VISUAL_R = 16
VISUAL_ALPHA = 32
VISUAL_DROPOUT = 0.0


def validate_representation_fepo_config(
    config: dict[str, Any], repo_root: str | Path = REPO_ROOT
) -> None:
    repo_root = Path(repo_root).resolve()
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Unsupported representation-aware FEPO stage: {stage}")
    steps, shadow_stage = STAGES[stage]
    if int(config["optimizer"]["max_steps"]) != steps:
        raise ValueError(f"{stage} must run exactly {steps} outer steps")

    shadow = deepcopy(config)
    shadow["stage"] = shadow_stage
    shadow["entropy_gr_cppo"] = shadow.pop("representation_entropy_gr_cppo")
    shadow_output = repo_root / "outputs" / "samtok_selective" / shadow_stage
    shadow["checkpoint"]["output_dir"] = str(shadow_output)
    shadow["provenance"]["manifest_path"] = str(
        shadow_output / "provenance_manifest.json"
    )
    validate_entropy_gr_cppo_config(shadow, repo_root)

    expected_output = repo_root / "outputs" / "samtok_selective" / str(stage)
    if Path(config["checkpoint"]["output_dir"]).resolve() != expected_output:
        raise ValueError(f"Representation-aware FEPO output must be {expected_output}")
    if config["model"].get("adapter_mode") != ADAPTER_MODE:
        raise ValueError(f"Representation-aware FEPO requires adapter_mode={ADAPTER_MODE}")
    lora = config["lora"]
    exact_lora = {
        "visual_r": VISUAL_R,
        "visual_alpha": VISUAL_ALPHA,
        "visual_dropout": VISUAL_DROPOUT,
    }
    for key, expected in exact_lora.items():
        if float(lora.get(key, float("nan"))) != float(expected):
            raise ValueError(f"Representation-aware FEPO requires {key}={expected}")
    representation = config.get("representation")
    expected_representation = {
        "anchor_adapter": "frozen",
        "trainable_adapter": "visual",
        "target_scope": "visual.merger_and_deepstack_mergers_only",
        "expected_target_linears": 8,
        "preupdate_equivalence_tolerance": 1e-5,
    }
    if representation != expected_representation:
        raise ValueError("Representation-aware FEPO mechanism contract changed")
