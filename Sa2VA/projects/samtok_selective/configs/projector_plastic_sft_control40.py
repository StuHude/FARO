import os

from projects.samtok_selective.config import build_config
from projects.samtok_selective.gr_cppo_contract import expected_frozen_anchor
from projects.samtok_selective.representation_fepo_contract import (
    ADAPTER_MODE,
    VISUAL_ALPHA,
    VISUAL_DROPOUT,
    VISUAL_R,
)


anchor = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER", str(expected_frozen_anchor())
)
config = build_config(continue_from=anchor, stage="projector_plastic_sft_control40")
config["model"]["adapter_mode"] = ADAPTER_MODE
config["lora"].update(
    {
        "visual_r": VISUAL_R,
        "visual_alpha": VISUAL_ALPHA,
        "visual_dropout": VISUAL_DROPOUT,
    }
)
config["data"]["pairs_per_device_batch"] = 4
config["optimizer"].update(
    {
        "lr": 5e-7,
        "warmup_ratio": 0.0,
        "max_steps": 40,
        "grad_accum_steps": 1,
        "updates_per_batch": 2,
    }
)
config["checkpoint"]["save_every"] = 0
config["representation"] = {
    "anchor_adapter": "frozen",
    "trainable_adapter": "visual",
    "target_scope": "visual.merger_and_deepstack_mergers_only",
    "expected_target_linears": 8,
    "matched_rl_outer_batches": 20,
    "matched_optimizer_updates": 40,
}
