import os

from projects.samtok_selective.config import build_config
from projects.samtok_selective.gr_cppo_contract import expected_frozen_anchor


anchor = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER", str(expected_frozen_anchor())
)
config = build_config(continue_from=anchor, stage="continued_sft_es_control40")
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
