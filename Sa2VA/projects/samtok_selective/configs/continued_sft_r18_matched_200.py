"""200-update continued-SFT control matched to R18's 100x2 policy updates."""

import os
from pathlib import Path

from projects.samtok_selective.config import build_config
from projects.samtok_selective.gr_cppo_contract import expected_frozen_anchor


anchor = os.environ.get("SAMTOK_STANDALONE_ADAPTER", str(expected_frozen_anchor()))
config = build_config(continue_from=anchor, stage="continued_sft_r18_matched_200")
config["data"]["jsonl"] = str(Path(config["data"]["jsonl"]).with_name("egfepo_train_5120.jsonl"))
config["data"]["expected_rows"] = 5120
config["data"]["expected_no_target_rows"] = 2560
config["data"]["pairs_per_device_batch"] = 4
config["optimizer"].update(
    {
        "lr": 5e-7,
        "warmup_ratio": 0.0,
        "max_steps": 200,
        "grad_accum_steps": 1,
        "updates_per_batch": 1,
    }
)
config["checkpoint"]["save_every"] = 0
