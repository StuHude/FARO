import os

from projects.samtok_selective.config import build_config


config = build_config(
    continue_from=os.environ.get("SAMTOK_STANDALONE_ADAPTER"),
    stage="continued_sft_to500",
)
config["optimizer"].update(
    {
        "max_steps": 200,
        "warmup_ratio": 0.05,
    }
)
config["checkpoint"]["save_every"] = 0
