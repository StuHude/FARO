import os

from projects.samtok_selective.config import build_config


config = build_config(
    continue_from=os.environ.get("SAMTOK_STANDALONE_ADAPTER"),
    stage="continued_sft_control20",
)
config["optimizer"].update(
    {
        "lr": 5e-7,
        "warmup_ratio": 0.0,
        "max_steps": 20,
        "grad_accum_steps": 1,
    }
)
config["checkpoint"]["save_every"] = 0
