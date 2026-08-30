from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_sft_anchor_rl_2gpu.py"))["config"]
)
ROOT = Path(config["paths"]["output_root"])

config["stage"] = "samtok_selective_anchor_relative_rl_2gpu"
config["run_name"] = "samtok_selective_anchor_relative_rl_2gpu"
config["routing"]["mode"] = "shared"
config["loss"]["lambda_opd"] = 0.0
config["rl"]["advantage_mode"] = "anchor_relative"
config["selective_outcome_loss_scales"] = {
    "positive": {"ce": 1.0, "rl": 1.0, "kl": 1.0},
    "negative": {"ce": 10.0 / 3.0, "rl": 1.0, "kl": 5.0},
}
config["risk_constraint"] = {"enabled": False}
config["selective_gradient_projection"] = {"enabled": False}
config["checkpoint"]["output_dir"] = str(
    ROOT / "samtok_selective_anchor_relative_rl_2gpu"
)
