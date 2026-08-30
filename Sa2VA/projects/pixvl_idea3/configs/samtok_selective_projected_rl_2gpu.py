from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_sft_anchor_rl_2gpu.py"))["config"]
)
ROOT = Path(config["paths"]["output_root"])
CONTINUED_SFT_ANCHOR = (
    ROOT / "samtok_selective_sft_anchor_continue_sft_2gpu" / "adapter"
)

config["stage"] = "samtok_selective_projected_rl_from_contsft_2gpu"
config["run_name"] = "samtok_selective_projected_rl_from_contsft_2gpu"
config["student_init"] = {"adapter_path": str(CONTINUED_SFT_ANCHOR)}
config["reference"] = {"adapter_path": str(CONTINUED_SFT_ANCHOR)}
config["routing"]["mode"] = "shared"
config["loss"]["lambda_opd"] = 0.0
config["selective_outcome_loss_scales"] = {
    "positive": {"ce": 1.0, "rl": 1.0, "kl": 1.0},
    # Global lambda_ce=0.3 and beta_kl=0.02.  These values restore unit
    # no-target CE and a 0.10 anchor KL while retaining binary null RL.
    "negative": {"ce": 10.0 / 3.0, "rl": 1.0, "kl": 5.0},
}
config["risk_constraint"] = {"enabled": False}
config["selective_gradient_projection"] = {
    "enabled": True,
    "objective": "target_geometry_policy",
    "constraint": "no_target_mask_or_null_loss",
    "epsilon": 1e-12,
}
config["checkpoint"]["output_dir"] = str(
    ROOT / "samtok_selective_projected_rl_from_contsft_2gpu"
)
