from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_sft_anchor_rl_2gpu.py"))["config"]
)
config["stage"] = "samtok_selective_risk_constrained_rl_2gpu"
config["run_name"] = "samtok_selective_risk_constrained_rl_2gpu"
config["selective_outcome_loss_scales"] = {
    "positive": {"ce": 1.0, "rl": 1.0, "kl": 1.0},
    # Global lambda_ce=0.3 and beta_kl=0.02; these scales restore SFT-strength
    # negative supervision and impose a tighter no-target trust region.
    "negative": {"ce": 10.0 / 3.0, "rl": 1.0, "kl": 5.0},
}
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_selective_risk_constrained_rl_2gpu"
)
config["risk_constraint"] = {
    "enabled": True,
    "outcome": "negative_null_ce",
    # Calibrate this budget from a frozen base rollout before GPU use.
    "budget": 1.5,
    "epsilon": 0.1,
    "lambda_init": 0.1,
    "dual_lr": 0.05,
    "lambda_max": 10.0,
    # This smoke uses dual ascent; it is not a gradient projection or exact
    # trust-region guarantee and must be reported as such.
    "projection": False,
}
