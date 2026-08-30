"""Minimal test of candidate-group soft local credit under matched FEPO data."""

from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_softlocal_group4_smoke_8gpu"
config["run_name"] = "fepo_softlocal_group4_smoke_8gpu"
config["routing"]["mode"] = "predicted_only_evidence"
config["routing"]["credit_assignment"] = "soft_local"
config["routing"]["soft_credit_gain"] = 1.0
config["rl"]["group_size"] = {"refseg": 4, "maskcap": 4}
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["save_every"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_softlocal_group4_smoke_8gpu"
)
