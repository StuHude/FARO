"""Matched shared control for the group-4 soft-local smoke."""

from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_shared_group4_smoke_8gpu"
config["run_name"] = "fepo_shared_group4_smoke_8gpu"
config["routing"]["mode"] = "shared"
config["rl"]["group_size"] = {"refseg": 4, "maskcap": 4}
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["save_every"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_shared_group4_smoke_8gpu"
)
