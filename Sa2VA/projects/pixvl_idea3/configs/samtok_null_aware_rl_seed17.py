from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_null_aware_rl_8gpu.py"))["config"]
)
config["seed"] = 17
config["stage"] = "samtok_null_aware_rl_seed17"
config["run_name"] = "samtok_null_aware_rl_seed17"
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_null_aware_rl_seed17"
)
