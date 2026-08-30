from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_token_scope_long_6gpu_corrected.py"))["config"]
)
config["seed"] = 17
config["stage"] = "fepo_token_scope_seed17_8gpu"
config["run_name"] = "fepo_token_scope_seed17_8gpu"
config["optimizer"]["max_steps"] = 200
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_token_scope_seed17_8gpu"
)
