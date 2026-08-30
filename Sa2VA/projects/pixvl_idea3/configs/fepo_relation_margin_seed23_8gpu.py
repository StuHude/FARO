from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_relation_margin_8gpu.py"))["config"]
)
config["seed"] = 23
config["stage"] = "fepo_relation_margin_seed23_8gpu"
config["run_name"] = "fepo_relation_margin_seed23_8gpu"
config["optimizer"]["max_steps"] = 200
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_relation_margin_seed23_8gpu"
)
