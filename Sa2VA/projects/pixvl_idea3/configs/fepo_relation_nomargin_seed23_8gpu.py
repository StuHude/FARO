from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_relation_margin_seed23_8gpu.py"))["config"]
)
config["stage"] = "fepo_relation_nomargin_seed23_8gpu"
config["run_name"] = "fepo_relation_nomargin_seed23_8gpu"
config["routing"]["rewards"]["relation"].update(
    {"target_weight": 0.85, "margin_weight": 0.0, "exact_weight": 0.15}
)
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_relation_nomargin_seed23_8gpu"
)
