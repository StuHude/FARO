from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_relation_margin_8gpu"
config["run_name"] = "fepo_relation_margin_8gpu"
config["routing"]["mode"] = "predicted_only_evidence"
config["routing"]["rewards"]["relation"].update(
    {"target_weight": 0.3, "margin_weight": 0.6, "exact_weight": 0.1}
)
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["save_every"] = 50
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_relation_margin_8gpu"
)
