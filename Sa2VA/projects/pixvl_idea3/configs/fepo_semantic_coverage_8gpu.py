from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_semantic_coverage_8gpu"
config["run_name"] = "fepo_semantic_coverage_8gpu"
config["routing"]["mode"] = "predicted_only_evidence"
config["routing"]["rewards"]["semantic"] = {
    "mode": "coverage_calibration",
    "rec_weight": 0.2,
    "pos_weight": 0.45,
    "neg_weight": 0.35,
}
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["save_every"] = 50
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_semantic_coverage_8gpu"
)
