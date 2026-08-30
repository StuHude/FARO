from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_shared_long_6gpu_corrected"
config["run_name"] = "fepo_shared_long_6gpu_corrected"
config["routing"]["mode"] = "shared"
config["optimizer"]["max_steps"] = 100
config["optimizer"]["grad_accum_steps"] = 1
config["checkpoint"]["save_every"] = 50
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_shared_long_6gpu_corrected"
)
