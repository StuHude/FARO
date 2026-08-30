from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"])
config["stage"] = "fepo_long_hard_6gpu"
config["run_name"] = "fepo_long_hard_6gpu"
config["routing"]["mode"] = "source_bucket"
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["save_every"] = 50
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "fepo_long_hard_6gpu")
