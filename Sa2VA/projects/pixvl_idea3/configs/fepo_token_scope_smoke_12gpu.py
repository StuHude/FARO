from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"])
config["stage"] = "fepo_token_scope_smoke_12gpu"
config["run_name"] = "fepo_token_scope_smoke_12gpu"
config["routing"]["mode"] = "predicted_only_evidence"
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "fepo_token_scope_smoke_12gpu")
