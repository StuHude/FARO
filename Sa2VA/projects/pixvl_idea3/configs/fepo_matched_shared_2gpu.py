from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
config = deepcopy(base)
config["stage"] = "fepo_matched_shared_2gpu"
config["run_name"] = "fepo_matched_shared_2gpu"
config["routing"]["mode"] = "shared"
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "fepo_matched_shared_2gpu")
