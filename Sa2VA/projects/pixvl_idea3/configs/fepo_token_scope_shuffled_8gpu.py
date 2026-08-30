from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_schema_smoke_2gpu.py"))["config"]
)
config["stage"] = "fepo_token_scope_shuffled_8gpu"
config["run_name"] = "fepo_token_scope_shuffled_8gpu"
config["routing"]["mode"] = "shuffled"
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["save_every"] = 50
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_token_scope_shuffled_8gpu"
)
