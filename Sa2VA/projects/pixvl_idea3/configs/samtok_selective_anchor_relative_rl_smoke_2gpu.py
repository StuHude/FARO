from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_anchor_relative_rl_2gpu.py"))["config"]
)
ROOT = Path(config["paths"]["output_root"])

config["stage"] = "samtok_selective_anchor_relative_rl_smoke_2gpu"
config["run_name"] = "samtok_selective_anchor_relative_rl_smoke_2gpu"
config["optimizer"]["max_steps"] = 8
config["data"]["fixed_record_order"] = True
config["checkpoint"]["save_every"] = 0
config["checkpoint"]["output_dir"] = str(
    ROOT / "samtok_selective_anchor_relative_rl_smoke_2gpu"
)
