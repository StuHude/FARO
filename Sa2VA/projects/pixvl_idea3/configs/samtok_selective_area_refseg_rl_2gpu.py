from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_refseg_rl_2gpu.py"))["config"]
)
config["stage"] = "samtok_selective_area_refseg_rl_2gpu"
config["run_name"] = "samtok_selective_area_refseg_rl_2gpu"
config["selective_negative_reward"] = {"mode": "area_penalty"}
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_selective_area_refseg_rl_2gpu"
)
