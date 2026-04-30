from copy import deepcopy
from pathlib import Path
import runpy

base = runpy.run_path(
    Path(__file__).with_name("idea3_mvp_scale100k_3gpu_routed_rl.py")
)["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_scale100k_3gpu_routed_rl_selfdist1500"
config["run_name"] = "idea3_mvp_scale100k_3gpu_routed_rl_selfdist1500"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl_selfdist1500"
config["optimizer"]["max_steps"] = 1500
config["checkpoint"]["save_every"] = 500
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
