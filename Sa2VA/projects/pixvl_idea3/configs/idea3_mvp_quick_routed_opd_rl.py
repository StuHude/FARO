from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_routed_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_quick_routed_opd_rl"
config["run_name"] = "idea3_mvp_quick_routed_opd_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/quick_routed_opd_rl"
config["optimizer"]["max_steps"] = 120
config["optimizer"]["grad_accum_steps"] = 4
config["checkpoint"]["save_every"] = 0
config["logging"]["log_every"] = 5
config["logging"]["snapshot_every"] = 20
