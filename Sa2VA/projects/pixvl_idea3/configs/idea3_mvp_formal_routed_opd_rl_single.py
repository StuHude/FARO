from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_routed_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_formal_routed_opd_rl_single"
config["run_name"] = "idea3_mvp_formal_routed_opd_rl_single"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/formal_routed_opd_rl_single"
config["optimizer"]["max_steps"] = 300
config["optimizer"]["grad_accum_steps"] = 4
config["checkpoint"]["save_every"] = 100
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
config["memory_optim"]["fsdp"]["enabled"] = False
