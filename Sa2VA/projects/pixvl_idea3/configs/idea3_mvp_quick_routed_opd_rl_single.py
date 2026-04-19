from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_quick_routed_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_quick_routed_opd_rl_single"
config["run_name"] = "idea3_mvp_quick_routed_opd_rl_single"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/quick_routed_opd_rl_single"
config["memory_optim"]["fsdp"]["enabled"] = False
