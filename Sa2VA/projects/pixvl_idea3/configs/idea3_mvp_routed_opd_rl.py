from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_unified_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_routed_opd_rl"
config["run_name"] = "idea3_mvp_routed_opd_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/stage3_routed_opd_rl"
