from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_unified_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_scale100k_2gpu_unified_opd_rl"
config["run_name"] = "idea3_mvp_scale100k_2gpu_unified_opd_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl"

# 2 GPUs, try to fill memory while keeping stability.
config["data"]["batch_size"] = 5
config["rl"]["group_size"] = {
    "refseg": 8,
    "maskcap": 4,
}

# 2 * 5 * 10000 = 100k samples seen.
config["optimizer"]["max_steps"] = 10000
config["optimizer"]["grad_accum_steps"] = 1
config["checkpoint"]["save_every"] = 500
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
config["memory_optim"]["fsdp"]["enabled"] = False
