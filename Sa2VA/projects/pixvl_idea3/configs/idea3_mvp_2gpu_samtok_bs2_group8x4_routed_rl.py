from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_routed_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_2gpu_samtok_bs2_group8x4_routed_rl"
config["run_name"] = "idea3_mvp_2gpu_samtok_bs2_group8x4_routed_rl"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/2gpu_samtok_bs2_group8x4_routed_rl"
config["data"]["batch_size"] = 2
config["rl"]["group_size"] = {
    "refseg": 8,
    "maskcap": 4,
}
config["optimizer"]["max_steps"] = 2000
config["optimizer"]["grad_accum_steps"] = 1
config["checkpoint"]["save_every"] = 200
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
config["memory_optim"]["fsdp"]["enabled"] = False
