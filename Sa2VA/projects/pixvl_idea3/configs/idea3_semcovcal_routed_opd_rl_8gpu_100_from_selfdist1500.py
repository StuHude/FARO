from copy import deepcopy
from pathlib import Path
import runpy

base = runpy.run_path(
    Path(__file__).with_name("idea3_mvp_scale100k_3gpu_routed_opd_rl_selfdist1500.py")
)["config"]
config = deepcopy(base)

config["stage"] = "idea3_semcovcal_routed_opd_rl_8gpu_100_from_selfdist1500"
config["run_name"] = "idea3_semcovcal_routed_opd_rl_8gpu_100_from_selfdist1500"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_from_selfdist1500"

config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_selfdist1500/checkpoint-step-1500/adapter",
}
config["resume"] = {
    "completed_steps": 0,
}

config["data"]["batch_size"] = 6
config["data"]["task_mix"] = {
    "refseg": 0.3,
    "maskcap": 0.7,
}
config["optimizer"]["grad_accum_steps"] = 1
config["optimizer"]["max_steps"] = 100
config["checkpoint"]["save_every"] = 50
config["logging"]["log_every"] = 5
config["logging"]["snapshot_every"] = 10

config["memory_optim"]["gradient_checkpointing"] = True
config["memory_optim"]["fsdp"]["enabled"] = False
config["memory_optim"]["gpu_reserve_target_gb"] = 0
config["memory_optim"]["gpu_reserve_headroom_gb"] = 8

config["routing"]["rewards"]["semantic"] = {
    "mode": "coverage_calibration",
    "rec_weight": 0.2,
    "pos_weight": 0.45,
    "neg_weight": 0.35,
}
