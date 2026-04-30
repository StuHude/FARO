from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(
    Path(__file__).with_name("idea3_mvp_scale100k_3gpu_routed_opd_rl_selfdist1500.py")
)["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_scale100k_4gpu_routed_opd_rl_selfdist1000_gtonly"
config["run_name"] = "idea3_mvp_scale100k_4gpu_routed_opd_rl_selfdist1000_gtonly"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_4gpu_routed_opd_rl_selfdist1000_gtonly"

# Keep the corrected self-distill OPD teacher mode, but remove overlay privilege:
# teacher sees the normal image and only gets GT rollout text as privileged input.
config["opd"]["teacher_mode"] = "self_privileged_rollout"
config["opd"]["teacher_image_key"] = "image"

# Re-run from baseline on 4 GPUs for 1000 steps.
config["optimizer"]["max_steps"] = 1000
config["checkpoint"]["save_every"] = 500
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20

