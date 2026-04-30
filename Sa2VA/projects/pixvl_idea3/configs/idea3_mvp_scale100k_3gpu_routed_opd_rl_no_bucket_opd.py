from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_scale100k_3gpu_routed_opd_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_scale100k_3gpu_routed_opd_rl_no_bucket_opd"
config["run_name"] = "idea3_mvp_scale100k_3gpu_routed_opd_rl_no_bucket_opd"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl_no_bucket_opd"

for bucket_cfg in config["routing"]["buckets"].values():
    bucket_cfg["opd_scale"] = 1.0
