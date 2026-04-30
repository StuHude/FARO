from copy import deepcopy
from pathlib import Path
import runpy


base = runpy.run_path(Path(__file__).with_name("idea3_mvp_scale100k_3gpu_routed_rl.py"))["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_scale100k_3gpu_routed_rl_shuffled_labels"
config["run_name"] = "idea3_mvp_scale100k_3gpu_routed_rl_shuffled_labels"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl_shuffled_labels"
config["paths"]["schema_root"] = "/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas_shuffled"
config["data"]["schema_files"] = [
    "/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas_shuffled/refseg_train_routed.jsonl",
    "/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas_shuffled/maskcap_train_routed.jsonl",
]
