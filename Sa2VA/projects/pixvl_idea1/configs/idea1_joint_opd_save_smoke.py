from pathlib import Path
from copy import deepcopy
import runpy


base = runpy.run_path(Path(__file__).with_name("idea1_joint_opd.py"))["config"]
config = deepcopy(base)

config["stage"] = "stage2_joint_opd_save_smoke"
config["run_name"] = "idea1_joint_opd_save_smoke"
config["checkpoint"]["output_dir"] = "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd_save_smoke"
config["checkpoint"]["save_every"] = 5
config["optimizer"]["max_steps"] = 120
config["data"]["schema_files"] = [
    "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_train.jsonl",
]
config["data"]["task_mix"] = {
    "refseg": 1.0,
    "maskcap": 0.0,
}
config["data"]["source_mix"] = {}
config["logging"]["snapshot_every"] = 5
