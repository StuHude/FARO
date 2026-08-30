from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"])
config["stage"] = "samtok_null_aware_sft_8gpu"
config["run_name"] = "samtok_null_aware_sft_8gpu"
config["student_init"] = {"adapter_path": None}
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [
    str(LOCAL_DATA / "refcoco_direct_sft_1024.jsonl"),
    str(LOCAL_DATA / "existence_train_256x2.jsonl"),
]
config["data"]["task_mix"] = {"direct_sft": 0.75, "existence": 0.25}
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["optimizer"]["max_steps"] = 100
config["optimizer"]["grad_accum_steps"] = 1
config["optimizer"]["lr"] = 1e-6
config["memory_optim"]["fsdp"]["enabled"] = False
config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "samtok_null_aware_sft_8gpu")
config["logging"]["log_every"] = 1
config["logging"]["snapshot_every"] = 10
