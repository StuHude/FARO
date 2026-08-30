from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"])
config["stage"] = "samtok_null_aware_rl_8gpu"
config["run_name"] = "samtok_null_aware_rl_8gpu"
config["student_init"] = {"adapter_path": None}
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [
    str(LOCAL_DATA / "refseg_samtok_refcoco_2048.jsonl"),
    str(LOCAL_DATA / "fepo_existence" / "refcoco_gres_image_disjoint_256.jsonl"),
]
config["data"]["task_mix"] = {"refseg": 0.75, "existence": 0.25}
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["optimizer"]["max_steps"] = 100
config["optimizer"]["grad_accum_steps"] = 1
config["optimizer"]["lr"] = 1e-6
config["memory_optim"]["fsdp"]["enabled"] = False
config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "samtok_null_aware_rl_8gpu")
config["routing"]["mode"] = "shared"
config["loss"].update({"lambda_ce": 0.3, "lambda_rl_seg": 1.0, "lambda_rl_cap": 0.5, "beta_kl": 0.02, "lambda_opd": 0.0})
config["rl"] = {"group_size": {"refseg": 4, "maskcap": 4, "existence": 4}, "tau_seg": 0.5, "tau_cap": 0.5}
config["generation"]["existence"] = {"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.95, "do_sample": True}
for _task in ("refseg", "existence"):
    config["generation"][_task].update({"temperature": 0.7, "top_p": 0.95, "do_sample": True})
