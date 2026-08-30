from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"])
config["stage"] = "samtok_selective_refseg_sft_2gpu"
config["run_name"] = "samtok_selective_refseg_sft_2gpu"
config["seed"] = 42
config["student_init"] = {"adapter_path": None}
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [
    str(LOCAL_DATA / "fepo_existence" / "grefcoco_selective_train_256.jsonl"),
]
config["data"]["task_mix"] = {"refseg": 1.0}
config["data"]["source_mix"] = None
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["data"]["visual_token_filter"] = {"enabled": False}
config["data"]["prompts"]["refseg"] = (
    'Please segment the region referred to by: "{query}". '
    'Return only the region mask; if the target is absent, return "No target."'
)
config["optimizer"].update({"max_steps": 100, "grad_accum_steps": 1, "lr": 1e-6})
config["memory_optim"]["fsdp"]["enabled"] = False
config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_selective_refseg_sft_2gpu"
)
config["logging"]["log_every"] = 1
config["logging"]["snapshot_every"] = 10
