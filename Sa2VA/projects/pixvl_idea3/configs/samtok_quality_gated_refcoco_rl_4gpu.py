from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(runpy.run_path(Path(__file__).with_name("idea3_mvp_base.py"))["config"])
config["stage"] = "samtok_quality_gated_refcoco_rl_4gpu"
config["run_name"] = "samtok_quality_gated_refcoco_rl_4gpu"
config["student_init"] = {"adapter_path": None}
local_data = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [str(local_data / "refseg_samtok_refcoco_2048.jsonl")]
config["data"]["task_mix"] = {"refseg": 1.0}
config["data"]["source_mix"] = {}
config["data"]["batch_size"] = 1
config["data"]["num_workers"] = 0
config["routing"]["mode"] = "shared"
config["routing"]["rewards"]["geometry"].update({
    "ciou_weight": 1.0,
    "boundary_weight": 0.0,
    "area_weight": 0.0,
    "exact_weight": 0.0,
})
config["reward_ranking"] = {"enabled": False}
config["optimizer"]["max_steps"] = 100
config["optimizer"]["grad_accum_steps"] = 1
config["optimizer"]["lr"] = 1e-6
config["memory_optim"]["fsdp"]["enabled"] = False
config["generation"]["refseg"].update({
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
})
config["rl"] = {
    "group_size": {"refseg": 4},
    "tau_seg": 0.5,
    "tau_cap": 0.5,
    "advantage_mode": "quality_gated",
    "quality_threshold": {"refseg": 0.5},
    "gate_temperature": 0.05,
}
config["loss"].update({
    "lambda_ce": 0.3,
    "lambda_rl_seg": 1.0,
    "lambda_rl_cap": 0.0,
    "beta_kl": 0.02,
    "lambda_opd": 0.0,
})
config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_quality_gated_refcoco_rl_4gpu"
)
