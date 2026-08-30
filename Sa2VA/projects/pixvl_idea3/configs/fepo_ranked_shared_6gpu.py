from copy import deepcopy
from pathlib import Path
import runpy

config = deepcopy(runpy.run_path(Path(__file__).with_name("fepo_long_shared_6gpu.py"))["config"])
config["stage"] = "fepo_ranked_shared_6gpu"
config["run_name"] = "fepo_ranked_shared_6gpu"
config["reward_ranking"] = {"enabled": True, "capacity": 16, "components": {"refseg": ["ciou"]}}
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [str(LOCAL_DATA / "refseg_samtok_refcoco_2048.jsonl")]
config["data"]["task_mix"] = {"refseg": 1.0}
config["data"]["source_mix"] = {}
config["routing"]["mode"] = "shared"
config["optimizer"]["max_steps"] = 100
config["optimizer"]["grad_accum_steps"] = 1
config["memory_optim"]["fsdp"]["enabled"] = False
config["generation"]["refseg"].update({"temperature": 0.7, "top_p": 0.95, "do_sample": True})
config["rl"] = {"group_size": {"refseg": 4}, "tau_seg": 0.5, "tau_cap": 0.5}
config["loss"].update({"lambda_ce": 0.3, "lambda_rl_seg": 1.0, "lambda_rl_cap": 0.0, "beta_kl": 0.02, "lambda_opd": 0.0})
config["checkpoint"]["output_dir"] = str(Path(config["paths"]["output_root"]) / "fepo_ranked_shared_6gpu")
