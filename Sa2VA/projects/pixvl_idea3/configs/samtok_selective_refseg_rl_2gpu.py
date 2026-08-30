from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(runpy.run_path(Path(__file__).with_name("samtok_null_aware_rl_8gpu.py"))["config"])
config["seed"] = 42
config["stage"] = "samtok_selective_refseg_rl_2gpu"
config["run_name"] = "samtok_selective_refseg_rl_2gpu"
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data"
config["data"]["schema_files"] = [
    str(LOCAL_DATA / "fepo_existence" / "grefcoco_selective_train_256.jsonl"),
]
config["data"]["task_mix"] = {"refseg": 1.0}
config["data"]["visual_token_filter"] = {"enabled": False}
config["data"]["prompts"]["refseg"] = (
    'Please segment the region referred to by: "{query}". '
    'Return only the region mask; if the target is absent, return "No target."'
)
config["loss"].update({"lambda_ce": 0.3, "lambda_rl_seg": 1.0, "lambda_rl_cap": 0.0, "beta_kl": 0.02, "lambda_opd": 0.0})
config["rl"] = {"group_size": {"refseg": 4}}
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_selective_refseg_rl_2gpu"
)
