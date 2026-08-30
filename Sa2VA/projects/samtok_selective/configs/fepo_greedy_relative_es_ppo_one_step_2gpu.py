import copy
import runpy
from pathlib import Path

from projects.samtok_selective.greedy_relative_gr_cppo_contract import METHOD, STAGE


base_path = Path(__file__).with_name("fepo_entropy_gr_cppo_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(base_path))["config"])
config["stage"] = STAGE
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
method = config.pop("entropy_gr_cppo")
method.update(
    {
        "method": METHOD,
        "advantage": "greedy_reward_delta_mean_abs_normalized",
        "advantage_epsilon": 1e-6,
    }
)
config["greedy_relative_entropy_gr_cppo"] = method
