import copy
import runpy
from pathlib import Path

from projects.samtok_selective.sign_balanced_gr_cppo_contract import METHOD, STAGE


base_path = Path(__file__).with_name("fepo_greedy_relative_es_ppo_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(base_path))["config"])
config["stage"] = STAGE
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
method = config.pop("greedy_relative_entropy_gr_cppo")
method.update({"method": METHOD, "advantage": "greedy_delta_equal_sign_l1_mass"})
config["sign_balanced_entropy_gr_cppo"] = method
