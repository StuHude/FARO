import copy
import runpy
from pathlib import Path

from projects.samtok_selective.greedy_preference_gr_cppo_contract import METHOD, STAGE


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
        "preference_pair": "best_improving_sample_vs_native_greedy",
        "preference_loss": "softplus_negative_native_log_odds_shift",
        "minimum_improvement": 1e-4,
        "native_scoring_temperature": 1.0,
        "max_epoch0_ratio_deviation": 0.01,
    }
)
config["greedy_preference_entropy_gr_cppo"] = method
