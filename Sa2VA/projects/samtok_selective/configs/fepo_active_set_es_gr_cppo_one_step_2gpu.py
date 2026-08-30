import os
import runpy
from pathlib import Path

from projects.samtok_selective.active_set_gr_cppo_contract import (
    METHOD,
    STAGE,
)
from projects.samtok_selective.config import REPO_ROOT
from projects.samtok_selective.gr_cppo_contract import expected_frozen_anchor


adapter = os.environ.get(
    "SAMTOK_STANDALONE_ADAPTER",
    str(expected_frozen_anchor(REPO_ROOT)),
)
source = Path(__file__).with_name("fepo_entropy_gr_cppo_one_step_2gpu.py")
config = runpy.run_path(str(source))["config"]
config["stage"] = STAGE
config["optimizer"]["max_steps"] = 1
config["checkpoint"]["adapter_init"] = adapter
config["checkpoint"]["output_dir"] = str(
    REPO_ROOT / "outputs" / "samtok_selective" / STAGE
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
method = dict(config.pop("entropy_gr_cppo"))
method["method"] = METHOD
method.update(
    {
        "selective_risk_mode": "fixed_training_sentinel_active_set",
        "sentinel_source": "sorted_training_no_target_ids",
        "sentinel_rows_total": 8,
        "anchor_budget_source": "frozen_initialization_pre_update",
        "null_ce_relative_slack": 0.10,
        "null_ce_absolute_slack": 1e-6,
        "margin_slack": 0.25,
        "holdout_access": False,
    }
)
config["active_set_entropy_gr_cppo"] = method
