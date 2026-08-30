"""Matched FEPO ablation with true evidence-weighted local credit."""

from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("fepo_relation_margin_seed17_8gpu.py"))["config"]
)
config["stage"] = "fepo_relation_softcredit_seed17_8gpu"
config["run_name"] = "fepo_relation_softcredit_seed17_8gpu"
config["routing"]["credit_assignment"] = "soft_local"
config["routing"]["soft_credit_gain"] = 1.0
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "fepo_relation_softcredit_seed17_8gpu"
)
