import runpy
from pathlib import Path

from projects.samtok_selective.entropy_gr_cppo_contract import TWENTY_STEP_STAGE


one_step = Path(__file__).resolve().with_name(
    "fepo_entropy_gr_cppo_one_step_2gpu.py"
)
config = runpy.run_path(str(one_step))["config"]
config["stage"] = TWENTY_STEP_STAGE
config["optimizer"]["max_steps"] = 20
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).resolve().with_name(TWENTY_STEP_STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
