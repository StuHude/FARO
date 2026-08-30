import copy
import runpy
from pathlib import Path

from projects.samtok_selective.representation_fepo_contract import TEN_STEP_STAGE


one_step = Path(__file__).with_name("fepo_projector_plastic_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(one_step))["config"])
config["stage"] = TEN_STEP_STAGE
config["data"].update({"expected_rows": 5120, "expected_no_target_rows": 2560})
config["optimizer"]["max_steps"] = 10
output = Path(config["checkpoint"]["output_dir"]).with_name(TEN_STEP_STAGE)
config["checkpoint"]["output_dir"] = str(output)
config["checkpoint"]["save_every"] = 0
config["provenance"]["manifest_path"] = str(output / "provenance_manifest.json")
