import copy
import runpy
from pathlib import Path

from projects.samtok_selective.representation_fepo_contract import TWENTY_STEP_STAGE


one_step = Path(__file__).with_name("fepo_projector_plastic_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(one_step))["config"])
config["stage"] = TWENTY_STEP_STAGE
config["optimizer"]["max_steps"] = 20
output = Path(config["checkpoint"]["output_dir"]).with_name(TWENTY_STEP_STAGE)
config["checkpoint"]["output_dir"] = str(output)
config["provenance"]["manifest_path"] = str(output / "provenance_manifest.json")
