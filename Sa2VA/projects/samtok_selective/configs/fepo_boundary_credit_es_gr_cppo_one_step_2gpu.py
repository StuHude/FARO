import copy
import runpy
from pathlib import Path

from projects.samtok_selective.boundary_credit_gr_cppo_contract import METHOD, STAGE
from projects.samtok_selective.tail_geometry import BOUNDARY_WIDTH


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
        "positive_reward": "raw_half_ciou_half_boundary_iou",
        "ciou_weight": 0.5,
        "boundary_iou_weight": 0.5,
        "boundary_width": BOUNDARY_WIDTH,
    }
)
config["boundary_entropy_gr_cppo"] = method
