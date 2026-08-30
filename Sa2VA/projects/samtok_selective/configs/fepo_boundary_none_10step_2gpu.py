import copy
import runpy
from pathlib import Path

from projects.samtok_selective.boundary_credit_gr_cppo_contract import (
    METHOD,
    TEN_STEP_NONE_STAGE,
)
from projects.samtok_selective.tail_geometry import BOUNDARY_WIDTH


base = Path(__file__).with_name("fepo_boundary_credit_es_gr_cppo_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(base))["config"])
config["stage"] = TEN_STEP_NONE_STAGE
config["optimizer"]["max_steps"] = 10
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(TEN_STEP_NONE_STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
method = config.pop("boundary_entropy_gr_cppo")
method.update(
    {
        "method": METHOD,
        "positive_reward": "raw_half_ciou_half_boundary_iou",
        "ciou_weight": 0.5,
        "boundary_iou_weight": 0.5,
        "boundary_width": BOUNDARY_WIDTH,
        "evidence_gate": {
            "mode": "none",
            "scale": 0.25,
            "clip_min": 0.25,
            "clip_max": 1.75,
            "noise_std": 0.01,
        },
    }
)
config["boundary_entropy_gr_cppo"] = method
