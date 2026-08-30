import copy
import runpy
from pathlib import Path

from projects.samtok_selective.representation_fepo_contract import (
    ADAPTER_MODE,
    ONE_STEP_STAGE,
    VISUAL_ALPHA,
    VISUAL_DROPOUT,
    VISUAL_R,
)


base_path = Path(__file__).with_name("fepo_entropy_gr_cppo_one_step_2gpu.py")
config = copy.deepcopy(runpy.run_path(str(base_path))["config"])
config["stage"] = ONE_STEP_STAGE
config["model"]["adapter_mode"] = ADAPTER_MODE
config["lora"].update(
    {
        "visual_r": VISUAL_R,
        "visual_alpha": VISUAL_ALPHA,
        "visual_dropout": VISUAL_DROPOUT,
    }
)
config["representation_entropy_gr_cppo"] = config.pop("entropy_gr_cppo")
config["representation"] = {
    "anchor_adapter": "frozen",
    "trainable_adapter": "visual",
    "target_scope": "visual.merger_and_deepstack_mergers_only",
    "expected_target_linears": 8,
    "preupdate_equivalence_tolerance": 1e-5,
}
output = Path(config["checkpoint"]["output_dir"]).with_name(ONE_STEP_STAGE)
config["checkpoint"]["output_dir"] = str(output)
config["provenance"]["manifest_path"] = str(output / "provenance_manifest.json")
