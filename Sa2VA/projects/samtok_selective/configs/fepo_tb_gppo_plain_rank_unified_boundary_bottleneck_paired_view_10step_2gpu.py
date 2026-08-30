"""BA-FEPO: boundary-bottleneck paired-view native-relative RL."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu import (
    config as _paired_view_config,
)
from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE,
)


config = copy.deepcopy(_paired_view_config)
config["stage"] = UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(
        UNIFIED_BOUNDARY_BOTTLENECK_PAIRED_VIEW_STAGE
    )
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
config["tail_gppo"]["paired_view_geometry"].update(
    {
        "mode": "gt_verified_boundary_bottleneck_paired_view_reward",
        "aggregation": "boundary_bottleneck_min",
    }
)
