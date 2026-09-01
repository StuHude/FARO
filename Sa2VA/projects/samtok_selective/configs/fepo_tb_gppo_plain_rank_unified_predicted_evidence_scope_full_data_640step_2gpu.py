"""Full-data PES-FEPO run with explicit pair coverage."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu import config as _base
from projects.samtok_selective.tail_gppo_contract import UNIFIED_PREDICTED_EVIDENCE_SCOPE_FULL_DATA_STAGE


config = copy.deepcopy(_base)
config["stage"] = UNIFIED_PREDICTED_EVIDENCE_SCOPE_FULL_DATA_STAGE
config["optimizer"]["max_steps"] = 640
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_PREDICTED_EVIDENCE_SCOPE_FULL_DATA_STAGE)
)
config["provenance"]["manifest_path"] = str(
    Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
)
config["tail_gppo"].update(
    {
        "full_data_schedule": True,
        "minimum_consumed_rows": 5120,
        "minimum_consumed_pairs": 2560,
    }
)
