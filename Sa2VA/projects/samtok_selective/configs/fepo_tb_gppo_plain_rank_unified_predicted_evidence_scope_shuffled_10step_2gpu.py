"""PES-FEPO shuffled-evidence negative control."""

import copy
from pathlib import Path

from projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu import config as _base
from projects.samtok_selective.tail_gppo_contract import UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE

config = copy.deepcopy(_base)
config["stage"] = UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE
config["checkpoint"]["output_dir"] = str(
    Path(config["checkpoint"]["output_dir"]).with_name(UNIFIED_PREDICTED_EVIDENCE_SCOPE_SHUFFLED_STAGE)
)
config["provenance"]["manifest_path"] = str(Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json")
config["tail_gppo"]["pes_evidence_shuffle"] = True
config["tail_gppo"]["pes_evidence_shuffle_seed"] = 1907
