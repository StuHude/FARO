from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_refseg_rl_2gpu.py"))["config"]
)
LOCAL_DATA = Path(__file__).resolve().parents[4] / "data" / "fepo_existence"
config["stage"] = "samtok_selective_curriculum_rl_2gpu"
config["run_name"] = "samtok_selective_curriculum_rl_2gpu"
config["data"]["schema_files"] = [
    str(LOCAL_DATA / "grefcoco_selective_curriculum_200.jsonl")
]
config["data"]["fixed_record_order"] = True
config["checkpoint"]["output_dir"] = str(
    Path(config["paths"]["output_root"]) / "samtok_selective_curriculum_rl_2gpu"
)
