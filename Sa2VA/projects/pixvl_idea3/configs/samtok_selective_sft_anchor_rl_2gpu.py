from copy import deepcopy
from pathlib import Path
import runpy


config = deepcopy(
    runpy.run_path(Path(__file__).with_name("samtok_selective_refseg_rl_2gpu.py"))["config"]
)
ROOT = Path(config["paths"]["output_root"])
SFT_ANCHOR = ROOT / "samtok_selective_refseg_sft_2gpu" / "adapter"

config["stage"] = "samtok_selective_sft_anchor_rl_2gpu"
config["run_name"] = "samtok_selective_sft_anchor_rl_2gpu"
config["student_init"] = {"adapter_path": str(SFT_ANCHOR)}
config["reference"] = {"adapter_path": str(SFT_ANCHOR)}
config["checkpoint"]["output_dir"] = str(ROOT / "samtok_selective_sft_anchor_rl_2gpu")
