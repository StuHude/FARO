from copy import deepcopy
from pathlib import Path
import runpy

base = runpy.run_path(
    Path(__file__).with_name("idea3_mvp_scale100k_2gpu_unified_opd_rl_selfdist1500.py")
)["config"]
config = deepcopy(base)

LOCAL_MODEL = "/dev/shm/xiaoyicheng_local/Qwen3-VL-4B-SAMTok"
config["model"]["base_model_name_or_path"] = LOCAL_MODEL
config["model"]["processor_name_or_path"] = LOCAL_MODEL
config["model"]["sam2_ckpt_path"] = f"{LOCAL_MODEL}/sam2.1_hiera_large.pt"
config["model"]["mask_tokenizer_path"] = f"{LOCAL_MODEL}/mask_tokenizer_256x2.pth"
