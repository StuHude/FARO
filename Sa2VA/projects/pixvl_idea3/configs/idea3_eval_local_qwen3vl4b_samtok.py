from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"
LOCAL_MODEL = Path("/dev/shm/models/Qwen3-VL-4B-SAMTok")

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_mvp_scale100k_3gpu_routed_opd_rl.py"
)["config"]
config = base

config["stage"] = "idea3_eval_local_qwen3vl4b_samtok"
config["run_name"] = "idea3_eval_local_qwen3vl4b_samtok"
config["model"]["base_model_name_or_path"] = str(LOCAL_MODEL)
config["model"]["processor_name_or_path"] = str(LOCAL_MODEL)
config["model"]["mask_tokenizer_path"] = str(LOCAL_MODEL / "mask_tokenizer_256x2.pth")
config["model"]["sam2_ckpt_path"] = str(LOCAL_MODEL / "sam2.1_hiera_large.pt")
