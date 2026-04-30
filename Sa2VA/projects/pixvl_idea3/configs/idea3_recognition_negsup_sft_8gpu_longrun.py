from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_recognition_negsup_sft_8gpu.py")["config"]
config = base

config["run_name"] = "idea3_recognition_negsup_sft_8gpu_longrun"
config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/adapter",
}
config["optimizer"]["max_steps"] = 1000
config["checkpoint"]["output_dir"] = str(ROOT / "outputs" / "pixvl_idea3" / "recognition_negsup_sft_8gpu_longrun")
