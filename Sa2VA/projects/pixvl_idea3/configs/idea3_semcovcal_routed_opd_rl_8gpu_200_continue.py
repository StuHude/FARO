from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_semcovcal_routed_opd_rl_8gpu_500.py"
)["config"]
config = base

config["stage"] = "idea3_semcovcal_routed_opd_rl_8gpu_200_run7_fast_continue"
config["run_name"] = "idea3_semcovcal_routed_opd_rl_8gpu_200_run7_fast_continue"
# Reuse the same output dir so trainer resumes from latest state/checkpoint at step 100.
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "semcovcal_routed_opd_rl_8gpu_100_run7_fast"
)
config["optimizer"]["max_steps"] = 200
config["checkpoint"]["save_every"] = 50
config["logging"]["log_every"] = 5
config["logging"]["snapshot_every"] = 10
