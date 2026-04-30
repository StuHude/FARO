from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_semcovcal_routed_opd_rl_8gpu_500.py"
)["config"]
config = base

config["stage"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_500_run1"
config["run_name"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_500_run1"
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "atom_semcovcal_routed_opd_rl_8gpu_500_run1"
)

config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/checkpoint-step-100/adapter",
}
config["resume"] = {"completed_steps": 0}

config["optimizer"]["max_steps"] = 500
config["checkpoint"]["save_every"] = 100
config["logging"]["log_every"] = 5
config["logging"]["snapshot_every"] = 10

config["routing"]["mode"] = "atom_conditioned"
