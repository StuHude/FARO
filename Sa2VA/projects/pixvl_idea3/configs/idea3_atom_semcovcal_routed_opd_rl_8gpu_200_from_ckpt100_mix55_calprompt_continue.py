from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt.py"
)["config"]
config = base

config["stage"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_200_from_ckpt100_mix55_calprompt_continue"
config["run_name"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_200_from_ckpt100_mix55_calprompt_continue"

# Reuse the same output directory so the continued run produces checkpoint-step-200
# alongside the existing checkpoint-step-100.
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt"
)

config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt/checkpoint-step-100/adapter",
}
config["resume"] = {"completed_steps": 100}

config["optimizer"]["max_steps"] = 200
config["checkpoint"]["save_every"] = 100
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
