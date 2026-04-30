from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_atom_semcovcal_routed_opd_rl_8gpu_500.py"
)["config"]
config = base

config["stage"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_1epoch_sample_conditioned"
config["run_name"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_1epoch_sample_conditioned"
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "atom_semcovcal_routed_opd_rl_8gpu_1epoch_sample_conditioned"
)

# Continue the automatic routing line from the semcov checkpoint while removing
# all source-conditioned routing priors.
config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/checkpoint-step-100/adapter",
}
config["resume"] = {"completed_steps": 0}

# Full training-data 1 epoch under the current visual-token filter:
# kept samples observed at runtime = 1297162
# global batch = 8 gpus * 4 per gpu = 32
# floor(1297162 / 32) = 40536 global steps
config["optimizer"]["max_steps"] = 40536
config["data"]["batch_size"] = 4
config["data"]["num_workers"] = 0
config["data"]["task_mix"] = {
    "refseg": 0.3,
    "maskcap": 0.7,
}

config["checkpoint"]["save_every"] = 1000
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20

config["routing"]["mode"] = "atom_conditioned"
