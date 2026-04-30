from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_atom_semcovcal_routed_opd_rl_8gpu_1epoch.py"
)["config"]
config = base

config["stage"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt"
config["run_name"] = "idea3_atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt"
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt"
)

config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/atom_semcovcal_routed_opd_rl_8gpu_1epoch_sample_conditioned/checkpoint-step-1000/adapter",
}
config["resume"] = {"completed_steps": 0}

config["optimizer"]["max_steps"] = 100
config["data"]["batch_size"] = 4
config["data"]["num_workers"] = 0
config["data"]["task_mix"] = {
    "refseg": 0.5,
    "maskcap": 0.5,
}

config["data"]["prompts"]["maskcap"] = (
    "Region: {mask_tokens}\n"
    "Describe this region precisely in one sentence. Mention only visually certain details and omit uncertain claims."
)

config["checkpoint"]["save_every"] = 100
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20

