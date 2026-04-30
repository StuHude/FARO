from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(
    WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt.py"
)["config"]
config = base

config["stage"] = "idea3_atom_triage_noevidence_routed_opd_rl_8gpu_100_from_baseline"
config["run_name"] = "idea3_atom_triage_noevidence_routed_opd_rl_8gpu_100_from_baseline"
config["checkpoint"]["output_dir"] = str(
    ROOT / "outputs" / "pixvl_idea3" / "atom_triage_noevidence_routed_opd_rl_8gpu_100_from_baseline"
)

# For the ablation against the previous auto-route line, start from the same
# baseline SAMTok model rather than a previously finetuned auto-route adapter.
config["student_init"] = {
    "adapter_path": None,
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

config["triage"] = {
    "enabled": True,
    "reward_weight": 0.5,
    "nll_weight": 0.25,
    "entropy_weight": 0.25,
    "clean_quantile": 0.35,
    "corrupted_quantile": 0.8,
}

config["checkpoint"]["save_every"] = 100
config["logging"]["log_every"] = 10
config["logging"]["snapshot_every"] = 20
