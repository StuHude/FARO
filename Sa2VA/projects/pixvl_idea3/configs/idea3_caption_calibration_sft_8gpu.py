from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_caption_calibration_sft.py")["config"]
config = base

config["run_name"] = "idea3_caption_calibration_sft_8gpu"
config["checkpoint"]["output_dir"] = str(ROOT / "outputs" / "pixvl_idea3" / "caption_calibration_sft_8gpu")

# Try to use more device memory per rank.
config["memory_optim"]["fsdp"]["enabled"] = False
config["memory_optim"]["gradient_checkpointing"] = False

# Batch is still sample-serial in the current trainer, but this is the largest
# safe knob before a fuller trainer refactor.
config["data"]["batch_size"] = 8
config["data"]["num_workers"] = 0
config["optimizer"]["grad_accum_steps"] = 4
config["logging"]["collect_peak_memory"] = False
