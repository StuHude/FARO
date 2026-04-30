from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea3" / "configs" / "idea3_caption_calibration_sft_8gpu.py")["config"]
config = base

config["stage"] = "recognition_negsup_sft"
config["run_name"] = "idea3_recognition_negsup_sft_8gpu"
config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_sft_8gpu/adapter",
}
config["data"]["schema_files"] = [
    str(ROOT / "data" / "pixvl_idea3" / "schemas" / "maskcap_train_semantic_only.jsonl"),
    str(ROOT / "data" / "pixvl_idea3" / "schemas" / "dlc_recognition_anchor.jsonl"),
    str(ROOT / "data" / "pixvl_idea3" / "schemas" / "dlc_negative_suppression.jsonl"),
]
config["data"]["source_mix"] = {
    "recognition_anchor": 8.0,
    "calibration_pseudo": 8.0,
}
config["data"]["batch_size"] = 4
config["data"]["prompts"]["maskcap"] = (
    "Region: {mask_tokens}\n"
    "Describe this region precisely in one sentence. Mention only visually certain details and omit uncertain claims."
)
config["data"]["prompts"]["recognition_anchor"] = (
    "Region: {mask_tokens}\n"
    "Identify the main object category of this region in 1 to 3 words. Output only the category."
)
config["data"]["prompts"]["calibration_caption"] = (
    "Region: {mask_tokens}\n"
    "Describe this region in one sentence. Mention only visually certain details and avoid unsupported claims."
)
config["optimizer"]["max_steps"] = 300
config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(ROOT / "outputs" / "pixvl_idea3" / "recognition_negsup_sft_8gpu")
