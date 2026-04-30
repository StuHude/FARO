from pathlib import Path
import runpy

ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

base = runpy.run_path(WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea1" / "configs" / "idea1_joint_sft.py")["config"]
config = base

config["stage"] = "caption_calibration_sft"
config["run_name"] = "idea3_caption_calibration_sft"
config["model"]["base_model_name_or_path"] = "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok"
config["model"]["processor_name_or_path"] = "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok"
config["model"]["sam2_ckpt_path"] = "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok/sam2.1_hiera_large.pt"
config["model"]["mask_tokenizer_path"] = "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth"

config["student_init"] = {
    "adapter_path": "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl/checkpoint-step-1500/adapter",
}

config["data"]["schema_files"] = [
    str(ROOT / "data" / "pixvl_idea3" / "schemas" / "maskcap_train_semantic_only.jsonl"),
]
config["data"]["task_mix"] = {"maskcap": 1.0}
config["data"]["source_mix"] = {}
config["data"]["batch_size"] = 1
config["data"]["prompts"]["maskcap"] = (
    "Region: {mask_tokens}\n"
    "Describe this region precisely in one sentence. Mention only visually certain details and omit uncertain claims."
)

config["optimizer"]["lr"] = 1e-5
config["optimizer"]["grad_accum_steps"] = 8
config["optimizer"]["max_steps"] = 500

config["generation"]["maskcap"]["max_new_tokens"] = 64

config["checkpoint"]["save_every"] = 100
config["checkpoint"]["output_dir"] = str(ROOT / "outputs" / "pixvl_idea3" / "caption_calibration_sft")
