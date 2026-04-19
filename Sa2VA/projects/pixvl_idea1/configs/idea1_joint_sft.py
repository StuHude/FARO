from pathlib import Path


ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"

config = {
    "stage": "stage1_joint_sft",
    "run_name": "idea1_joint_sft_maxmem",
    "seed": 3407,
    "paths": {
        "workspace": str(WORKSPACE),
        "project_root": str(WORKSPACE / "Sa2VA"),
        "data_root": str(ROOT / "data" / "pixvl_idea1"),
        "schema_root": str(ROOT / "data" / "pixvl_idea1" / "schemas"),
        "smoke_root": str(ROOT / "data" / "pixvl_idea1" / "smoke"),
        "output_root": str(ROOT / "outputs" / "pixvl_idea1"),
        "hf_cache": str(ROOT / ".cache" / "huggingface"),
        "mask_code_cache": str(ROOT / "data" / "pixvl_idea1" / "mask_codes.sqlite3"),
    },
    "model": {
        "base_model_name_or_path": "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co",
        "processor_name_or_path": "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co",
        "sam2_ckpt_path": "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/sam2.1_hiera_large.pt",
        "mask_tokenizer_path": "/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth",
        "codebook_size": 256,
        "codebook_depth": 2,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
    },
    "data": {
        "schema_files": [
            str(ROOT / "data" / "pixvl_idea1" / "schemas" / "refseg_train.jsonl"),
            str(ROOT / "data" / "pixvl_idea1" / "schemas" / "dam_train.jsonl"),
            str(ROOT / "data" / "pixvl_idea1" / "schemas" / "fine_grained_dataset_part1_train.jsonl"),
        ],
        "split": "train",
        "batch_size": 1,
        "num_workers": 0,
        "task_mix": {
            "refseg": 0.6,
            "maskcap": 0.4,
        },
        "source_mix": {
            "cocostuff": 0.233,
            "lvis": 0.233,
            "paco": 0.233,
            "fine_grained_dataset_part1": 0.3,
        },
        "overlay": {
            "darken_alpha": 0.4,
            "boundary_px": 2,
            "boundary_color": [255, 64, 64],
        },
        "visual_token_filter": {
            "enabled": True,
            "max_ratio_to_avg": 1.5,
        },
        "prompts": {
            "refseg": 'Please segment the region referred to by: "{query}". Return only the region mask.',
            "maskcap": "Region: {mask_tokens}\nDescribe this region precisely in one sentence.",
        },
    },
    "optimizer": {
        "lr": 2e-5,
        "weight_decay": 0.05,
        "betas": [0.9, 0.999],
        "grad_accum_steps": 8,
        "max_steps": 2000,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
    },
    "lora": {
        "enabled": True,
        "r": 128,
        "alpha": 256,
        "dropout": 0.05,
        "bias": "none",
        "modules_to_save": [],
    },
    "memory_optim": {
        "gradient_checkpointing": True,
        "fsdp": {
            "enabled": True,
            "activation_checkpointing": True,
            "state_dict_type": "sharded_state_dict",
            "transformer_cls_names_to_wrap": ["Qwen3VLTextDecoderLayer"],
        },
    },
    "logging": {
        "log_every": 1,
        "snapshot_every": 10,
    },
    "loss": {
        "lambda_cap_ce": 1.0,
    },
    "generation": {
        "refseg": {
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "maskcap": {
            "max_new_tokens": 48,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
    },
    "checkpoint": {
        "save_every": 200,
        "output_dir": str(ROOT / "outputs" / "pixvl_idea1" / "stage1_joint_sft_maxmem"),
    },
}
