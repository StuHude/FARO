from copy import deepcopy
from pathlib import Path
import runpy


ROOT = Path("/mnt/pfs/xiaoyicheng")
WORKSPACE = ROOT / "BRIDGE-OPD"
SAMTOK_MODEL = ROOT / "models" / "Qwen3-VL-4B-SAMTok"

base = runpy.run_path(WORKSPACE / "Sa2VA" / "projects" / "pixvl_idea1" / "configs" / "idea1_joint_sft.py")["config"]
config = deepcopy(base)

config["stage"] = "idea3_mvp_base"
config["run_name"] = "idea3_mvp_base"
config["model"].update(
    {
        "base_model_name_or_path": str(SAMTOK_MODEL),
        "processor_name_or_path": str(SAMTOK_MODEL),
        "sam2_ckpt_path": str(SAMTOK_MODEL / "sam2.1_hiera_large.pt"),
        "mask_tokenizer_path": str(SAMTOK_MODEL / "mask_tokenizer_256x2.pth"),
    }
)
config["paths"].update(
    {
        "data_root": str(ROOT / "data" / "pixvl_idea3"),
        "schema_root": str(ROOT / "data" / "pixvl_idea3" / "schemas"),
        "output_root": str(ROOT / "outputs" / "pixvl_idea3"),
        "mask_code_cache": str(ROOT / "data" / "pixvl_idea3" / "mask_codes.sqlite3"),
    }
)
config["data"].update(
    {
        "schema_files": [
            str(ROOT / "data" / "pixvl_idea3" / "schemas" / "refseg_train_routed.jsonl"),
            str(ROOT / "data" / "pixvl_idea3" / "schemas" / "maskcap_train_routed.jsonl"),
        ],
        "task_mix": {
            "refseg": 0.7,
            "maskcap": 0.3,
        },
        "bucket_mix": {
            "semantic": 1.0,
            "relation": 1.2,
            "geometry": 1.0,
            "default": 1.0,
        },
        "source_mix": {
            "cocostuff": 0.34,
            "lvis": 0.33,
            "paco": 0.33,
            "fine_grained_dataset_part1": 1.0,
        },
    }
)
config["student_init"] = {
    "adapter_path": None,
}
config["teacher"] = {
    "adapter_path": None,
}
config["reference"] = {
    "adapter_path": None,
}
config["resume"] = {
    "completed_steps": 0,
}
config["checkpoint"]["save_every"] = 200
config["routing"] = {
    "max_confusers": 16,
    "failure_thresholds": {
        "semantic": 0.68,
        "relation": 0.58,
        "geometry": 0.60,
    },
    "buckets": {
        "semantic": {
            "ce_scale": 1.0,
            "rl_scale": 1.0,
            "opd_scale": 1.0,
        },
        "relation": {
            "ce_scale": 1.0,
            "rl_scale": 1.15,
            "opd_scale": 1.0,
        },
        "geometry": {
            "ce_scale": 0.9,
            "rl_scale": 1.2,
            "opd_scale": 1.2,
        },
    },
    "rewards": {
        "semantic": {
            "base_weight": 0.75,
            "keyword_weight": 0.25,
        },
        "relation": {
            "target_weight": 0.7,
            "margin_weight": 0.2,
            "exact_weight": 0.1,
        },
        "relation_caption": {
            "base_weight": 0.7,
            "relation_weight": 0.3,
        },
        "geometry": {
            "ciou_weight": 0.55,
            "boundary_weight": 0.25,
            "area_weight": 0.1,
            "exact_weight": 0.1,
            "boundary_width": 2,
        },
    },
}
