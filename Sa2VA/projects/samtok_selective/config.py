from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SA2VA_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = SA2VA_ROOT.parent

DEFAULT_PROMPT = (
    'Please segment the region referred to by: "{query}". '
    'Return only the region mask; if the target is absent, return "No target."'
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} to an explicitly approved artifact path")
    return value


def build_config(
    *,
    smoke: bool = False,
    continue_from: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Build a standalone config without executing another project's config."""
    base_checkpoint = Path(_required_env("SAMTOK_BASE_CHECKPOINT")).resolve()
    data_jsonl = Path(
        os.environ.get(
            "SAMTOK_SELECTIVE_DATA",
            REPO_ROOT / "data" / "fepo_existence" / "grefcoco_selective_train_256.jsonl",
        )
    ).resolve()
    run_name = stage or ("sft_smoke_2gpu" if smoke else "sft_stage1")
    output_dir = REPO_ROOT / "outputs" / "samtok_selective" / run_name
    cache_dir = REPO_ROOT / "data" / "samtok_selective_cache"

    config: dict[str, Any] = {
        "schema_version": 1,
        "stage": run_name,
        "seed": 42,
        "model": {
            "base_checkpoint": str(base_checkpoint),
            "processor_checkpoint": str(base_checkpoint),
            "sam2_checkpoint": str(base_checkpoint / "sam2.1_hiera_large.pt"),
            "mask_tokenizer_checkpoint": str(base_checkpoint / "mask_tokenizer_256x2.pth"),
            "codebook_size": 256,
            "codebook_depth": 2,
            "attention_backend": "flash_attention_2",
            "trust_remote_code": True,
        },
        "data": {
            "jsonl": str(data_jsonl),
            "prompt": DEFAULT_PROMPT,
            "pairs_per_device_batch": 1,
            "num_workers": 0,
            "cache_path": str(cache_dir / "mask_codes.sqlite3"),
            # The filename denotes 256 matched pairs: 256 target-present and
            # 256 no-target rows.
            "expected_rows": 512,
            "expected_no_target_rows": 256,
        },
        "lora": {
            "r": 128,
            "alpha": 256,
            "dropout": 0.05,
            "bias": "none",
        },
        "optimizer": {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "weight_decay": 0.05,
            "warmup_ratio": 0.05,
            "max_steps": 100,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
        },
        "runtime": {
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "expected_world_size": 2,
        },
        "checkpoint": {
            "output_dir": str(output_dir),
            "save_every": 100,
            "adapter_init": continue_from,
        },
        "logging": {"log_every": 1},
        "provenance": {
            "hash_large_files": True,
            "manifest_path": str(output_dir / "provenance_manifest.json"),
        },
    }
    if smoke:
        config["optimizer"]["max_steps"] = 2
        config["optimizer"]["warmup_ratio"] = 0.0
        config["checkpoint"]["save_every"] = 0
    return deepcopy(config)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported standalone config schema")
    if int(config["runtime"]["expected_world_size"]) != 2:
        raise ValueError("The initial standalone gate is registered as a two-GPU run")
    if int(config["data"]["pairs_per_device_batch"]) < 1:
        raise ValueError("Each device batch must contain at least one positive/negative pair")
    if int(config["optimizer"]["max_steps"]) < 1:
        raise ValueError("max_steps must be positive")
    if str(config["stage"]).startswith("continued") and not config["checkpoint"].get("adapter_init"):
        raise ValueError("continued-SFT requires SAMTOK_STANDALONE_ADAPTER")
