from __future__ import annotations

import argparse
import json
import math
import random
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import REPO_ROOT, validate_config
from .data import PairedBatchSampler, SelectiveRefSegDataset, identity_collate
from .manifests import (
    assert_training_source_clean,
    build_manifest,
    guard_runtime_environment,
    runtime_module_files,
    validate_base_checkpoint,
    validate_declared_paths,
    write_json_atomic,
)
from .mask_codec import SAMTokMaskCodec
from .modeling import (
    answer_token_cross_entropy,
    assert_only_lora_trainable,
    build_model_and_processor,
    build_supervised_inputs,
    move_tensors,
    save_trainable_lora_adapter,
    validate_adapter_init,
)
from .sft_schedule import batch_update_indices, validate_updates_per_batch


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = runpy.run_path(str(Path(path).resolve()))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must expose a config dictionary: {path}")
    validate_config(config)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloader(config: dict[str, Any]) -> DataLoader:
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model_config = config["model"]
    codec = SAMTokMaskCodec(
        model_name_or_path=model_config["processor_checkpoint"],
        mask_tokenizer_path=model_config["mask_tokenizer_checkpoint"],
        sam2_ckpt_path=model_config["sam2_checkpoint"],
        codebook_size=int(model_config["codebook_size"]),
        codebook_depth=int(model_config["codebook_depth"]),
        device=device,
    )
    data_config = config["data"]
    dataset = SelectiveRefSegDataset(
        data_config["jsonl"],
        data_config["prompt"],
        codec,
        data_config["cache_path"],
        expected_rows=int(data_config["expected_rows"]),
        expected_no_target_rows=int(data_config["expected_no_target_rows"]),
    )
    sampler = PairedBatchSampler(
        dataset,
        pairs_per_batch=int(data_config["pairs_per_device_batch"]),
        seed=int(config["seed"]),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(data_config["num_workers"]),
        collate_fn=identity_collate,
    )


def main() -> None:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from transformers import get_cosine_schedule_with_warmup

    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    assert_training_source_clean(PACKAGE_ROOT)
    guard_runtime_environment()
    validate_declared_paths(config, REPO_ROOT)
    validate_base_checkpoint(config["model"]["base_checkpoint"])
    seed_everything(int(config["seed"]))

    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["optimizer"]["grad_accum_steps"]),
        mixed_precision=config["runtime"]["mixed_precision"] if torch.cuda.is_available() else "no",
        kwargs_handlers=[ddp],
    )
    # The cluster's Accelerate exposes this setting as a mutable property but
    # does not yet accept the newer constructor keyword.
    accelerator.even_batches = False
    expected_world_size = int(config["runtime"]["expected_world_size"])
    if accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            f"Registered gate requires {expected_world_size} processes, got {accelerator.num_processes}"
        )
    output_dir = Path(config["checkpoint"]["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(config, config_path, PACKAGE_ROOT)
        write_json_atomic(config["provenance"]["manifest_path"], manifest)
    accelerator.wait_for_everyone()

    adapter_root = REPO_ROOT / "outputs" / "samtok_selective"
    adapter_init = validate_adapter_init(config["checkpoint"].get("adapter_init"), adapter_root)
    model, processor = build_model_and_processor(config, adapter_path=adapter_init)
    trainable_summary = assert_only_lora_trainable(model)
    dataloader = _build_dataloader(config)
    if accelerator.is_main_process:
        manifest_path = Path(config["provenance"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_modules"] = runtime_module_files()
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({"runtime_modules": manifest["runtime_modules"]}, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_config = config["optimizer"]
    optimizer = AdamW(
        parameters,
        lr=float(optimizer_config["lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    max_steps = int(optimizer_config["max_steps"])
    updates_per_batch = validate_updates_per_batch(
        optimizer_config.get("updates_per_batch", 1)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * float(optimizer_config["warmup_ratio"])),
        num_training_steps=max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    metrics: dict[str, Any] = {
        "stage": config["stage"],
        "status": "running",
        "trainable": trainable_summary,
        "steps": [],
    }
    model.train()
    step = 0
    while step < max_steps:
        for samples in dataloader:
            if step >= max_steps:
                break
            outcomes = [bool(sample["no_target"]) for sample in samples]
            if not outcomes or all(outcomes) or not any(outcomes):
                raise RuntimeError("Every device batch must retain both selective outcomes")
            for batch_update in batch_update_indices(step, max_steps, updates_per_batch):
                with accelerator.accumulate(model):
                    inputs, answer_mask = build_supervised_inputs(processor, samples)
                    inputs = move_tensors(inputs, accelerator.device)
                    answer_mask = answer_mask.to(accelerator.device)
                    outputs = model(**inputs, use_cache=False)
                    loss = answer_token_cross_entropy(
                        outputs.logits,
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        answer_mask,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite SFT loss at step {step}: {loss}")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, float(optimizer_config["max_grad_norm"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                gathered_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                if not math.isfinite(gathered_loss):
                    raise FloatingPointError(f"Non-finite gathered loss at step {step}")
                if accelerator.is_main_process:
                    item = {
                        "step": step,
                        "batch_update": batch_update,
                        "loss": gathered_loss,
                        "batch_rows_per_rank": len(samples),
                    }
                    metrics["steps"].append(item)
                    if step % int(config["logging"]["log_every"]) == 0:
                        print(json.dumps(item), flush=True)
                step += 1
        if len(dataloader) == 0:
            raise RuntimeError("Paired dataloader is empty")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        adapter_dir = output_dir / "adapter"
        adapter_artifact = save_trainable_lora_adapter(
            unwrapped,
            adapter_dir,
            representation_mode=(
                config["model"].get("adapter_mode")
                == "frozen_anchor_plus_visual_projector"
            ),
        )
        processor.save_pretrained(adapter_dir)
        metrics["adapter_artifact"] = str(adapter_artifact)
        metrics["status"] = "finished"
        metrics["steps_completed"] = step
        metrics["updates_per_batch"] = updates_per_batch
        write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
