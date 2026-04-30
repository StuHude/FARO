from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType

from projects.pixvl_idea1.trainers.common import (
    build_dataloader,
    build_model_bundle,
    build_optimizer_and_scheduler,
    build_prompt_and_answer_ids,
    build_supervised_batch_inputs,
    clean_generated_text,
    compute_answer_cross_entropy,
    compute_answer_cross_entropy_batch,
    extract_adapter_state_dict,
    find_latest_adapter_checkpoint,
    find_latest_state_checkpoint,
    forward_answer_logits,
    load_sampler_state,
    load_config,
    move_inputs_to_device,
    save_sampler_state,
    save_adapter_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    fsdp_plugin = None
    if cfg.get("memory_optim", {}).get("fsdp", {}).get("enabled", False):
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=cfg["memory_optim"]["fsdp"].get(
                "transformer_cls_names_to_wrap",
                ["Qwen3VLTextDecoderLayer"],
            ),
            state_dict_type=cfg["memory_optim"]["fsdp"].get("state_dict_type", "sharded_state_dict"),
            use_orig_params=True,
            limit_all_gathers=True,
            activation_checkpointing=cfg["memory_optim"]["fsdp"].get("activation_checkpointing", False),
        )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=bool(cfg.get("memory_optim", {}).get("gradient_checkpointing", False)),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["optimizer"]["grad_accum_steps"],
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs],
    )
    output_dir = Path(cfg["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_state_dir, latest_state_step = find_latest_state_checkpoint(output_dir)
    latest_ckpt_dir, latest_ckpt_step = find_latest_adapter_checkpoint(output_dir)
    resume_steps = 0
    adapter_path = cfg.get("student_init", {}).get("adapter_path")
    sampler_state = None
    if latest_state_dir is not None and latest_state_step >= 0:
        resume_steps = latest_state_step
        sampler_state = load_sampler_state(latest_state_dir)
        matching_ckpt_dir = output_dir / f"checkpoint-step-{latest_state_step}"
        if (matching_ckpt_dir / "adapter").exists():
            adapter_path = str(matching_ckpt_dir / "adapter")
    elif latest_ckpt_dir is not None and latest_ckpt_step >= 0:
        adapter_path = str(latest_ckpt_dir / "adapter")
        resume_steps = latest_ckpt_step
        sampler_state = load_sampler_state(latest_ckpt_dir)
    else:
        resume_cfg = cfg.get("resume", {})
        resume_steps = int(resume_cfg.get("completed_steps", 0))
    model, processor = build_model_bundle(
        cfg,
        trainable=True,
        adapter_path=adapter_path,
    )
    dataloader, batch_sampler = build_dataloader(cfg, resume_steps=resume_steps, sampler_state=sampler_state)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if latest_state_dir is not None and latest_state_step >= 0:
        accelerator.load_state(str(latest_state_dir))
    elif resume_steps > 0:
        for _ in range(resume_steps):
            scheduler.step()

    metrics = {
        "stage": cfg["stage"],
        "status": "running",
        "steps": [],
    }
    log_every = int(cfg.get("logging", {}).get("log_every", 1))
    snapshot_every = int(cfg.get("logging", {}).get("snapshot_every", 10))
    save_every = int(cfg["checkpoint"].get("save_every", 0))
    resume_metrics_path = output_dir / "metrics.resume_seed.json"
    if accelerator.is_main_process and resume_steps > 0 and resume_metrics_path.exists():
        with resume_metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
            metrics["status"] = "running"

    model.train()
    step = resume_steps
    while step < cfg["optimizer"]["max_steps"]:
        for batch in dataloader:
            if step >= cfg["optimizer"]["max_steps"]:
                break
            with accelerator.accumulate(model):
                total_loss = 0.0
                task_loss = {"refseg": 0.0, "maskcap": 0.0}
                batch_inputs, answer_mask = build_supervised_batch_inputs(processor, batch)
                batch_inputs = move_inputs_to_device(batch_inputs, accelerator.device)
                answer_mask = answer_mask.to(accelerator.device)

                outputs = model(**batch_inputs, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                labels = batch_inputs["input_ids"][:, 1:]
                token_mask = answer_mask[:, 1:] & batch_inputs["attention_mask"][:, 1:].bool()

                total_loss = compute_answer_cross_entropy_batch(logits, labels, token_mask)
                batch_task = batch[0]["task"]
                if batch_task == "maskcap":
                    total_loss = total_loss * cfg["loss"]["lambda_cap_ce"]
                task_loss[batch_task] = float(total_loss.detach().cpu())
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss_value = float(total_loss.detach().cpu())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            peak_alloc_mb = None
            peak_reserved_mb = None
            collect_peak_memory = bool(cfg.get("logging", {}).get("collect_peak_memory", True))
            should_collect_peak = collect_peak_memory and torch.cuda.is_available() and (
                step % snapshot_every == 0 or (save_every > 0 and step % save_every == 0) or step < 10
            )
            if should_collect_peak:
                peak_alloc = torch.tensor(
                    [torch.cuda.max_memory_allocated(accelerator.device) / (1024**2)],
                    device=accelerator.device,
                )
                peak_reserved = torch.tensor(
                    [torch.cuda.max_memory_reserved(accelerator.device) / (1024**2)],
                    device=accelerator.device,
                )
                gathered_alloc = accelerator.gather(peak_alloc).detach().cpu().tolist()
                gathered_reserved = accelerator.gather(peak_reserved).detach().cpu().tolist()
                peak_alloc_mb = [round(float(value), 1) for value in gathered_alloc]
                peak_reserved_mb = [round(float(value), 1) for value in gathered_reserved]
                torch.cuda.reset_peak_memory_stats(accelerator.device)

            if accelerator.is_main_process:
                step_metrics = {
                    "step": step,
                    "loss": step_loss_value,
                    "refseg_loss": task_loss["refseg"],
                    "maskcap_loss": task_loss["maskcap"],
                }
                if peak_alloc_mb is not None and peak_reserved_mb is not None:
                    step_metrics["peak_alloc_mb"] = peak_alloc_mb
                    step_metrics["peak_reserved_mb"] = peak_reserved_mb
                metrics["steps"].append(step_metrics)
                if step % log_every == 0:
                    print(json.dumps(step_metrics, ensure_ascii=False), flush=True)
                if step % snapshot_every == 0:
                    with (output_dir / "metrics.partial.json").open("w", encoding="utf-8") as handle:
                        json.dump(metrics, handle, ensure_ascii=False, indent=2)
            if save_every > 0 and (step + 1) % save_every == 0:
                state_dir = output_dir / f"state-step-{step + 1:06d}"
                accelerator.save_state(str(state_dir))
                ckpt_dir = output_dir / f"checkpoint-step-{step + 1}"
                periodic_model = accelerator.unwrap_model(model)
                periodic_state = None
                if accelerator.distributed_type == DistributedType.FSDP:
                    gathered_state = accelerator.get_state_dict(model)
                    periodic_state = extract_adapter_state_dict(periodic_model, state_dict=gathered_state)
                if accelerator.is_main_process:
                    save_sampler_state(state_dir, batch_sampler.state_dict())
                    save_adapter_checkpoint(
                        periodic_model,
                        processor,
                        str(ckpt_dir),
                        {
                            "stage": cfg["stage"],
                            "config_path": args.config,
                            "steps": step + 1,
                        },
                        state_dict=periodic_state,
                    )
                    save_sampler_state(ckpt_dir, batch_sampler.state_dict())
                accelerator.wait_for_everyone()
            del total_loss
            step += 1

    final_model = accelerator.unwrap_model(model)
    final_adapter_state = None
    if accelerator.distributed_type == DistributedType.FSDP:
        gathered_state = accelerator.get_state_dict(model)
        final_adapter_state = extract_adapter_state_dict(final_model, state_dict=gathered_state)

    if accelerator.is_main_process:
        metrics["status"] = "finished"
        save_adapter_checkpoint(
            final_model,
            processor,
            str(output_dir),
            {
                "stage": cfg["stage"],
                "config_path": args.config,
                "steps": step,
            },
            state_dict=final_adapter_state,
        )
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
