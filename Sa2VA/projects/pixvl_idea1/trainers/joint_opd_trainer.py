from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from projects.pixvl_idea1.rewards import compute_cap_reward, compute_seg_reward, is_cap_failure, is_seg_failure
from projects.pixvl_idea1.rewards.text_similarity import SentenceSimilarityScorer
from projects.pixvl_idea1.trainers.common import (
    build_dataloader,
    build_model_bundle,
    build_optimizer_and_scheduler,
    build_prompt_and_answer_ids,
    clean_generated_text,
    compute_answer_cross_entropy,
    compute_jsd,
    compute_teacher_confidence_weights,
    extract_adapter_state_dict,
    find_latest_adapter_checkpoint,
    find_latest_state_checkpoint,
    forward_answer_logits,
    generate_answer,
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
        find_unused_parameters=True,
        static_graph=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["optimizer"]["grad_accum_steps"],
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs],
    )

    def debug_phase(message: str) -> None:
        if accelerator.is_main_process:
            print(f"[stage2-debug {time.strftime('%F %T')}] {message}", flush=True)

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_state_dir, latest_state_step = find_latest_state_checkpoint(output_dir)
    latest_ckpt_dir, latest_ckpt_step = find_latest_adapter_checkpoint(output_dir)
    resume_steps = max(latest_ckpt_step, 0)
    adapter_path = cfg["student_init"]["adapter_path"]
    sampler_state = None
    if latest_state_dir is not None and latest_state_step >= 0:
        resume_steps = latest_state_step
        sampler_state = load_sampler_state(latest_state_dir)
        matching_ckpt_dir = output_dir / f"checkpoint-step-{latest_state_step}"
        if (matching_ckpt_dir / "adapter").exists():
            adapter_path = str(matching_ckpt_dir / "adapter")
    elif latest_ckpt_dir is not None and latest_ckpt_step >= 0:
        adapter_path = str(latest_ckpt_dir / "adapter")
    debug_phase("before_build_student")
    model, processor = build_model_bundle(cfg, trainable=True, adapter_path=adapter_path)
    debug_phase("after_build_student")
    teacher_model, _ = build_model_bundle(cfg, trainable=False, adapter_path=cfg["teacher"]["adapter_path"])
    debug_phase("after_build_teacher")

    dataloader, batch_sampler = build_dataloader(cfg, resume_steps=resume_steps, sampler_state=sampler_state)
    debug_phase("after_build_dataloader")
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    debug_phase("after_build_optimizer")
    model, teacher_model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, teacher_model, optimizer, dataloader, scheduler
    )
    debug_phase("after_accelerator_prepare")
    teacher_model.eval()
    similarity_scorer = SentenceSimilarityScorer()
    debug_phase("after_similarity_scorer")
    if latest_state_dir is not None and latest_state_step >= 0:
        accelerator.load_state(str(latest_state_dir))
        debug_phase(f"after_load_state step={latest_state_step}")
    elif resume_steps > 0:
        for _ in range(resume_steps):
            scheduler.step()
        debug_phase(f"after_scheduler_fastforward step={resume_steps}")

    metrics = {
        "stage": cfg["stage"],
        "status": "running",
        "steps": [],
    }

    log_every = int(cfg.get("logging", {}).get("log_every", 1))
    snapshot_every = int(cfg.get("logging", {}).get("snapshot_every", 10))
    save_every = int(cfg["checkpoint"].get("save_every", 0))

    debug_phase("trainer_ready")
    model.train()
    step = resume_steps
    while step < cfg["optimizer"]["max_steps"]:
        if step == resume_steps:
            debug_phase("enter_dataloader_loop")
        for batch in dataloader:
            if step >= cfg["optimizer"]["max_steps"]:
                break
            if step == resume_steps:
                debug_phase(f"first_batch_loaded size={len(batch)}")
            with accelerator.accumulate(model):
                total_loss = 0.0
                for sample_idx, sample in enumerate(batch):
                    if step == resume_steps and sample_idx == 0:
                        debug_phase(f"first_sample task={sample['task']} source={sample['source']}")
                    prompt_inputs, answer_ids = build_prompt_and_answer_ids(
                        processor,
                        sample["image"],
                        sample["prompt_text"],
                        sample["answer_text"],
                    )
                    prompt_inputs = move_inputs_to_device(prompt_inputs, accelerator.device)
                    answer_ids = answer_ids.to(accelerator.device)
                    ce_logits = forward_answer_logits(model, prompt_inputs, answer_ids)
                    ce_loss = compute_answer_cross_entropy(ce_logits, answer_ids)
                    if sample["task"] == "maskcap":
                        ce_loss = ce_loss * cfg["loss"]["lambda_cap_ce"]
                    if step == resume_steps and sample_idx == 0:
                        debug_phase("first_sample_ce_done")

                    generate_ctx = nullcontext()
                    if accelerator.distributed_type == DistributedType.FSDP:
                        generate_ctx = FSDP.summon_full_params(model, writeback=False, recurse=True)
                    with generate_ctx:
                        sample_ids, sample_text = generate_answer(
                            accelerator.unwrap_model(model),
                            processor,
                            prompt_inputs,
                            cfg["generation"][sample["task"]],
                        )
                    if step == resume_steps and sample_idx == 0:
                        debug_phase("first_sample_generate_done")

                    if sample["task"] == "refseg":
                        parsed_codes = dataloader.dataset.codec.text_to_codes(sample_text)
                        pred_mask = dataloader.dataset.codec.decode_codes(sample["image"], parsed_codes)
                        reward = compute_seg_reward(pred_mask, sample["mask_binary"], parsed_codes, sample["gt_codes"])
                        fail = is_seg_failure(reward, cfg["opd"]["tau_seg"])
                    else:
                        reward = compute_cap_reward(
                            clean_generated_text(sample_text),
                            sample["caption"],
                            similarity_scorer=similarity_scorer,
                        )
                        fail = is_cap_failure(reward, cfg["opd"]["tau_cap"])
                    if step == resume_steps and sample_idx == 0:
                        debug_phase("first_sample_reward_done")

                    overlay_prompt_inputs, _ = build_prompt_and_answer_ids(
                        processor,
                        sample["overlay_image"],
                        sample["prompt_text"],
                        sample["answer_text"],
                    )
                    overlay_prompt_inputs = move_inputs_to_device(overlay_prompt_inputs, accelerator.device)
                    sample_ids = sample_ids.to(accelerator.device)
                    student_logits = forward_answer_logits(model, prompt_inputs, sample_ids)
                    with torch.no_grad():
                        teacher_logits = forward_answer_logits(teacher_model, overlay_prompt_inputs, sample_ids)
                    weights = compute_teacher_confidence_weights(teacher_logits)
                    opd_loss = compute_jsd(student_logits, teacher_logits, weights)
                    opd_scale = 1.0 if (cfg["opd"]["all_sample_distill"] or fail) else 0.0
                    opd_loss = opd_loss * opd_scale
                    if step == resume_steps and sample_idx == 0:
                        debug_phase("first_sample_opd_done")

                    loss = ce_loss + cfg["opd"]["lambda_opd"] * opd_loss
                    total_loss = total_loss + loss

                total_loss = total_loss / max(len(batch), 1)
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if step == resume_steps:
                    debug_phase("first_optimizer_step_done")

            step_loss_value = float(total_loss.detach().cpu())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if accelerator.is_main_process:
                step_metrics = {"step": step, "loss": step_loss_value}
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
