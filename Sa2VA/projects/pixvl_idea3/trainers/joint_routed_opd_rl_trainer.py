from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType

from projects.pixvl_idea1.trainers.common import (
    build_dataloader,
    build_model_bundle,
    build_optimizer_and_scheduler,
    build_prompt_and_answer_ids,
    clean_generated_text,
    compute_answer_cross_entropy,
    compute_answer_logprob_sums,
    compute_jsd_values,
    compute_reference_kl_values,
    compute_teacher_confidence_weights_batch,
    extract_adapter_state_dict,
    find_latest_adapter_checkpoint,
    find_latest_state_checkpoint,
    forward_answer_logits,
    forward_answer_logits_batch,
    generate_answers,
    load_config,
    load_sampler_state,
    move_inputs_to_device,
    normalize_rewards,
    save_adapter_checkpoint,
    save_sampler_state,
    seed_everything,
)
from projects.pixvl_idea1.rewards.text_similarity import SentenceSimilarityScorer
from projects.pixvl_idea3.routing import (
    build_relation_confuser_map,
    compute_geometry_reward,
    compute_relation_caption_reward,
    compute_relation_reward,
    compute_semantic_caption_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def route_bucket_for_sample(sample: dict[str, object]) -> str:
    meta = sample.get("meta") or {}
    if isinstance(meta, dict):
        return str(meta.get("failure_route", sample.get("route_bucket", "geometry")))
    return str(sample.get("route_bucket", "geometry"))


def failure_threshold(cfg: dict[str, object], bucket: str, task: str) -> float:
    routing = cfg.get("routing", {})
    thresholds = routing.get("failure_thresholds", {}) if isinstance(routing, dict) else {}
    if bucket in thresholds:
        return float(thresholds[bucket])
    rl_cfg = cfg.get("rl", {})
    if task == "maskcap":
        return float(rl_cfg.get("tau_cap", 0.65))
    return float(rl_cfg.get("tau_seg", 0.5))


def routed_reward(
    cfg: dict[str, object],
    sample: dict[str, object],
    sample_text: str,
    dataset,
    relation_confusers: dict[str, list[dict[str, object]]],
    similarity_scorer: SentenceSimilarityScorer,
) -> tuple[float, dict[str, float]]:
    bucket = route_bucket_for_sample(sample)
    reward_cfg = (cfg.get("routing", {}) or {}).get("rewards", {})
    if sample["task"] == "maskcap":
        if bucket == "relation":
            details = compute_relation_caption_reward(
                clean_generated_text(sample_text),
                sample["caption"],
                similarity_scorer=similarity_scorer,
                base_weight=float(reward_cfg.get("relation_caption", {}).get("base_weight", 0.7)),
                relation_weight=float(reward_cfg.get("relation_caption", {}).get("relation_weight", 0.3)),
            )
        else:
            details = compute_semantic_caption_reward(
                clean_generated_text(sample_text),
                sample["caption"],
                similarity_scorer=similarity_scorer,
                base_weight=float(reward_cfg.get("semantic", {}).get("base_weight", 0.75)),
                keyword_weight=float(reward_cfg.get("semantic", {}).get("keyword_weight", 0.25)),
            )
        return details["total"], details

    parsed_codes = dataset.codec.text_to_codes(sample_text)
    pred_mask = dataset.codec.decode_codes(sample["image"], parsed_codes)
    if bucket == "relation":
        details = compute_relation_reward(
            pred_mask,
            sample["mask_binary"],
            confuser_masks=relation_confusers.get(sample["id"], []),
            pred_codes=parsed_codes,
            gt_codes=sample["gt_codes"],
            target_weight=float(reward_cfg.get("relation", {}).get("target_weight", 0.7)),
            margin_weight=float(reward_cfg.get("relation", {}).get("margin_weight", 0.2)),
            exact_weight=float(reward_cfg.get("relation", {}).get("exact_weight", 0.1)),
        )
    else:
        details = compute_geometry_reward(
            pred_mask,
            sample["mask_binary"],
            pred_codes=parsed_codes,
            gt_codes=sample["gt_codes"],
            ciou_weight=float(reward_cfg.get("geometry", {}).get("ciou_weight", 0.55)),
            boundary_weight=float(reward_cfg.get("geometry", {}).get("boundary_weight", 0.25)),
            area_weight=float(reward_cfg.get("geometry", {}).get("area_weight", 0.1)),
            exact_weight=float(reward_cfg.get("geometry", {}).get("exact_weight", 0.1)),
            boundary_width=int(reward_cfg.get("geometry", {}).get("boundary_width", 2)),
        )
    return details["total"], details


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

    model, processor = build_model_bundle(cfg, trainable=True, adapter_path=adapter_path)
    teacher_model, _ = build_model_bundle(cfg, trainable=False, adapter_path=cfg["teacher"]["adapter_path"])
    reference_model, _ = build_model_bundle(cfg, trainable=False, adapter_path=cfg["reference"]["adapter_path"])

    dataloader, batch_sampler = build_dataloader(cfg, resume_steps=resume_steps, sampler_state=sampler_state)
    relation_confusers = build_relation_confuser_map(
        dataloader.dataset.records,
        max_confusers=int(cfg.get("routing", {}).get("max_confusers", 16)),
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    model, teacher_model, reference_model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, teacher_model, reference_model, optimizer, dataloader, scheduler
    )
    teacher_model.eval()
    reference_model.eval()
    similarity_scorer = SentenceSimilarityScorer()
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

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

    model.train()
    step = resume_steps
    while step < cfg["optimizer"]["max_steps"]:
        for batch in dataloader:
            if step >= cfg["optimizer"]["max_steps"]:
                break
            step_bucket_rewards: dict[str, list[float]] = defaultdict(list)
            step_bucket_failures: dict[str, int] = defaultdict(int)
            step_bucket_counts: dict[str, int] = defaultdict(int)
            with accelerator.accumulate(model):
                total_loss = 0.0
                for sample in batch:
                    bucket = route_bucket_for_sample(sample)
                    bucket_cfg = cfg.get("routing", {}).get("buckets", {}).get(bucket, {})
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
                    ce_scale = float(bucket_cfg.get("ce_scale", 1.0))
                    if sample["task"] == "maskcap":
                        ce_loss = ce_loss * float(cfg["loss"].get("lambda_cap_ce", 1.0))

                    rewards = []
                    group_size = cfg["rl"]["group_size"][sample["task"]]
                    max_group_size = max(cfg["rl"]["group_size"].values())
                    effective_mask = torch.tensor(
                        [1.0 if sample_idx < group_size else 0.0 for sample_idx in range(max_group_size)],
                        dtype=torch.float32,
                        device=accelerator.device,
                    )

                    overlay_prompt_inputs, _ = build_prompt_and_answer_ids(
                        processor,
                        sample["overlay_image"],
                        sample["prompt_text"],
                        sample["answer_text"],
                    )
                    overlay_prompt_inputs = move_inputs_to_device(overlay_prompt_inputs, accelerator.device)

                    sample_ids_batch, sample_texts = generate_answers(
                        accelerator.unwrap_model(model),
                        processor,
                        prompt_inputs,
                        cfg["generation"][sample["task"]],
                        num_return_sequences=max_group_size,
                    )
                    sample_ids_batch = sample_ids_batch.to(accelerator.device)
                    student_logits_batch, answer_attention = forward_answer_logits_batch(
                        model,
                        prompt_inputs,
                        sample_ids_batch,
                        pad_token_id,
                    )
                    logprob_sums = compute_answer_logprob_sums(
                        student_logits_batch,
                        sample_ids_batch,
                        answer_attention,
                    )

                    fail_mask: list[float] = []
                    for sample_idx, sample_text in enumerate(sample_texts):
                        effective_scale = float(effective_mask[sample_idx].item())
                        if effective_scale:
                            reward_value, reward_details = routed_reward(
                                cfg,
                                sample,
                                sample_text,
                                dataloader.dataset,
                                relation_confusers,
                                similarity_scorer,
                            )
                            rewards.append(reward_value)
                            step_bucket_rewards[bucket].append(float(reward_value))
                            threshold = failure_threshold(cfg, bucket, sample["task"])
                            fail = reward_value < threshold
                            if fail:
                                step_bucket_failures[bucket] += 1
                            step_bucket_counts[bucket] += 1
                        else:
                            reward_value = 0.0
                            reward_details = {}
                            fail = False
                        fail_mask.append(1.0 if (effective_scale and fail) else 0.0)

                    if cfg["loss"].get("lambda_opd", 0.0) > 0.0 and (
                        cfg.get("opd", {}).get("all_sample_distill", False) or any(value > 0.0 for value in fail_mask)
                    ):
                        with torch.no_grad():
                            teacher_logits_batch, _ = forward_answer_logits_batch(
                                teacher_model,
                                overlay_prompt_inputs,
                                sample_ids_batch,
                                pad_token_id,
                            )
                        teacher_weights = compute_teacher_confidence_weights_batch(teacher_logits_batch)
                        opd_values = compute_jsd_values(
                            student_logits_batch,
                            teacher_logits_batch,
                            teacher_weights,
                            answer_attention,
                        )
                    else:
                        opd_values = torch.zeros(max_group_size, dtype=torch.float32, device=accelerator.device)
                    opd_mask = torch.tensor(fail_mask, dtype=torch.float32, device=accelerator.device)

                    with torch.no_grad():
                        ref_logits_batch, _ = forward_answer_logits_batch(
                            reference_model,
                            prompt_inputs,
                            sample_ids_batch,
                            pad_token_id,
                        )
                    kl_values = compute_reference_kl_values(
                        student_logits_batch,
                        ref_logits_batch,
                        answer_attention,
                    )

                    advantages = normalize_rewards(rewards).to(accelerator.device)
                    if max_group_size > group_size:
                        pad = torch.zeros(max_group_size - group_size, dtype=advantages.dtype, device=advantages.device)
                        advantages = torch.cat([advantages, pad], dim=0)
                    rl_loss = torch.stack(
                        [-advantage * logprob for advantage, logprob in zip(advantages, logprob_sums)]
                    ).sum() / effective_mask.sum().clamp_min(1.0)
                    opd_loss = (opd_values * opd_mask).sum() / opd_mask.sum().clamp_min(1.0)
                    kl_loss = (kl_values * effective_mask).sum() / effective_mask.sum().clamp_min(1.0)

                    rl_scale = float(bucket_cfg.get("rl_scale", 1.0))
                    opd_scale = float(bucket_cfg.get("opd_scale", 1.0))
                    if sample["task"] == "refseg":
                        base_rl_lambda = float(cfg["loss"]["lambda_rl_seg"])
                    else:
                        base_rl_lambda = float(cfg["loss"]["lambda_rl_cap"])
                    loss = (
                        float(cfg["loss"]["lambda_ce"]) * ce_scale * ce_loss
                        + base_rl_lambda * rl_scale * rl_loss
                        + float(cfg["loss"].get("lambda_opd", 0.0)) * opd_scale * opd_loss
                        + float(cfg["loss"]["beta_kl"]) * kl_loss
                    )
                    total_loss = total_loss + loss

                total_loss = total_loss / max(len(batch), 1)
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss_value = float(total_loss.detach().cpu())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if accelerator.is_main_process:
                step_metrics = {"step": step, "loss": step_loss_value}
                for bucket in sorted(step_bucket_counts):
                    count = max(step_bucket_counts[bucket], 1)
                    step_metrics[f"{bucket}_mean_reward"] = sum(step_bucket_rewards[bucket]) / max(
                        len(step_bucket_rewards[bucket]), 1
                    )
                    step_metrics[f"{bucket}_failure_rate"] = step_bucket_failures[bucket] / count
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
