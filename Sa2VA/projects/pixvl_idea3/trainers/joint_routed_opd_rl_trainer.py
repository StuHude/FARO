from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    InitProcessGroupKwargs,
)
from accelerate.utils import DistributedType

from projects.pixvl_idea1.trainers.common import (
    build_dataloader,
    build_model_bundle,
    build_optimizer_and_scheduler,
    build_prompt_batch_inputs,
    build_supervised_batch_inputs,
    clean_generated_text,
    compute_answer_cross_entropy_per_sample,
    compute_answer_logprob_sums,
    build_answer_token_scope,
    compute_jsd_values,
    compute_reference_kl_values,
    compute_teacher_confidence_weights_batch,
    extract_adapter_state_dict,
    find_latest_adapter_checkpoint,
    find_latest_state_checkpoint,
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
    compute_semantic_coverage_calibration_reward,
    compute_semantic_caption_reward,
    infer_atom_failure_route,
)
from projects.pixvl_idea3.failure_evidence import predicted_only_evidence_route, soft_local_scale
from projects.pixvl_idea3.reward_ranking import RunningComponentRanker
from projects.pixvl_idea3.selective_policy import (
    anchor_relative_advantages,
    project_conflicting_gradient,
    selective_outcome_loss_scales,
)
from projects.pixvl_idea3.existence import predicts_target_exists
from projects.pixvl_idea3.risk_constraints import outcome_constraint, update_dual


def quality_gated_advantages(
    rewards: list[float],
    *,
    threshold: float,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Use GRPO advantages only for rollouts above an absolute quality floor.

    Relative ranking can reinforce a uniformly bad candidate group.  The gate
    makes such groups contribute zero policy-gradient signal while retaining
    the ordinary CE/KL preservation terms.  Above the floor, the smooth gate
    preserves the within-group advantage ordering.
    """
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() <= 1:
        return torch.zeros_like(values)
    centered = (values - values.mean()) / (values.std() + 1e-4)
    scale = max(float(temperature), 1e-4)
    gate = torch.sigmoid((values - float(threshold)) / scale)
    # A hard floor avoids tiny updates from uniformly failed groups; the
    # sigmoid remains useful for mixed groups near the acceptance boundary.
    gate = gate * (values >= float(threshold)).to(values.dtype)
    return centered * gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def routing_payload_for_sample(cfg: dict[str, object], sample: dict[str, object]) -> dict[str, object]:
    runtime_key = "_runtime_routing_payload"
    runtime_mode_key = "_runtime_routing_mode"
    routing_mode = str((cfg.get("routing", {}) or {}).get("mode", "source_bucket"))
    cached_mode = sample.get(runtime_mode_key)
    cached_payload = sample.get(runtime_key)
    if cached_mode == routing_mode and isinstance(cached_payload, dict):
        return cached_payload

    if routing_mode in {"shared", "unified"}:
        # Common task-matched objective: segmentation uses geometry reward and
        # captioning uses semantic reward, without any per-sample router.
        bucket = "geometry" if str(sample.get("task", "")) == "refseg" else "semantic"
        payload = {
            "failure_route": bucket,
            "failure_route_reasons": ["shared_task_matched_objective"],
            "route_weights": {name: 0.0 for name in ("semantic", "relation", "geometry")},
            "shared_objective": True,
        }
        sample[runtime_mode_key] = routing_mode
        sample[runtime_key] = payload
        return payload

    if routing_mode == "shuffled":
        # Matched-compute negative control.  The assignment is deterministic
        # per sample, independent of prompt, GT, source bucket, and rollout.
        sample_id = str(sample.get("id", ""))
        seed = int((cfg.get("seed", 0) if isinstance(cfg, dict) else 0))
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        bucket = ("semantic", "relation", "geometry")[digest[0] % 3]
        payload = {
            "failure_route": bucket,
            "failure_route_reasons": ["deterministic_shuffled_control"],
            "route_weights": {
                name: 1.0 if name == bucket else 0.0
                for name in ("semantic", "relation", "geometry")
            },
            "shuffled": True,
        }
        sample[runtime_mode_key] = routing_mode
        sample[runtime_key] = payload
        return payload

    if routing_mode in {"predicted_only_evidence", "predicted_evidence"}:
        # A rollout has not been produced yet.  Keep the pre-rollout fallback
        # neutral; update_rollout_predicted_route replaces it immediately after
        # generation.  This preserves callers that inspect a sample early.
        payload = {
            "failure_route": "geometry",
            "failure_route_reasons": ["awaiting_predicted_rollout"],
            "route_weights": {name: 1.0 / 3.0 for name in ("semantic", "relation", "geometry")},
            "predicted_only": True,
        }
        sample[runtime_mode_key] = routing_mode
        sample[runtime_key] = payload
        return payload

    if routing_mode == "atom_conditioned":
        payload = infer_atom_failure_route(sample, include_slice_tags=False)
    else:
        meta = sample.get("meta") or {}
        if isinstance(meta, dict):
            bucket = str(meta.get("failure_route", sample.get("route_bucket", "geometry")))
        else:
            bucket = str(sample.get("route_bucket", "geometry"))
        payload = {
            "failure_route": bucket,
            "failure_route_reasons": ["precomputed_source_bucket"],
            "route_weights": {
                "semantic": 1.0 if bucket == "semantic" else 0.0,
                "relation": 1.0 if bucket == "relation" else 0.0,
                "geometry": 1.0 if bucket == "geometry" else 0.0,
            },
        }

    sample[runtime_mode_key] = routing_mode
    sample[runtime_key] = payload
    return payload


def update_rollout_predicted_route(
    cfg: dict[str, object],
    sample: dict[str, object],
    predicted_text: str,
    *,
    answer_confidence: float | None = None,
) -> dict[str, object]:
    """Update a sample route from generated text, without target-derived data."""

    routing = cfg.get("routing", {}) or {}
    mode = str(routing.get("mode", "source_bucket")) if isinstance(routing, dict) else "source_bucket"
    if mode not in {"predicted_only_evidence", "predicted_evidence"}:
        return routing_payload_for_sample(cfg, sample)
    fepo_cfg = routing.get("predicted_only_evidence", {}) if isinstance(routing, dict) else {}
    if not isinstance(fepo_cfg, dict):
        fepo_cfg = {}
    payload = predicted_only_evidence_route(
        prompt_text=str(sample.get("prompt_text", "")),
        predicted_text=str(predicted_text),
        task=str(sample.get("task", "")),
        answer_confidence=answer_confidence,
        temperature=float(fepo_cfg.get("temperature", 0.25)),
        min_failure=float(fepo_cfg.get("min_failure", 0.25)),
    )
    sample["_runtime_routing_mode"] = mode
    sample["_runtime_routing_payload"] = payload
    return payload


def route_bucket_for_sample(cfg: dict[str, object], sample: dict[str, object]) -> str:
    payload = routing_payload_for_sample(cfg, sample)
    return str(payload.get("failure_route", "geometry"))


def route_weights_for_sample(cfg: dict[str, object], sample: dict[str, object]) -> dict[str, float]:
    payload = routing_payload_for_sample(cfg, sample)
    weights = payload.get("route_weights")
    if isinstance(weights, dict):
        return {
            "semantic": float(weights.get("semantic", 0.0)),
            "relation": float(weights.get("relation", 0.0)),
            "geometry": float(weights.get("geometry", 0.0)),
        }
    bucket = str(payload.get("failure_route", "geometry"))
    return {
        "semantic": 1.0 if bucket == "semantic" else 0.0,
        "relation": 1.0 if bucket == "relation" else 0.0,
        "geometry": 1.0 if bucket == "geometry" else 0.0,
    }


def bucket_scale_with_route_weight(
    bucket_cfg: dict[str, object],
    route_weights: dict[str, float],
    bucket: str,
    key: str,
) -> float:
    base = float(bucket_cfg.get(key, 1.0))
    # Keep legacy bucket scales, but let atom routing softly amplify the chosen branch.
    return base * (1.0 + float(route_weights.get(bucket, 0.0)))


def build_privileged_rollout_prompt(sample: dict[str, object], best_text: str) -> str:
    prompt_text = str(sample["prompt_text"])
    if sample["task"] == "refseg":
        return (
            prompt_text
            + "\n[Training-only privileged correct rollout]\n"
            + best_text
            + "\nUse the privileged correct rollout above as hidden guidance when evaluating candidate mask tokens."
        )
    return (
        prompt_text
        + "\n[Training-only privileged correct rollout]\n"
        + clean_generated_text(best_text)
        + "\nUse the privileged correct rollout above as hidden guidance when evaluating candidate description tokens."
    )


def self_teacher_logits_batch(
    model: torch.nn.Module,
    processor,
    task_samples: list[dict[str, object]],
    privileged_texts: list[str | None],
    image_key: str,
    sample_ids_batch: torch.Tensor,
    accelerator_device: torch.device,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_samples: list[dict[str, object]] = []
    for sample, privileged_text in zip(task_samples, privileged_texts):
        teacher_sample = dict(sample)
        if privileged_text is not None:
            teacher_sample["prompt_text"] = build_privileged_rollout_prompt(sample, privileged_text)
        teacher_samples.append(teacher_sample)
    overlay_prompt_inputs = build_prompt_batch_inputs(processor, teacher_samples, image_key=image_key)
    overlay_prompt_inputs = move_inputs_to_device(overlay_prompt_inputs, accelerator_device)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        teacher_logits_batch, teacher_answer_attention = forward_answer_logits_batch(
            model,
            overlay_prompt_inputs,
            sample_ids_batch,
            pad_token_id,
        )
    if model_was_training:
        model.train()
    return teacher_logits_batch, teacher_answer_attention


def route_bucket_for_sample_legacy(sample: dict[str, object]) -> str:
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


def compute_answer_entropy_scores(
    answer_logits: torch.Tensor,
    answer_attention: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.softmax(answer_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    if answer_attention is not None:
        entropy = entropy * answer_attention.to(entropy.dtype)
        lengths = answer_attention.sum(dim=-1).clamp_min(1).to(entropy.dtype)
    else:
        lengths = torch.full(
            (entropy.shape[0],),
            entropy.shape[1],
            dtype=entropy.dtype,
            device=entropy.device,
        ).clamp_min(1)
    return entropy.sum(dim=-1) / lengths


def rollout_answer_confidence(
    answer_logits: torch.Tensor,
    answer_attention: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map predictive entropy to a bounded rollout-only confidence probe."""

    entropy = compute_answer_entropy_scores(answer_logits, answer_attention)
    max_entropy = torch.log(
        torch.tensor(float(answer_logits.shape[-1]), device=answer_logits.device)
    ).clamp_min(1e-6)
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)


def minmax_normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    vmin = values.min()
    vmax = values.max()
    denom = (vmax - vmin).clamp_min(1e-6)
    return (values - vmin) / denom


def triage_labels_from_stats(
    cfg: dict[str, object],
    reward_values: torch.Tensor,
    nll_values: torch.Tensor,
    entropy_values: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    triage_cfg = (cfg.get("triage", {}) or {})
    reward_weight = float(triage_cfg.get("reward_weight", 0.5))
    nll_weight = float(triage_cfg.get("nll_weight", 0.25))
    entropy_weight = float(triage_cfg.get("entropy_weight", 0.25))
    clean_quantile = float(triage_cfg.get("clean_quantile", 0.35))
    corrupted_quantile = float(triage_cfg.get("corrupted_quantile", 0.8))

    reward_deficit = 1.0 - reward_values
    q_values = (
        reward_weight * minmax_normalize_tensor(reward_deficit)
        + nll_weight * minmax_normalize_tensor(nll_values)
        + entropy_weight * minmax_normalize_tensor(entropy_values)
    )
    if q_values.numel() <= 1:
        return ["suspicious"], q_values

    clean_tau = torch.quantile(q_values, clean_quantile)
    corr_tau = torch.quantile(q_values, corrupted_quantile)
    labels: list[str] = []
    for q in q_values:
        qv = float(q.item())
        if qv <= float(clean_tau.item()):
            labels.append("clean")
        elif qv >= float(corr_tau.item()):
            labels.append("corrupted")
        else:
            labels.append("suspicious")
    return labels, q_values


def routed_reward(
    cfg: dict[str, object],
    sample: dict[str, object],
    sample_text: str,
    dataset,
    relation_confusers: dict[str, list[dict[str, object]]],
    similarity_scorer: SentenceSimilarityScorer,
) -> tuple[float, dict[str, float]]:
    if sample.get("task") == "existence":
        text = clean_generated_text(sample_text).lower()
        target_exists = "no target" not in str(sample.get("answer_text", "")).lower()
        predicted_exists = predicts_target_exists(text)
        reward = 1.0 if predicted_exists == target_exists else 0.0
        return reward, {"total": reward, "existence_total": reward}
    if sample.get("task") == "refseg" and bool((sample.get("meta") or {}).get("no_target", False)):
        explicit_null = not predicts_target_exists(clean_generated_text(sample_text))
        negative_cfg = cfg.get("selective_negative_reward", {}) or {}
        mode = str(negative_cfg.get("mode", "binary"))
        if explicit_null:
            reward = 1.0
            predicted_area_ratio = 0.0
        elif mode == "area_penalty":
            parsed_codes = dataset.codec.text_to_codes(sample_text)
            if len(parsed_codes) != dataset.codec.codebook_depth:
                reward = -1.0
                predicted_area_ratio = 1.0
            else:
                pred_mask = dataset.codec.decode_codes(sample["image"], parsed_codes)
                if torch.is_tensor(pred_mask):
                    predicted_area_ratio = float(pred_mask.float().mean().item())
                else:
                    predicted_area_ratio = float(pred_mask.mean())
                predicted_area_ratio = max(0.0, min(1.0, predicted_area_ratio))
                reward = -(predicted_area_ratio ** 0.5)
        else:
            reward = 0.0
            predicted_area_ratio = 0.0
        return reward, {
            "total": reward,
            "ciou": reward,
            "boundary": reward,
            "area": reward,
            "exact": reward,
            "selective_null": reward,
            "predicted_area_ratio": predicted_area_ratio,
        }
    routing_mode = str((cfg.get("routing", {}) or {}).get("mode", "source_bucket"))
    if routing_mode in {"predicted_only_evidence", "predicted_evidence"}:
        # FEPO's defining contract: evaluate the same rollout with every
        # task-valid verifier and blend their scores by predicted evidence.
        # The route may still select a dominant local credit branch, but the
        # reward itself is no longer a hard argmax over capabilities.
        payload = routing_payload_for_sample(cfg, sample)
        raw_weights = payload.get("route_weights", {})
        weights = {
            name: float(raw_weights.get(name, 0.0)) if isinstance(raw_weights, dict) else 0.0
            for name in ("semantic", "relation", "geometry")
        }
        reward_cfg = (cfg.get("routing", {}) or {}).get("rewards", {})
        text = clean_generated_text(sample_text)

        if sample["task"] == "maskcap":
            semantic_cfg = reward_cfg.get("semantic", {})
            semantic_mode = str(semantic_cfg.get("mode", "base_keyword"))
            if semantic_mode == "coverage_calibration":
                semantic_details = compute_semantic_coverage_calibration_reward(
                    text,
                    sample["caption"],
                    rec_weight=float(semantic_cfg.get("rec_weight", 0.2)),
                    pos_weight=float(semantic_cfg.get("pos_weight", 0.45)),
                    neg_weight=float(semantic_cfg.get("neg_weight", 0.35)),
                )
            else:
                semantic_details = compute_semantic_caption_reward(
                    text,
                    sample["caption"],
                    similarity_scorer=similarity_scorer,
                    base_weight=float(semantic_cfg.get("base_weight", 0.75)),
                    keyword_weight=float(semantic_cfg.get("keyword_weight", 0.25)),
                )
            relation_details = compute_relation_caption_reward(
                text,
                sample["caption"],
                similarity_scorer=similarity_scorer,
                base_weight=float(reward_cfg.get("relation_caption", {}).get("base_weight", 0.7)),
                relation_weight=float(reward_cfg.get("relation_caption", {}).get("relation_weight", 0.3)),
            )
            active = {"semantic": weights["semantic"], "relation": weights["relation"]}
            active_total = sum(active.values())
            if active_total <= 1e-8:
                active = {"semantic": 1.0, "relation": 0.0}
                active_total = 1.0
            total = (
                active["semantic"] * float(semantic_details["total"])
                + active["relation"] * float(relation_details["total"])
            ) / active_total
            return total, {
                "total": total,
                "semantic_total": float(semantic_details["total"]),
                "relation_total": float(relation_details["total"]),
                "route_semantic_weight": active["semantic"] / active_total,
                "route_relation_weight": active["relation"] / active_total,
            }

        parsed_codes = dataset.codec.text_to_codes(sample_text)
        pred_mask = dataset.codec.decode_codes(sample["image"], parsed_codes)
        relation_details = compute_relation_reward(
            pred_mask,
            sample["mask_binary"],
            confuser_masks=relation_confusers.get(sample["id"], []),
            pred_codes=parsed_codes,
            gt_codes=sample["gt_codes"],
            target_weight=float(reward_cfg.get("relation", {}).get("target_weight", 0.7)),
            margin_weight=float(reward_cfg.get("relation", {}).get("margin_weight", 0.2)),
            exact_weight=float(reward_cfg.get("relation", {}).get("exact_weight", 0.1)),
        )
        geometry_details = compute_geometry_reward(
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
        # Refseg has no independent semantic verifier; its semantic mass is
        # conservatively folded into the geometry verifier rather than dropped.
        active = {
            "geometry": weights["geometry"] + weights["semantic"],
            "relation": weights["relation"],
        }
        active_total = sum(active.values())
        if active_total <= 1e-8:
            active = {"geometry": 1.0, "relation": 0.0}
            active_total = 1.0
        total = (
            active["geometry"] * float(geometry_details["total"])
            + active["relation"] * float(relation_details["total"])
        ) / active_total
        fepo_cfg = (cfg.get("routing", {}) or {}).get("predicted_only_evidence", {})
        anchor_weight = float(fepo_cfg.get("geometry_anchor", 0.0)) if isinstance(fepo_cfg, dict) else 0.0
        anchor_weight = max(0.0, min(1.0, anchor_weight))
        if anchor_weight > 0.0:
            # Preserve a task-matched geometry floor while allowing evidence
            # to redirect the remaining correction budget to relation credit.
            total = anchor_weight * float(geometry_details["total"]) + (1.0 - anchor_weight) * total
        return total, {
            "total": total,
            "geometry_total": float(geometry_details["total"]),
            "relation_total": float(relation_details["total"]),
            "geometry_anchor": anchor_weight,
            "route_geometry_weight": active["geometry"] / active_total,
            "route_relation_weight": active["relation"] / active_total,
        }

    bucket = route_bucket_for_sample(cfg, sample)
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
            semantic_cfg = reward_cfg.get("semantic", {})
            semantic_mode = str(semantic_cfg.get("mode", "base_keyword"))
            if semantic_mode == "coverage_calibration":
                details = compute_semantic_coverage_calibration_reward(
                    clean_generated_text(sample_text),
                    sample["caption"],
                    rec_weight=float(semantic_cfg.get("rec_weight", 0.2)),
                    pos_weight=float(semantic_cfg.get("pos_weight", 0.45)),
                    neg_weight=float(semantic_cfg.get("neg_weight", 0.35)),
                )
            else:
                details = compute_semantic_caption_reward(
                    clean_generated_text(sample_text),
                    sample["caption"],
                    similarity_scorer=similarity_scorer,
                    base_weight=float(semantic_cfg.get("base_weight", 0.75)),
                    keyword_weight=float(semantic_cfg.get("keyword_weight", 0.25)),
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


def ensure_gpu_reserve_buffers(
    reserve_buffers: list[torch.Tensor],
    *,
    target_gb: float,
    headroom_gb: float,
    device: torch.device,
) -> None:
    if not torch.cuda.is_available() or device.type != "cuda":
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    target_bytes = int(target_gb * (1024 ** 3))
    headroom_bytes = int(headroom_gb * (1024 ** 3))
    need_bytes = target_bytes - used_bytes
    max_alloc = free_bytes - headroom_bytes
    if need_bytes <= 0 or max_alloc <= 0:
        return
    alloc_bytes = min(need_bytes, max_alloc)
    chunk_bytes = 256 * 1024 * 1024
    allocated_total = 0
    while alloc_bytes > 0:
        current = min(chunk_bytes, alloc_bytes)
        reserve_buffers.append(torch.zeros(current, dtype=torch.uint8, device=device))
        alloc_bytes -= current
        allocated_total += current
    if allocated_total > 0:
        print(
            json.dumps(
                {
                    "reserve_mb": round(allocated_total / (1024 ** 2), 2),
                    "target_gb": target_gb,
                    "headroom_gb": headroom_gb,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

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
        static_graph=False,
        broadcast_buffers=False,
    )
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["optimizer"]["grad_accum_steps"],
        mixed_precision="no",
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
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
    teacher_model.to(accelerator.device)
    reference_model.to(accelerator.device)

    dataloader, batch_sampler = build_dataloader(cfg, resume_steps=resume_steps, sampler_state=sampler_state)
    relation_confusers = build_relation_confuser_map(
        dataloader.dataset.records,
        max_confusers=int(cfg.get("routing", {}).get("max_confusers", 16)),
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    teacher_model.eval()
    reference_model.eval()
    similarity_scorer = SentenceSimilarityScorer()
    rank_cfg = cfg.get("reward_ranking", {}) or {}
    reward_rankers = {}
    if bool(rank_cfg.get("enabled", False)):
        capacity = int(rank_cfg.get("capacity", 16))
        configured_components = rank_cfg.get("components", {}) or {}
        reward_rankers = {
            "refseg": RunningComponentRanker(
                capacity,
                tuple(configured_components.get("refseg", ("ciou", "boundary", "area", "exact"))),
            ),
            "maskcap": RunningComponentRanker(
                capacity,
                tuple(configured_components.get("maskcap", ("base", "relation"))),
            ),
        }
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
    risk_cfg = cfg.get("risk_constraint", {}) or {}
    risk_enabled = bool(risk_cfg.get("enabled", False))
    null_dual = float(risk_cfg.get("lambda_init", 0.0))
    null_budget = float(risk_cfg.get("budget", 0.0))
    null_epsilon = float(risk_cfg.get("epsilon", 0.0))
    null_dual_lr = float(risk_cfg.get("dual_lr", 0.0))
    null_dual_max = float(risk_cfg.get("lambda_max", 10.0))
    log_every = int(cfg.get("logging", {}).get("log_every", 1))
    snapshot_every = int(cfg.get("logging", {}).get("snapshot_every", 10))
    save_every = int(cfg["checkpoint"].get("save_every", 0))
    reserve_buffers: list[torch.Tensor] = []
    reserve_target_gb = float(cfg.get("memory_optim", {}).get("gpu_reserve_target_gb", 0.0))
    reserve_headroom_gb = float(cfg.get("memory_optim", {}).get("gpu_reserve_headroom_gb", 8.0))
    projection_cfg = cfg.get("selective_gradient_projection", {}) or {}
    projection_enabled = bool(projection_cfg.get("enabled", False))
    if projection_enabled and int(cfg["optimizer"]["grad_accum_steps"]) != 1:
        raise ValueError("selective gradient projection requires grad_accum_steps=1")

    model.train()
    step = resume_steps
    while step < cfg["optimizer"]["max_steps"]:
        for batch in dataloader:
            if step >= cfg["optimizer"]["max_steps"]:
                break
            step_bucket_rewards: dict[str, list[float]] = defaultdict(list)
            step_bucket_failures: dict[str, int] = defaultdict(int)
            step_bucket_counts: dict[str, int] = defaultdict(int)
            step_quality_gate_active = 0
            step_quality_gate_total = 0
            step_anchor_improved = 0
            step_anchor_total = 0
            step_selective_positive_groups = 0
            step_selective_positive_active_groups = 0
            step_selective_negative_groups = 0
            step_selective_negative_active_groups = 0
            step_triage_counts: dict[str, int] = defaultdict(int)
            step_triage_q_values: list[float] = []
            projection_stats: dict[str, float | bool] = {
                "active": False,
                "dot": 0.0,
                "cosine": 0.0,
                "coefficient": 0.0,
            }
            with accelerator.accumulate(model):
                total_loss = torch.zeros((), device=accelerator.device)
                objective_loss = torch.zeros((), device=accelerator.device)
                constraint_loss = torch.zeros((), device=accelerator.device)
                objective_count = 0
                constraint_count = 0
                batch_inputs, answer_mask = build_supervised_batch_inputs(processor, batch)
                batch_inputs = move_inputs_to_device(batch_inputs, accelerator.device)
                answer_mask = answer_mask.to(accelerator.device)
                outputs = model(**batch_inputs, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                labels = batch_inputs["input_ids"][:, 1:]
                token_mask = answer_mask[:, 1:] & batch_inputs["attention_mask"][:, 1:].bool()
                ce_losses = compute_answer_cross_entropy_per_sample(logits, labels, token_mask)
                null_violation, null_loss_value, null_mask = outcome_constraint(
                    ce_losses,
                    batch,
                    budget=null_budget,
                    epsilon=null_epsilon,
                )
                task_to_indices: dict[str, list[int]] = defaultdict(list)
                for sample_idx_in_batch, sample in enumerate(batch):
                    task_to_indices[str(sample["task"])].append(sample_idx_in_batch)

                task_count_tensor = torch.tensor(
                    [
                        len(task_to_indices.get("refseg", [])),
                        len(task_to_indices.get("maskcap", [])),
                        len(task_to_indices.get("existence", [])),
                    ],
                    dtype=torch.int64,
                    device=accelerator.device,
                )
                if accelerator.num_processes > 1:
                    gathered_task_counts = accelerator.gather(task_count_tensor).view(accelerator.num_processes, -1)
                    if not torch.equal(gathered_task_counts, gathered_task_counts[:1].expand_as(gathered_task_counts)):
                        raise RuntimeError(
                            f"Per-rank task counts diverged: {gathered_task_counts.detach().cpu().tolist()}"
                        )

                for task in ("refseg", "maskcap", "existence"):
                    batch_indices = task_to_indices.get(task, [])
                    if not batch_indices:
                        continue
                    task_samples = [batch[idx] for idx in batch_indices]
                    group_size = int(cfg["rl"]["group_size"][task])

                    prompt_inputs = build_prompt_batch_inputs(processor, task_samples, image_key="image")
                    prompt_inputs = move_inputs_to_device(prompt_inputs, accelerator.device)

                    sample_ids_batch, sample_texts = generate_answers(
                        accelerator.unwrap_model(model),
                        processor,
                        prompt_inputs,
                        cfg["generation"][task],
                        num_return_sequences=group_size,
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
                        token_scope=build_answer_token_scope(sample_ids_batch, processor, task),
                    )

                    per_sample_rewards: list[list[float]] = []
                    per_sample_anchor_rewards: list[float] = []
                    per_sample_fail_masks: list[list[float]] = []
                    per_sample_privileged_texts: list[str | None] = []
                    for local_idx, global_idx in enumerate(batch_indices):
                        sample = batch[global_idx]
                        routing_mode = str((cfg.get("routing", {}) or {}).get("mode", "source_bucket"))
                        bucket = route_bucket_for_sample(cfg, sample)
                        start = local_idx * group_size
                        end = start + group_size
                        if routing_mode in {"predicted_only_evidence", "predicted_evidence"}:
                            fepo_cfg = (cfg.get("routing", {}) or {}).get("predicted_only_evidence", {})
                            if not isinstance(fepo_cfg, dict):
                                fepo_cfg = {}
                            probe_index = int(fepo_cfg.get("probe_index", 0))
                            probe_index = max(0, min(probe_index, max(group_size - 1, 0)))
                            update_rollout_predicted_route(
                                cfg,
                                sample,
                                sample_texts[start + probe_index],
                                answer_confidence=float(
                                    rollout_answer_confidence(
                                        student_logits_batch[start + probe_index : start + probe_index + 1],
                                        answer_attention[start + probe_index : start + probe_index + 1],
                                    )[0].item()
                                ),
                            )
                            bucket = route_bucket_for_sample(cfg, sample)
                        rewards: list[float] = []
                        fail_mask: list[float] = []
                        for sample_offset, sample_text in enumerate(sample_texts[start:end]):
                            reward_value, reward_details = routed_reward(
                                cfg,
                                sample,
                                sample_text,
                                dataloader.dataset,
                                relation_confusers,
                                similarity_scorer,
                            )
                            if reward_rankers:
                                task_name = str(sample["task"])
                                component_values = {
                                    name: float(reward_details.get(name, reward_details.get(f"{name}_total", reward_details.get("total", reward_value))))
                                    for name in reward_rankers[task_name].components
                                }
                                reward_value = reward_rankers[task_name].score(component_values)
                            rewards.append(float(reward_value))
                            step_bucket_rewards[bucket].append(float(reward_value))
                            threshold = failure_threshold(cfg, bucket, sample["task"])
                            fail = reward_value < threshold
                            if fail:
                                step_bucket_failures[bucket] += 1
                            step_bucket_counts[bucket] += 1
                            fail_mask.append(1.0 if fail else 0.0)
                        per_sample_rewards.append(rewards)
                        per_sample_anchor_rewards.append(float("nan"))
                        if sample.get("task") == "refseg" and bool(
                            (sample.get("meta") or {}).get("no_target", False)
                        ):
                            step_selective_negative_groups += 1
                            if rewards and max(rewards) - min(rewards) > 1e-8:
                                step_selective_negative_active_groups += 1
                        elif sample.get("task") == "refseg":
                            step_selective_positive_groups += 1
                            if rewards and max(rewards) - min(rewards) > 1e-8:
                                step_selective_positive_active_groups += 1
                        per_sample_fail_masks.append(fail_mask)
                        per_sample_privileged_texts.append(str(sample["answer_text"]))

                    anchor_relative_enabled = (
                        str(cfg.get("rl", {}).get("advantage_mode", "group_normalized"))
                        == "anchor_relative"
                    )
                    if anchor_relative_enabled:
                        anchor_generation_cfg = dict(cfg["generation"][task])
                        anchor_generation_cfg["do_sample"] = False
                        with torch.no_grad():
                            _, anchor_texts = generate_answers(
                                accelerator.unwrap_model(reference_model),
                                processor,
                                prompt_inputs,
                                anchor_generation_cfg,
                                num_return_sequences=1,
                            )
                        for local_idx, global_idx in enumerate(batch_indices):
                            sample = batch[global_idx]
                            anchor_reward, _ = routed_reward(
                                cfg,
                                sample,
                                anchor_texts[local_idx],
                                dataloader.dataset,
                                relation_confusers,
                                similarity_scorer,
                            )
                            per_sample_anchor_rewards[local_idx] = float(anchor_reward)

                    triage_enabled = bool((cfg.get("triage", {}) or {}).get("enabled", False))
                    teacher_mode = str((cfg.get("opd", {}) or {}).get("teacher_mode", "frozen_overlay"))
                    teacher_image_key = str((cfg.get("opd", {}) or {}).get("teacher_image_key", "overlay_image"))
                    need_teacher_branch = triage_enabled or float(cfg["loss"].get("lambda_opd", 0.0)) > 0.0
                    teacher_logits_batch = None
                    if need_teacher_branch:
                        if teacher_mode == "self_privileged_rollout":
                            teacher_logits_batch, _ = self_teacher_logits_batch(
                                accelerator.unwrap_model(model),
                                processor,
                                task_samples,
                                per_sample_privileged_texts,
                                teacher_image_key,
                                sample_ids_batch,
                                accelerator.device,
                                pad_token_id,
                            )
                        else:
                            overlay_prompt_inputs = build_prompt_batch_inputs(processor, task_samples, image_key="overlay_image")
                            overlay_prompt_inputs = move_inputs_to_device(overlay_prompt_inputs, accelerator.device)
                            with torch.no_grad():
                                teacher_logits_batch, _ = forward_answer_logits_batch(
                                    teacher_model,
                                    overlay_prompt_inputs,
                                    sample_ids_batch,
                                    pad_token_id,
                                )

                    if triage_enabled:
                        reward_tensor = torch.tensor(
                            [sum(rewards) / max(len(rewards), 1) for rewards in per_sample_rewards],
                            dtype=torch.float32,
                            device=accelerator.device,
                        )
                        sample_nlls: list[torch.Tensor] = []
                        sample_entropies: list[torch.Tensor] = []
                        for local_idx, _global_idx in enumerate(batch_indices):
                            start = local_idx * group_size
                            end = start + group_size
                            lengths = answer_attention[start:end].sum(dim=-1).clamp_min(1).to(logprob_sums.dtype)
                            seq_nll = (-logprob_sums[start:end] / lengths).mean()
                            assert teacher_logits_batch is not None
                            seq_entropy = compute_answer_entropy_scores(
                                teacher_logits_batch[start:end],
                                answer_attention[start:end],
                            ).mean()
                            sample_nlls.append(seq_nll)
                            sample_entropies.append(seq_entropy)
                        triage_labels, triage_q_values = triage_labels_from_stats(
                            cfg,
                            reward_tensor,
                            torch.stack(sample_nlls),
                            torch.stack(sample_entropies),
                        )
                        step_triage_q_values.extend(float(x.item()) for x in triage_q_values)
                        gated_masks: list[list[float]] = []
                        for triage_label, fail_mask in zip(triage_labels, per_sample_fail_masks):
                            step_triage_counts[triage_label] += 1
                            if triage_label == "suspicious":
                                gated_masks.append([1.0] * len(fail_mask))
                            else:
                                gated_masks.append([0.0] * len(fail_mask))
                        per_sample_fail_masks = gated_masks

                    need_opd_local = (
                        cfg["loss"].get("lambda_opd", 0.0) > 0.0
                        and any(any(mask_value > 0.0 for mask_value in fail_mask) for fail_mask in per_sample_fail_masks)
                    )
                    if need_opd_local:
                        teacher_weights = compute_teacher_confidence_weights_batch(teacher_logits_batch)
                        opd_values_all = compute_jsd_values(
                            student_logits_batch,
                            teacher_logits_batch,
                            teacher_weights,
                            answer_attention,
                        )
                    else:
                        opd_values_all = torch.zeros(sample_ids_batch.shape[0], dtype=torch.float32, device=accelerator.device)

                    with torch.no_grad():
                        ref_logits_batch, _ = forward_answer_logits_batch(
                            reference_model,
                            prompt_inputs,
                            sample_ids_batch,
                            pad_token_id,
                        )
                    kl_values_all = compute_reference_kl_values(
                        student_logits_batch,
                        ref_logits_batch,
                        answer_attention,
                    )

                    for local_idx, global_idx in enumerate(batch_indices):
                        sample = batch[global_idx]
                        bucket = route_bucket_for_sample(cfg, sample)
                        route_weights = route_weights_for_sample(cfg, sample)
                        routing_cfg = cfg.get("routing", {})
                        bucket_cfg = routing_cfg.get("buckets", {}).get(bucket, {})
                        use_soft_local = (
                            isinstance(routing_cfg, dict)
                            and routing_cfg.get("credit_assignment") == "soft_local"
                            and str(routing_cfg.get("mode", ""))
                            in {"predicted_only_evidence", "predicted_evidence"}
                        )
                        ce_loss = ce_losses[global_idx]
                        if use_soft_local:
                            ce_scale = soft_local_scale(
                                routing_cfg,
                                route_weights,
                                "ce_scale",
                                fallback_bucket=bucket,
                            )
                        else:
                            ce_scale = bucket_scale_with_route_weight(bucket_cfg, route_weights, bucket, "ce_scale")
                        if sample["task"] == "maskcap":
                            ce_loss = ce_loss * float(cfg["loss"].get("lambda_cap_ce", 1.0))

                        start = local_idx * group_size
                        end = start + group_size
                        advantage_mode = str(cfg.get("rl", {}).get("advantage_mode", "group_normalized"))
                        if advantage_mode == "quality_gated":
                            threshold_cfg = cfg.get("rl", {}).get("quality_threshold", {})
                            if isinstance(threshold_cfg, dict):
                                quality_threshold = float(threshold_cfg.get(str(sample["task"]), 0.5))
                            else:
                                quality_threshold = float(threshold_cfg)
                            advantages = quality_gated_advantages(
                                per_sample_rewards[local_idx],
                                threshold=quality_threshold,
                                temperature=float(cfg.get("rl", {}).get("gate_temperature", 0.05)),
                            ).to(accelerator.device)
                            step_quality_gate_total += len(per_sample_rewards[local_idx])
                            step_quality_gate_active += sum(
                                float(value) >= quality_threshold
                                for value in per_sample_rewards[local_idx]
                            )
                        elif advantage_mode == "group_normalized":
                            advantages = normalize_rewards(per_sample_rewards[local_idx]).to(accelerator.device)
                        elif advantage_mode == "anchor_relative":
                            no_target = sample.get("task") == "refseg" and bool(
                                (sample.get("meta") or {}).get("no_target", False)
                            )
                            if no_target:
                                advantages = torch.zeros(
                                    group_size,
                                    dtype=torch.float32,
                                    device=accelerator.device,
                                )
                            else:
                                advantages = anchor_relative_advantages(
                                    per_sample_rewards[local_idx],
                                    per_sample_anchor_rewards[local_idx],
                                ).to(accelerator.device)
                            step_anchor_improved += int((advantages > 0).sum().item())
                            step_anchor_total += int(advantages.numel())
                        else:
                            raise ValueError(f"unknown rl.advantage_mode: {advantage_mode}")
                        logprob_slice = logprob_sums[start:end]
                        opd_values = opd_values_all[start:end]
                        kl_values = kl_values_all[start:end]
                        opd_mask = torch.tensor(
                            per_sample_fail_masks[local_idx],
                            dtype=torch.float32,
                            device=accelerator.device,
                        )
                        rl_loss = -(advantages * logprob_slice).mean()
                        opd_loss = (opd_values * opd_mask).sum() / opd_mask.sum().clamp_min(1.0)
                        kl_loss = kl_values.mean()

                        if use_soft_local:
                            rl_scale = soft_local_scale(
                                routing_cfg,
                                route_weights,
                                "rl_scale",
                                fallback_bucket=bucket,
                            )
                            opd_scale = soft_local_scale(
                                routing_cfg,
                                route_weights,
                                "opd_scale",
                                fallback_bucket=bucket,
                            )
                        else:
                            rl_scale = bucket_scale_with_route_weight(bucket_cfg, route_weights, bucket, "rl_scale")
                            opd_scale = bucket_scale_with_route_weight(bucket_cfg, route_weights, bucket, "opd_scale")
                        if sample["task"] == "refseg":
                            base_rl_lambda = float(cfg["loss"]["lambda_rl_seg"])
                        else:
                            base_rl_lambda = float(cfg["loss"]["lambda_rl_cap"])
                        outcome_scales = selective_outcome_loss_scales(cfg, sample)
                        loss = (
                            float(cfg["loss"]["lambda_ce"])
                            * outcome_scales["ce"]
                            * ce_scale
                            * ce_loss
                            + base_rl_lambda
                            * outcome_scales["rl"]
                            * rl_scale
                            * rl_loss
                            + float(cfg["loss"].get("lambda_opd", 0.0))
                            * outcome_scales["opd"]
                            * opd_scale
                            * opd_loss
                            + float(cfg["loss"]["beta_kl"])
                            * outcome_scales["kl"]
                            * kl_loss
                        )
                        total_loss = total_loss + loss
                        # V7 keeps the positive policy objective separate from
                        # explicit no-target supervision.  This lets us
                        # project only the geometry/policy update when it
                        # would increase the negative-outcome loss.
                        if (
                            projection_enabled
                            and sample["task"] == "refseg"
                            and bool((sample.get("meta") or {}).get("no_target", False))
                        ):
                            constraint_loss = constraint_loss + loss
                            constraint_count += 1
                        else:
                            objective_loss = objective_loss + loss
                            objective_count += 1

                if risk_enabled and bool(null_mask.any()):
                    # Add one outcome-specific constraint per optimizer batch.
                    total_loss = total_loss + null_dual * null_violation
                    violation_value = float(null_violation.detach().cpu())
                    null_dual = update_dual(
                        null_dual,
                        violation_value,
                        learning_rate=null_dual_lr,
                        maximum=null_dual_max,
                    )

                if projection_enabled:
                    # Projection is intentionally restricted to one optimizer
                    # step per batch (validated above).  Each branch captures
                    # local gradients without DDP synchronization, then uses
                    # explicit reductions before projection.
                    outcome_counts = torch.tensor(
                        [objective_count, constraint_count],
                        dtype=torch.float32,
                        device=accelerator.device,
                    )
                    outcome_counts = accelerator.reduce(outcome_counts, reduction="sum")
                    global_objective_count = int(outcome_counts[0].item())
                    global_constraint_count = int(outcome_counts[1].item())
                    # Every rank must participate in both DDP backward passes.
                    # A rank without one outcome contributes a differentiable
                    # zero; the all-reduce then yields each global gradient.
                    if objective_count == 0:
                        objective_loss = total_loss * 0.0
                    if constraint_count == 0:
                        constraint_loss = total_loss * 0.0
                    objective_loss = objective_loss / max(global_objective_count, 1)
                    constraint_loss = constraint_loss / max(global_constraint_count, 1)
                    total_loss = objective_loss + constraint_loss
                    params = [p for p in model.parameters() if p.requires_grad]

                    # Always take both no-sync backward passes, including when
                    # one outcome is absent globally.  Falling back to ordinary
                    # DDP backward in that case averages an already globally
                    # normalized local loss and shrinks it by world size.
                    optimizer.zero_grad(set_to_none=True)
                    with accelerator.no_sync(model):
                        accelerator.backward(objective_loss, retain_graph=True)
                    objective_grads = [
                        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                        for p in params
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    with accelerator.no_sync(model):
                        accelerator.backward(constraint_loss)
                    constraint_grads = [
                        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                        for p in params
                    ]
                    if accelerator.num_processes > 1:
                        objective_grads = [
                            accelerator.reduce(grad, reduction="sum") for grad in objective_grads
                        ]
                        constraint_grads = [
                            accelerator.reduce(grad, reduction="sum") for grad in constraint_grads
                        ]
                    if global_objective_count and global_constraint_count:
                        projected_grads, projection_stats = project_conflicting_gradient(
                            objective_grads,
                            constraint_grads,
                            epsilon=float(projection_cfg.get("epsilon", 1e-12)),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        for parameter, projected, constraint in zip(
                            params, projected_grads, constraint_grads
                        ):
                            parameter.grad = projected + constraint
                    else:
                        # With only one global outcome, there is nothing to
                        # project; retain the explicitly reduced available
                        # gradient and keep the absent side at zero.
                        for parameter, objective, constraint in zip(
                            params, objective_grads, constraint_grads
                        ):
                            parameter.grad = objective + constraint
                else:
                    total_loss = total_loss / max(len(batch), 1)
                    accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss_value = float(total_loss.detach().cpu())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if reserve_target_gb > 0.0:
                    ensure_gpu_reserve_buffers(
                        reserve_buffers,
                        target_gb=reserve_target_gb,
                        headroom_gb=reserve_headroom_gb,
                        device=accelerator.device,
                    )

            if str(cfg.get("rl", {}).get("advantage_mode", "group_normalized")) == "anchor_relative":
                anchor_counts = accelerator.reduce(
                    torch.tensor(
                        [step_anchor_improved, step_anchor_total],
                        dtype=torch.float32,
                        device=accelerator.device,
                    ),
                    reduction="sum",
                )
                step_anchor_improved = int(anchor_counts[0].item())
                step_anchor_total = int(anchor_counts[1].item())

            if accelerator.is_main_process:
                step_metrics = {"step": step, "loss": step_loss_value}
                if risk_enabled:
                    negative_kl = kl_values_all[null_mask.to(kl_values_all.device)]
                    step_metrics.update(
                        {
                            "null_loss": float(null_loss_value.cpu()),
                            "null_budget": null_budget + null_epsilon,
                            "null_violation": float(null_violation.detach().cpu()),
                            "null_lambda": null_dual,
                            "null_constraint_active": int(bool(null_mask.any())),
                            "null_kl_mean": float(negative_kl.mean().detach().cpu()),
                        }
                    )
                if projection_enabled:
                    step_metrics.update(
                        {
                            "projection_active": int(bool(projection_stats["active"])),
                            "projection_dot": float(projection_stats["dot"]),
                            "projection_cosine": float(projection_stats["cosine"]),
                            "projection_coefficient": float(projection_stats["coefficient"]),
                            "projection_objective_groups": int(global_objective_count),
                            "projection_constraint_groups": int(global_constraint_count),
                        }
                    )
                for bucket in sorted(step_bucket_counts):
                    count = max(step_bucket_counts[bucket], 1)
                    step_metrics[f"{bucket}_mean_reward"] = sum(step_bucket_rewards[bucket]) / max(
                        len(step_bucket_rewards[bucket]), 1
                    )
                    step_metrics[f"{bucket}_failure_rate"] = step_bucket_failures[bucket] / count
                for triage_label in ("clean", "suspicious", "corrupted"):
                    if triage_label in step_triage_counts:
                        step_metrics[f"triage_{triage_label}_count"] = step_triage_counts[triage_label]
                if step_triage_q_values:
                    step_metrics["triage_mean_q"] = sum(step_triage_q_values) / len(step_triage_q_values)
                if str(cfg.get("rl", {}).get("advantage_mode", "group_normalized")) == "quality_gated":
                    step_metrics["quality_gate_active_rate"] = step_quality_gate_active / max(step_quality_gate_total, 1)
                if str(cfg.get("rl", {}).get("advantage_mode", "group_normalized")) == "anchor_relative":
                    step_metrics["anchor_improved_rollout_rate"] = (
                        step_anchor_improved / max(step_anchor_total, 1)
                    )
                if step_selective_negative_groups:
                    step_metrics["selective_negative_active_group_rate"] = (
                        step_selective_negative_active_groups / step_selective_negative_groups
                    )
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
