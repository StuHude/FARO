from __future__ import annotations

import argparse
import json
import math
import os
import random
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .ampcpo_contract import validate_ampcpo_config, validate_continued_adapter
from .config import REPO_ROOT
from .data import PairedBatchSampler, SelectiveRefSegDataset, identity_collate
from .geometry_reward import ciou
from .manifests import (
    assert_training_source_clean,
    build_manifest,
    guard_runtime_environment,
    runtime_module_files,
    validate_base_checkpoint,
    validate_declared_paths,
    write_json_atomic,
)
from .mask_codec import SAMTokMaskCodec, decode_rle_mask
from .modeling import (
    assert_only_lora_trainable,
    build_model_and_processor,
    build_supervised_inputs,
    move_tensors,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
CANONICAL_NULL = "No target."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = runpy.run_path(str(Path(path).resolve()))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must expose a config dictionary: {path}")
    validate_ampcpo_config(config, REPO_ROOT)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_sample_answer_nll(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids[:, 1:]
    token_mask = answer_mask[:, 1:] & attention_mask[:, 1:].bool()
    counts = token_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise RuntimeError("Every AM-CPPO sample must contain answer tokens")
    nll = -log_probs[:, :-1, :].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (nll * token_mask.to(nll.dtype)).sum(dim=1) / counts.to(nll.dtype)


def clipped_policy_loss(
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_log_probs.shape != old_action_log_probs.shape or action_log_probs.shape != advantages.shape:
        raise ValueError("AM-CPPO policy tensors must have identical shapes")
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    clip_fraction = (ratio != clipped_ratio).to(torch.float32).mean()
    return -objective.mean(), clip_fraction


def canonical_null_margin_terms(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    no_target: torch.Tensor,
    tokenizer: Any,
    mask_start_token_ids: list[int],
) -> torch.Tensor:
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    margins: list[torch.Tensor] = []
    for row_index in torch.nonzero(no_target, as_tuple=False).flatten().tolist():
        positions = torch.nonzero(answer_mask[row_index], as_tuple=False).flatten().tolist()
        canonical_positions: list[int] = []
        for position in positions:
            token_id = int(input_ids[row_index, position].item())
            if token_id == im_end_id:
                break
            canonical_positions.append(position)
        if not canonical_positions:
            raise RuntimeError("No-target answer has no canonical tokens")
        canonical_ids = [int(input_ids[row_index, position].item()) for position in canonical_positions]
        decoded = tokenizer.decode(canonical_ids, skip_special_tokens=False).strip()
        if decoded != CANONICAL_NULL:
            raise RuntimeError(f"Expected canonical {CANONICAL_NULL!r}, decoded {decoded!r}")
        first_position = canonical_positions[0]
        # The constraint is on the mask-or-null action at the answer boundary.
        # Comparing the full null phrase probability to one mask token would
        # mix sequence lengths and turn the margin into an arbitrary length
        # penalty.  Validate the complete phrase above, but use only its first
        # token for the action decision.
        null_log_prob = log_probs[
            row_index, first_position - 1, input_ids[row_index, first_position]
        ]
        candidate_ids = torch.tensor(
            mask_start_token_ids, dtype=torch.long, device=log_probs.device
        )
        mask_log_prob = torch.logsumexp(
            log_probs[row_index, first_position - 1, candidate_ids], dim=0
        )
        margins.append(null_log_prob - mask_log_prob)
    if not margins:
        raise RuntimeError("Paired AM-CPPO batch has no no-target sample")
    return torch.stack(margins).float()


def _mask_token_ids(tokenizer: Any, codec: SAMTokMaskCodec) -> tuple[int, list[list[int]]]:
    start_id = int(tokenizer.convert_tokens_to_ids(codec.catalog["start"]))
    if start_id < 0 or start_id == tokenizer.unk_token_id:
        raise ValueError("SAMTok mask-start token is unavailable")
    by_depth: list[list[int]] = []
    for depth in range(codec.codebook_depth):
        token_ids = []
        for code in range(depth * codec.codebook_size, (depth + 1) * codec.codebook_size):
            token = codec.catalog["index_to_token"].get(code)
            if token is None:
                raise ValueError(f"SAMTok catalog is missing mask code {code}")
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            if token_id < 0 or token_id == tokenizer.unk_token_id:
                raise ValueError(f"SAMTok mask code is unavailable: {token}")
            token_ids.append(token_id)
        by_depth.append(token_ids)
    return start_id, by_depth


def positive_actions_and_ciou(
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    samples: list[dict[str, Any]],
    codec: SAMTokMaskCodec,
    mask_start_id: int,
    code_token_ids: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    action_log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    for row_index, sample in enumerate(samples):
        if bool(sample["no_target"]):
            continue
        positions = torch.nonzero(answer_mask[row_index], as_tuple=False).flatten()
        if positions.numel() < codec.codebook_depth + 2:
            raise RuntimeError(f"Positive sample has incomplete SAMTok answer: {sample['id']}")
        first_position = int(positions[0].item())
        if int(input_ids[row_index, first_position].item()) != mask_start_id:
            raise RuntimeError(f"Positive sample does not start with mask action: {sample['id']}")
        selected_log_probs = [log_probs[row_index, first_position - 1, mask_start_id]]
        predicted_codes: list[int] = []
        for depth, candidates in enumerate(code_token_ids):
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=log_probs.device)
            prediction_logits = log_probs[row_index, first_position + depth, candidate_tensor]
            selected = int(torch.argmax(prediction_logits).item())
            predicted_codes.append(depth * codec.codebook_size + selected)
            selected_log_probs.append(prediction_logits[selected])
        prediction = codec.decode_codes(sample["image"], predicted_codes)
        target = decode_rle_mask(sample["mask"])
        reward = ciou(prediction, target)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise FloatingPointError(f"Invalid positive cIoU reward for {sample['id']}: {reward}")
        action_log_probs.append(torch.stack(selected_log_probs).mean().float())
        rewards.append(reward)
    if not action_log_probs:
        raise RuntimeError("Paired AM-CPPO batch has no positive sample")
    return torch.stack(action_log_probs), torch.tensor(rewards, device=log_probs.device)


def _build_dataloader(config: dict[str, Any]) -> tuple[DataLoader, SAMTokMaskCodec]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(data_config["num_workers"]),
        collate_fn=identity_collate,
    )
    return dataloader, codec


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
    initialization = validate_continued_adapter(
        config["checkpoint"]["adapter_init"], repo_root=REPO_ROOT, hash_model=False
    )
    seed_everything(int(config["seed"]))

    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["optimizer"]["grad_accum_steps"]),
        mixed_precision=config["runtime"]["mixed_precision"] if torch.cuda.is_available() else "no",
        kwargs_handlers=[ddp],
    )
    accelerator.even_batches = False
    expected_world_size = int(config["runtime"]["expected_world_size"])
    if accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            f"Registered AM-CPPO smoke requires {expected_world_size} processes, "
            f"got {accelerator.num_processes}"
        )

    output_dir = Path(config["checkpoint"]["output_dir"])
    if accelerator.is_main_process:
        initialization = validate_continued_adapter(
            config["checkpoint"]["adapter_init"], repo_root=REPO_ROOT, hash_model=True
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(config, config_path, PACKAGE_ROOT)
        manifest["method"] = config["ampcpo"]
        manifest["initialization"] = initialization
        write_json_atomic(config["provenance"]["manifest_path"], manifest)
    accelerator.wait_for_everyone()

    model, processor = build_model_and_processor(
        config, adapter_path=initialization["path"]
    )
    trainable_summary = assert_only_lora_trainable(model)
    dataloader, codec = _build_dataloader(config)
    mask_start_id, code_token_ids = _mask_token_ids(processor.tokenizer, codec)
    if accelerator.is_main_process:
        manifest_path = Path(config["provenance"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_modules"] = runtime_module_files()
        manifest["mask_action"] = {
            "start_token": codec.catalog["start"],
            "start_token_id": mask_start_id,
            "codebook_size": codec.codebook_size,
            "codebook_depth": codec.codebook_depth,
        }
        write_json_atomic(manifest_path, manifest)
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * float(optimizer_config["warmup_ratio"])),
        num_training_steps=max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    method = config["ampcpo"]
    metrics: dict[str, Any] = {
        "stage": config["stage"],
        "status": "running",
        "method": method,
        "initialization": initialization,
        "trainable": trainable_summary,
        "steps": [],
    }
    if accelerator.is_main_process:
        write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()
    model.train()
    step = 0
    while step < max_steps:
        for samples in dataloader:
            if step >= max_steps:
                break
            outcomes = torch.tensor(
                [bool(sample["no_target"]) for sample in samples], dtype=torch.bool
            )
            if outcomes.numel() == 0 or bool(outcomes.all()) or not bool(outcomes.any()):
                raise RuntimeError("Every AM-CPPO device batch must retain a complete outcome pair")
            with accelerator.accumulate(model):
                inputs, answer_mask = build_supervised_inputs(processor, samples)
                inputs = move_tensors(inputs, accelerator.device)
                answer_mask = answer_mask.to(accelerator.device)
                outcomes = outcomes.to(accelerator.device)
                outputs = model(**inputs, use_cache=False)
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                sample_nll = per_sample_answer_nll(
                    log_probs,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    answer_mask,
                ).float()
                null_ce = sample_nll[outcomes].mean()
                margins = canonical_null_margin_terms(
                    log_probs,
                    inputs["input_ids"],
                    answer_mask,
                    outcomes,
                    processor.tokenizer,
                    [mask_start_id],
                )
                margin_penalty = F.relu(float(method["margin_target"]) - margins).mean()
                action_log_probs, rewards = positive_actions_and_ciou(
                    log_probs,
                    inputs["input_ids"],
                    answer_mask,
                    samples,
                    codec,
                    mask_start_id,
                    code_token_ids,
                )
                policy_loss, clip_fraction = clipped_policy_loss(
                    action_log_probs,
                    action_log_probs.detach(),
                    rewards,
                    float(method["clip_epsilon"]),
                )
                loss = (
                    float(method["policy_weight"]) * policy_loss
                    + float(method["null_ce_weight"]) * null_ce
                    + float(method["margin_weight"]) * margin_penalty
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite AM-CPPO loss at step {step}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, float(optimizer_config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            scalars = torch.stack(
                (
                    loss.detach().float(),
                    policy_loss.detach().float(),
                    null_ce.detach().float(),
                    margin_penalty.detach().float(),
                    margins.detach().mean().float(),
                    rewards.detach().mean().float(),
                    clip_fraction.detach().float(),
                )
            )
            gathered = accelerator.gather(scalars).reshape(-1, scalars.numel()).mean(dim=0)
            values = [float(value.item()) for value in gathered]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite gathered AM-CPPO metric at step {step}")
            if accelerator.is_main_process:
                item = {
                    "step": step,
                    "loss": values[0],
                    "policy_loss": values[1],
                    "null_ce": values[2],
                    "margin_penalty": values[3],
                    "mean_null_action_margin": values[4],
                    "positive_mean_ciou": values[5],
                    "clip_fraction": values[6],
                    "batch_rows_per_rank": len(samples),
                }
                metrics["steps"].append(item)
                write_json_atomic(output_dir / "metrics.json", metrics)
                print(json.dumps(item), flush=True)
            step += 1
        if len(dataloader) == 0:
            raise RuntimeError("AM-CPPO paired dataloader is empty")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        adapter_dir = output_dir / "adapter"
        unwrapped.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(adapter_dir)
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.is_file() or adapter_file.stat().st_size == 0:
            raise RuntimeError("AM-CPPO smoke did not produce a nonempty LoRA adapter")
        metrics["status"] = "finished"
        metrics["steps_completed"] = step
        metrics["summary"] = {
            key: sum(float(item[key]) for item in metrics["steps"]) / len(metrics["steps"])
            for key in (
                "loss",
                "policy_loss",
                "null_ce",
                "margin_penalty",
                "mean_null_action_margin",
                "positive_mean_ciou",
                "clip_fraction",
            )
        }
        write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
