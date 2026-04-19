from __future__ import annotations

import json
import importlib.metadata
import math
import os
import random
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file as safetensors_save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from projects.pixvl_idea1.datasets import (
    HomogeneousTaskBatchSampler,
    UnifiedRegionDataset,
    identity_collate,
)


def load_config(path: str) -> dict[str, Any]:
    config = runpy.run_path(path).get("config")
    if config is None:
        raise ValueError(f"{path} 没有暴露 config 字典")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def resolve_attention_backend(preferred: str) -> str:
    if preferred != "flash_attention_2":
        return preferred
    try:
        importlib.metadata.version("flash_attn")
        from flash_attn import flash_attn_func  # noqa: F401
        return preferred
    except Exception:
        return "sdpa"


def discover_lora_target_modules(model: torch.nn.Module) -> list[str]:
    modules = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and "visual" not in name
            and "lm_head" not in name
            and "embed_tokens" not in name
        ):
            modules.add(name.split(".")[-1])
    return sorted(modules)


def build_model_bundle(
    cfg: dict[str, Any],
    trainable: bool,
    adapter_path: str | None = None,
):
    model_cfg = cfg["model"]
    base_model = model_cfg["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(
        model_cfg["processor_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    attn_backend = resolve_attention_backend(model_cfg.get("attn_implementation", "flash_attention_2"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        dtype=_resolve_dtype(),
        attn_implementation=attn_backend,
    )

    if trainable and cfg.get("memory_optim", {}).get("gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable and cfg["lora"]["enabled"]:
        target_modules = discover_lora_target_modules(model)
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=cfg["lora"].get("modules_to_save") or None,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
    else:
        model.requires_grad_(False)

    if torch.cuda.is_available():
        target_dtype = _resolve_dtype()
        model.to(dtype=target_dtype)

    return model, processor


def build_dataloader(
    cfg: dict[str, Any],
    resume_steps: int = 0,
    sampler_state: dict[str, Any] | None = None,
) -> tuple[DataLoader, HomogeneousTaskBatchSampler]:
    codec_device = "cpu"
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        codec_device = f"cuda:{local_rank}"
    dataset = UnifiedRegionDataset(
        schema_files=cfg["data"]["schema_files"],
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        cache_path=cfg["paths"]["mask_code_cache"],
        task_mix=cfg["data"]["task_mix"],
        bucket_mix=cfg["data"].get("bucket_mix"),
        source_mix=cfg["data"].get("source_mix"),
        prompt_templates=cfg["data"]["prompts"],
        overlay_cfg=cfg["data"]["overlay"],
        visual_token_filter=cfg["data"].get("visual_token_filter"),
        codec_device=codec_device,
    )
    batch_sampler = HomogeneousTaskBatchSampler(
        dataset,
        cfg["data"]["batch_size"],
        seed=int(cfg.get("seed", 0)),
    )
    if sampler_state is not None:
        batch_sampler.load_state_dict(sampler_state)
    elif resume_steps > 0:
        batch_sampler.set_start_step(resume_steps)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=identity_collate,
    )
    return dataloader, batch_sampler


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict[str, Any]):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg["optimizer"]["lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    total_steps = cfg["optimizer"]["max_steps"]
    warmup_steps = int(total_steps * cfg["optimizer"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def render_chat_text(processor: Any, prompt_text: str, answer_text: str | None, add_generation_prompt: bool) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if answer_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def encode_chat(processor: Any, image: Any, prompt_text: str, answer_text: str | None, add_generation_prompt: bool) -> dict[str, torch.Tensor]:
    text = render_chat_text(processor, prompt_text, answer_text, add_generation_prompt=add_generation_prompt)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def build_prompt_and_answer_ids(
    processor: Any,
    image: Any,
    prompt_text: str,
    answer_text: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt_inputs = encode_chat(processor, image, prompt_text, None, add_generation_prompt=True)
    full_inputs = encode_chat(processor, image, prompt_text, answer_text, add_generation_prompt=False)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    answer_ids = full_inputs["input_ids"][0, prompt_len:].clone()
    return prompt_inputs, answer_ids


def move_inputs_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def forward_answer_logits(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    prompt_len = prompt_inputs["input_ids"].shape[1]
    answer_ids = answer_ids.unsqueeze(0)
    input_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in prompt_inputs:
            model_inputs[key] = prompt_inputs[key]
    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits[:, :-1]
    answer_logits = logits[:, prompt_len - 1 : prompt_len - 1 + answer_ids.shape[1], :]
    return answer_logits


def _repeat_prompt_inputs(prompt_inputs: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    repeated: dict[str, torch.Tensor] = {}
    for key, value in prompt_inputs.items():
        if value.shape[0] == repeats:
            repeated[key] = value
        elif value.shape[0] == 1:
            repeat_dims = [repeats] + [1] * (value.dim() - 1)
            repeated[key] = value.repeat(*repeat_dims)
        elif key == "pixel_values" and prompt_inputs.get("image_grid_thw") is not None and prompt_inputs["image_grid_thw"].shape[0] == 1:
            repeat_dims = [repeats] + [1] * (value.dim() - 1)
            repeated[key] = value.repeat(*repeat_dims)
        elif key == "pixel_values_videos" and prompt_inputs.get("video_grid_thw") is not None and prompt_inputs["video_grid_thw"].shape[0] == 1:
            repeat_dims = [repeats] + [1] * (value.dim() - 1)
            repeated[key] = value.repeat(*repeat_dims)
        else:
            raise ValueError(f"Cannot repeat prompt input {key} with batch {value.shape[0]} to {repeats}")
    return repeated


def forward_answer_logits_batch(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    answer_ids: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if answer_ids.dim() == 1:
        answer_ids = answer_ids.unsqueeze(0)
    batch_size = answer_ids.shape[0]
    repeated_prompt_inputs = _repeat_prompt_inputs(prompt_inputs, batch_size)
    prompt_len = repeated_prompt_inputs["input_ids"].shape[1]
    prompt_attention = repeated_prompt_inputs.get("attention_mask")
    if prompt_attention is None:
        prompt_attention = torch.ones_like(repeated_prompt_inputs["input_ids"], dtype=torch.long)
    answer_attention = (answer_ids != pad_token_id).long()
    input_ids = torch.cat([repeated_prompt_inputs["input_ids"], answer_ids], dim=1)
    attention_mask = torch.cat([prompt_attention, answer_attention], dim=1)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in repeated_prompt_inputs:
            model_inputs[key] = repeated_prompt_inputs[key]
    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits[:, :-1]
    answer_logits = logits[:, prompt_len - 1 : prompt_len - 1 + answer_ids.shape[1], :]
    return answer_logits, answer_attention


def compute_answer_cross_entropy(answer_logits: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(answer_logits, dim=-1)
    nll = -log_probs.gather(-1, answer_ids.view(1, -1, 1)).squeeze(-1)
    return nll.mean()


def compute_answer_logprob_sum(answer_logits: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, answer_ids.view(1, -1, 1)).squeeze(-1)
    return token_log_probs.sum(dim=-1).mean()


def compute_answer_logprob_sums(
    answer_logits: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_attention: torch.Tensor | None = None,
) -> torch.Tensor:
    if answer_ids.dim() == 1:
        answer_ids = answer_ids.unsqueeze(0)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    if answer_attention is not None:
        token_log_probs = token_log_probs * answer_attention.to(token_log_probs.dtype)
    return token_log_probs.sum(dim=-1)


def generate_answer(
    model: torch.nn.Module,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generation_cfg: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    model_inputs = {key: value for key, value in prompt_inputs.items()}
    generated = model.generate(
        **model_inputs,
        use_cache=False,
        max_new_tokens=generation_cfg["max_new_tokens"],
        do_sample=generation_cfg["do_sample"],
        temperature=generation_cfg["temperature"],
        top_p=generation_cfg["top_p"],
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    answer_ids = generated[0, prompt_len:].detach().cpu()
    text = processor.tokenizer.decode(
        answer_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return answer_ids, text


def generate_answers(
    model: torch.nn.Module,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generation_cfg: dict[str, Any],
    num_return_sequences: int,
) -> tuple[torch.Tensor, list[str]]:
    model_inputs = {key: value for key, value in prompt_inputs.items()}
    generated = model.generate(
        **model_inputs,
        use_cache=False,
        max_new_tokens=generation_cfg["max_new_tokens"],
        do_sample=generation_cfg["do_sample"],
        temperature=generation_cfg["temperature"],
        top_p=generation_cfg["top_p"],
        num_return_sequences=num_return_sequences,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    answer_ids = generated[:, prompt_len:].detach()
    texts = processor.tokenizer.batch_decode(
        answer_ids.detach().cpu(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return answer_ids, texts


def clean_generated_text(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "").replace("<|end|>", "")
    return " ".join(cleaned.strip().split())


def compute_jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    mean_probs = 0.5 * (student_probs + teacher_probs)
    mean_log_probs = torch.log(mean_probs.clamp_min(1e-8))
    kl_s = F.kl_div(student_log_probs, mean_probs, reduction="none").sum(dim=-1)
    kl_t = F.kl_div(teacher_log_probs, mean_probs, reduction="none").sum(dim=-1)
    jsd = 0.5 * (kl_s + kl_t)
    if weights is not None:
        jsd = jsd * weights.view(1, -1)
    return jsd.mean()


def compute_jsd_values(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    mean_probs = 0.5 * (student_probs + teacher_probs)
    kl_s = F.kl_div(student_log_probs, mean_probs, reduction="none").sum(dim=-1)
    kl_t = F.kl_div(teacher_log_probs, mean_probs, reduction="none").sum(dim=-1)
    jsd = 0.5 * (kl_s + kl_t)
    if weights is not None:
        jsd = jsd * weights.to(jsd.dtype)
    if token_mask is None:
        return jsd.mean(dim=-1)
    token_mask = token_mask.to(jsd.dtype)
    return (jsd * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)


def compute_teacher_confidence_weights(teacher_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(teacher_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    vocab_size = teacher_logits.shape[-1]
    return (1.0 - entropy / math.log(vocab_size)).clamp(0.0, 1.0)[0]


def compute_teacher_confidence_weights_batch(teacher_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(teacher_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    vocab_size = teacher_logits.shape[-1]
    return (1.0 - entropy / math.log(vocab_size)).clamp(0.0, 1.0)


def compute_reference_kl(student_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    reference_probs = F.softmax(reference_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, reference_probs, reduction="batchmean")


def compute_reference_kl_values(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    reference_probs = F.softmax(reference_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    kl = F.kl_div(student_log_probs, reference_probs, reduction="none").sum(dim=-1)
    if token_mask is None:
        return kl.mean(dim=-1)
    token_mask = token_mask.to(kl.dtype)
    return (kl * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)


def normalize_rewards(rewards: list[float]) -> torch.Tensor:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    if rewards_tensor.numel() == 1:
        return torch.zeros_like(rewards_tensor)
    return (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-4)


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def checkpoint_step_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def find_latest_adapter_checkpoint(output_dir: str | Path) -> tuple[Path | None, int]:
    root = Path(output_dir)
    candidates = sorted(root.glob("checkpoint-step-*"), key=checkpoint_step_from_dir)
    latest_dir: Path | None = None
    latest_step = -1
    for candidate in candidates:
        run_state_path = candidate / "run_state.json"
        adapter_path = candidate / "adapter" / "adapter_model.safetensors"
        if not run_state_path.exists() or not adapter_path.exists():
            continue
        step = checkpoint_step_from_dir(candidate)
        if step < 0:
            try:
                payload = json.loads(run_state_path.read_text())
                step = int(payload.get("steps", -1))
            except Exception:
                step = -1
        if step > latest_step:
            latest_dir = candidate
            latest_step = step
    return latest_dir, latest_step


def find_latest_state_checkpoint(output_dir: str | Path) -> tuple[Path | None, int]:
    root = Path(output_dir)
    candidates = sorted(root.glob("state-step-*"), key=checkpoint_step_from_dir)
    latest_dir: Path | None = None
    latest_step = -1
    for candidate in candidates:
        if not any(candidate.rglob(".metadata")):
            continue
        step = checkpoint_step_from_dir(candidate)
        if step > latest_step:
            latest_dir = candidate
            latest_step = step
    return latest_dir, latest_step


def save_sampler_state(output_dir: str | Path, state: dict[str, Any]) -> None:
    path = Path(output_dir) / "sampler_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def load_sampler_state(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "sampler_state.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _active_adapter_name(model: PeftModel) -> str:
    active = getattr(model, "active_adapter", None)
    if isinstance(active, str) and active:
        return active
    active_list = getattr(model, "active_adapters", None)
    if callable(active_list):
        names = active_list()
        if names:
            return names[0]
    return "default"


def _extract_modules_to_save_tensors(
    adapter_name: str,
    modules_to_save: list[str],
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    mapping = {
        "lm_head": (
            "base_model.model.lm_head.weight",
            [
                f"base_model.model.lm_head.modules_to_save.{adapter_name}.weight",
                "base_model.model.lm_head.weight",
                f"lm_head.modules_to_save.{adapter_name}.weight",
                "lm_head.weight",
            ],
        ),
        "embed_tokens": (
            "base_model.model.model.language_model.embed_tokens.weight",
            [
                f"base_model.model.model.language_model.embed_tokens.modules_to_save.{adapter_name}.weight",
                "base_model.model.model.language_model.embed_tokens.weight",
                f"model.model.language_model.embed_tokens.modules_to_save.{adapter_name}.weight",
                "model.model.language_model.embed_tokens.weight",
            ],
        ),
    }
    for module_name in modules_to_save:
        if module_name not in mapping:
            continue
        dest_key, source_candidates = mapping[module_name]
        for source_key in source_candidates:
            value = state_dict.get(source_key)
            if value is not None:
                output[dest_key] = value.detach().cpu().contiguous()
                break
    return output


def extract_adapter_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor] | None:
    if not isinstance(model, PeftModel):
        return state_dict
    adapter_name = _active_adapter_name(model)
    peft_config = model.peft_config[adapter_name]
    modules_to_save = list(peft_config.modules_to_save or [])
    source_state = state_dict or model.state_dict()
    adapter_state: dict[str, torch.Tensor] = {}

    adapter_token = f".{adapter_name}."
    modules_to_save_token = f".modules_to_save.{adapter_name}."

    for key, value in source_state.items():
        normalized_key = None
        if ".lora_" in key:
            if adapter_token in key:
                normalized_key = key.replace(adapter_token, ".")
            elif ".lora_A.weight" in key or ".lora_B.weight" in key or ".lora_embedding_" in key or ".lora_magnitude_vector" in key:
                normalized_key = key
        elif modules_to_save_token in key:
            normalized_key = key.replace(modules_to_save_token, ".")

        if normalized_key is not None:
            adapter_state[normalized_key] = value.detach().cpu().contiguous()

    if modules_to_save:
        for key, value in _extract_modules_to_save_tensors(adapter_name, modules_to_save, source_state).items():
            adapter_state.setdefault(key, value)

    return adapter_state


def save_adapter_checkpoint(
    model: torch.nn.Module,
    processor: Any,
    output_dir: str,
    extra_state: dict[str, Any],
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    output = ensure_output_dir(output_dir)
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if state_dict is not None and isinstance(model, PeftModel):
        adapter_name = _active_adapter_name(model)
        model.peft_config[adapter_name].save_pretrained(str(adapter_dir))
        safetensors_save_file(state_dict, str(adapter_dir / "adapter_model.safetensors"))
    else:
        save_kwargs: dict[str, Any] = {}
        if state_dict is not None:
            save_kwargs["state_dict"] = state_dict
        model.save_pretrained(adapter_dir, **save_kwargs)
    processor.save_pretrained(adapter_dir)
    with (output / "run_state.json").open("w", encoding="utf-8") as handle:
        json.dump(extra_state, handle, ensure_ascii=False, indent=2)
