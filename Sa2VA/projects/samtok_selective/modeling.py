from __future__ import annotations

import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import ImageEnhance


def normalize_qwen3vl_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Map the v5-dev RoPE field to its v4.57 name without mutating input."""
    payload = deepcopy(payload)
    text_config = payload.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("SAMTok checkpoint has no text_config object")
    rope_parameters = text_config.pop("rope_parameters", None)
    if text_config.get("rope_scaling") is None and isinstance(rope_parameters, dict):
        text_config["rope_scaling"] = rope_parameters
    rope_scaling = text_config.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or rope_scaling.get("mrope_section") != [24, 20, 20]:
        raise ValueError("SAMTok checkpoint must preserve mrope_section=[24, 20, 20]")
    return payload


def load_compatible_qwen3vl_config(checkpoint: str | Path):
    """Load v5-dev SAMTok config losslessly with the pinned v4.57 runtime."""
    from transformers import Qwen3VLConfig

    config_path = Path(checkpoint) / "config.json"
    payload = normalize_qwen3vl_config_payload(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    return Qwen3VLConfig.from_dict(payload)


def resolve_attention_backend(preferred: str) -> str:
    if preferred != "flash_attention_2":
        return preferred
    try:
        importlib.metadata.version("flash_attn")
    except importlib.metadata.PackageNotFoundError:
        return "sdpa"
    return preferred


def discover_lora_target_modules(model: torch.nn.Module) -> list[str]:
    targets = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "visual" not in name
        and "lm_head" not in name
        and "embed_tokens" not in name
    }
    if not targets:
        raise RuntimeError("No language-model linear modules were found for LoRA")
    return sorted(targets)


def discover_visual_projector_lora_targets(model: torch.nn.Module) -> list[str]:
    """Return the fixed SAMTok visual-to-language merger subspace.

    The visual transformer stays frozen.  Only the final merger and its three
    DeepStack mergers are allowed to receive the representation adapter.
    Exact module paths are used so generic ``linear_fc1``/``linear_fc2``
    suffix matching cannot accidentally patch the 24 visual blocks.
    """
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ("visual.merger." in name or "visual.deepstack_merger_list." in name)
        and name.rsplit(".", 1)[-1] in {"linear_fc1", "linear_fc2"}
    ]
    targets = sorted(targets)
    if len(targets) != 8:
        raise RuntimeError(
            "SAMTok visual projector inventory changed: expected 8 merger "
            f"linears, found {len(targets)} ({targets})"
        )
    return targets


def _lora_adapter_name(name: str) -> str | None:
    """Extract a PEFT adapter key without confusing it with module names."""
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"lora_A", "lora_B"}:
            return parts[index + 1]
    return None


def visual_projector_adapter_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Audit the two-adapter representation configuration before training."""
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if isinstance(active, str):
        active = [active]
    active = list(active or [])
    if set(active) != {"anchor", "visual"}:
        raise RuntimeError(f"Expected active adapters anchor+visual, got {active}")
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    # Adapter names are nested below ``lora_A``/``lora_B``.  Matching the
    # generic ``.visual`` substring would count the frozen Qwen visual module
    # itself as a trainable visual adapter and would hide an anchor weight.
    visual_trainable = [name for name in trainable if _lora_adapter_name(name) == "visual"]
    anchor_trainable = [name for name in trainable if _lora_adapter_name(name) == "anchor"]
    forbidden_visual = [
        name
        for name in visual_trainable
            if "visual.merger." not in name and "visual.deepstack_merger_list." not in name
    ]
    if not visual_trainable or anchor_trainable or forbidden_visual:
        raise RuntimeError(
            "Invalid representation adapter trainability: "
            f"visual={len(visual_trainable)}, anchor={len(anchor_trainable)}, "
            f"forbidden={forbidden_visual[:4]}"
        )
    return {
        "active_adapters": active,
        "visual_trainable_tensors": len(visual_trainable),
        "visual_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _lora_adapter_name(name) == "visual"
        ),
        "anchor_trainable_tensors": len(anchor_trainable),
        "target_scope": "visual.merger_and_deepstack_mergers_only",
    }


def activate_visual_projector_adapters(model: torch.nn.Module) -> None:
    """Activate anchor+visual inference while keeping only visual LoRA trainable."""
    # The pinned cluster PEFT exposes multi-adapter composition on the tuner
    # (`base_model`) while its older PeftModel facade accepts only one name.
    tuner = getattr(model, "base_model", None)
    if tuner is None or not hasattr(tuner, "set_adapter"):
        raise RuntimeError("PEFT model does not expose its adapter tuner")
    tuner.set_adapter(["anchor", "visual"])
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(_lora_adapter_name(name) == "visual")
        else:
            parameter.requires_grad_(False)


def save_trainable_lora_adapter(
    model: torch.nn.Module,
    adapter_dir: str | Path,
    *,
    representation_mode: bool,
) -> Path:
    """Save the trainable adapter and return its concrete artifact path."""
    adapter_dir = Path(adapter_dir)
    kwargs: dict[str, Any] = {"safe_serialization": True}
    if representation_mode:
        kwargs["selected_adapters"] = ["visual"]
    model.save_pretrained(adapter_dir, **kwargs)
    candidates = [adapter_dir / "adapter_model.safetensors"]
    if representation_mode:
        candidates.insert(0, adapter_dir / "visual" / "adapter_model.safetensors")
    artifacts = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if len(artifacts) != 1:
        raise RuntimeError(
            f"Expected one nonempty saved trainable adapter under {adapter_dir}, got {artifacts}"
        )
    return artifacts[0]


def validate_adapter_init(adapter_path: str | None, output_root: str | Path) -> str | None:
    if not adapter_path:
        return None
    path = Path(adapter_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root not in path.parents:
        raise ValueError("continued-SFT may initialize only from a standalone reproduction adapter")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / filename).is_file():
            raise ValueError(f"Adapter initialization is missing {filename}: {path}")
    return str(path)


def build_model_and_processor(config: dict[str, Any], adapter_path: str | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_config = config["model"]
    base = model_config["base_checkpoint"]
    processor = AutoProcessor.from_pretrained(
        model_config["processor_checkpoint"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    checkpoint_config = load_compatible_qwen3vl_config(base)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base,
        config=checkpoint_config,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=dtype,
        attn_implementation=resolve_attention_backend(model_config["attention_backend"]),
    )
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)
    if config["runtime"].get("gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
    adapter_mode = str(model_config.get("adapter_mode", "language_only"))
    if adapter_mode == "frozen_anchor_plus_visual_projector":
        if not adapter_path:
            raise ValueError("Representation adapter mode requires the frozen anchor")
        visual_targets = discover_visual_projector_lora_targets(model)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name="anchor",
            is_trainable=False,
        )
        lora = config["lora"]
        model.add_adapter(
            "visual",
            LoraConfig(
                r=int(lora.get("visual_r", 16)),
                lora_alpha=int(lora.get("visual_alpha", 32)),
                lora_dropout=float(lora.get("visual_dropout", 0.0)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=visual_targets,
            ),
        )
        # PEFT activates adapters for both forward and training; re-apply the
        # frozen-anchor rule after every adapter switch.
        activate_visual_projector_adapters(model)
        visual_projector_adapter_summary(model)
    elif adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        targets = discover_lora_target_modules(model)
        lora = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias=str(lora["bias"]),
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    assert_only_lora_trainable(model)
    return model, processor


def assert_only_lora_trainable(model: torch.nn.Module) -> dict[str, int]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")
    invalid = [name for name, _ in trainable if "lora_" not in name]
    if invalid:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid[:8]}")
    return {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
    }


def render_chat(processor: Any, prompt: str, answer: str | None) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=answer is None,
    )


def build_supervised_inputs(
    processor: Any, samples: list[dict[str, Any]]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt_texts = [render_chat(processor, sample["prompt_text"], None) for sample in samples]
    full_texts = [
        render_chat(processor, sample["prompt_text"], sample["answer_text"]) for sample in samples
    ]
    images = [sample["image"] for sample in samples]
    prompt_inputs = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
    full_inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
    full_lengths = full_inputs["attention_mask"].sum(dim=1)
    answer_mask = torch.zeros_like(full_inputs["input_ids"], dtype=torch.bool)
    for index, (start, end) in enumerate(zip(prompt_lengths.tolist(), full_lengths.tolist())):
        if end <= start:
            raise RuntimeError(f"Sample {samples[index]['id']} has an empty supervised answer")
        answer_mask[index, int(start) : int(end)] = True
    tensors = {name: value for name, value in full_inputs.items() if isinstance(value, torch.Tensor)}
    return tensors, answer_mask


def build_target_preserving_view_samples(
    samples: list[dict[str, Any]],
    *,
    brightness: float = 1.03,
    contrast: float = 0.97,
) -> list[dict[str, Any]]:
    """Create a deterministic photometric view whose SAMTok target is unchanged.

    Geometry is deliberately untouched: the answer text and mask-code target are
    copied from the same row, while only the pixels are mildly perturbed.  This
    keeps the auxiliary CE a supervised interface consistency signal rather than
    a PixVL-style self-supervised cycle or an implicit label transform.
    """
    if brightness <= 0.0 or contrast <= 0.0:
        raise ValueError("Target-preserving view factors must be positive")
    transformed: list[dict[str, Any]] = []
    for sample in samples:
        view = dict(sample)
        image = sample.get("image")
        if image is None or not hasattr(image, "copy"):
            raise TypeError("Every sample must provide a PIL image")
        view["image"] = ImageEnhance.Contrast(
            ImageEnhance.Brightness(image.copy()).enhance(brightness)
        ).enhance(contrast)
        transformed.append(view)
    return transformed


def answer_token_cross_entropy(
    logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, answer_mask: torch.Tensor
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = answer_mask[:, 1:] & attention_mask[:, 1:].bool()
    token_counts = token_mask.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise RuntimeError("Every sample must select at least one answer token")
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    per_sample = (nll * token_mask.to(nll.dtype)).sum(dim=1) / token_counts.to(nll.dtype)
    return per_sample.mean()


def move_tensors(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in inputs.items()}
