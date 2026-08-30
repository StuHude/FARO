#!/usr/bin/env python
"""Score canonical-null versus mask-action margins for SAMTok refseg prompts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from tools.action_margin_contract import (
    EXPECTED_ROWS,
    MARGIN_FORMAT,
    MARGIN_SCHEMA_VERSION,
    load_schema_contract,
    validate_formal_adapter,
)


PROMPT = (
    'Please segment the region referred to by: "{query}". '
    'Return only the region mask; if the target is absent, return "No target."'
)
CANONICAL_NULL_TEXT = "No target."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--policy", required=True, choices=("continued", "candidate"))
    parser.add_argument(
        "--mask-start-token",
        action="append",
        dest="mask_start_tokens",
        help="Valid one-token mask action; repeat to score a candidate set",
    )
    return parser.parse_args()


def render_prompt(processor, query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": PROMPT.format(query=query)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def resolve_single_token_ids(tokenizer, tokens: list[str]) -> list[int]:
    token_ids: list[int] = []
    for token in tokens:
        encoded = [int(value) for value in tokenizer.encode(token, add_special_tokens=False)]
        if len(encoded) != 1:
            raise ValueError(f"Mask-start candidate must encode to one token: {token!r} -> {encoded}")
        token_id = encoded[0]
        if token_id < 0 or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Mask-start candidate is unavailable: {token!r}")
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Mask-start candidates must map to unique token IDs")
    return token_ids


def append_teacher_forcing_target(
    inputs: dict[str, torch.Tensor], target_ids: list[int]
) -> tuple[dict[str, torch.Tensor], int]:
    if not target_ids:
        raise ValueError("Canonical null sequence has no token IDs")
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Action-margin scoring requires one unpadded prompt")
    prompt_length = int(inputs["attention_mask"][0].sum().item())
    if prompt_length != input_ids.shape[1]:
        raise ValueError("Action-margin scoring does not accept padded prompts")
    target = torch.tensor([target_ids], dtype=input_ids.dtype, device=input_ids.device)
    result = dict(inputs)
    result["input_ids"] = torch.cat((input_ids, target), dim=1)
    attention_extension = torch.ones(
        (1, len(target_ids)),
        dtype=inputs["attention_mask"].dtype,
        device=inputs["attention_mask"].device,
    )
    result["attention_mask"] = torch.cat((inputs["attention_mask"], attention_extension), dim=1)
    return result, prompt_length


def action_log_probabilities(
    logits: torch.Tensor,
    prompt_length: int,
    null_token_ids: list[int],
    mask_start_token_ids: list[int],
) -> tuple[float, float, float]:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected [1, sequence, vocabulary] logits")
    start = prompt_length - 1
    if start < 0 or start >= logits.shape[1]:
        raise ValueError("Logits do not cover the canonical null sequence")
    log_probs = torch.log_softmax(logits[0, start].float(), dim=-1)
    null_log_prob = log_probs[int(null_token_ids[0])]
    mask_ids = torch.tensor(mask_start_token_ids, dtype=torch.long, device=log_probs.device)
    mask_log_prob = torch.logsumexp(log_probs[mask_ids], dim=0)
    margin = null_log_prob - mask_log_prob
    values = tuple(float(value.item()) for value in (null_log_prob, mask_log_prob, margin))
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite action log probability")
    return values


def main() -> None:
    args = parse_args()
    if args.num_tasks < 1 or not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task-id must be in [0, num-tasks)")
    repo_root = Path(__file__).resolve().parents[1]
    adapter_identity = validate_formal_adapter(
        args.adapter, repo_root / "outputs" / "samtok_selective"
    )

    from peft import PeftModel
    from projects.samtok_selective.modeling import (
        load_compatible_qwen3vl_config,
        resolve_attention_backend,
    )
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        config=load_compatible_qwen3vl_config(args.model),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attention_backend("flash_attention_2"),
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.cuda().eval()

    tokenizer = processor.tokenizer
    mask_start_tokens = args.mask_start_tokens or ["<|mt_start|>"]
    mask_start_ids = resolve_single_token_ids(tokenizer, mask_start_tokens)
    null_token_ids = [
        int(value) for value in tokenizer.encode(CANONICAL_NULL_TEXT, add_special_tokens=False)
    ]
    if not null_token_ids:
        raise ValueError("Canonical null sequence has no token IDs")

    records: list[dict[str, object]] = []
    schema = load_schema_contract(args.schema)
    rows = schema.rows
    for index in range(args.task_id, len(rows), args.num_tasks):
        row = rows[index]
        image = Image.open(str(row["image_path"])).convert("RGB")
        inputs = processor(
            text=[render_prompt(processor, str(row["query"]))],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.cuda(non_blocking=True)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        model_inputs, prompt_length = append_teacher_forcing_target(inputs, null_token_ids)
        with torch.inference_mode():
            logits = model(**model_inputs, use_cache=False).logits
        null_log_prob, mask_log_prob, margin = action_log_probabilities(
            logits, prompt_length, null_token_ids, mask_start_ids
        )
        meta = row["meta"]
        records.append(
            {
                "index": index,
                "id": str(row["id"]),
                "pair_id": str(row.get("pair_id", "")),
                "no_target": bool(meta["no_target"]),
                "source_image_id": str(meta["source_image_id"]),
                "null_action_log_prob": null_log_prob,
                "mask_start_log_prob": mask_log_prob,
                "margin": margin,
            }
        )
        if len(records) % 25 == 0:
            print(json.dumps({"rows": len(records), "last_index": index}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format": MARGIN_FORMAT,
                "schema_version": MARGIN_SCHEMA_VERSION,
                "schema": str(schema.path),
                "schema_sha256": schema.sha256,
                "schema_row_ids_sha256": schema.row_ids_sha256,
                "schema_row_count": EXPECTED_ROWS,
                "model": str(Path(args.model).resolve()),
                "adapter": str(Path(args.adapter).resolve()),
                "adapter_metrics_sha256": adapter_identity["metrics_sha256"],
                "adapter_provenance_sha256": adapter_identity["provenance_sha256"],
                "policy": args.policy,
                "task_id": args.task_id,
                "num_tasks": args.num_tasks,
                "scoring": {
                    "canonical_null_text": CANONICAL_NULL_TEXT,
                    "null_token_ids": null_token_ids,
                    "mask_start_tokens": mask_start_tokens,
                    "mask_start_token_ids": mask_start_ids,
                    "margin_definition": "log_p_first_null_token_minus_logsumexp_mask_start",
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
