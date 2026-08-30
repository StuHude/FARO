"""Deterministic text-only calibration for abstention/no-target prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    encode_chat,
    generate_answer,
    load_config,
    move_inputs_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--input", required=True, help="JSONL rows with image/image_path and prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def is_abstention(text: str) -> bool:
    cleaned = re.sub(r"<[^>]+>", " ", text).lower()
    return bool(re.search(r"\bno\s+target\b|\bno\s+such\s+target\b|\bdoes\s+not\s+exist\b", cleaned))


def main() -> None:
    args = parse_args()
    if args.num_tasks < 1 or not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task-id must be in [0, num-tasks)")
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    cfg = load_config(args.config)
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    generation_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    records = []
    with torch.inference_mode():
        for index in range(args.task_id, len(rows), args.num_tasks):
            row = rows[index]
            image_path = row.get("image_path") or row.get("image")
            prompt = row.get("prompt") or row.get("query") or ""
            if not image_path or not prompt:
                continue
            image = Image.open(image_path).convert("RGB")
            prompt_inputs = encode_chat(processor, image, prompt, None, add_generation_prompt=True)
            prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
            _, text = generate_answer(model, processor, prompt_inputs, generation_cfg)
            target = str(row.get("answer") or row.get("target") or "")
            records.append(
                {
                    "id": row.get("id", str(index)),
                    "class": row.get("class"),
                    "source": row.get("source"),
                    "target": target,
                    "prediction": text,
                    "abstention": is_abstention(text),
                    "target_abstention": is_abstention(target),
                }
            )

    abstention_hits = sum(int(r["abstention"] == r["target_abstention"]) for r in records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "num_samples": len(records),
                "abstention_accuracy": abstention_hits / max(len(records), 1),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
