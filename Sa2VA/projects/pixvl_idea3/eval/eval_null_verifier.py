"""Frozen SAMTok target-vs-null verifier calibration.

This is deliberately a language-head probe on the original SAMTok checkpoint.
It does not load PixVL weights, a PixVL verifier, or any trainable adapter.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    forward_answer_logits,
    load_config,
    move_inputs_to_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def _mean_logprob(logits: torch.Tensor, answer_ids: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, dim=-1)[0]
    return float(logp.gather(-1, answer_ids.to(logits.device).unsqueeze(-1)).squeeze(-1).mean().item())


def score_hypothesis(model, processor, device, image, prompt: str, answer: str) -> float:
    prompt_inputs, answer_ids = build_prompt_and_answer_ids(processor, image, prompt, answer)
    prompt_inputs = move_inputs_to_device(prompt_inputs, device)
    logits = forward_answer_logits(model, prompt_inputs, answer_ids.to(device))
    return _mean_logprob(logits, answer_ids)


def target_description(prompt: str) -> str:
    """Remove task and abstention instructions that would leak the label."""
    text = re.sub(r"^\s*please\s+segment\s+", "", prompt, flags=re.IGNORECASE)
    text = re.split(r"\s*\bif\s+no\s+described\s+target\s+exists\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip().rstrip(".")


def main() -> None:
    args = parse_args()
    if args.num_tasks < 1 or not 0 <= args.task_id < args.num_tasks:
        raise ValueError("task-id must be in [0, num-tasks)")
    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    cfg = load_config(args.config)
    base = str(cfg["model"]["base_model_name_or_path"]).lower()
    if "samtok" not in base:
        raise ValueError(f"NC verifier must use the original SAMTok checkpoint, got {base}")
    if args.adapter_path and Path(args.adapter_path).is_dir():
        adapter_cfg = Path(args.adapter_path) / "adapter_config.json"
        if adapter_cfg.exists() and "samtok" not in adapter_cfg.read_text(encoding="utf-8").lower():
            raise ValueError("adapter must be derived from the original SAMTok checkpoint")
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    records = []
    with torch.inference_mode():
        for index in range(args.task_id, len(rows), args.num_tasks):
            row = rows[index]
            image_path = row.get("image_path") or row.get("image")
            prompt = str(row.get("prompt") or row.get("query") or "").strip()
            if not image_path or not prompt:
                continue
            image = Image.open(image_path).convert("RGB")
            description = target_description(prompt)
            verifier_prompt = (
                "Answer exactly one word, Yes or No. Does the image contain the target described below?\n"
                f"Target description: {description}\n"
                "Answer:"
            )
            yes = score_hypothesis(model, processor, device, image, verifier_prompt, "Yes")
            no = score_hypothesis(model, processor, device, image, verifier_prompt, "No")
            margin = yes - no
            records.append({
                "id": row.get("id", str(index)),
                "class": row.get("class"),
                "source": row.get("source"),
                "image_path": image_path,
                "prompt": prompt,
                "target_description": description,
                "target_exists": row.get("class") == "positive",
                "yes_logprob": yes,
                "no_logprob": no,
                "target_margin": margin,
                "predicted_exists": bool(margin >= 0.0),
            })
    correct = sum(int(bool(r["predicted_exists"]) == bool(r["target_exists"])) for r in records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "num_samples": len(records),
        "accuracy": correct / max(len(records), 1),
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
