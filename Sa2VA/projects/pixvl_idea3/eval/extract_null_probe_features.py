"""Extract frozen original-SAMTok prompt representations for a null probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from projects.pixvl_idea1.trainers.common import build_model_bundle, encode_chat, load_config, move_inputs_to_device
from projects.pixvl_idea3.eval.eval_null_verifier import target_description


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    args = p.parse_args()
    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    cfg = load_config(args.config)
    if "samtok" not in str(cfg["model"]["base_model_name_or_path"]).lower():
        raise ValueError("null probe must start from original SAMTok")
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
            description = target_description(prompt)
            question = (
                "Answer exactly one word, Yes or No. Does the image contain the target described below?\n"
                f"Target description: {description}\nAnswer:"
            )
            image = Image.open(image_path).convert("RGB")
            inputs = move_inputs_to_device(encode_chat(processor, image, question, None, True), device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = outputs.hidden_states[-1]
            last = int(inputs["attention_mask"][0].sum().item()) - 1
            feature = hidden[0, last].float().cpu().tolist()
            records.append({
                "id": row.get("id", str(index)),
                "class": row.get("class"),
                "source": row.get("source"),
                "target_exists": row.get("class") == "positive",
                "feature": feature,
            })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records}, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
