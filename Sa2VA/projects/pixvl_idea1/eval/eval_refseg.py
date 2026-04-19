from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from projects.pixvl_idea1.rewards import compute_ciou
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    generate_answer,
    load_config,
    move_inputs_to_device,
)
from projects.pixvl_idea1.datasets import UnifiedRegionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dataset = UnifiedRegionDataset(
        schema_files=[args.schema_file],
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        cache_path=cfg["paths"]["mask_code_cache"],
        task_mix={"refseg": 1.0},
        source_mix=None,
        prompt_templates=cfg["data"]["prompts"],
        overlay_cfg=cfg["data"]["overlay"],
    )
    results = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["task"] != "refseg":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["refseg"])
        pred_codes = dataset.codec.text_to_codes(sample_text)
        pred_mask = dataset.codec.decode_codes(sample["image"], pred_codes)
        ciou = compute_ciou(pred_mask, sample["mask_binary"])
        results.append({"id": sample["id"], "ciou": ciou})
    payload = {
        "num_samples": len(results),
        "mean_ciou": sum(item["ciou"] for item in results) / max(len(results), 1),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
