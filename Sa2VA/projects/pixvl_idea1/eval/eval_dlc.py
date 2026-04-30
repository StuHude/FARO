from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from projects.pixvl_idea1.rewards import compute_cap_reward
from projects.pixvl_idea1.rewards.text_similarity import SentenceSimilarityScorer
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    clean_generated_text,
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
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    similarity_scorer = SentenceSimilarityScorer()
    dataset = UnifiedRegionDataset(
        schema_files=[args.schema_file],
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        cache_path=cfg["paths"]["mask_code_cache"],
        task_mix={"maskcap": 1.0},
        source_mix=None,
        prompt_templates=cfg["data"]["prompts"],
        overlay_cfg=cfg["data"]["overlay"],
    )
    results = []
    indices = list(range(len(dataset)))[args.task_id :: args.num_tasks]
    for idx in indices:
        sample = dataset[idx]
        if sample["task"] != "maskcap":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["maskcap"])
        reward = compute_cap_reward(clean_generated_text(sample_text), sample["caption"], similarity_scorer=similarity_scorer)
        results.append({"id": sample["id"], "reward": reward})
    payload = {
        "num_samples": len(results),
        "mean_reward": sum(item["reward"] for item in results) / max(len(results), 1),
        "results": results,
        "task_id": args.task_id,
        "num_tasks": args.num_tasks,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
