from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from projects.pixvl_idea1.datasets import UnifiedRegionDataset
from projects.pixvl_idea1.rewards import compute_cap_reward, compute_ciou
from projects.pixvl_idea1.rewards.text_similarity import SentenceSimilarityScorer
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    clean_generated_text,
    generate_answer,
    load_config,
    move_inputs_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--relation-schema")
    parser.add_argument("--geometry-schema")
    parser.add_argument("--semantic-schema")
    parser.add_argument("--refseg-overall-schema")
    parser.add_argument("--dlc-schema")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def eval_refseg_schema(model, processor, cfg, schema_file: str) -> dict[str, float]:
    dataset = UnifiedRegionDataset(
        schema_files=[schema_file],
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
        results.append(ciou)
    return {
        "num_samples": len(results),
        "mean_ciou": sum(results) / max(len(results), 1),
    }


def eval_maskcap_schema(model, processor, cfg, schema_file: str) -> dict[str, float]:
    similarity_scorer = SentenceSimilarityScorer()
    dataset = UnifiedRegionDataset(
        schema_files=[schema_file],
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
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["task"] != "maskcap":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["maskcap"])
        reward = compute_cap_reward(clean_generated_text(sample_text), sample["caption"], similarity_scorer=similarity_scorer)
        results.append(reward)
    return {
        "num_samples": len(results),
        "mean_reward": sum(results) / max(len(results), 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    payload: dict[str, dict[str, float]] = {}
    if args.relation_schema:
        payload["relation"] = eval_refseg_schema(model, processor, cfg, args.relation_schema)
    if args.geometry_schema:
        payload["geometry"] = eval_refseg_schema(model, processor, cfg, args.geometry_schema)
    if args.semantic_schema:
        payload["semantic"] = eval_maskcap_schema(model, processor, cfg, args.semantic_schema)
    if args.refseg_overall_schema:
        payload["refseg_overall"] = eval_refseg_schema(model, processor, cfg, args.refseg_overall_schema)
    if args.dlc_schema:
        payload["dlc"] = eval_maskcap_schema(model, processor, cfg, args.dlc_schema)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
