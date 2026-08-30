from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

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
from projects.pixvl_idea3.existence import predicts_target_exists
from projects.samtok_selective.tail_geometry import boundary_iou, geometry_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--anchor-adapter-path")
    parser.add_argument("--relation-schema")
    parser.add_argument("--geometry-schema")
    parser.add_argument("--semantic-schema")
    parser.add_argument("--refseg-overall-schema")
    parser.add_argument("--dlc-schema")
    parser.add_argument("--existence-schema")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-registry")
    return parser.parse_args()


_CANONICAL_MASK = re.compile(
    r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>(?:<\|im_end\|>)?"
)
_CANONICAL_NULL = re.compile(r"No target\.(?:<\|im_end\|>)?")


def is_canonical_response(text: str, truth_exists: bool) -> bool:
    normalized = str(text).strip()
    pattern = _CANONICAL_MASK if truth_exists else _CANONICAL_NULL
    return pattern.fullmatch(normalized) is not None


def _mask_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute the square morphological boundary used by FEPO diagnostics."""
    result = np.asarray(mask, dtype=bool)
    for _ in range(width):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    dilated = result
    result = np.asarray(mask, dtype=bool)
    for _ in range(width):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    return dilated & ~result


def _boundary_iou(prediction: np.ndarray, target: np.ndarray, width: int = 2) -> float:
    pred_boundary = _mask_boundary(prediction, width)
    target_boundary = _mask_boundary(target, width)
    union = np.logical_or(pred_boundary, target_boundary).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_boundary, target_boundary).sum() / union)


def _target_geometry_slices(
    target: np.ndarray,
    thresholds: dict[str, float] | None = None,
    training_records: list[dict] | None = None,
) -> dict[str, object]:
    binary = np.asarray(target, dtype=bool)
    area = int(binary.sum())
    if area <= 0 or binary.size == 0:
        return {"target_area_ratio": None, "target_compactness": None,
                "target_boundary_density": None, "small": False,
                "thin": False, "boundary_hard": False}
    boundary = _mask_boundary(binary)
    perimeter = int(boundary.sum())
    area_ratio = float(area / binary.size)
    compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1))
    density = float(perimeter / max(math.sqrt(area), 1.0))
    thresholds = dict(thresholds or {})
    # Older training registries predate the scale-stratified screen and only
    # store q25. Derive q75 from their training records, never from holdout
    # masks, so every evaluation uses a fixed training-only area partition.
    if "area_ratio_q75" not in thresholds:
        training_ratios = [float(row["area_ratio"]) for row in (training_records or []) if "area_ratio" in row]
        if training_ratios:
            thresholds["area_ratio_q75"] = float(np.quantile(training_ratios, 0.75, method="linear"))
    area_q25 = float(thresholds.get("area_ratio_q25", 0.01))
    area_q75 = thresholds.get("area_ratio_q75")
    compactness_q25 = float(thresholds.get("compactness_q25", 0.15))
    density_q75 = float(thresholds.get("boundary_density_q75", 8.0))
    return {"target_area_ratio": area_ratio, "target_compactness": compactness,
            "target_boundary_density": density,
            "small": area_ratio <= area_q25, "thin": compactness <= compactness_q25,
            "boundary_hard": density >= density_q75,
            "area_stratum": (
                "small" if area_ratio <= area_q25 else
                "large" if area_q75 is not None and area_ratio >= float(area_q75)
                else "medium"
            ) if area_q75 is not None else None}


def build_eval_model_bundle(cfg, adapter_path: str | None, anchor_adapter_path: str | None):
    if not anchor_adapter_path:
        return build_model_bundle(cfg, trainable=False, adapter_path=adapter_path)
    if not adapter_path:
        raise ValueError("anchor_adapter_path requires a visual adapter_path")

    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=None)
    model = PeftModel.from_pretrained(
        model,
        anchor_adapter_path,
        adapter_name="anchor",
        is_trainable=False,
    )
    model.load_adapter(adapter_path, adapter_name="visual", is_trainable=False)
    # The installed PEFT runtime reports both names for list composition but
    # silently applies only the first adapter when their LoRA ranks differ
    # (anchor r=128, visual r=16).  Merge the disjoint adapters explicitly so
    # the evaluator cannot produce an anchor-only result while claiming a
    # combined run.  ``cat`` preserves the exact additive LoRA update and is
    # valid for different ranks; SVD is unnecessary here because the anchor
    # language targets and visual merger targets are disjoint, and is very
    # expensive for the 4B model.
    if not hasattr(model, "add_weighted_adapter"):
        raise RuntimeError("PEFT runtime lacks weighted adapter merge support")
    try:
        model.add_weighted_adapter(
            ["anchor", "visual"],
            [1.0, 1.0],
            adapter_name="faro_combined",
            combination_type="cat",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to merge anchor+visual adapters: {exc}") from exc
    tuner = getattr(model, "base_model", None)
    if tuner is None or not hasattr(tuner, "set_adapter"):
        raise RuntimeError("PEFT model does not expose adapter activation")
    tuner.set_adapter("faro_combined")
    active = list(getattr(model, "active_adapters", []))
    if active != ["faro_combined"]:
        raise RuntimeError(f"Expected active faro_combined adapter, got {active}")
    return model, processor


def eval_refseg_schema(model, processor, cfg, schema_file: str, task_id: int = 0, num_tasks: int = 1, geometry_registry: dict | None = None) -> dict[str, object]:
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
    records = []
    for idx in range(task_id, len(dataset), num_tasks):
        sample = dataset[idx]
        if sample["task"] != "refseg":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["refseg"])
        pred_codes = dataset.codec.text_to_codes(sample_text)
        pred_mask = dataset.codec.decode_codes(sample["image"], pred_codes)
        no_target = bool((sample.get("meta") or {}).get("no_target", False))
        explicit_null = not predicts_target_exists(clean_generated_text(sample_text))
        ciou = (1.0 if explicit_null else 0.0) if no_target else compute_ciou(pred_mask, sample["mask_binary"])
        target_geometry = geometry_features(sample["mask_binary"]) if not no_target else None
        pred_boundary_iou = boundary_iou(pred_mask, sample["mask_binary"]) if not no_target else None
        fixed_slices = _target_geometry_slices(
            sample["mask_binary"],
            (geometry_registry or {}).get("thresholds"),
            list((geometry_registry or {}).get("records", {}).values()),
        ) if not no_target else {}
        pair_id = str(sample.get("pair_id", sample.get("id", idx)))
        registry_row = (geometry_registry or {}).get("records", {}).get(pair_id, {})
        results.append(ciou)
        records.append({
            "id": sample.get("id", str(idx)),
            "task": "refseg",
            "ciou": float(ciou),
            "truth_exists": not no_target,
            "pred_exists": bool(pred_codes),
            "explicit_null": explicit_null,
            "valid_mask_tokens": len(pred_codes) == dataset.codec.codebook_depth,
            "canonical_response": is_canonical_response(sample_text, not no_target),
            "raw_text": sample_text,
            "pair_id": pair_id,
            "boundary_iou": None if pred_boundary_iou is None else float(pred_boundary_iou),
            "target_area_ratio": None if target_geometry is None else float(target_geometry.area_ratio),
            "target_compactness": None if target_geometry is None else float(target_geometry.compactness),
            "target_boundary_density": None if target_geometry is None else float(target_geometry.boundary_density),
            "slice_metadata": {
                "small": bool(registry_row.get("small", fixed_slices.get("small", False))),
                "thin": bool(registry_row.get("thin", fixed_slices.get("thin", False))),
                "boundary_hard": bool(registry_row.get("boundary_hard", fixed_slices.get("boundary_hard", False))),
                "area_stratum": registry_row.get("area_stratum", fixed_slices.get("area_stratum")),
                "source": "training_geometry_registry_thresholds" if geometry_registry else "fixed_evaluator_bins",
            },
        })
    positives = [row for row in records if row["truth_exists"]]
    negatives = [row for row in records if not row["truth_exists"]]
    return {
        "num_samples": len(results),
        "mean_ciou": sum(results) / max(len(results), 1),
        "mean_boundary_iou": sum(float(row["boundary_iou"]) for row in records if row["boundary_iou"] is not None) / max(sum(row["boundary_iou"] is not None for row in records), 1),
        "positive_mean_ciou": sum(row["ciou"] for row in positives) / max(len(positives), 1),
        "positive_mask_rate": sum(row["pred_exists"] for row in positives) / max(len(positives), 1),
        "no_target_explicit_recall": sum(row["explicit_null"] for row in negatives) / max(len(negatives), 1),
        "invalid_output_rate": sum(not (row["valid_mask_tokens"] or row["explicit_null"]) for row in records) / max(len(records), 1),
        "canonical_response_rate": sum(row["canonical_response"] for row in records) / max(len(records), 1),
        "positive_canonical_response_rate": sum(row["canonical_response"] for row in positives) / max(len(positives), 1),
        "negative_canonical_response_rate": sum(row["canonical_response"] for row in negatives) / max(len(negatives), 1),
        "records": records,
    }


def eval_maskcap_schema(model, processor, cfg, schema_file: str, task_id: int = 0, num_tasks: int = 1) -> dict[str, object]:
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
    records = []
    for idx in range(task_id, len(dataset), num_tasks):
        sample = dataset[idx]
        if sample["task"] != "maskcap":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["maskcap"])
        reward = compute_cap_reward(clean_generated_text(sample_text), sample["caption"], similarity_scorer=similarity_scorer)
        results.append(reward)
        records.append({
            "id": sample.get("id", str(idx)),
            "task": "maskcap",
            "reward": float(reward),
        })
    return {
        "num_samples": len(results),
        "mean_reward": sum(results) / max(len(results), 1),
        "records": records,
    }


def eval_existence_schema(model, processor, cfg, schema_file: str, task_id: int = 0, num_tasks: int = 1) -> dict[str, object]:
    dataset = UnifiedRegionDataset(
        schema_files=[schema_file],
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        cache_path=cfg["paths"]["mask_code_cache"],
        task_mix={"existence": 1.0},
        source_mix=None,
        prompt_templates=cfg["data"]["prompts"],
        overlay_cfg=cfg["data"]["overlay"],
    )
    records = []
    generation_cfg = {
        **cfg.get("generation", {}).get("existence", {}),
        "max_new_tokens": 8,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    for idx in range(task_id, len(dataset), num_tasks):
        sample = dataset[idx]
        if sample["task"] != "existence":
            continue
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, sample["image"], sample["prompt_text"], sample["answer_text"])
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, generation_cfg)
        text = clean_generated_text(sample_text).lower()
        truth = "no target" not in str(sample["answer_text"]).lower()
        pred = predicts_target_exists(text)
        records.append({"id": sample.get("id", str(idx)), "truth_exists": truth, "pred_exists": pred, "raw_text": sample_text})
    pos = [row for row in records if row["truth_exists"]]
    neg = [row for row in records if not row["truth_exists"]]
    pos_recall = sum(bool(row["pred_exists"]) for row in pos) / max(len(pos), 1)
    neg_recall = sum(not bool(row["pred_exists"]) for row in neg) / max(len(neg), 1)
    return {"num_samples": len(records), "positive_recall": pos_recall, "no_target_recall": neg_recall, "balanced_accuracy": 0.5 * (pos_recall + neg_recall), "records": records}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    adapter_path = args.adapter_path
    if str(adapter_path).strip().lower() in {"", "none", "null"}:
        adapter_path = None
    anchor_adapter_path = args.anchor_adapter_path
    if str(anchor_adapter_path).strip().lower() in {"", "none", "null"}:
        anchor_adapter_path = None
    model, processor = build_eval_model_bundle(
        cfg,
        adapter_path=adapter_path,
        anchor_adapter_path=anchor_adapter_path,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    geometry_registry = None
    if args.geometry_registry:
        geometry_registry = json.loads(Path(args.geometry_registry).read_text(encoding="utf-8"))
    payload: dict[str, dict[str, float]] = {}
    if args.relation_schema:
        payload["relation"] = eval_refseg_schema(model, processor, cfg, args.relation_schema, args.task_id, args.num_tasks, geometry_registry)
    if args.geometry_schema:
        payload["geometry"] = eval_refseg_schema(model, processor, cfg, args.geometry_schema, args.task_id, args.num_tasks, geometry_registry)
    if args.semantic_schema:
        payload["semantic"] = eval_maskcap_schema(model, processor, cfg, args.semantic_schema, args.task_id, args.num_tasks)
    if args.refseg_overall_schema:
        payload["refseg_overall"] = eval_refseg_schema(model, processor, cfg, args.refseg_overall_schema, args.task_id, args.num_tasks, geometry_registry)
    if args.dlc_schema:
        payload["dlc"] = eval_maskcap_schema(model, processor, cfg, args.dlc_schema, args.task_id, args.num_tasks)
    if args.existence_schema:
        payload["existence"] = eval_existence_schema(model, processor, cfg, args.existence_schema, args.task_id, args.num_tasks)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
