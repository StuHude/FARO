from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from projects.pixvl_idea1.datasets.schema import decode_rle_mask
from projects.pixvl_idea1.rewards.cap_reward import clean_caption_text, compute_cap_reward
from projects.pixvl_idea1.rewards.seg_reward import compute_ciou, compute_exact_pair


SEMANTIC_KEYWORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "wooden",
    "metal",
    "metallic",
    "plastic",
    "glass",
    "striped",
    "spotted",
    "hat",
    "shirt",
    "hand",
    "arm",
    "leg",
    "wheel",
    "door",
    "window",
    "tail",
    "ear",
    "face",
}

RELATION_KEYWORDS = [
    "left",
    "right",
    "behind",
    "in front of",
    "front",
    "next to",
    "between",
    "holding",
    "closest",
    "farthest",
    "nearest",
    "under",
    "over",
    "near",
    "beside",
    "not",
]

MASKCAP_SOURCES = {
    "cocostuff",
    "lvis",
    "paco",
    "fine_grained_dataset_part1",
    "dlc_bench",
}

RELATION_SOURCES = {
    "relation_dataset",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(clean_caption_text(text).lower().split())


def extract_word_hits(text: str | None, lexicon: set[str]) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    words = set(re.findall(r"[a-z0-9]+", normalized))
    hits = sorted(word for word in lexicon if word in words)
    return hits


def has_relation_keyword(text: str | None) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in RELATION_KEYWORDS)


def extract_relation_hits(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return sorted(keyword for keyword in RELATION_KEYWORDS if keyword in normalized)


def mask_area_ratio(mask_obj: dict[str, Any]) -> float:
    mask = decode_rle_mask(mask_obj)
    return float(mask.sum()) / float(max(mask.size, 1))


def _tensor_mask(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)).view(1, 1, mask.shape[0], mask.shape[1])


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    tensor = _tensor_mask(mask)
    for _ in range(iterations):
        tensor = (-F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1) > 0.5).float()
    return tensor[0, 0].numpy().astype(np.uint8)


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    tensor = _tensor_mask(mask)
    for _ in range(iterations):
        tensor = (F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1) > 0.5).float()
    return tensor[0, 0].numpy().astype(np.uint8)


def extract_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    width = max(int(width), 1)
    mask = mask.astype(np.uint8)
    eroded = _binary_erode(mask, iterations=width)
    dilated = _binary_dilate(mask, iterations=width)
    boundary = np.logical_and(dilated > 0, eroded == 0)
    return boundary.astype(np.uint8)


def compute_boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, width: int = 2) -> float:
    pred_boundary = extract_boundary(pred_mask, width=width).astype(bool)
    gt_boundary = extract_boundary(gt_mask, width=width).astype(bool)
    union = float((pred_boundary | gt_boundary).sum())
    if union == 0.0:
        return 1.0
    inter = float((pred_boundary & gt_boundary).sum())
    return inter / union


def compute_boundary_complexity(mask_obj: dict[str, Any], width: int = 2) -> float:
    mask = decode_rle_mask(mask_obj)
    area = float(max(mask.sum(), 1))
    boundary = float(extract_boundary(mask, width=width).sum())
    return boundary / math.sqrt(area)


def keyword_f1(prediction: str, reference: str, lexicon: set[str]) -> float:
    pred_hits = set(extract_word_hits(prediction, lexicon))
    ref_hits = set(extract_word_hits(reference, lexicon))
    if not pred_hits and not ref_hits:
        return 1.0
    if not pred_hits or not ref_hits:
        return 0.0
    inter = len(pred_hits & ref_hits)
    precision = inter / len(pred_hits)
    recall = inter / len(ref_hits)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def stable_hash_ratio(text: str, denominator: int = 10_000) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return float(value % denominator) / float(denominator)


def derive_slice_tags(
    record: dict[str, Any],
    *,
    semantic_lexicon: set[str] | None = None,
    geometry_area_threshold: float = 0.035,
    geometry_complexity_threshold: float | None = 5.5,
) -> list[str]:
    semantic_lexicon = semantic_lexicon or SEMANTIC_KEYWORDS
    tags: list[str] = []
    task = record.get("task")
    source = record.get("source")
    if task == "maskcap":
        caption = record.get("caption")
        if extract_word_hits(caption, semantic_lexicon) or source == "fine_grained_dataset_part1":
            tags.append("semantic")
        if source in RELATION_SOURCES or has_relation_keyword(caption):
            tags.append("relation")
    query = record.get("query")
    if task == "refseg" and has_relation_keyword(query):
        tags.append("relation")
    if task == "refseg":
        area_ratio = mask_area_ratio(record["mask"])
        geometry_hard = area_ratio <= geometry_area_threshold
        if geometry_complexity_threshold is not None:
            boundary_complexity = compute_boundary_complexity(record["mask"])
            geometry_hard = geometry_hard or boundary_complexity >= geometry_complexity_threshold
        if geometry_hard:
            tags.append("geometry")
    return sorted(set(tags))


def infer_failure_route(
    record: dict[str, Any],
    *,
    include_slice_tags: bool = True,
    include_geometry_metrics: bool = True,
) -> dict[str, Any]:
    task = record.get("task")
    source = record.get("source")
    reasons: list[str] = []
    if source in RELATION_SOURCES:
        reasons.append("relation_source_bucket")
        bucket = "relation"
    elif task == "maskcap" or source in MASKCAP_SOURCES:
        reasons.append("maskcap_semantic_bucket")
        bucket = "semantic"
    elif has_relation_keyword(record.get("query")):
        reasons.append("relation_keyword")
        bucket = "relation"
    else:
        reasons.append("default_refseg_geometry_bucket")
        bucket = "geometry"

    semantic_hits = extract_word_hits(record.get("query") or record.get("caption"), SEMANTIC_KEYWORDS)
    tags = derive_slice_tags(
        record,
        geometry_complexity_threshold=5.5 if include_geometry_metrics else None,
    ) if include_slice_tags else []
    if include_slice_tags and bucket not in tags:
        tags.append(bucket)
    payload = {
        "failure_route": bucket,
        "failure_route_reasons": reasons,
        "failure_slice_tags": sorted(set(tags)),
        "semantic_keyword_hits": semantic_hits,
    }
    if task == "refseg" and include_geometry_metrics:
        payload["mask_area_ratio"] = round(mask_area_ratio(record["mask"]), 6)
        payload["mask_boundary_complexity"] = round(compute_boundary_complexity(record["mask"]), 6)
    return payload


def annotate_record(
    record: dict[str, Any],
    *,
    include_slice_tags: bool = True,
    include_geometry_metrics: bool = True,
) -> dict[str, Any]:
    meta = dict(record.get("meta") or {})
    meta.update(
        infer_failure_route(
            record,
            include_slice_tags=include_slice_tags,
            include_geometry_metrics=include_geometry_metrics,
        )
    )
    updated = dict(record)
    updated["meta"] = meta
    return updated


def build_relation_confuser_map(records: list[dict[str, Any]], max_confusers: int = 16) -> dict[str, list[dict[str, Any]]]:
    image_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("task") == "refseg":
            image_to_records[record["image_path"]].append(record)

    confuser_map: dict[str, list[dict[str, Any]]] = {}
    for group in image_to_records.values():
        signatures = {
            record["id"]: f'{record["mask"].get("size")}::{record["mask"].get("counts")}'
            for record in group
        }
        for record in group:
            target_signature = signatures[record["id"]]
            seen_signatures = {target_signature}
            confusers: list[dict[str, Any]] = []
            for other in group:
                if other["id"] == record["id"]:
                    continue
                other_signature = signatures[other["id"]]
                if other_signature in seen_signatures:
                    continue
                seen_signatures.add(other_signature)
                confusers.append(other["mask"])
                if len(confusers) >= max_confusers:
                    break
            confuser_map[record["id"]] = confusers
    return confuser_map


def compute_semantic_caption_reward(
    prediction: str,
    reference: str,
    *,
    similarity_scorer: Any | None = None,
    base_weight: float = 0.75,
    keyword_weight: float = 0.25,
) -> dict[str, float]:
    base = compute_cap_reward(prediction, reference, similarity_scorer=similarity_scorer)
    keyword = keyword_f1(prediction, reference, SEMANTIC_KEYWORDS)
    total = base_weight * base + keyword_weight * keyword
    return {
        "bucket": "semantic",
        "base": float(base),
        "keyword": float(keyword),
        "total": float(total),
    }


def compute_relation_caption_reward(
    prediction: str,
    reference: str,
    *,
    similarity_scorer: Any | None = None,
    base_weight: float = 0.7,
    relation_weight: float = 0.3,
) -> dict[str, float]:
    base = compute_cap_reward(prediction, reference, similarity_scorer=similarity_scorer)
    pred_hits = set(extract_relation_hits(prediction))
    ref_hits = set(extract_relation_hits(reference))
    if not pred_hits and not ref_hits:
        relation_score = 1.0
    elif not pred_hits or not ref_hits:
        relation_score = 0.0
    else:
        inter = len(pred_hits & ref_hits)
        precision = inter / len(pred_hits)
        recall = inter / len(ref_hits)
        relation_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    total = base_weight * base + relation_weight * relation_score
    return {
        "bucket": "relation",
        "base": float(base),
        "relation": float(relation_score),
        "total": float(total),
    }


def compute_relation_reward(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    confuser_masks: list[dict[str, Any]] | None = None,
    pred_codes: list[int] | None = None,
    gt_codes: list[int] | None = None,
    target_weight: float = 0.7,
    margin_weight: float = 0.2,
    exact_weight: float = 0.1,
) -> dict[str, float]:
    ciou = compute_ciou(pred_mask, gt_mask)
    confuser_iou = 0.0
    for mask_obj in confuser_masks or []:
        confuser_iou = max(confuser_iou, compute_ciou(pred_mask, decode_rle_mask(mask_obj)))
    margin = (ciou - confuser_iou + 1.0) / 2.0
    margin = max(0.0, min(1.0, margin))
    exact = compute_exact_pair(pred_codes or [], gt_codes or []) if pred_codes is not None and gt_codes is not None else 0.0
    total = target_weight * ciou + margin_weight * margin + exact_weight * exact
    return {
        "bucket": "relation",
        "ciou": float(ciou),
        "margin": float(margin),
        "best_confuser_ciou": float(confuser_iou),
        "exact": float(exact),
        "total": float(total),
    }


def compute_geometry_reward(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    pred_codes: list[int] | None = None,
    gt_codes: list[int] | None = None,
    ciou_weight: float = 0.55,
    boundary_weight: float = 0.25,
    area_weight: float = 0.1,
    exact_weight: float = 0.1,
    boundary_width: int = 2,
) -> dict[str, float]:
    ciou = compute_ciou(pred_mask, gt_mask)
    boundary = compute_boundary_iou(pred_mask, gt_mask, width=boundary_width)
    gt_area = float(max(gt_mask.sum(), 1))
    pred_area = float(pred_mask.sum())
    area_score = max(0.0, 1.0 - abs(pred_area - gt_area) / gt_area)
    exact = compute_exact_pair(pred_codes or [], gt_codes or []) if pred_codes is not None and gt_codes is not None else 0.0
    total = ciou_weight * ciou + boundary_weight * boundary + area_weight * area_score + exact_weight * exact
    return {
        "bucket": "geometry",
        "ciou": float(ciou),
        "boundary": float(boundary),
        "area": float(area_score),
        "exact": float(exact),
        "total": float(total),
    }
