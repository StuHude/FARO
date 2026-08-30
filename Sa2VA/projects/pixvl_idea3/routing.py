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

ACTION_KEYWORDS = {
    "standing",
    "sitting",
    "lying",
    "holding",
    "wearing",
    "walking",
    "riding",
    "flying",
    "parked",
    "leaning",
    "looking",
    "eating",
    "driving",
    "resting",
}

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "with", "of", "in", "on", "at", "to", "for", "from", "and", "or", "by", "as",
    "it", "its", "their", "his", "her", "there", "here", "region", "object",
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

COUNT_ORDINAL_KEYWORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "last",
]

NEGATION_KEYWORDS = [
    "not",
    "no",
    "none",
    "without",
    "except",
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


def extract_condition_atoms(text: str | None) -> dict[str, list[str]]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    token_set = set(tokens)
    slots = extract_semantic_slots(text)
    relation_atoms = set(extract_relation_hits(text))
    relation_atoms.update(word for word in COUNT_ORDINAL_KEYWORDS if word in token_set)
    relation_atoms.update(word for word in NEGATION_KEYWORDS if word in token_set)
    if any(action in slots.get("action", []) for action in {"holding", "wearing", "riding"}):
        relation_atoms.update(action for action in slots.get("action", []) if action in {"holding", "wearing", "riding"})
    slots["relation"] = sorted(relation_atoms)
    return slots


def compute_atom_route_weights(
    record: dict[str, Any],
    *,
    atom_weight: float = 0.4,
    geometry_weight: float = 0.35,
    tau: float = 2.0,
    epsilon: float = 0.05,
) -> dict[str, Any]:
    text = record.get("query") or record.get("caption") or ""
    atoms = extract_condition_atoms(text)

    semantic_score = 0.0
    semantic_score += 0.45 if atoms.get("category") else 0.0
    semantic_score += min(len(atoms.get("attribute", [])), 3) * 0.12
    semantic_score += min(len(atoms.get("part", [])), 2) * 0.10
    semantic_score += min(len(atoms.get("action", [])), 2) * 0.06

    relation_score = 0.0
    relation_atoms = atoms.get("relation", [])
    relation_score += min(len(relation_atoms), 4) * 0.18
    if any(atom in relation_atoms for atom in NEGATION_KEYWORDS):
        relation_score += 0.15
    if any(atom in relation_atoms for atom in COUNT_ORDINAL_KEYWORDS):
        relation_score += 0.10

    meta = record.get("meta") or {}
    area_ratio = float(meta.get("mask_area_ratio", 0.0) or 0.0)
    boundary_complexity = float(meta.get("mask_boundary_complexity", 0.0) or 0.0)
    geometry_score = 0.15
    if record.get("task") == "refseg":
        geometry_score += 0.35 if 0.0 < area_ratio <= 0.035 else 0.0
        geometry_score += 0.30 if boundary_complexity >= 5.5 else 0.0
        geometry_score += 0.15 if not relation_atoms else 0.0

    # Keep routing strictly sample-conditioned: only use signals present in the
    # current sample itself (text atoms + geometry metadata), not dataset source.
    semantic_deficit = atom_weight * semantic_score + epsilon
    relation_deficit = atom_weight * relation_score + epsilon
    geometry_deficit = geometry_weight * geometry_score + epsilon

    deficit_tensor = torch.tensor(
        [semantic_deficit, relation_deficit, geometry_deficit],
        dtype=torch.float32,
    )
    weight_tensor = torch.softmax(tau * deficit_tensor, dim=0)
    weights = {
        "semantic": float(weight_tensor[0].item()),
        "relation": float(weight_tensor[1].item()),
        "geometry": float(weight_tensor[2].item()),
    }
    bucket = max(weights, key=weights.get)
    return {
        "bucket": bucket,
        "weights": weights,
        "atoms": atoms,
        "deficits": {
            "semantic": float(semantic_deficit),
            "relation": float(relation_deficit),
            "geometry": float(geometry_deficit),
        },
    }


def infer_atom_failure_route(
    record: dict[str, Any],
    *,
    include_slice_tags: bool = True,
) -> dict[str, Any]:
    routed = compute_atom_route_weights(record)
    bucket = routed["bucket"]
    tags = derive_slice_tags(
        record,
        geometry_complexity_threshold=5.5,
    ) if include_slice_tags else []
    if include_slice_tags and bucket not in tags:
        tags.append(bucket)
    return {
        "failure_route": bucket,
        "failure_route_reasons": ["atom_conditioned_route"],
        "failure_slice_tags": sorted(set(tags)),
        "route_weights": routed["weights"],
        "condition_atoms": routed["atoms"],
        "route_deficits": routed["deficits"],
    }


def extract_semantic_slots(text: str | None) -> dict[str, list[str]]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    token_set = set(tokens)
    attributes = sorted(token for token in SEMANTIC_KEYWORDS if token in token_set)
    actions = sorted(token for token in ACTION_KEYWORDS if token in token_set)
    relation_tokens = sorted(token for token in RELATION_KEYWORDS if token in normalized)
    reserved = set(attributes) | set(actions) | set(relation_tokens) | STOPWORDS
    category = [token for token in tokens if token not in reserved]
    dedup_category: list[str] = []
    for token in category:
        if token not in dedup_category:
            dedup_category.append(token)
        if len(dedup_category) >= 3:
            break
    parts = [token for token in attributes if token in {"hat", "shirt", "hand", "arm", "leg", "wheel", "door", "window", "tail", "ear", "face"}]
    attrs = [token for token in attributes if token not in parts]
    return {
        "category": dedup_category,
        "attribute": attrs,
        "part": parts,
        "action": actions,
        "relation": relation_tokens,
    }


def slot_recall(pred_slots: dict[str, list[str]], ref_slots: dict[str, list[str]]) -> float:
    ref_atoms = []
    pred_atoms = set()
    for key in ("category", "attribute", "part", "action", "relation"):
        ref_atoms.extend(ref_slots.get(key, []))
        pred_atoms.update(pred_slots.get(key, []))
    ref_atoms = list(dict.fromkeys(ref_atoms))
    if not ref_atoms:
        return 1.0
    hits = sum(1 for atom in ref_atoms if atom in pred_atoms)
    return hits / len(ref_atoms)


def slot_precision(pred_slots: dict[str, list[str]], ref_slots: dict[str, list[str]]) -> float:
    pred_atoms = []
    ref_atoms = set()
    for key in ("category", "attribute", "part", "action", "relation"):
        pred_atoms.extend(pred_slots.get(key, []))
        ref_atoms.update(ref_slots.get(key, []))
    pred_atoms = list(dict.fromkeys(pred_atoms))
    if not pred_atoms:
        return 1.0
    hits = sum(1 for atom in pred_atoms if atom in ref_atoms)
    return hits / len(pred_atoms)


def category_anchor_score(pred_slots: dict[str, list[str]], ref_slots: dict[str, list[str]]) -> float:
    pred = set(pred_slots.get("category", []))
    ref = set(ref_slots.get("category", []))
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    return 1.0 if (pred & ref) else 0.0


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
        # Token-only RefCOCO rows are decoded lazily by UnifiedRegionDataset;
        # they have no RLE object to use as a confuser mask.  Keep the map
        # empty for such rows rather than rejecting an otherwise valid label.
        if any(not isinstance(record.get("mask"), dict) for record in group):
            for record in group:
                confuser_map[record["id"]] = []
            continue
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


def compute_semantic_coverage_calibration_reward(
    prediction: str,
    reference: str,
    *,
    rec_weight: float = 0.2,
    pos_weight: float = 0.45,
    neg_weight: float = 0.35,
) -> dict[str, float]:
    pred_slots = extract_semantic_slots(prediction)
    ref_slots = extract_semantic_slots(reference)
    rec = category_anchor_score(pred_slots, ref_slots)
    coverage = slot_recall(pred_slots, ref_slots)
    calibration = slot_precision(pred_slots, ref_slots)
    total = rec_weight * rec + pos_weight * coverage + neg_weight * calibration
    return {
        "bucket": "semantic",
        "recognition": float(rec),
        "coverage": float(coverage),
        "calibration": float(calibration),
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
