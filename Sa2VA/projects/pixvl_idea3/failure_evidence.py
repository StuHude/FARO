"""Predicted-only failure evidence and soft local-credit routing.

This module deliberately contains no model or dataset code.  It is the small,
deterministic contract shared by training and diagnostics for FEPO.  A router
must consume evidence available at rollout time; GT-derived geometry is only an
oracle upper bound and must never be passed here by the deployment path.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Sequence


CAPABILITIES = ("semantic", "relation", "geometry")

_ROLLOUT_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are",
    "was", "were", "with", "of", "in", "on", "at", "to", "for", "from",
    "and", "or", "by", "as", "it", "its", "their", "his", "her", "there",
    "here", "region", "object",
}
_RELATION_WORDS = {
    "left", "right", "behind", "front", "under", "over", "near", "beside",
    "between", "holding", "wearing", "riding", "closest", "farthest", "nearest",
    "not", "no", "none", "without", "except", "one", "two", "three", "first",
    "second", "third",
}
_NO_TARGET_WORDS = {
    "nonexistent", "absent", "missing", "nobody", "nothing",
}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _softmax(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not values:
        return ()
    temperature = max(float(temperature), 1e-6)
    scaled = [float(value) / temperature for value in values]
    pivot = max(scaled)
    exp_values = [math.exp(value - pivot) for value in scaled]
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


def failure_evidence(
    *,
    semantic_coverage: float,
    unsupported_claim_rate: float,
    target_confuser_margin: float,
    boundary_iou: float,
    area_error: float,
    no_target_error: float = 0.0,
) -> dict[str, float]:
    """Return capability deficits in ``[0, 1]`` from rollout-time signals.

    ``unsupported_claim_rate`` and ``no_target_error`` are explicit because
    DLC-style negative prompts punish overconfident captions even when keyword
    recall looks good.  The geometric deficit uses boundary and area, rather
    than a source label or ground-truth mask metadata.
    """

    semantic_deficit = 0.5 * (1.0 - _clip(semantic_coverage)) + 0.35 * _clip(
        unsupported_claim_rate
    ) + 0.25 * _clip(no_target_error)
    relation_deficit = 1.0 - _clip(target_confuser_margin)
    geometry_deficit = 0.65 * (1.0 - _clip(boundary_iou)) + 0.35 * _clip(area_error)
    return {
        "semantic": _clip(semantic_deficit),
        "relation": _clip(relation_deficit),
        "geometry": _clip(geometry_deficit),
    }


def soft_route(
    deficits: Mapping[str, float],
    *,
    temperature: float = 0.25,
    min_failure: float = 0.25,
) -> dict[str, object]:
    """Convert deficits into a mixed route and a failure gate.

    The returned weights are never hard-argmaxed.  ``failure`` is a separate
    gate so correct samples retain the ordinary policy objective, while failed
    samples receive local correction in proportion to all active deficits.
    """

    ordered = [_clip(deficits.get(name, 0.0)) for name in CAPABILITIES]
    weights = _softmax(ordered, temperature)
    max_deficit = max(ordered, default=0.0)
    return {
        "weights": dict(zip(CAPABILITIES, weights)),
        "deficits": dict(zip(CAPABILITIES, ordered)),
        "failure": max_deficit >= float(min_failure),
        "max_deficit": max_deficit,
    }


def _rollout_tokens(text: str | None) -> set[str]:
    """Extract cheap lexical probes without consulting labels or dataset state."""

    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in _ROLLOUT_STOPWORDS
    }


def predicted_only_evidence(
    *,
    prompt_text: str,
    predicted_text: str,
    task: str,
    answer_confidence: float | None = None,
    no_target_error: float | None = None,
) -> dict[str, float]:
    """Estimate FEPO deficits using only the current prompt and rollout.

    This is intentionally a conservative adapter rather than a reward.  It
    never accepts masks, captions, source buckets, or any other target-derived
    value.  For captioning, prompt/answer token overlap supplies semantic and
    relation probes.  For refseg, generated mask codes are opaque to this
    module, so geometry is driven by the model confidence probe and text
    semantics are held neutral.
    """

    predicted_tokens = _rollout_tokens(predicted_text)
    prompt_tokens = _rollout_tokens(prompt_text)
    prompt_lower = str(prompt_text or "").lower()
    prompt_no_target = any(token in prompt_lower for token in _NO_TARGET_WORDS)
    prompt_no_target = prompt_no_target or "no target" in prompt_lower
    prompt_no_target = prompt_no_target or "not present" in prompt_lower
    prompt_no_target = prompt_no_target or "without target" in prompt_lower
    relation_prompt = prompt_tokens & _RELATION_WORDS
    if prompt_no_target:
        relation_prompt -= {"not", "no", "none", "without", "except"}
    relation_predicted = predicted_tokens & _RELATION_WORDS
    empty_prediction = no_target_error is None and not predicted_tokens and not str(predicted_text or "").strip()
    no_target = _clip(no_target_error) if no_target_error is not None else float(empty_prediction or prompt_no_target)
    confidence = 0.5 if answer_confidence is None else _clip(answer_confidence)

    if task == "refseg":
        # Segmentation code tokens do not expose caption semantics.  Do not
        # mistake opaque codes for unsupported language claims.
        semantic_coverage = 1.0
        unsupported_claim_rate = 0.0
    elif prompt_tokens:
        overlap = len(predicted_tokens & prompt_tokens)
        semantic_coverage = overlap / len(prompt_tokens)
        unsupported_claim_rate = 1.0 - overlap / max(len(predicted_tokens), 1)
    else:
        semantic_coverage = 1.0 if predicted_tokens else 0.0
        unsupported_claim_rate = 0.0

    if task == "refseg":
        # Mask-token rollouts do not expose relation words.  A relation query
        # can still use the rollout confidence as a predicted-only proxy for
        # target-vs-confuser uncertainty; non-relation prompts stay neutral.
        relation_margin = confidence if relation_prompt else 1.0
    elif relation_prompt:
        relation_margin = len(relation_predicted & relation_prompt) / len(relation_prompt)
    else:
        # No relation constraint is visible in this prompt, so relation routing
        # must not be activated by an absent signal.
        relation_margin = 1.0

    evidence = failure_evidence(
        semantic_coverage=semantic_coverage,
        unsupported_claim_rate=unsupported_claim_rate,
        target_confuser_margin=relation_margin,
        boundary_iou=confidence,
        area_error=1.0 - confidence,
        no_target_error=no_target,
    )
    if prompt_no_target:
        # Explicit no-target requests are an abstention/semantic failure, not
        # a geometry failure, even when the model emits opaque mask tokens.
        evidence["semantic"] = max(evidence["semantic"], 0.9)
    return evidence


def predicted_only_evidence_route(
    *,
    prompt_text: str,
    predicted_text: str,
    task: str,
    answer_confidence: float | None = None,
    no_target_error: float | None = None,
    temperature: float = 0.25,
    min_failure: float = 0.25,
) -> dict[str, object]:
    """Build the trainer routing payload from rollout-time probes only."""

    deficits = predicted_only_evidence(
        prompt_text=prompt_text,
        predicted_text=predicted_text,
        task=task,
        answer_confidence=answer_confidence,
        no_target_error=no_target_error,
    )
    route = soft_route(deficits, temperature=temperature, min_failure=min_failure)
    weights = route["weights"]
    assert isinstance(weights, dict)
    bucket = max(CAPABILITIES, key=lambda name: float(weights[name]))
    route.update(
        {
            "failure_route": bucket,
            "failure_route_reasons": ["predicted_only_evidence"],
            "route_weights": dict(weights),
            "route_deficits": dict(route["deficits"]),
            "predicted_only": True,
        }
    )
    return route


def local_scope(capability: str, task: str) -> str:
    """Return the only answer-token scope that a capability may update."""

    if capability == "semantic":
        return "semantic_text"
    if capability == "relation":
        return "relation_and_mask" if task == "refseg" else "semantic_text"
    if capability == "geometry":
        return "mask" if task == "refseg" else "task_matched"
    raise ValueError(f"unknown FEPO capability: {capability}")


def mix_local_losses(
    local_losses: Mapping[str, float],
    route: Mapping[str, object],
    *,
    preservation_loss: float = 0.0,
    preservation_weight: float = 1.0,
) -> float:
    """Mix local losses without changing their signs or inventing branches."""

    weights = route.get("weights", {})
    if not isinstance(weights, Mapping):
        raise TypeError("route['weights'] must be a mapping")
    correction = sum(
        float(weights.get(name, 0.0)) * float(local_losses.get(name, 0.0))
        for name in CAPABILITIES
    )
    if not bool(route.get("failure", False)):
        correction = 0.0
    return correction + float(preservation_weight) * float(preservation_loss)


def soft_local_scale(
    routing: Mapping[str, object],
    route_weights: Mapping[str, float],
    key: str,
    *,
    fallback_bucket: str,
) -> float:
    """Blend capability scales around the shared objective's neutral value."""

    buckets = routing.get("buckets", {})
    if not isinstance(buckets, Mapping):
        buckets = {}
    fallback = buckets.get(fallback_bucket, {})
    if not isinstance(fallback, Mapping):
        fallback = {}
    weighted_delta = 0.0
    total_weight = 0.0
    for capability in CAPABILITIES:
        weight = max(0.0, float(route_weights.get(capability, 0.0)))
        if weight <= 0.0:
            continue
        branch = buckets.get(capability, fallback)
        if not isinstance(branch, Mapping):
            branch = fallback
        scale = float(branch.get(key, fallback.get(key, 1.0)))
        weighted_delta += weight * (scale - 1.0)
        total_weight += weight
    if total_weight <= 1e-8:
        return float(fallback.get(key, 1.0))
    gain = float(routing.get("soft_credit_gain", 1.0))
    return max(0.0, 1.0 + gain * weighted_delta / total_weight)
