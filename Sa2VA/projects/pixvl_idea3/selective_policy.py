"""Loss controls for the shared mask-or-null selective pixel policy."""

from __future__ import annotations

from typing import Mapping

import torch


def anchor_relative_advantages(
    rewards: list[float],
    anchor_reward: float,
    *,
    positive_only: bool = True,
) -> torch.Tensor:
    """Keep only rollout improvements over a frozen anchor policy."""
    values = torch.tensor(rewards, dtype=torch.float32) - float(anchor_reward)
    if positive_only:
        values = values.clamp_min(0.0)
    return values.clamp(-1.0, 1.0)


def project_conflicting_gradient(
    objective: list[torch.Tensor],
    constraint: list[torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Project an objective gradient so it cannot increase constraint loss."""
    if len(objective) != len(constraint):
        raise ValueError("objective and constraint gradients must have equal length")
    dot = torch.zeros((), device=objective[0].device if objective else "cpu")
    constraint_norm = torch.zeros_like(dot)
    objective_norm = torch.zeros_like(dot)
    for objective_grad, constraint_grad in zip(objective, constraint):
        if objective_grad.shape != constraint_grad.shape:
            raise ValueError("objective and constraint gradient shapes differ")
        dot = dot + (objective_grad * constraint_grad).sum()
        constraint_norm = constraint_norm + constraint_grad.square().sum()
        objective_norm = objective_norm + objective_grad.square().sum()

    active = bool(dot.item() < 0.0 and constraint_norm.item() > epsilon)
    coefficient = dot / constraint_norm.clamp_min(epsilon) if active else torch.zeros_like(dot)
    projected = [
        objective_grad - coefficient * constraint_grad
        for objective_grad, constraint_grad in zip(objective, constraint)
    ]
    cosine = dot / (objective_norm.sqrt() * constraint_norm.sqrt()).clamp_min(epsilon)
    return projected, {
        "active": active,
        "dot": float(dot.item()),
        "cosine": float(cosine.item()),
        "coefficient": float(coefficient.item()),
    }


def selective_outcome_loss_scales(
    cfg: Mapping[str, object], sample: Mapping[str, object]
) -> dict[str, float]:
    scales = {"ce": 1.0, "rl": 1.0, "opd": 1.0, "kl": 1.0}
    configured = cfg.get("selective_outcome_loss_scales", {})
    if not isinstance(configured, Mapping) or sample.get("task") != "refseg":
        return scales

    meta = sample.get("meta", {})
    no_target = bool(meta.get("no_target", False)) if isinstance(meta, Mapping) else False
    outcome = "negative" if no_target else "positive"
    outcome_scales = configured.get(outcome, {})
    if not isinstance(outcome_scales, Mapping):
        return scales
    for name in scales:
        if name in outcome_scales:
            value = float(outcome_scales[name])
            if value < 0.0:
                raise ValueError(f"selective {outcome} {name} scale must be non-negative")
            scales[name] = value
    return scales
