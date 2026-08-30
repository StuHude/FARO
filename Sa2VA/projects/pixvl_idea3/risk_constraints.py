"""Outcome-specific constraints for safe policy-improvement smoke tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def negative_outcome_mask(samples: Sequence[dict[str, object]]) -> torch.Tensor:
    """Return the target-derived training mask for explicit no-target rows."""

    return torch.tensor(
        [
            bool(sample.get("task") == "refseg" and (sample.get("meta") or {}).get("no_target", False))
            for sample in samples
        ],
        dtype=torch.bool,
    )


def outcome_constraint(
    losses: torch.Tensor,
    samples: Sequence[dict[str, object]],
    *,
    budget: float,
    epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable null-loss violation and detached diagnostics."""

    mask = negative_outcome_mask(samples).to(losses.device)
    if not bool(mask.any()):
        zero = losses.sum() * 0.0
        return zero, zero.detach(), mask
    null_loss = losses[mask].mean()
    violation = torch.relu(null_loss - float(budget) - float(epsilon))
    return violation, null_loss.detach(), mask


def update_dual(
    value: float,
    violation: float,
    *,
    learning_rate: float,
    maximum: float,
) -> float:
    """Projected ascent update for a non-negative Lagrange multiplier."""

    return min(maximum, max(0.0, float(value) + float(learning_rate) * float(violation)))
