from __future__ import annotations

import math
from typing import Any


def validate_evidence_gate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("evidence_gate must be a dictionary")
    mode = config.get("mode")
    if mode not in {"view_drop", "shuffled", "none"}:
        raise ValueError("evidence_gate.mode must be view_drop, shuffled, or none")
    for key in ("scale", "clip_min", "clip_max", "noise_std"):
        value = float(config.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"evidence_gate.{key} must be finite")
    if float(config["scale"]) < 0.0:
        raise ValueError("evidence_gate.scale must be nonnegative")
    if float(config["clip_min"]) < 0.0 or float(config["clip_max"]) < float(config["clip_min"]):
        raise ValueError("evidence_gate clip bounds are invalid")
    if float(config["noise_std"]) < 0.0:
        raise ValueError("evidence_gate.noise_std must be nonnegative")


def detached_group_gate(
    evidence_gap: "torch.Tensor",
    *,
    mode: str,
    scale: float,
    clip_min: float,
    clip_max: float,
    generator: torch.Generator | None = None,
) -> "torch.Tensor":
    """Convert a sibling-group view evidence gap into a detached RL multiplier."""
    import torch

    if evidence_gap.ndim != 1 or evidence_gap.numel() < 2:
        raise ValueError("evidence_gap must be a sibling-group vector")
    if not torch.isfinite(evidence_gap).all():
        raise FloatingPointError("evidence_gap must be finite")
    if mode == "none":
        gate = torch.ones_like(evidence_gap)
    elif mode == "shuffled":
        permutation = torch.randperm(evidence_gap.numel(), device=evidence_gap.device, generator=generator)
        shuffled = evidence_gap.index_select(0, permutation)
        centered = shuffled - shuffled.mean()
        gate = 1.0 + float(scale) * centered / (centered.std(unbiased=False) + 1e-6)
    elif mode == "view_drop":
        centered = evidence_gap - evidence_gap.mean()
        gate = 1.0 + float(scale) * centered / (centered.std(unbiased=False) + 1e-6)
    else:
        raise ValueError(f"unsupported evidence gate mode: {mode}")
    return gate.clamp(float(clip_min), float(clip_max)).detach()
