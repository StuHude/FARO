#!/usr/bin/env python3
"""CPU-only contract probe for the conditional A-PES scope variant.

This module deliberately lives outside the registered trainer.  A-PES is a
follow-up candidate, so its probability-gap evidence can be audited without
changing the currently registered PES objective or creating a checkpoint.
"""

from __future__ import annotations

import math

import torch


def probability_gap_scope_masks(
    entropies: torch.Tensor,
    native_probabilities: torch.Tensor,
    sampled_probabilities: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    support_size: int = 8,
    confident_entropy: float = 0.35,
    ambiguous_entropy: float = 0.70,
    confident_gap: float = 0.40,
    ambiguous_gap: float = 0.10,
    shuffle_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached A-PES scope, state, and probability-gap tensors.

    ``g = p_native - p_sampled`` is computed at each mask-code depth.  As in
    the registered trainer, a larger native-vs-sampled gap is stronger
    evidence: state 0 (confident) scopes the first changed depth, state 1
    (ambiguous) scopes the first two changed depths, and state 2 (unsupported)
    has an empty scope.  ``shuffle_seed`` permutes only the evidence states,
    preserving the same rows, code changes, and probability gaps for the
    negative control.
    """
    tensors = (entropies, native_probabilities, sampled_probabilities)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("A-PES evidence tensors must have shape [N, depth]")
    if not (
        entropies.shape == native_probabilities.shape == sampled_probabilities.shape
    ):
        raise ValueError("A-PES evidence tensors must have identical shapes")
    if not native_codes or len(sampled_codes) != entropies.shape[0]:
        raise ValueError("A-PES evidence/code shapes are inconsistent")
    depth = len(native_codes)
    if entropies.shape[1] != depth or any(len(row) != depth for row in sampled_codes):
        raise ValueError("A-PES evidence depth does not match SAMTok grammar")
    if int(support_size) < 2:
        raise ValueError("A-PES support_size must be at least two")
    thresholds = (
        confident_entropy,
        ambiguous_entropy,
        confident_gap,
        ambiguous_gap,
    )
    if not all(math.isfinite(float(value)) for value in thresholds):
        raise ValueError("A-PES thresholds must be finite")
    if not (0.0 < confident_entropy < ambiguous_entropy <= 1.0):
        raise ValueError("A-PES entropy thresholds must be ordered in (0, 1]")
    if confident_gap < ambiguous_gap:
        raise ValueError("A-PES gap thresholds must be ordered")
    for name, values in (
        ("entropy", entropies),
        ("native probability", native_probabilities),
        ("sampled probability", sampled_probabilities),
    ):
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"A-PES {name} tensor must be finite")
    if bool(((native_probabilities < 0) | (native_probabilities > 1)).any().item()):
        raise ValueError("A-PES native probabilities must lie in [0, 1]")
    if bool(((sampled_probabilities < 0) | (sampled_probabilities > 1)).any().item()):
        raise ValueError("A-PES sampled probabilities must lie in [0, 1]")

    # Detach before any state or scope decision: evidence is a rollout
    # diagnostic and must never carry a policy gradient.
    entropy = entropies.detach().float().clamp_min(0.0)
    gap = (native_probabilities.detach().float() - sampled_probabilities.detach().float()).clamp_min(0.0)
    normalized_entropy = entropy / math.log(float(support_size))
    mean_entropy = normalized_entropy.mean(dim=1)
    mean_gap = gap.mean(dim=1)
    confident = (mean_entropy < confident_entropy) & (mean_gap >= confident_gap)
    ambiguous = (~confident) & (
        (mean_entropy < ambiguous_entropy) | (mean_gap >= ambiguous_gap)
    )
    states = torch.full(
        (len(sampled_codes),), 2, dtype=torch.long, device=entropies.device
    )
    states[ambiguous] = 1
    states[confident] = 0
    if shuffle_seed is not None:
        generator = torch.Generator(device=states.device)
        generator.manual_seed(int(shuffle_seed))
        states = states[torch.randperm(len(states), generator=generator, device=states.device)]

    scope = torch.zeros_like(entropy, dtype=torch.float32)
    for row, codes in enumerate(sampled_codes):
        changed = [
            index
            for index, (code, native) in enumerate(zip(codes, native_codes))
            if code != native
        ]
        state = int(states[row].item())
        if not changed or state == 2:
            continue
        scope[row, changed[: (1 if state == 0 else 2)]] = 1.0
    return scope.detach(), states.detach(), gap.detach()
