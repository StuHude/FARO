from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .active_set_gr_cppo_contract import (
    active_set_flags,
    derive_active_set_budgets,
)
from .config import REPO_ROOT
from .data import (
    PairedBatchSampler,
    RegisteredPairedBatchSampler,
    SelectiveRefSegDataset,
    identity_collate,
)
from .evidence_gate import detached_group_gate
from .geometry_reward import ciou
from .gr_cppo_contract import validate_frozen_anchor, validate_gr_cppo_config
from .manifests import (
    assert_training_source_clean,
    build_manifest,
    guard_runtime_environment,
    runtime_module_files,
    validate_base_checkpoint,
    validate_declared_paths,
    write_json_atomic,
)
from .mask_codec import SAMTokMaskCodec, decode_rle_mask
from .tail_geometry import (
    ANCHOR_BUFFER_SIZE,
    FIFOEmpiricalRank,
    build_geometry_registry,
    boundary_iou,
    select_anchor_buffer_pair_ids,
    select_registered_ids,
    shuffled_hard_flags,
)
from .modeling import (
    activate_visual_projector_adapters,
    assert_only_lora_trainable,
    build_model_and_processor,
    build_supervised_inputs,
    build_target_preserving_view_samples,
    answer_token_cross_entropy,
    move_tensors,
    render_chat,
    save_trainable_lora_adapter,
    visual_projector_adapter_summary,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
CANONICAL_NULL = "No target."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = runpy.run_path(str(Path(path).resolve()))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must expose a config dictionary: {path}")
    if "representation_entropy_gr_cppo" in config:
        from .representation_fepo_contract import validate_representation_fepo_config

        validate_representation_fepo_config(config, REPO_ROOT)
    elif "tail_gppo" in config:
        from .tail_gppo_contract import validate_tail_gppo_config

        validate_tail_gppo_config(config, REPO_ROOT)
    elif "sign_balanced_entropy_gr_cppo" in config:
        from .sign_balanced_gr_cppo_contract import (
            validate_sign_balanced_gr_cppo_config,
        )

        validate_sign_balanced_gr_cppo_config(config, REPO_ROOT)
    elif "greedy_relative_entropy_gr_cppo" in config:
        from .greedy_relative_gr_cppo_contract import (
            validate_greedy_relative_gr_cppo_config,
        )

        validate_greedy_relative_gr_cppo_config(config, REPO_ROOT)
    elif "gain_preference_entropy_gr_cppo" in config:
        from .gain_preference_gr_cppo_contract import (
            validate_gain_preference_gr_cppo_config,
        )

        validate_gain_preference_gr_cppo_config(config, REPO_ROOT)
    elif "greedy_preference_entropy_gr_cppo" in config:
        from .greedy_preference_gr_cppo_contract import (
            validate_greedy_preference_gr_cppo_config,
        )

        validate_greedy_preference_gr_cppo_config(config, REPO_ROOT)
    elif "improvement_entropy_gr_cppo" in config:
        from .improvement_only_gr_cppo_contract import (
            validate_improvement_only_gr_cppo_config,
        )

        validate_improvement_only_gr_cppo_config(config, REPO_ROOT)
    elif "boundary_entropy_gr_cppo" in config:
        from .boundary_credit_gr_cppo_contract import (
            validate_boundary_credit_gr_cppo_config,
        )

        validate_boundary_credit_gr_cppo_config(config, REPO_ROOT)
    elif "active_set_entropy_gr_cppo" in config:
        from .active_set_gr_cppo_contract import (
            validate_active_set_gr_cppo_config,
        )

        validate_active_set_gr_cppo_config(config, REPO_ROOT)
    elif "entropy_gr_cppo" in config:
        from .entropy_gr_cppo_contract import validate_entropy_gr_cppo_config

        validate_entropy_gr_cppo_config(config, REPO_ROOT)
    else:
        validate_gr_cppo_config(config, REPO_ROOT)
    return config


def _method_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "representation_entropy_gr_cppo",
        "grounded_interface_fepo",
        "tail_gppo",
        "sign_balanced_entropy_gr_cppo",
        "greedy_relative_entropy_gr_cppo",
        "gain_preference_entropy_gr_cppo",
        "greedy_preference_entropy_gr_cppo",
        "improvement_entropy_gr_cppo",
        "boundary_entropy_gr_cppo",
        "active_set_entropy_gr_cppo",
        "entropy_gr_cppo",
        "gr_cppo",
    )
    for key in keys:
        method = config.get(key)
        if method is not None:
            if not isinstance(method, dict):
                raise ValueError(f"GR-CPPO method configuration {key} must be a dictionary")
            return method
    raise ValueError("Missing GR-CPPO method configuration")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_standardized_advantages(
    rewards: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A rollout group must contain at least two scalar rewards")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Rollout rewards must be finite")
    centered = rewards.float() - rewards.float().mean()
    std = rewards.float().std(unbiased=False)
    if float(std.item()) <= eps:
        return torch.zeros_like(centered)
    return centered / (std + eps)


def greedy_relative_advantages(
    rewards: torch.Tensor, greedy_reward: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if rewards.ndim != 1 or greedy_reward.numel() != 1:
        raise ValueError("Greedy-relative advantages require vector rewards and one baseline")
    deltas = rewards.float() - greedy_reward.float()
    if not torch.isfinite(deltas).all():
        raise FloatingPointError("Greedy-relative reward deltas must be finite")
    return deltas / deltas.abs().mean().clamp_min(eps)


def sign_balanced_greedy_advantages(
    rewards: torch.Tensor, greedy_reward: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if rewards.ndim != 1 or greedy_reward.numel() != 1:
        raise ValueError("Sign-balanced advantages require vector rewards and one baseline")
    deltas = rewards.float() - greedy_reward.float()
    if not torch.isfinite(deltas).all():
        raise FloatingPointError("Sign-balanced reward deltas must be finite")
    positive = F.relu(deltas)
    negative = F.relu(-deltas)
    has_positive = float(positive.sum().item()) > eps
    has_negative = float(negative.sum().item()) > eps
    if has_positive and has_negative:
        half_mass = rewards.numel() / 2.0
        return half_mass * (
            positive / positive.sum().clamp_min(eps)
            - negative / negative.sum().clamp_min(eps)
        )
    if has_positive:
        return positive / positive.mean().clamp_min(eps)
    if has_negative:
        return -negative / negative.mean().clamp_min(eps)
    return torch.zeros_like(deltas)


def hierarchical_prefix_credit_advantages(
    rewards: torch.Tensor,
    greedy_reward: torch.Tensor,
    sampled_codes: list[list[int]],
    greedy_codes: list[int],
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    novelty_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Assign positive geometry gains to informative mask-code prefixes.

    The final decoded-mask reward remains the only target signal.  Prefixes
    receive more of a trajectory's positive gain when they are rare within
    the rollout group or diverge from the native greedy prefix.  This is a
    label-free hierarchical credit assignment: it changes only the detached
    PPO advantage, never the SAMTok decoder or the rollout reward.
    """
    if rewards.ndim != 1 or greedy_reward.numel() != 1:
        raise ValueError("Hierarchical credit requires vector rewards and one greedy reward")
    if len(sampled_codes) != rewards.numel() or not greedy_codes:
        raise ValueError("Hierarchical credit rollout/code shapes are inconsistent")
    depth = len(greedy_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Hierarchical credit requires complete code trajectories")
    if not 0.0 <= depth_decay <= 1.0 or not 0.0 <= novelty_weight <= 1.0:
        raise ValueError("Hierarchical credit weights must lie in [0, 1]")
    positive = F.relu(rewards.float() - greedy_reward.float() - minimum_improvement)
    if not bool((positive > 0).any().item()):
        return torch.zeros_like(positive)
    count = float(len(sampled_codes))
    prefix_scores = torch.ones(
        (len(sampled_codes), depth), dtype=torch.float32, device=rewards.device
    )
    for d in range(depth):
        prefixes = [tuple(codes[: d + 1]) for codes in sampled_codes]
        frequencies = {prefix: prefixes.count(prefix) for prefix in set(prefixes)}
        greedy_prefix = tuple(greedy_codes[: d + 1])
        for row, prefix in enumerate(prefixes):
            rarity = 1.0 - frequencies[prefix] / count
            diverges = float(prefix != greedy_prefix)
            novelty = novelty_weight * rarity + (1.0 - novelty_weight) * diverges
            prefix_scores[row, d] += (depth_decay**d) * novelty
    credit = positive * prefix_scores.mean(dim=1)
    active = credit > 0
    return credit / credit[active].mean().clamp_min(eps) if bool(active.any()) else credit


def pareto_geometry_improvement_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    minimum_improvement: float = 1e-4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Credit only rollouts that improve both geometry objectives.

    The two positive gains are combined by a geometric mean, so a large gain
    in cIoU cannot compensate for a regression in boundary IoU.  The returned
    values are detached-policy advantages and preserve the raw mask metrics as
    the only rollout supervision.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Pareto geometry requires rollout metrics shaped [N, 2]")
    if native_geometry.numel() != 2:
        raise ValueError("Pareto geometry requires one native [cIoU, boundary IoU] pair")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Pareto geometry metrics must be finite")
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    positive = F.relu(gains - minimum_improvement)
    active = (gains[:, 0] > minimum_improvement) & (gains[:, 1] > minimum_improvement)
    credit = torch.sqrt((positive[:, 0] * positive[:, 1]).clamp_min(0.0))
    credit = torch.where(active, credit, torch.zeros_like(credit))
    if not bool(active.any().item()):
        return torch.zeros(raw_geometry.shape[0], dtype=torch.float32, device=raw_geometry.device)
    return credit / credit[active].mean().clamp_min(eps)


def rank_pareto_geometry_advantages(
    raw_geometry: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Rank cIoU and boundary quality jointly within one rollout group.

    Unlike absolute-gain Pareto credit, this keeps near-miss trajectories in
    the learning signal: each axis is converted to a tie-aware empirical rank
    over the K sampled masks, then the geometric mean is standardized.  The
    operation is detached from the decoder and is therefore a pure policy
    credit allocation signal.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Rank-Pareto geometry requires rollout metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2:
        raise ValueError("Rank-Pareto geometry requires at least two rollouts")
    if not torch.isfinite(raw_geometry).all():
        raise FloatingPointError("Rank-Pareto geometry metrics must be finite")
    values = raw_geometry.float()
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    score = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    centered = score - score.mean()
    std = score.std(unbiased=False)
    if float(std.item()) <= eps:
        return torch.zeros_like(score)
    return centered / (std + eps)


def native_anchored_rank_pareto_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    *,
    minimum_improvement: float = 1e-4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rank sampled geometry against native greedy and suppress regressions.

    Native greedy is inserted as an explicit reference point on both axes.
    A sampled mask may receive positive credit only when it exceeds native on
    cIoU *and* boundary IoU; mixed or regressive candidates are forced to the
    non-positive side even if one axis is high.  This is a detached PPO credit
    signal and leaves SAMTok decoding and the target-free sentinel unchanged.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Native-anchored geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Native-anchored geometry requires K>=2 and one [2] baseline")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Native-anchored geometry metrics must be finite")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    baseline = scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    advantages = scores[1:] - baseline
    # Mixed/regressive masks are never rewarded for a single-axis win.
    advantages = torch.where(jointly_better, advantages, -advantages.abs())
    centered = advantages - advantages.mean()
    std = advantages.std(unbiased=False)
    if float(std.item()) <= eps:
        return torch.zeros_like(centered)
    return centered / (std + eps)


def depth_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    rarity_weight: float = 0.5,
    depth_permutation: list[int] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Allocate joint geometry gains to the first changed SAMTok depth.

    A complete mask reward is still computed from the decoded mask.  This
    detached credit signal only changes how a positive, jointly better sample
    is weighted: gains are attributed to the earliest code depth that differs
    from native greedy, with a deterministic rarity bonus for prefixes that
    are less frequent in the rollout group.  An optional permutation is used
    only for the depth-decay lookup by the R15 localization control; prefix
    rarity and the joint geometry gain remain tied to the original first
    divergence.  The resulting scalar is used by the existing grammar-action
    PPO objective, preserving the SAMTok-only decoder and preventing
    sequence-level gains from being copied uniformly across all code depths.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Depth-local geometry requires rollout metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Depth-local geometry requires K>=2 and one [cIoU, boundary] baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Depth-local geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Depth-local geometry requires complete code trajectories")
    if depth_permutation is not None:
        if len(depth_permutation) != depth or sorted(depth_permutation) != list(range(depth)):
            raise ValueError(
                "Depth-local geometry permutation must be a bijection over code depths"
            )
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0 or not 0.0 <= rarity_weight <= 1.0:
        raise ValueError("Depth-local geometry weights must lie in [0, 1]")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Depth-local geometry metrics must be finite")

    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    joint_gain = torch.sqrt(
        F.relu(gains[:, 0] - minimum_improvement)
        * F.relu(gains[:, 1] - minimum_improvement)
    )
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    credit = torch.zeros_like(joint_gain)
    count = float(len(sampled_codes))
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if not changed_depths or not bool(jointly_better[row].item()):
            continue
        first_depth = changed_depths[0]
        prefixes = [tuple(other[: first_depth + 1]) for other in sampled_codes]
        own_prefix = tuple(codes[: first_depth + 1])
        frequency = prefixes.count(own_prefix) / count
        rarity = 1.0 - frequency
        assigned_depth = (
            first_depth
            if depth_permutation is None
            else int(depth_permutation[first_depth])
        )
        locality = (depth_decay**assigned_depth) * (1.0 + rarity_weight * rarity)
        credit[row] = joint_gain[row] * locality
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def asymmetric_signed_native_relative_depth_local_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    beta: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Credit joint native-relative gains at the first code divergence.

    Positive credit requires improvement on both cIoU and boundary IoU. A
    weak fixed negative credit is assigned only when both axes regress;
    mixed-axis trade-offs and unchanged trajectories remain neutral. This is
    the preregistered asymmetric signed R20 rule.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Signed depth-local geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Signed depth-local geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Signed depth-local geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Signed depth-local geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isclose(float(beta), 0.25, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R20 fixes beta=0.25; beta sweeps are not registered")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Signed depth-local geometry metrics must be finite")

    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    signed_gain = (1.0 - float(beta)) * gains[:, 0] + float(beta) * gains[:, 1]
    better = (gains[:, 0] > minimum_improvement) & (gains[:, 1] > minimum_improvement)
    worse = (gains[:, 0] < -minimum_improvement) & (gains[:, 1] < -minimum_improvement)
    credit = torch.zeros_like(signed_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if not changed_depths:
            continue
        value = signed_gain[row]
        if not bool(better[row].item()) and not bool(worse[row].item()):
            continue
        if bool(worse[row].item()):
            value = float(beta) * value
        credit[row] = value * (depth_decay ** changed_depths[0])
    active = credit != 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    scale = credit[active].abs().mean().clamp_min(eps)
    return credit / scale


def native_anchored_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Use native-reference midranks, then localize positive credit by depth.

    Midranks retain the continuous sibling ordering advocated by grouped RL,
    while the native greedy mask is an explicit reference point.  Only masks
    that improve both cIoU and boundary IoU over that reference receive
    credit; the first changed SAMTok code receives the depth-decayed signal.
    The operation is detached policy credit and introduces no extra decoder or
    target.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Native rank-local geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Native rank-local geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Native rank-local geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Native rank-local geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Native rank-local geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            credit[row] = rank_gain[row] * (depth_decay ** changed_depths[0])
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def paired_view_native_rank_local_geometry_advantages(
    clean_geometry: torch.Tensor,
    augmented_geometry: torch.Tensor,
    clean_native_geometry: torch.Tensor,
    augmented_native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    augmented_native_codes: list[int] | None = None,
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Conservatively combine two GT-verified views of the same action.

    Each view retains R18 native-relative rank-local credit
    (``native_reference_midrank_first_divergence``).  The geometric
    mean makes a gain count only when it is useful on both views; no detached
    evidence score, teacher distribution, or second policy is introduced.

    ``native_codes`` and ``augmented_native_codes`` are view-specific greedy
    trajectories.  The latter is optional only for compatibility with older
    offline probes; runtime training always supplies it explicitly.
    """
    if augmented_native_codes is None:
        augmented_native_codes = native_codes
    clean = native_anchored_rank_local_geometry_advantages(
        clean_geometry,
        clean_native_geometry,
        sampled_codes,
        native_codes,
        minimum_improvement=minimum_improvement,
        depth_decay=depth_decay,
        eps=eps,
    )
    augmented = native_anchored_rank_local_geometry_advantages(
        augmented_geometry,
        augmented_native_geometry,
        sampled_codes,
        augmented_native_codes,
        minimum_improvement=minimum_improvement,
        depth_decay=depth_decay,
        eps=eps,
    )
    if clean.shape != augmented.shape:
        raise ValueError("Paired-view credits must have identical shapes")
    return torch.sqrt((clean.clamp_min(0.0) * augmented.clamp_min(0.0)).clamp_min(0.0))


def boundary_bottleneck_paired_view_geometry_advantages(
    clean_geometry: torch.Tensor,
    augmented_geometry: torch.Tensor,
    clean_native_geometry: torch.Tensor,
    augmented_native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    augmented_native_codes: list[int] | None = None,
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Use the weaker of clean/view rank-local credits as the bottleneck.

    BA-FEPO keeps the paired-view eligibility and first-divergence scope of
    R18, but replaces the geometric mean with a minimum.  A trajectory that
    improves only one view therefore cannot receive a large update.  Both
    inputs remain GT-verified geometry; no view is a teacher target.
    """
    clean = native_anchored_rank_local_geometry_advantages(
        clean_geometry,
        clean_native_geometry,
        sampled_codes,
        native_codes,
        minimum_improvement=minimum_improvement,
        depth_decay=depth_decay,
        eps=eps,
    )
    augmented = native_anchored_rank_local_geometry_advantages(
        augmented_geometry,
        augmented_native_geometry,
        sampled_codes,
        native_codes if augmented_native_codes is None else augmented_native_codes,
        minimum_improvement=minimum_improvement,
        depth_decay=depth_decay,
        eps=eps,
    )
    if clean.shape != augmented.shape:
        raise ValueError("Boundary-bottleneck credits must have identical shapes")
    return torch.minimum(clean.clamp_min(0.0), augmented.clamp_min(0.0))


def native_rank_signed_depth_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R31: native rank gain with signed credit for jointly-worse masks.

    The native greedy mask anchors a two-axis midrank.  Jointly-better and
    jointly-worse trajectories receive the signed rank displacement, localized
    to their first changed code depth; mixed-axis trade-offs stay neutral.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Native signed rank geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Native signed rank geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Native signed rank geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Native signed rank geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Native signed rank geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    better = (gains[:, 0] > minimum_improvement) & (gains[:, 1] > minimum_improvement)
    worse = (gains[:, 0] < -minimum_improvement) & (gains[:, 1] < -minimum_improvement)
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed = [d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native]
        if not changed or not bool((better[row] | worse[row]).item()):
            continue
        credit[row] = rank_gain[row] * (depth_decay ** changed[0])
    active = credit != 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].abs().mean().clamp_min(eps)


def soft_native_dominance_depth_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    temperature: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R34: continuous native-relative dominance with local credit.

    A fixed-temperature softplus maps each detached geometry gain to a smooth,
    strictly monotone dominance score.  Only trajectories that improve both
    native cIoU and boundary IoU receive credit; the first changed SAMTok
    depth scopes the update.  This removes the K=4 midrank quantization without
    adding a teacher, router, or second objective.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Soft native dominance requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Soft native dominance requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Soft native dominance rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Soft native dominance requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Soft native dominance metrics must be finite")

    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    # Center at the improvement gate so that the smooth score remains
    # geometry-monotone even when the temperature is small.  A tanh saturates
    # both gains in the common regime and can let depth decay reverse ranking.
    scaled = (gains - float(minimum_improvement)) / float(temperature)
    dominance = torch.sqrt(
        F.softplus(scaled[:, 0]).clamp_min(0.0)
        * F.softplus(scaled[:, 1]).clamp_min(0.0)
    )
    credit = torch.zeros_like(dominance)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            credit[row] = dominance[row] * (depth_decay ** changed_depths[0])
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def uncertainty_calibrated_native_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    uncertainty: torch.Tensor,
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    confidence_floor: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R25: uncertainty-calibrated native rank-local geometry credit.

    Geometry remains the sole training reward.  A detached uncertainty score
    from the calibrated grammar rollout (entropy plus missing top-support
    mass) only rescales masks that already improve both native cIoU and
    boundary IoU.  This is a confidence calibration of R18 credit, not a
    teacher, OPD target, synthetic label, or second policy.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Uncertainty native rank-local geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Uncertainty native rank-local geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Uncertainty native geometry rollout/code shapes are inconsistent")
    if uncertainty.ndim != 1 or uncertainty.numel() != raw_geometry.shape[0]:
        raise ValueError("Uncertainty scores must have shape [N]")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Uncertainty native rank-local geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isfinite(confidence_floor) or not 0.0 < confidence_floor <= 1.0:
        raise ValueError("confidence_floor must lie in (0, 1]")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Uncertainty native geometry metrics must be finite")
    if not torch.isfinite(uncertainty).all():
        raise FloatingPointError("Uncertainty scores must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    # Entropy/missing-mass uncertainty is clipped to [0, 1] by its caller.
    # The floor prevents high-uncertainty but valid improvements from vanishing.
    confidence = (1.0 - uncertainty.float()).clamp(
        min=float(confidence_floor), max=1.0
    )
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            credit[row] = (
                rank_gain[row]
                * confidence[row]
                * (depth_decay ** changed_depths[0])
            )
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def action_budget_native_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    action_budget: int = 2,
    excess_penalty: float = 0.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """AB-FEPO: soft fixed action-budget credit over native rank-local gains.

    Action count is the number of SAMTok code depths changed from the native
    trajectory. Joint geometry improvements retain R18 credit, while edits
    beyond the fixed budget are softly discounted. All measurements are
    detached rollout/code metadata; this is not an inference router or a
    second policy objective.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Action-budget geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Action-budget geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Action-budget rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Action-budget geometry requires complete code trajectories")
    if not isinstance(action_budget, int) or not 0 <= action_budget <= depth:
        raise ValueError("action_budget must be an integer in [0, depth]")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isfinite(excess_penalty) or excess_penalty < 0.0:
        raise ValueError("excess_penalty must be finite and nonnegative")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Action-budget geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            excess = max(0, len(changed_depths) - action_budget)
            budget_scale = 1.0 / (1.0 + float(excess_penalty) * float(excess))
            credit[row] = rank_gain[row] * budget_scale * (
                depth_decay ** changed_depths[0]
            )
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def margin_calibrated_native_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    margin_power: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R28: continuous joint-geometry margin calibration.

    Native tie-aware rank remains the ordering signal.  A detached geometric
    margin, the geometric mean of cIoU and boundary gains, downweights brittle
    near-threshold wins while preserving first-divergence localization.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Margin-calibrated geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Margin-calibrated geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Margin-calibrated geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Margin-calibrated geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isfinite(margin_power) or margin_power <= 0.0:
        raise ValueError("margin_power must be finite and positive")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Margin-calibrated geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    joint_margin = (
        gains[:, 0].clamp_min(0.0) * gains[:, 1].clamp_min(0.0)
    ).pow(float(margin_power))
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            credit[row] = rank_gain[row] * joint_margin[row] * (
                depth_decay ** changed_depths[0]
            )
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def confidence_gated_native_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    uncertainty: torch.Tensor,
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    confidence_threshold: float = 0.5,
    confidence_floor: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R27: hard confidence gate over native rank-local geometry credit.

    The detached calibration uncertainty is used only as an abstention gate:
    a rollout must have confidence at or above the fixed threshold to receive
    geometry credit.  Geometry remains the sole reward and the gate cannot
    create labels, a teacher, a router, or a second training objective.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Confidence-gated geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Confidence-gated geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Confidence-gated geometry rollout/code shapes are inconsistent")
    if uncertainty.ndim != 1 or uncertainty.numel() != raw_geometry.shape[0]:
        raise ValueError("Confidence scores must have shape [N]")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Confidence-gated geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not math.isfinite(confidence_threshold) or not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in (0, 1]")
    if not math.isfinite(confidence_floor) or not 0.0 < confidence_floor <= 1.0:
        raise ValueError("confidence_floor must lie in (0, 1]")
    if confidence_floor > confidence_threshold:
        raise ValueError("confidence_floor cannot exceed confidence_threshold")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Confidence-gated geometry metrics must be finite")
    if not torch.isfinite(uncertainty).all():
        raise FloatingPointError("Confidence scores must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    scores = torch.sqrt((ranks[0] * ranks[1]).clamp_min(0.0))
    rank_gain = scores[1:] - scores[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    confidence = (1.0 - uncertainty.float()).clamp(
        min=float(confidence_floor), max=1.0
    )
    confident = confidence >= float(confidence_threshold)
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()) and bool(confident[row].item()):
            credit[row] = rank_gain[row] * (depth_decay ** changed_depths[0])
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def calibrated_rollout_uncertainty(
    controlled_entropies: torch.Tensor,
    top_support_masses: torch.Tensor,
    *,
    support_size: int = 8,
) -> torch.Tensor:
    """Reduce detached per-depth calibration diagnostics to one [0,1] score."""
    if controlled_entropies.ndim != 2 or top_support_masses.shape != controlled_entropies.shape:
        raise ValueError("Calibration diagnostics must both have shape [N, depth]")
    if int(support_size) < 2:
        raise ValueError("support_size must be at least two")
    if not torch.isfinite(controlled_entropies).all() or not torch.isfinite(top_support_masses).all():
        raise FloatingPointError("Calibration diagnostics must be finite")
    entropy = controlled_entropies.float().clamp_min(0.0) / math.log(float(support_size))
    missing_mass = (1.0 - top_support_masses.float()).clamp(0.0, 1.0)
    score = 0.5 * entropy.clamp(0.0, 1.0) + 0.5 * missing_mass
    return score.mean(dim=1).clamp(0.0, 1.0).detach()


def scale_stratified_native_rank_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    area_stratum: str,
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    axis_weights: tuple[float, float] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rank native-relative geometry within a training-only area stratum.

    The sampled masks are sibling rollouts for one target, so their ranks are
    computed conditionally on that target's registered area stratum.  Small
    targets use a boundary-heavy axis mix, large targets use cIoU-heavy mix,
    and medium targets remain balanced.  Only joint improvements over native
    receive the first-divergence depth-local credit.  All inputs are detached
    rollout measurements; this function cannot create a second decoder or a
    PixVL training path.
    """
    if area_stratum not in {"small", "medium", "large"}:
        raise ValueError("area_stratum must be small, medium, or large")
    expected_weights = {
        "small": (0.35, 0.65),
        "medium": (0.50, 0.50),
        "large": (0.65, 0.35),
    }[area_stratum]
    weights = expected_weights if axis_weights is None else tuple(float(x) for x in axis_weights)
    if len(weights) != 2 or not all(math.isfinite(x) and x > 0.0 for x in weights):
        raise ValueError("area rank weights must be finite and positive")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("area rank weights must sum to one")
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(weights, expected_weights)):
        raise ValueError("area rank weights are fixed by the registered area stratum")
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Scale-stratified geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Scale-stratified geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Scale-stratified geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Scale-stratified geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Scale-stratified geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    score = weights[0] * ranks[0] + weights[1] * ranks[1]
    rank_gain = score[1:] - score[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (
        gains[:, 1] > minimum_improvement
    )
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if changed_depths and bool(jointly_better[row].item()):
            credit[row] = rank_gain[row] * (depth_decay ** changed_depths[0])
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def bidirectional_coarse_fine_native_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    coarse_weight: float = 0.5,
    fine_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Credit native-relative geometry at both coarse and fine divergences.

    Early SAMTok codes provide coarse extent decisions while late codes refine
    boundaries.  For a rollout that improves both native cIoU and boundary
    IoU, its native-reference rank gain is weighted by the first changed depth
    (coarse) and the last changed depth (fine).  This is one scalar detached
    advantage for the existing PPO objective, not a router or second policy.
    Mixed/regressive geometry receives no credit and unchanged trajectories
    remain neutral.
    """
    if raw_geometry.ndim != 2 or raw_geometry.shape[1] != 2:
        raise ValueError("Bidirectional geometry requires metrics shaped [N, 2]")
    if raw_geometry.shape[0] < 2 or native_geometry.numel() != 2:
        raise ValueError("Bidirectional geometry requires K>=2 and one baseline")
    if len(sampled_codes) != raw_geometry.shape[0] or not native_codes:
        raise ValueError("Bidirectional geometry rollout/code shapes are inconsistent")
    depth = len(native_codes)
    if any(len(codes) != depth for codes in sampled_codes):
        raise ValueError("Bidirectional geometry requires complete code trajectories")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and nonnegative")
    if not 0.0 <= depth_decay <= 1.0:
        raise ValueError("depth_decay must lie in [0, 1]")
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (coarse_weight, fine_weight)):
        raise ValueError("coarse/fine weights must be finite and positive")
    if not math.isclose(float(coarse_weight + fine_weight), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("coarse/fine weights must sum to one")
    if not torch.isfinite(raw_geometry).all() or not torch.isfinite(native_geometry).all():
        raise FloatingPointError("Bidirectional geometry metrics must be finite")

    values = torch.cat((native_geometry.float().reshape(1, 2), raw_geometry.float()), dim=0)
    ranks: list[torch.Tensor] = []
    denominator = float(values.shape[0])
    for column in range(2):
        axis = values[:, column]
        less = (axis[:, None] > axis[None, :]).sum(dim=1).float()
        equal = (axis[:, None] == axis[None, :]).sum(dim=1).float()
        ranks.append((less + 0.5 * equal) / denominator)
    rank_gain = (0.5 * (ranks[0] + ranks[1]))[1:] - (0.5 * (ranks[0] + ranks[1]))[0]
    gains = raw_geometry.float() - native_geometry.float().reshape(1, 2)
    jointly_better = (gains[:, 0] > minimum_improvement) & (gains[:, 1] > minimum_improvement)
    credit = torch.zeros_like(rank_gain)
    for row, codes in enumerate(sampled_codes):
        changed_depths = [
            d for d, (code, native) in enumerate(zip(codes, native_codes)) if code != native
        ]
        if not changed_depths or not bool(jointly_better[row].item()):
            continue
        first_depth, last_depth = changed_depths[0], changed_depths[-1]
        coarse_locality = float(coarse_weight) * (depth_decay**first_depth)
        fine_locality = float(fine_weight) * (depth_decay ** (depth - 1 - last_depth))
        credit[row] = rank_gain[row] * (coarse_locality + fine_locality)
    active = credit > 0
    if not bool(active.any().item()):
        return torch.zeros_like(credit)
    return credit / credit[active].mean().clamp_min(eps)


def shuffled_depth_local_geometry_advantages(
    raw_geometry: torch.Tensor,
    native_geometry: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    minimum_improvement: float = 1e-4,
    depth_decay: float = 0.85,
    rarity_weight: float = 0.5,
    shuffle_seed: int = 20260827,
    eps: float = 1e-6,
) -> torch.Tensor:
    """R15 control: retain R14 gains and rarity while cyclically shuffling depth.

    The offset is deterministic and nonzero whenever more than one code depth
    exists.  This breaks the causal interpretation of earliest-divergence
    depth without changing which rollouts are jointly better or how prefix
    rarity is measured.
    """
    if not isinstance(shuffle_seed, int):
        raise TypeError("Depth shuffle seed must be an integer")
    depth = len(native_codes)
    if depth < 1:
        raise ValueError("Depth shuffle requires at least one native code")
    offset = shuffle_seed % depth
    if depth > 1 and offset == 0:
        offset = 1
    permutation = [(index + offset) % depth for index in range(depth)]
    return depth_local_geometry_advantages(
        raw_geometry,
        native_geometry,
        sampled_codes,
        native_codes,
        minimum_improvement=minimum_improvement,
        depth_decay=depth_decay,
        rarity_weight=rarity_weight,
        depth_permutation=permutation,
        eps=eps,
    )


def clipped_policy_loss(
    current_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        current_log_probs.shape != behavior_log_probs.shape
        or current_log_probs.shape != advantages.shape
    ):
        raise ValueError("GR-CPPO policy tensors must have identical shapes")
    if behavior_log_probs.requires_grad:
        raise ValueError("Behavior log-probabilities must be detached")
    ratio = torch.exp(current_log_probs - behavior_log_probs)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("GR-CPPO importance ratios must be finite")
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    clip_fraction = (ratio != clipped_ratio).to(torch.float32).mean()
    return -surrogate.mean(), ratio, clip_fraction


def predicted_evidence_scope_masks(
    controlled_entropies: torch.Tensor,
    top_support_masses: torch.Tensor,
    sampled_codes: list[list[int]],
    native_codes: list[int],
    *,
    native_margins: torch.Tensor,
    support_size: int = 8,
    confident_entropy: float = 0.35,
    ambiguous_entropy: float = 0.70,
    confident_margin: float = 0.70,
    ambiguous_margin: float = 0.25,
    shuffle_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed predicted-only local credit scope for PES-FEPO.

    Evidence is detached sampler information. State 0 is confident (one
    first-divergence depth), state 1 is ambiguous (the first two changed
    depths), and state 2 is unsupported (no positive scope). The reward itself
    remains the native-relative geometry reward; this function only gates the
    action log-probability terms that receive it.
    """
    if controlled_entropies.ndim != 2 or top_support_masses.shape != controlled_entropies.shape:
        raise ValueError("PES evidence tensors must both have shape [N, depth]")
    if native_margins.shape != controlled_entropies.shape:
        raise ValueError("PES native margin tensor must match entropy shape")
    if len(sampled_codes) != controlled_entropies.shape[0] or not native_codes:
        raise ValueError("PES evidence/code shapes are inconsistent")
    depth = len(native_codes)
    if controlled_entropies.shape[1] != depth or any(len(row) != depth for row in sampled_codes):
        raise ValueError("PES evidence depth does not match SAMTok grammar")
    if int(support_size) < 2:
        raise ValueError("PES support_size must be at least two")
    thresholds = (
        confident_entropy,
        ambiguous_entropy,
        confident_margin,
        ambiguous_margin,
    )
    if not all(math.isfinite(float(x)) for x in thresholds):
        raise ValueError("PES evidence thresholds must be finite")
    if not (0.0 < confident_entropy < ambiguous_entropy <= 1.0):
        raise ValueError("PES entropy thresholds must be ordered in (0, 1]")
    if confident_margin <= ambiguous_margin:
        raise ValueError("PES margin thresholds must be ordered")
    if not torch.isfinite(controlled_entropies).all() or not torch.isfinite(top_support_masses).all() or not torch.isfinite(native_margins).all():
        raise FloatingPointError("PES evidence tensors must be finite")
    normalized_entropy = controlled_entropies.float().clamp_min(0.0) / math.log(float(support_size))
    margin = native_margins.float().clamp_min(0.0)
    confident = (normalized_entropy.mean(dim=1) < confident_entropy) & (
        margin.mean(dim=1) >= confident_margin
    )
    ambiguous = (~confident) & (
        (normalized_entropy.mean(dim=1) < ambiguous_entropy)
        | (margin.mean(dim=1) >= ambiguous_margin)
    )
    states = torch.full(
        (len(sampled_codes),), 2, dtype=torch.long, device=controlled_entropies.device
    )
    states[ambiguous] = 1
    states[confident] = 0
    if shuffle_seed is not None:
        generator = torch.Generator(device=controlled_entropies.device)
        generator.manual_seed(int(shuffle_seed))
        states = states[torch.randperm(len(states), generator=generator, device=states.device)]
    masks = torch.zeros_like(controlled_entropies, dtype=torch.float32)
    for row, codes in enumerate(sampled_codes):
        changed = [idx for idx, (code, native) in enumerate(zip(codes, native_codes)) if code != native]
        if not changed or int(states[row].item()) == 2:
            continue
        width = 1 if int(states[row].item()) == 0 else 2
        masks[row, changed[:width]] = 1.0
    return masks.detach(), states.detach()


def clipped_scope_policy_loss(
    current_action_log_probs: torch.Tensor,
    behavior_action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO loss over the detached evidence-selected code-depth scope."""
    if current_action_log_probs.shape != behavior_action_log_probs.shape:
        raise ValueError("PES current/behavior action tensors must match")
    if current_action_log_probs.ndim != 2 or action_mask.shape != current_action_log_probs.shape:
        raise ValueError("PES action tensors and mask must have shape [batch, depth]")
    if advantages.ndim != 1 or advantages.shape[0] != current_action_log_probs.shape[0]:
        raise ValueError("PES advantages must have shape [batch]")
    if behavior_action_log_probs.requires_grad or action_mask.requires_grad:
        raise ValueError("PES behavior probabilities and scope mask must be detached")
    ratio = torch.exp(current_action_log_probs - behavior_action_log_probs)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("PES importance ratios must be finite")
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    token_advantages = advantages.detach()[:, None] * action_mask
    surrogate = torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
    active = action_mask.sum().clamp_min(1.0)
    clip_fraction = ((ratio != clipped_ratio).to(torch.float32) * action_mask).sum() / active
    return -surrogate.sum() / active, ratio, clip_fraction


def _last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().sum(dim=1) - 1
    if bool((positions < 0).any()):
        raise RuntimeError("Cannot sample from an empty prompt")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, positions]


def append_tokens(
    inputs: dict[str, torch.Tensor], token_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    if token_ids.ndim == 1:
        token_ids = token_ids[:, None]
    if token_ids.ndim != 2 or token_ids.shape[0] != inputs["input_ids"].shape[0]:
        raise ValueError("Appended tokens must have shape [batch] or [batch, tokens]")
    attention = inputs["attention_mask"]
    if not bool(attention.bool().all()):
        raise ValueError("GR-CPPO append requires an unpadded, equal-length prompt batch")
    result = dict(inputs)
    result["input_ids"] = torch.cat(
        (inputs["input_ids"], token_ids.to(inputs["input_ids"].device)), dim=1
    )
    result["attention_mask"] = torch.cat(
        (
            attention,
            torch.ones(
                token_ids.shape,
                dtype=attention.dtype,
                device=attention.device,
            ),
        ),
        dim=1,
    )
    return result


def build_rollout_prompt_batch(
    processor: Any,
    sample: dict[str, Any],
    rollouts_per_prompt: int,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    if rollouts_per_prompt < 1:
        raise ValueError("rollouts_per_prompt must be positive")
    prompt = render_chat(processor, sample["prompt_text"], None)
    encoded = processor(
        text=[prompt] * rollouts_per_prompt,
        images=[sample["image"]] * rollouts_per_prompt,
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: value for name, value in encoded.items() if isinstance(value, torch.Tensor)}
    if not bool(inputs["attention_mask"].bool().all()):
        raise RuntimeError("Repeated identical rollout prompts unexpectedly contain padding")
    return inputs if device is None else move_tensors(inputs, torch.device(device))


def _valid_token_id(tokenizer: Any, token: str) -> int:
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    if token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"SAMTok grammar token is unavailable: {token}")
    return token_id


def _mask_grammar_ids(
    tokenizer: Any, codec: SAMTokMaskCodec
) -> tuple[int, list[list[int]], int]:
    start_id = _valid_token_id(tokenizer, codec.catalog["start"])
    end_id = _valid_token_id(tokenizer, codec.catalog["end"])
    by_depth: list[list[int]] = []
    for depth in range(codec.codebook_depth):
        candidates: list[int] = []
        first_code = depth * codec.codebook_size
        for code in range(first_code, first_code + codec.codebook_size):
            token = codec.catalog["index_to_token"].get(code)
            if token is None:
                raise ValueError(f"SAMTok catalog is missing mask code {code}")
            candidates.append(_valid_token_id(tokenizer, token))
        by_depth.append(candidates)
    return start_id, by_depth, end_id


def _grammar_step_log_prob(
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    candidates: list[int] | None,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(device=logits.device, dtype=torch.float32)
        if temperature.ndim == 0:
            scaled = logits.float() / temperature
        elif temperature.ndim == 1 and temperature.shape[0] == logits.shape[0]:
            scaled = logits.float() / temperature[:, None]
        else:
            raise ValueError("Action temperatures must be scalar or shape [batch]")
    else:
        scaled = logits.float() / float(temperature)
    if candidates is None:
        return F.log_softmax(scaled, dim=-1).gather(1, selected_ids[:, None]).squeeze(1)
    candidate_ids = torch.tensor(candidates, dtype=torch.long, device=logits.device)
    candidate_logits = scaled.index_select(1, candidate_ids)
    matches = candidate_ids[None, :] == selected_ids[:, None]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("Selected code token is outside its depth-specific grammar")
    local = matches.to(torch.int64).argmax(dim=1)
    return F.log_softmax(candidate_logits, dim=-1).gather(1, local[:, None]).squeeze(1)


def categorical_entropy(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(device=logits.device, dtype=torch.float32)
        scaled = logits.float() / (temperature if temperature.ndim == 0 else temperature[:, None])
    else:
        scaled = logits.float() / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def collision_effective_support(
    logits: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(device=logits.device, dtype=torch.float32)
        scaled = logits.float() / (temperature if temperature.ndim == 0 else temperature[:, None])
    else:
        scaled = logits.float() / float(temperature)
    probabilities = F.softmax(scaled, dim=-1)
    return (probabilities.square().sum(dim=-1)).reciprocal()


def calibrate_temperatures_to_effective_support(
    logits: torch.Tensor,
    target_effective_support: float,
    temperature_min: float,
    temperature_max: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Effective-support calibration requires [batch, candidates] logits")
    low = torch.full(
        (logits.shape[0],), temperature_min, dtype=torch.float32, device=logits.device
    )
    high = torch.full_like(low, temperature_max)
    support_at_min = collision_effective_support(logits, low)
    support_at_max = collision_effective_support(logits, high)
    for _ in range(iterations):
        midpoint = (low + high) * 0.5
        effective_support = collision_effective_support(logits, midpoint)
        below_target = effective_support < target_effective_support
        low = torch.where(below_target, midpoint, low)
        high = torch.where(below_target, high, midpoint)
    calibrated = (low + high) * 0.5
    calibrated = torch.where(
        support_at_min >= target_effective_support,
        torch.full_like(calibrated, temperature_min),
        calibrated,
    )
    calibrated = torch.where(
        support_at_max < target_effective_support,
        torch.full_like(calibrated, temperature_max),
        calibrated,
    )
    achieved = collision_effective_support(logits, calibrated)
    return calibrated.detach(), support_at_min.detach(), achieved.detach()


def _supported_step_log_prob(
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    support_token_ids: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    if support_token_ids.ndim != 2 or support_token_ids.shape[0] != logits.shape[0]:
        raise ValueError("Frozen support must have shape [batch, support]")
    support_logits = logits.float().gather(1, support_token_ids)
    log_probs = F.log_softmax(support_logits / temperatures[:, None], dim=-1)
    matches = support_token_ids == selected_ids[:, None]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("Sampled token is outside its frozen old-policy support")
    local = matches.to(torch.int64).argmax(dim=1)
    return log_probs.gather(1, local[:, None]).squeeze(1)


def sample_grammar_rollouts(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    grammar_ids: tuple[int, list[list[int]], int],
    temperature: float,
) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
    start_id, code_token_ids, end_id = grammar_ids
    batch_size = prompt_inputs["input_ids"].shape[0]
    working = prompt_inputs
    sequence: list[torch.Tensor] = []
    log_prob_terms: list[torch.Tensor] = []
    sampled_codes: list[list[int]] = [[] for _ in range(batch_size)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for action_index in range(len(code_token_ids) + 2):
            outputs = model(**working, use_cache=False)
            next_logits = _last_token_logits(outputs.logits, working["attention_mask"])
            if action_index == 0:
                selected = torch.full(
                    (batch_size,), start_id, dtype=torch.long, device=next_logits.device
                )
                candidates = None
            elif action_index == len(code_token_ids) + 1:
                selected = torch.full(
                    (batch_size,), end_id, dtype=torch.long, device=next_logits.device
                )
                candidates = None
            else:
                depth = action_index - 1
                candidates = code_token_ids[depth]
                candidate_tensor = torch.tensor(
                    candidates, dtype=torch.long, device=next_logits.device
                )
                distribution = torch.distributions.Categorical(
                    logits=next_logits.index_select(1, candidate_tensor).float() / temperature
                )
                local_ids = distribution.sample()
                selected = candidate_tensor[local_ids]
                for row, local_id in enumerate(local_ids.tolist()):
                    sampled_codes[row].append(depth * len(candidates) + int(local_id))
            # Boundary tokens are forced by the grammar. Their probability is
            # one under the constrained policy and must not enter the PPO
            # importance ratio; only depth-specific code actions are sampled.
            if candidates is not None:
                log_prob_terms.append(
                    _grammar_step_log_prob(next_logits, selected, candidates, temperature)
                )
            sequence.append(selected)
            working = append_tokens(working, selected)
    if was_training:
        model.train()
    token_ids = torch.stack(sequence, dim=1)
    behavior_log_probs = torch.stack(log_prob_terms, dim=1).sum(dim=1).detach()
    if behavior_log_probs.requires_grad:
        raise RuntimeError("Behavior log-probabilities were not detached")
    return token_ids, sampled_codes, behavior_log_probs


def sample_effective_support_grammar_rollouts(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    grammar_ids: tuple[int, list[list[int]], int],
    *,
    support_size: int,
    target_effective_support: float,
    temperature_min: float,
    temperature_max: float,
    calibration_iterations: int,
) -> tuple[
    torch.Tensor,
    list[list[int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    start_id, code_token_ids, end_id = grammar_ids
    batch_size = prompt_inputs["input_ids"].shape[0]
    working = prompt_inputs
    sequence: list[torch.Tensor] = []
    log_prob_terms: list[torch.Tensor] = []
    action_temperatures: list[torch.Tensor] = []
    action_supports: list[torch.Tensor] = []
    native_effective_supports: list[torch.Tensor] = []
    calibrated_effective_supports: list[torch.Tensor] = []
    native_entropies: list[torch.Tensor] = []
    controlled_entropies: list[torch.Tensor] = []
    top_support_masses: list[torch.Tensor] = []
    native_margins: list[torch.Tensor] = []
    controlled_native_kls: list[torch.Tensor] = []
    sampled_codes: list[list[int]] = [[] for _ in range(batch_size)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for action_index in range(len(code_token_ids) + 2):
            outputs = model(**working, use_cache=False)
            next_logits = _last_token_logits(outputs.logits, working["attention_mask"])
            if action_index == 0:
                selected = torch.full(
                    (batch_size,), start_id, dtype=torch.long, device=next_logits.device
                )
                candidates = None
                temperatures = None
                support_ids = None
            elif action_index == len(code_token_ids) + 1:
                selected = torch.full(
                    (batch_size,), end_id, dtype=torch.long, device=next_logits.device
                )
                candidates = None
                temperatures = None
                support_ids = None
            else:
                depth = action_index - 1
                candidates = code_token_ids[depth]
                candidate_tensor = torch.tensor(
                    candidates, dtype=torch.long, device=next_logits.device
                )
                candidate_logits = next_logits.index_select(1, candidate_tensor).float()
                # Stable argsort makes token-id order the tie breaker because
                # candidate_tensor follows increasing code index.
                support_local = torch.argsort(
                    candidate_logits, dim=-1, descending=True, stable=True
                )[:, :support_size]
                support_ids = candidate_tensor[support_local]
                support_logits = candidate_logits.gather(1, support_local)
                temperatures, native_support, calibrated_support = (
                    calibrate_temperatures_to_effective_support(
                        support_logits,
                        target_effective_support,
                        temperature_min,
                        temperature_max,
                        calibration_iterations,
                    )
                )
                distribution = torch.distributions.Categorical(
                    logits=support_logits / temperatures[:, None]
                )
                support_choice = distribution.sample()
                selected = support_ids.gather(1, support_choice[:, None]).squeeze(1)
                selected_local = support_local.gather(1, support_choice[:, None]).squeeze(1)
                # Compare the native action with the code sampled by this
                # rollout. The tensor is detached with the surrounding
                # no-grad rollout and is used only for PES scope selection.
                native_logit = candidate_logits.max(dim=-1).values
                # ``selected`` contains a vocabulary token id, while
                # ``candidate_logits`` is indexed by the depth-local code
                # position.  Use the local index to avoid out-of-range
                # gathers whenever the code vocabulary is offset or the
                # calibrated support is smaller than the full vocabulary.
                sampled_logit = candidate_logits.gather(1, selected_local[:, None]).squeeze(1)
                native_margins.append(native_logit - sampled_logit)
                action_temperatures.append(temperatures)
                action_supports.append(support_ids)
                native_effective_supports.append(native_support)
                calibrated_effective_supports.append(calibrated_support)
                native_log_probs = F.log_softmax(candidate_logits, dim=-1)
                controlled_log_probs = F.log_softmax(
                    support_logits / temperatures[:, None], dim=-1
                )
                controlled_probs = controlled_log_probs.exp()
                support_native_log_probs = native_log_probs.gather(1, support_local)
                native_entropies.append(
                    -(native_log_probs.exp() * native_log_probs).sum(dim=-1)
                )
                controlled_entropies.append(
                    -(controlled_probs * controlled_log_probs).sum(dim=-1)
                )
                top_support_masses.append(support_native_log_probs.exp().sum(dim=-1))
                controlled_native_kls.append(
                    (
                        controlled_probs
                        * (controlled_log_probs - support_native_log_probs)
                    ).sum(dim=-1)
                )
                for row, local_id in enumerate(selected_local.tolist()):
                    sampled_codes[row].append(depth * len(candidates) + int(local_id))
            if candidates is not None and temperatures is not None and support_ids is not None:
                log_prob_terms.append(
                    _supported_step_log_prob(
                        next_logits, selected, support_ids, temperatures
                    )
                )
            sequence.append(selected)
            working = append_tokens(working, selected)
    if was_training:
        model.train()
    token_ids = torch.stack(sequence, dim=1)
    behavior_log_probs = torch.stack(log_prob_terms, dim=1).sum(dim=1).detach()
    temperatures_by_action = torch.stack(action_temperatures, dim=1).detach()
    supports_by_action = torch.stack(action_supports, dim=1).detach()
    return (
        token_ids,
        sampled_codes,
        behavior_log_probs,
        temperatures_by_action,
        supports_by_action,
        torch.stack(native_effective_supports, dim=1).detach(),
        torch.stack(calibrated_effective_supports, dim=1).detach(),
        torch.stack(native_entropies, dim=1).detach(),
        torch.stack(controlled_entropies, dim=1).detach(),
        torch.stack(top_support_masses, dim=1).detach(),
        torch.stack(native_margins, dim=1).detach(),
        torch.stack(controlled_native_kls, dim=1).detach(),
    )


def greedy_grammar_codes(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    grammar_ids: tuple[int, list[list[int]], int],
) -> list[int]:
    start_id, code_token_ids, end_id = grammar_ids
    if prompt_inputs["input_ids"].shape[0] != 1:
        raise ValueError("Native greedy reference requires exactly one prompt")
    working = prompt_inputs
    codes: list[int] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for action_index in range(len(code_token_ids) + 2):
            outputs = model(**working, use_cache=False)
            next_logits = _last_token_logits(outputs.logits, working["attention_mask"])
            if action_index == 0:
                selected = torch.tensor([start_id], device=next_logits.device)
            elif action_index == len(code_token_ids) + 1:
                selected = torch.tensor([end_id], device=next_logits.device)
            else:
                depth = action_index - 1
                candidates = code_token_ids[depth]
                candidate_ids = torch.tensor(
                    candidates, dtype=torch.long, device=next_logits.device
                )
                local = int(
                    next_logits.index_select(1, candidate_ids).argmax(dim=1).item()
                )
                selected = candidate_ids[local : local + 1]
                codes.append(depth * len(candidates) + local)
            working = append_tokens(working, selected)
    if was_training:
        model.train()
    return codes


def collect_anchor_policy_logits(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    grammar_ids: tuple[int, list[list[int]], int],
    device: torch.device | str,
    *,
    native_codes: list[int] | None = None,
    no_grad: bool = False,
) -> tuple[list[torch.Tensor], list[int]]:
    """Collect depth-wise codebook logits along a fixed native trajectory.

    The anchor trajectory is generated once before optimization.  Subsequent
    calls use those detached codes, making the KL a policy-drift constraint and
    not a teacher or target-prediction objective.
    """
    start_id, code_token_ids, _end_id = grammar_ids
    working = build_rollout_prompt_batch(processor, sample, 1, device)
    selected_codes: list[int] = []
    values: list[torch.Tensor] = []

    def run() -> tuple[list[torch.Tensor], list[int]]:
        nonlocal working
        working = append_tokens(
            working,
            torch.tensor([start_id], dtype=torch.long, device=working["input_ids"].device),
        )
        for depth, candidates in enumerate(code_token_ids):
            outputs = model(**working, use_cache=False)
            logits = _last_token_logits(outputs.logits, working["attention_mask"])
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=logits.device
            )
            candidate_logits = logits.index_select(1, candidate_tensor).float()
            values.append(candidate_logits.squeeze(0))
            if native_codes is None:
                local = int(candidate_logits.argmax(dim=1).item())
                selected_codes.append(depth * len(candidates) + local)
            else:
                if len(native_codes) != len(code_token_ids):
                    raise ValueError("Anchor native code depth does not match SAMTok grammar")
                code = int(native_codes[depth])
                local = code - depth * len(candidates)
                if local < 0 or local >= len(candidates):
                    raise ValueError("Anchor native code lies outside its depth vocabulary")
                selected_codes.append(code)
                local = int(local)
            working = append_tokens(
                working,
                candidate_tensor[local].reshape(1),
            )
        return values, selected_codes
    was_training = model.training
    model.eval()
    try:
        if no_grad:
            with torch.no_grad():
                return run()
        return run()
    finally:
        if was_training:
            model.train()


def anchor_categorical_kl(
    current_logits: list[torch.Tensor], anchor_logits: list[torch.Tensor]
) -> torch.Tensor:
    """Mean categorical KL(current || detached frozen anchor) over code depths."""
    if not current_logits or len(current_logits) != len(anchor_logits):
        raise ValueError("Anchor KL requires matching non-empty depth logits")
    values: list[torch.Tensor] = []
    for current, anchor in zip(current_logits, anchor_logits):
        if current.shape != anchor.shape or current.ndim != 1:
            raise ValueError("Anchor KL logits must be one-dimensional matching vectors")
        if not torch.isfinite(current).all() or not torch.isfinite(anchor).all():
            raise FloatingPointError("Anchor KL logits must be finite")
        current_log_probs = F.log_softmax(current.float(), dim=-1)
        anchor_log_probs = F.log_softmax(anchor.detach().float(), dim=-1)
        values.append(
            (current_log_probs.exp() * (current_log_probs - anchor_log_probs)).sum()
        )
    result = torch.stack(values).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("Anchor categorical KL is non-finite")
    return result


@contextmanager
def _without_gradient_checkpointing(model: torch.nn.Module):
    """Run an auxiliary anchor pass without checkpoint state sharing.

    The policy loss keeps SAMTok checkpointing enabled for memory efficiency.
    Anchor-KL replays rows with different sequence lengths, however, and a
    checkpointed auxiliary forward can collide with the live policy graph
    during backward.  Disable it only for this short replay and restore the
    original non-reentrant configuration afterwards.
    """
    target = model
    while hasattr(target, "module"):
        target = target.module
    disable = getattr(target, "gradient_checkpointing_disable", None)
    enable = getattr(target, "gradient_checkpointing_enable", None)
    was_enabled = bool(getattr(target, "is_gradient_checkpointing", False))
    if not (was_enabled and callable(disable) and callable(enable)):
        yield
        return
    disable()
    try:
        yield
    finally:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def grammar_sequence_from_codes(
    codes: list[int],
    grammar_ids: tuple[int, list[list[int]], int],
    device: torch.device,
) -> torch.Tensor:
    start_id, code_token_ids, end_id = grammar_ids
    if len(codes) != len(code_token_ids):
        raise ValueError("Greedy code depth does not match the SAMTok grammar")
    sequence = [start_id]
    for depth, code in enumerate(codes):
        width = len(code_token_ids[depth])
        local = int(code) - depth * width
        if local < 0 or local >= width:
            raise ValueError("Greedy code lies outside its depth-specific vocabulary")
        sequence.append(int(code_token_ids[depth][local]))
    sequence.append(end_id)
    return torch.tensor([sequence], dtype=torch.long, device=device)


def greedy_crossing_preference_loss(
    current_best_log_prob: torch.Tensor,
    current_greedy_log_prob: torch.Tensor,
    old_best_log_prob: torch.Tensor,
    old_log_odds: torch.Tensor,
    active: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if current_best_log_prob.shape != current_greedy_log_prob.shape:
        raise ValueError("Preference log-probabilities must have matching shapes")
    current_log_odds = current_best_log_prob - current_greedy_log_prob
    shift = current_log_odds - old_log_odds.detach()
    if active:
        loss = F.softplus(-shift).mean()
    else:
        loss = shift.sum() * 0.0
    best_ratio = torch.exp(
        (current_best_log_prob - old_best_log_prob.detach()).clamp(-20.0, 20.0)
    )
    return loss, shift, best_ratio


def score_sampled_sequences(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    sequence_token_ids: torch.Tensor,
    grammar_ids: tuple[int, list[list[int]], int],
    temperature: float | torch.Tensor,
    support_token_ids: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    return_action_terms: bool = False,
) -> torch.Tensor:
    start_id, code_token_ids, end_id = grammar_ids
    if sequence_token_ids.ndim != 2 or sequence_token_ids.shape[1] != len(code_token_ids) + 2:
        raise ValueError("Sampled sequences do not match the SAMTok grammar depth")
    if not bool((sequence_token_ids[:, 0] == start_id).all()):
        raise ValueError("Sampled sequence does not start with mask-start")
    if not bool((sequence_token_ids[:, -1] == end_id).all()):
        raise ValueError("Sampled sequence does not end with mask-end")
    if action_mask is not None:
        if action_mask.shape != (sequence_token_ids.shape[0], len(code_token_ids)):
            raise ValueError("Action mask must cover one entry per code depth")
        action_mask = action_mask.to(device=sequence_token_ids.device, dtype=torch.float32)
    prompt_width = prompt_inputs["input_ids"].shape[1]
    extended = append_tokens(prompt_inputs, sequence_token_ids)
    outputs = model(**extended, use_cache=False)
    terms: list[torch.Tensor] = []
    for action_index in range(sequence_token_ids.shape[1]):
        logits = outputs.logits[:, prompt_width - 1 + action_index, :]
        candidates = None
        if 0 < action_index < sequence_token_ids.shape[1] - 1:
            candidates = code_token_ids[action_index - 1]
        if candidates is not None:
            action_temperature = (
                temperature[:, action_index - 1]
                if isinstance(temperature, torch.Tensor) and temperature.ndim == 2
                else temperature
            )
            if support_token_ids is None:
                terms.append(
                    _grammar_step_log_prob(
                        logits,
                        sequence_token_ids[:, action_index],
                        candidates,
                        action_temperature,
                    )
                )
            else:
                terms.append(
                    _supported_step_log_prob(
                        logits,
                        sequence_token_ids[:, action_index],
                        support_token_ids[:, action_index - 1],
                        action_temperature,
                    )
                )
    scored = torch.stack(terms, dim=1)
    if return_action_terms:
        return scored
    if action_mask is not None:
        scored = scored * action_mask
    return scored.sum(dim=1)


def view_drop_evidence_gap(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    sequence_ids: torch.Tensor,
    grammar_ids: tuple[int, list[list[int]], int],
    temperature: float,
    noise_std: float,
) -> torch.Tensor:
    """Score sampled mask actions on the original and a mild corrupted view.

    The gap is a rollout-time visual-support diagnostic. It is detached before
    entering the policy objective, so it cannot create a hidden supervised or
    OPD training path.
    """
    pixel_values = prompt_inputs.get("pixel_values")
    if pixel_values is None or not torch.is_floating_point(pixel_values):
        return torch.zeros(sequence_ids.shape[0], device=sequence_ids.device)
    perturbed = dict(prompt_inputs)
    view = pixel_values.float()
    if view.ndim >= 3:
        # A deterministic cyclic patch displacement preserves image statistics
        # while weakening exact target evidence. Metadata stays unchanged.
        shifted = torch.roll(view, shifts=1, dims=-1)
        if noise_std:
            shifted = shifted + float(noise_std) * torch.randn_like(shifted)
        perturbed["pixel_values"] = shifted.to(dtype=pixel_values.dtype)
    else:
        perturbed["pixel_values"] = view.to(dtype=pixel_values.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean = score_sampled_sequences(
            model, prompt_inputs, sequence_ids, grammar_ids, temperature
        )
        corrupted = score_sampled_sequences(
            model, perturbed, sequence_ids, grammar_ids, temperature
        )
    if was_training:
        model.train()
    return (clean.float() - corrupted.float()).detach()


def canonical_null_ce_and_first_action_margin(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    no_target: torch.Tensor,
    tokenizer: Any,
    mask_start_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    losses: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for row in torch.nonzero(no_target, as_tuple=False).flatten().tolist():
        positions = torch.nonzero(answer_mask[row] & attention_mask[row].bool(), as_tuple=False)
        canonical_positions: list[int] = []
        for position in positions.flatten().tolist():
            if int(input_ids[row, position].item()) == im_end_id:
                break
            canonical_positions.append(int(position))
        if not canonical_positions:
            raise RuntimeError("No-target answer has no canonical tokens")
        canonical_ids = [int(input_ids[row, position].item()) for position in canonical_positions]
        decoded = tokenizer.decode(canonical_ids, skip_special_tokens=False).strip()
        if decoded != CANONICAL_NULL:
            raise RuntimeError(f"Expected canonical {CANONICAL_NULL!r}, decoded {decoded!r}")
        token_losses = [
            -log_probs[row, position - 1, input_ids[row, position]]
            for position in canonical_positions
        ]
        losses.extend(token_losses)
        first = canonical_positions[0]
        margins.append(
            log_probs[row, first - 1, input_ids[row, first]]
            - log_probs[row, first - 1, mask_start_id]
        )
    if not losses:
        raise RuntimeError("Paired GR-CPPO batch has no no-target sample")
    return torch.stack(losses).mean(), torch.stack(margins)


def select_training_null_sentinel(
    dataset: SelectiveRefSegDataset,
    *,
    total_rows: int,
    process_index: int,
    world_size: int,
) -> list[dict[str, Any]]:
    if total_rows < world_size or total_rows % world_size != 0:
        raise ValueError("Sentinel rows must be positive and divisible by world size")
    if not 0 <= process_index < world_size:
        raise ValueError("Invalid sentinel process index")
    negative_indices = sorted(
        (
            index
            for index, row in enumerate(dataset.rows)
            if dataset._is_no_target(row)
        ),
        key=lambda index: str(dataset.rows[index]["id"]),
    )
    if len(negative_indices) < total_rows:
        raise ValueError(
            f"Requested {total_rows} sentinel rows, found {len(negative_indices)}"
        )
    selected = negative_indices[:total_rows]
    local_indices = selected[process_index::world_size]
    samples = [dataset[index] for index in local_indices]
    if len(samples) != total_rows // world_size:
        raise RuntimeError("Sentinel sharding produced an unexpected row count")
    if any(not bool(sample["no_target"]) for sample in samples):
        raise RuntimeError("Active-set sentinel must contain only no-target rows")
    return samples


def samples_for_pair_ids(
    dataset: SelectiveRefSegDataset,
    pair_ids: list[str],
    *,
    no_target: bool,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        positive_index, negative_index = dataset.pair_to_indices[str(pair_id)]
        sample = dataset[negative_index if no_target else positive_index]
        if bool(sample["no_target"]) != no_target:
            raise RuntimeError("registered pair outcome does not match requested sample")
        samples.append(sample)
    return samples


def evaluate_null_margins(
    model: torch.nn.Module,
    processor: Any,
    samples: list[dict[str, Any]],
    mask_start_id: int,
    device: torch.device,
    *,
    microbatch: int,
) -> torch.Tensor:
    if not samples or microbatch < 1:
        raise ValueError("null sentinel requires samples and a positive microbatch")
    values: list[torch.Tensor] = []
    for start in range(0, len(samples), microbatch):
        batch = samples[start : start + microbatch]
        inputs, answer_mask = build_supervised_inputs(processor, batch)
        inputs = move_tensors(inputs, device)
        answer_mask = answer_mask.to(device)
        no_target = torch.ones(len(batch), dtype=torch.bool, device=device)
        outputs = model(**inputs, use_cache=False)
        _, margins = canonical_null_ce_and_first_action_margin(
            outputs.logits,
            inputs["input_ids"],
            inputs["attention_mask"],
            answer_mask,
            no_target,
            processor.tokenizer,
            mask_start_id,
        )
        values.append(margins)
    return torch.cat(values)


def _rollout_rewards(
    codec: SAMTokMaskCodec,
    sample: dict[str, Any],
    sampled_codes: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    target = decode_rle_mask(sample["mask"])
    rewards: list[float] = []
    for codes in sampled_codes:
        prediction = codec.decode_codes(sample["image"], codes)
        reward = float(ciou(prediction, target))
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise FloatingPointError(f"Invalid cIoU reward for {sample['id']}: {reward}")
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def _rollout_geometry_metrics(
    codec: SAMTokMaskCodec,
    sample: dict[str, Any],
    sampled_codes: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    target = decode_rle_mask(sample["mask"])
    values: list[tuple[float, float]] = []
    for codes in sampled_codes:
        prediction = codec.decode_codes(sample["image"], codes)
        ciou_value = float(ciou(prediction, target))
        boundary_value = float(boundary_iou(prediction, target))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (ciou_value, boundary_value)):
            raise FloatingPointError(f"Invalid tail geometry reward for {sample['id']}")
        values.append((ciou_value, boundary_value))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _paired_photometric_sample(
    sample: dict[str, Any], *, brightness: float, contrast: float
) -> dict[str, Any]:
    """Return a shallow target-preserving photometric view for training credit."""
    if not math.isfinite(float(brightness)) or not math.isfinite(float(contrast)):
        raise ValueError("paired-view photometric factors must be finite")
    if float(brightness) <= 0.0 or float(contrast) <= 0.0:
        raise ValueError("paired-view photometric factors must be positive")
    view = dict(sample)
    image = ImageEnhance.Brightness(sample["image"]).enhance(float(brightness))
    view["image"] = ImageEnhance.Contrast(image).enhance(float(contrast))
    return view


def _disable_dropout(model: torch.nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def _build_dataloader(
    config: dict[str, Any]
) -> tuple[DataLoader, SAMTokMaskCodec, SelectiveRefSegDataset, dict[str, Any] | None]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model_config = config["model"]
    codec = SAMTokMaskCodec(
        model_name_or_path=model_config["processor_checkpoint"],
        mask_tokenizer_path=model_config["mask_tokenizer_checkpoint"],
        sam2_ckpt_path=model_config["sam2_checkpoint"],
        codebook_size=int(model_config["codebook_size"]),
        codebook_depth=int(model_config["codebook_depth"]),
        device=device,
    )
    data_config = config["data"]
    dataset = SelectiveRefSegDataset(
        data_config["jsonl"],
        data_config["prompt"],
        codec,
        data_config["cache_path"],
        expected_rows=int(data_config["expected_rows"]),
        expected_no_target_rows=int(data_config["expected_no_target_rows"]),
    )
    schedule = None
    if "tail_gppo" in config:
        area_stratified_schedule = bool(
            config.get("tail_gppo", {}).get("area_stratified_schedule", False)
        )
        boundary_stratified_schedule = bool(
            config.get("tail_gppo", {}).get("boundary_stratified_schedule", False)
        )
        full_data_schedule = bool(
            config.get("tail_gppo", {}).get("full_data_schedule", False)
        )
        registry = build_geometry_registry(
            dataset.rows,
            area_stratified=area_stratified_schedule,
            boundary_stratified=boundary_stratified_schedule,
        )
        schedule = select_registered_ids(
            registry,
            area_stratified=area_stratified_schedule,
            boundary_stratified=boundary_stratified_schedule,
            include_sentinel=full_data_schedule,
        )
        sampler = RegisteredPairedBatchSampler(dataset, schedule["batches"])
    else:
        sampler = PairedBatchSampler(
            dataset,
            pairs_per_batch=int(data_config["pairs_per_device_batch"]),
            seed=int(config["seed"]),
        )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(data_config["num_workers"]),
            collate_fn=identity_collate,
        ),
        codec,
        dataset,
        {"registry": registry, "schedule": schedule} if schedule is not None else None,
    )


def main() -> None:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from transformers import get_cosine_schedule_with_warmup

    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    method = _method_config(config)
    assert_training_source_clean(PACKAGE_ROOT)
    guard_runtime_environment()
    validate_declared_paths(config, REPO_ROOT)
    validate_base_checkpoint(config["model"]["base_checkpoint"])
    initialization = validate_frozen_anchor(
        config["checkpoint"]["adapter_init"], repo_root=REPO_ROOT, hash_model=False
    )
    seed_everything(int(config["seed"]))

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=config["runtime"]["mixed_precision"] if torch.cuda.is_available() else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    accelerator.even_batches = False
    expected_world_size = int(config["runtime"]["expected_world_size"])
    if accelerator.num_processes != expected_world_size:
        raise RuntimeError(
            f"Registered GR-CPPO requires {expected_world_size} processes, "
            f"got {accelerator.num_processes}"
        )

    output_dir = Path(config["checkpoint"]["output_dir"])
    if accelerator.is_main_process:
        initialization = validate_frozen_anchor(
            config["checkpoint"]["adapter_init"], repo_root=REPO_ROOT, hash_model=True
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(config, config_path, PACKAGE_ROOT)
        manifest["method"] = method
        manifest["initialization"] = initialization
        write_json_atomic(config["provenance"]["manifest_path"], manifest)
    accelerator.wait_for_everyone()

    model, processor = build_model_and_processor(config, adapter_path=initialization["path"])
    trainable_summary = assert_only_lora_trainable(model)
    representation_mode_enabled = (
        "representation_entropy_gr_cppo" in config
        or config["model"].get("adapter_mode") == "frozen_anchor_plus_visual_projector"
    )
    representation_summary = None
    if representation_mode_enabled:
        representation_summary = visual_projector_adapter_summary(model)
    dataloader, codec, dataset, tail_data = _build_dataloader(config)
    if representation_summary is not None:
        model.to(accelerator.device)
        probe = next(
            dataset[index]
            for index, row in enumerate(dataset.rows)
            if not bool(row["meta"]["no_target"])
        )
        probe_inputs, _ = build_supervised_inputs(processor, [probe])
        probe_inputs = move_tensors(probe_inputs, accelerator.device)
        _disable_dropout(model)
        model.set_adapter("anchor")
        with torch.no_grad():
            anchor_logits = model(**probe_inputs, use_cache=False).logits.float()
        activate_visual_projector_adapters(model)
        with torch.no_grad():
            combined_logits = model(**probe_inputs, use_cache=False).logits.float()
        max_abs_delta = float((combined_logits - anchor_logits).abs().max().item())
        tolerance = float(config["representation"]["preupdate_equivalence_tolerance"])
        if not math.isfinite(max_abs_delta) or max_abs_delta > tolerance:
            raise RuntimeError(
                "Zero-effect visual adapter changed pre-update logits: "
                f"max_abs_delta={max_abs_delta}, tolerance={tolerance}"
            )
        # Keep the tiny probe batch alive for the post-update visual-effect
        # check required by grounded-interface FEPO.
        del anchor_logits, combined_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        representation_summary = visual_projector_adapter_summary(model)
        representation_summary.update(
            {
                "preupdate_probe_id": probe["id"],
                "preupdate_anchor_max_abs_logit_delta": max_abs_delta,
                "preupdate_equivalence_tolerance": tolerance,
            }
        )
        trainable_summary["representation"] = representation_summary
    active_set_enabled = method.get("selective_risk_mode") == (
        "fixed_training_sentinel_active_set"
    )
    # R4 deliberately uses the tail schedule's no-target rows as the sole
    # feasibility buffer.  Keeping this flag explicit makes it impossible to
    # accidentally reintroduce the old two-sentinel collective protocol.
    unified_sentinel_enabled = bool(
        "tail_gppo" in config and method.get("unified_sentinel") is True
    )
    if unified_sentinel_enabled and not active_set_enabled:
        raise RuntimeError("Unified sentinel requires the active-set risk mode")
    sentinel_samples: list[dict[str, Any]] | None = None
    if active_set_enabled and not unified_sentinel_enabled:
        if not isinstance(dataset, SelectiveRefSegDataset):
            raise TypeError("Active-set sentinel requires SelectiveRefSegDataset")
        sentinel_samples = select_training_null_sentinel(
            dataset,
            total_rows=int(method["sentinel_rows_total"]),
            process_index=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    grammar_ids = _mask_grammar_ids(processor.tokenizer, codec)
    tail_enabled = "tail_gppo" in config
    grounded_interface_cfg = method.get("grounded_interface")
    grounded_interface_enabled = isinstance(grounded_interface_cfg, dict)
    paired_view_cfg = method.get("paired_view_geometry")
    paired_view_enabled = isinstance(paired_view_cfg, dict)
    boundary_bottleneck_paired_view_enabled = bool(
        paired_view_enabled
        and paired_view_cfg.get("aggregation") == "boundary_bottleneck_min"
    )
    hierarchical_prefix_enabled = bool(
        tail_enabled and method.get("prefix_credit_mode") == "hierarchical_geometry_prefix"
    )
    pareto_geometry_enabled = bool(
        tail_enabled and method.get("pareto_credit_mode") == "pareto_geometry_improvement"
    )
    rank_pareto_geometry_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "rank_pareto_geometry"
    )
    native_rank_pareto_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "native_anchored_rank_pareto"
    )
    depth_local_geometry_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        in {
            "depth_local_geometry",
            "depth_local_geometry_shuffled",
            "depth_local_geometry_rarity_free",
        }
    )
    signed_native_depth_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "asymmetric_signed_native_depth_local"
    )
    native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "native_anchored_rank_local"
    )
    native_rank_signed_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "native_rank_signed_depth_local"
    )
    soft_native_dominance_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "soft_native_dominance_depth_local"
    )
    scale_stratified_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "scale_stratified_native_rank_local"
    )
    bidirectional_coarse_fine_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "bidirectional_coarse_fine_native_geometry"
    )
    uncertainty_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "uncertainty_calibrated_native_rank_local"
    )
    action_budget_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "action_budget_native_rank_local"
    )
    boundary_stratified_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "boundary_stratified_native_rank_local"
    )
    confidence_gated_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "confidence_gated_native_rank_local"
    )
    predicted_evidence_scope_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "predicted_evidence_scope"
    )
    margin_calibrated_native_rank_local_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "margin_calibrated_native_rank_local"
    )
    primal_dual_null_risk_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode")
        == "primal_dual_null_risk_native_rank_local"
    )
    anchor_kl_enabled = bool(tail_enabled and method.get("anchor_kl_enabled") is True)
    depth_local_rarity_free_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "depth_local_geometry_rarity_free"
    )
    depth_local_shuffle_enabled = bool(
        tail_enabled
        and method.get("pareto_credit_mode") == "depth_local_geometry_shuffled"
    )
    verified_replay_enabled = bool(
        tail_enabled and method.get("verified_replay_mode") == "best_sampled_cIoU_replay"
    )
    verified_prefix_replay_enabled = bool(
        tail_enabled
        and method.get("verified_replay_mode") == "best_sampled_prefix_replay"
    )
    pareto_prefix_replay_enabled = bool(
        tail_enabled
        and method.get("verified_replay_mode") == "best_sampled_pareto_prefix_replay"
    )
    tail_positive_only_enabled = bool(
        tail_enabled
        and (
            method.get("positive_only_credit") is True
            or hierarchical_prefix_enabled
            or pareto_geometry_enabled
            or rank_pareto_geometry_enabled
            or native_rank_pareto_enabled
            or depth_local_geometry_enabled
            or signed_native_depth_local_enabled
            or native_rank_local_enabled
            or soft_native_dominance_enabled
            or scale_stratified_native_rank_local_enabled
            or bidirectional_coarse_fine_enabled
            or uncertainty_native_rank_local_enabled
            or action_budget_native_rank_local_enabled
            or boundary_stratified_native_rank_local_enabled
            or confidence_gated_native_rank_local_enabled
            or predicted_evidence_scope_enabled
            or margin_calibrated_native_rank_local_enabled
            or primal_dual_null_risk_enabled
            or anchor_kl_enabled
        )
    )
    boundary_credit_enabled = "boundary_entropy_gr_cppo" in config
    improvement_only_enabled = "improvement_entropy_gr_cppo" in config
    sign_balanced_enabled = "sign_balanced_entropy_gr_cppo" in config
    greedy_relative_enabled = (
        "greedy_relative_entropy_gr_cppo" in config or sign_balanced_enabled
    )
    gain_preference_enabled = "gain_preference_entropy_gr_cppo" in config
    greedy_preference_enabled = (
        "greedy_preference_entropy_gr_cppo" in config or gain_preference_enabled
    )
    evidence_gate_cfg = method.get("evidence_gate")
    evidence_gate_enabled = isinstance(evidence_gate_cfg, dict) and evidence_gate_cfg.get(
        "mode", "none"
    ) != "none"
    tail_state: dict[str, Any] | None = None
    tail_ciou_queue: FIFOEmpiricalRank | None = None
    tail_boundary_queue: FIFOEmpiricalRank | None = None
    tail_sentinel_samples: list[dict[str, Any]] | None = None
    tail_anchor_margins: torch.Tensor | None = None
    tail_hard_flags: dict[str, bool] | None = None
    if tail_enabled:
        # TB-GPPO performs frozen-anchor FIFO and sentinel forwards before the
        # optimizer is constructed, so place the model explicitly first.
        model.to(accelerator.device)
        if tail_data is None:
            raise RuntimeError("TB-GPPO registry and schedule were not initialized")
        registry = tail_data["registry"]
        schedule = tail_data["schedule"]
        true_flags = {
            pair_id: bool(record["hard_geometry"])
            for pair_id, record in registry["records"].items()
        }
        if method["arm"] == "shuffled_labels":
            tail_hard_flags = shuffled_hard_flags(
                schedule["schedule_pair_ids"], true_flags
            )
        else:
            tail_hard_flags = true_flags

        fifo_ids = list(schedule["fifo_init_pair_ids"])
        per_rank_fifo = len(fifo_ids) // accelerator.num_processes
        fifo_start = accelerator.process_index * per_rank_fifo
        local_fifo_ids = fifo_ids[fifo_start : fifo_start + per_rank_fifo]
        local_fifo_values: list[tuple[float, float]] = []
        for sample in samples_for_pair_ids(dataset, local_fifo_ids, no_target=False):
            prompt_inputs = build_rollout_prompt_batch(
                processor, sample, 1, accelerator.device
            )
            codes = greedy_grammar_codes(model, prompt_inputs, grammar_ids)
            local_fifo_values.append(
                tuple(
                    _rollout_geometry_metrics(
                        codec, sample, [codes], accelerator.device
                    )[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            )
        gathered_fifo = accelerator.gather(
            torch.tensor(
                local_fifo_values, dtype=torch.float32, device=accelerator.device
            )
        ).reshape(len(fifo_ids), 2)
        tail_ciou_queue = FIFOEmpiricalRank(gathered_fifo[:, 0].cpu().tolist())
        tail_boundary_queue = FIFOEmpiricalRank(gathered_fifo[:, 1].cpu().tolist())

        sentinel_ids = list(schedule["sentinel_pair_ids"])
        per_rank_sentinel = len(sentinel_ids) // accelerator.num_processes
        sentinel_start = accelerator.process_index * per_rank_sentinel
        local_sentinel_ids = sentinel_ids[
            sentinel_start : sentinel_start + per_rank_sentinel
        ]
        tail_sentinel_samples = samples_for_pair_ids(
            dataset, local_sentinel_ids, no_target=True
        )
        if not unified_sentinel_enabled:
            _disable_dropout(model)
            with torch.no_grad():
                local_anchor_margins = evaluate_null_margins(
                    model,
                    processor,
                    tail_sentinel_samples,
                    grammar_ids[0],
                    accelerator.device,
                    microbatch=int(method["sentinel_microbatch"]),
                )
            tail_anchor_margins = accelerator.gather(
                local_anchor_margins.detach().float()
            )
        if unified_sentinel_enabled:
            # The tail schedule is the source of truth for R4's shared
            # no-target sentinel; active-set initialization below reuses these
            # exact local rows rather than selecting a second buffer.
            sentinel_samples = tail_sentinel_samples
        tail_state = {
            "registry_sha256": registry["registry_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "sentinel_pair_ids": sentinel_ids,
            "fifo_init_pair_ids": fifo_ids,
            "fifo_initial_ciou": list(tail_ciou_queue.values),
            "fifo_initial_boundary_iou": list(tail_boundary_queue.values),
            "fifo_initial_ciou_sha256": tail_ciou_queue.sha256(),
            "fifo_initial_boundary_sha256": tail_boundary_queue.sha256(),
            "anchor_sentinel_margins": (
                tail_anchor_margins.cpu().tolist()
                if tail_anchor_margins is not None
                else []
            ),
            "arm": method["arm"],
            "full_data_schedule": bool(schedule.get("full_data_schedule", False)),
            "scheduled_pair_count": int(schedule.get("scheduled_pair_count", len(schedule["schedule_pair_ids"]))),
            "scheduled_row_count": int(schedule.get("scheduled_row_count", len(schedule["schedule_pair_ids"]) * 2)),
        }
        if accelerator.is_main_process:
            write_json_atomic(output_dir / "tail_geometry_registry.json", registry)
            write_json_atomic(output_dir / "tail_schedule.json", schedule)
    if accelerator.is_main_process:
        manifest_path = Path(config["provenance"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_modules"] = runtime_module_files()
        if representation_summary is not None:
            manifest["representation_adapter"] = representation_summary
        if tail_state is not None:
            manifest["tail_gppo"] = tail_state
        manifest["rollout_contract"] = {
            "grammar_token_ids": {
                "mask_start": grammar_ids[0],
                "codes_by_depth": grammar_ids[1],
                "mask_end": grammar_ids[2],
            },
            "rollouts_per_prompt": int(method["rollouts_per_prompt"]),
            "policy_epochs": int(method["policy_epochs"]),
            "multimodal_batching": method["multimodal_batching"],
            "ppo_action_logprob_scope": "sampled_depth_specific_code_tokens_only",
            "forced_boundary_probability": 1.0,
            "validity_gate": {
                "require_nonconstant_rewards": method["require_nonconstant_rewards"],
                "reward_std_epsilon": method["reward_std_epsilon"],
                "require_epoch2_ratio_change": method["require_epoch2_ratio_change"],
                "min_epoch2_ratio_abs_deviation": method[
                    "min_epoch2_ratio_abs_deviation"
                ],
            },
        }
        if method.get("exploration") == "per_prefix_topm_collision_support":
            manifest["rollout_contract"]["effective_support_control"] = {
                "support_size": method["support_size"],
                "target_effective_support": method["target_effective_support"],
                "temperature_min": method["temperature_min"],
                "temperature_max": method["temperature_max"],
                "calibration_iterations": method["calibration_iterations"],
                "rescore_policy": "frozen_old_support_and_temperature",
                "selection_data": "training_logits_only_no_holdout_tuning",
            }
        if method.get("evidence_gate") is not None:
            manifest["rollout_contract"]["evidence_gate"] = {
                **dict(method["evidence_gate"]),
                "target": "detached_advantage_multiplier",
                "view": "same-image-cyclic-patch-displacement-plus-noise",
                "label_free": True,
                "self_supervised_loop": False,
            }
        if grounded_interface_enabled:
            manifest["rollout_contract"]["grounded_interface"] = {
                **dict(grounded_interface_cfg),
                "loss": "answer_token_cross_entropy",
                "target_scope": "same_row_answer_text_including_canonical_null",
                "teacher": "none",
                "holdout_access": False,
            }
        if paired_view_enabled:
            manifest["rollout_contract"]["paired_view_geometry"] = {
                **dict(paired_view_cfg),
                "reward": "geometric_mean_clean_augmented_native_rank_local",
                "target": "same_row_ground_truth_mask_geometry",
                "teacher": "none",
                "opd": False,
                "ema": False,
                "self_supervised_loop": False,
                "holdout_access": False,
            }
        if active_set_enabled:
            manifest["rollout_contract"]["selective_risk_control"] = {
                "mode": method["selective_risk_mode"],
                "sentinel_source": method["sentinel_source"],
                "sentinel_rows_total": method["sentinel_rows_total"],
                "anchor_budget_source": method["anchor_budget_source"],
                "null_ce_relative_slack": method["null_ce_relative_slack"],
                "null_ce_absolute_slack": method["null_ce_absolute_slack"],
                "margin_slack": method["margin_slack"],
                "holdout_access": False,
            }
        write_json_atomic(manifest_path, manifest)
    accelerator.wait_for_everyone()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_config = config["optimizer"]
    optimizer = AdamW(
        parameters,
        lr=float(optimizer_config["lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    max_steps = int(optimizer_config["max_steps"])
    policy_epochs = int(method["policy_epochs"])
    update_steps = max_steps * policy_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(update_steps * float(optimizer_config["warmup_ratio"])),
        num_training_steps=update_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    # Model construction is synchronized from the common seed; rollout draws
    # use rank-offset seeds so data-parallel workers do not duplicate groups.
    seed_everything(int(config["seed"]) + accelerator.process_index)

    anchor_kl_samples: list[dict[str, Any]] = []
    anchor_kl_logits: list[list[torch.Tensor]] = []
    anchor_kl_codes: list[list[int]] = []
    anchor_kl_pair_ids: list[str] = []
    anchor_kl_active_observations = 0
    anchor_kl_observations = 0
    anchor_kl_history: list[float] = []
    if anchor_kl_enabled:
        if tail_data is None:
            raise RuntimeError("Anchor-KL requires the registered tail geometry registry")
        registry = tail_data["registry"]
        schedule = tail_data["schedule"]
        anchor_kl_pair_ids = select_anchor_buffer_pair_ids(
            registry,
            schedule,
            total_rows=int(method.get("anchor_buffer_rows", ANCHOR_BUFFER_SIZE)),
        )
        per_rank = len(anchor_kl_pair_ids) // accelerator.num_processes
        if per_rank * accelerator.num_processes != len(anchor_kl_pair_ids):
            raise RuntimeError("Anchor-KL buffer must divide evenly across workers")
        local_start = accelerator.process_index * per_rank
        local_ids = anchor_kl_pair_ids[local_start : local_start + per_rank]
        anchor_kl_samples = samples_for_pair_ids(dataset, local_ids, no_target=False)
        _disable_dropout(model)
        with _without_gradient_checkpointing(model):
            for sample in anchor_kl_samples:
                logits, codes = collect_anchor_policy_logits(
                    model, processor, sample, grammar_ids, accelerator.device, no_grad=True
                )
                if len(logits) != len(grammar_ids[1]) or not all(
                    torch.isfinite(value).all() for value in logits
                ):
                    raise FloatingPointError("Frozen anchor codebook logits are non-finite")
                anchor_kl_logits.append([value.detach().float().cpu() for value in logits])
                anchor_kl_codes.append(codes)
        if len(anchor_kl_logits) != len(anchor_kl_samples):
            raise RuntimeError("Anchor-KL cache size does not match its training buffer")
        if tail_state is not None:
            tail_state["anchor_kl_buffer_pair_ids"] = list(anchor_kl_pair_ids)
            tail_state["anchor_kl_buffer_rows"] = len(anchor_kl_pair_ids)
            tail_state["anchor_kl_buffer_local_rows"] = len(anchor_kl_samples)
        if accelerator.is_main_process:
            manifest_path = Path(config["provenance"]["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("rollout_contract", {})["anchor_kl"] = {
                "enabled": True,
                "buffer_rows": len(anchor_kl_pair_ids),
                "buffer_pair_ids": list(anchor_kl_pair_ids),
                "epsilon_nats": float(method["anchor_kl_epsilon"]),
                "lambda": float(method["anchor_kl_lambda"]),
                "target": "frozen_initialization_codebook_logits",
                "holdout_access": False,
            }
            write_json_atomic(manifest_path, manifest)
        accelerator.wait_for_everyone()
        model.train()

    sentinel_inputs: dict[str, torch.Tensor] | None = None
    sentinel_answer_mask: torch.Tensor | None = None
    sentinel_no_target: torch.Tensor | None = None
    active_set_state: dict[str, Any] | None = None
    if active_set_enabled:
        if sentinel_samples is None:
            raise RuntimeError("Active-set sentinel was not initialized")
        sentinel_inputs, sentinel_answer_mask = build_supervised_inputs(
            processor, sentinel_samples
        )
        sentinel_inputs = move_tensors(sentinel_inputs, accelerator.device)
        sentinel_answer_mask = sentinel_answer_mask.to(accelerator.device)
        sentinel_no_target = torch.ones(
            len(sentinel_samples), dtype=torch.bool, device=accelerator.device
        )
        _disable_dropout(model)
        with torch.no_grad():
            anchor_outputs = model(**sentinel_inputs, use_cache=False)
            anchor_null_ce, anchor_margins = canonical_null_ce_and_first_action_margin(
                anchor_outputs.logits,
                sentinel_inputs["input_ids"],
                sentinel_inputs["attention_mask"],
                sentinel_answer_mask,
                sentinel_no_target,
                processor.tokenizer,
                grammar_ids[0],
            )
        if unified_sentinel_enabled:
            local_sentinel_count = len(sentinel_samples)
            packed_anchor = accelerator.gather(
                torch.cat(
                    (
                        anchor_null_ce.detach().float().reshape(1),
                        anchor_margins.detach().float(),
                    )
                )
            )
            expected_anchor = accelerator.num_processes * (local_sentinel_count + 1)
            if packed_anchor.numel() != expected_anchor:
                raise RuntimeError("Unified anchor gather returned an unexpected shape")
            anchor_stats = packed_anchor.reshape(
                accelerator.num_processes, local_sentinel_count + 1
            )
            global_anchor_ce = float(anchor_stats[:, 0].mean().item())
            tail_anchor_margins = anchor_stats[:, 1:].reshape(-1)
            global_anchor_margin_min = float(tail_anchor_margins.min().item())
        else:
            global_anchor_ce = float(
                accelerator.gather(anchor_null_ce.detach().float().reshape(1))
                .mean()
                .item()
            )
            global_anchor_margin_min = float(
                accelerator.gather(anchor_margins.detach().float()).min().item()
            )
        null_ce_budget, margin_budget = derive_active_set_budgets(
            global_anchor_ce,
            global_anchor_margin_min,
            null_ce_relative_slack=float(method["null_ce_relative_slack"]),
            null_ce_absolute_slack=float(method["null_ce_absolute_slack"]),
            margin_slack=float(method["margin_slack"]),
        )
        active_set_state = {
            "sentinel_ids_local": [sample["id"] for sample in sentinel_samples],
            "anchor_null_ce": global_anchor_ce,
            "anchor_margin_min": global_anchor_margin_min,
            "null_ce_budget": null_ce_budget,
            "margin_budget": margin_budget,
        }
        if accelerator.is_main_process:
            manifest_path = Path(config["provenance"]["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rollout_contract"]["selective_risk_control"].update(
                {
                    "anchor_null_ce": global_anchor_ce,
                    "anchor_margin_min": global_anchor_margin_min,
                    "null_ce_budget": null_ce_budget,
                    "margin_budget": margin_budget,
                }
            )
            if unified_sentinel_enabled:
                if tail_state is None or tail_anchor_margins is None:
                    raise RuntimeError("Unified tail anchor was not initialized")
                tail_state["anchor_sentinel_margins"] = tail_anchor_margins.cpu().tolist()
                manifest["tail_gppo"]["anchor_sentinel_margins"] = (
                    tail_anchor_margins.cpu().tolist()
                )
            write_json_atomic(manifest_path, manifest)
        accelerator.wait_for_everyone()

    metrics: dict[str, Any] = {
        "stage": config["stage"],
        "status": "running",
        "method": method,
        "initialization": initialization,
        "trainable": trainable_summary,
        "steps": [],
    }
    if active_set_state is not None:
        metrics["active_set"] = active_set_state
    if tail_state is not None:
        metrics["tail_gppo"] = tail_state
    if accelerator.is_main_process:
        write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()

    outer_step = 0
    nonconstant_reward_groups = 0
    total_rollout_groups = 0
    multitrajectory_groups = 0
    improved_over_greedy_rollouts = 0
    improved_over_greedy_groups = 0
    effective_support_hits = 0
    effective_support_decisions = 0
    ratio_change_observations = 0
    effective_support_target_reached_fraction: float | None = None
    positive_policy_grad_observations = 0
    visual_gradient_norm_observations = 0
    visual_gradient_norms: list[float] = []
    null_ce_active_observations = 0
    margin_active_observations = 0
    active_set_observations = 0
    tail_margin_violation_rates: list[float] = []
    primal_dual_lambda = float(method.get("primal_dual_lambda_init", 0.0))
    primal_dual_lambda_history: list[float] = []
    primal_dual_active_observations = 0
    primal_dual_observations = 0
    pes_state_counts = torch.zeros(3, dtype=torch.long, device=accelerator.device)
    pes_state_observations = 0
    consumed_pair_ids: set[str] = set()
    while outer_step < max_steps:
        for samples in dataloader:
            if outer_step >= max_steps:
                break
            consumed_pair_ids.update(str(sample["pair_id"]) for sample in samples)
            positives = [sample for sample in samples if not bool(sample["no_target"])]
            negatives = [sample for sample in samples if bool(sample["no_target"])]
            if not positives or not negatives or len(positives) != len(negatives):
                raise RuntimeError("Every GR-CPPO device batch must retain complete outcome pairs")

            rollout_groups: list[dict[str, Any]] = []
            for sample in positives:
                prompt_inputs = build_rollout_prompt_batch(
                    processor,
                    sample,
                    int(method["rollouts_per_prompt"]),
                    accelerator.device,
                )
                if method.get("exploration") == "per_prefix_topm_collision_support":
                    (
                        sequence_ids,
                        sampled_codes,
                        behavior_log_probs,
                        action_temperatures,
                        action_supports,
                        native_effective_supports,
                        calibrated_effective_supports,
                        native_entropies,
                        controlled_entropies,
                        top_support_masses,
                        native_margins,
                        controlled_native_kls,
                    ) = sample_effective_support_grammar_rollouts(
                        model,
                        prompt_inputs,
                        grammar_ids,
                        support_size=int(method["support_size"]),
                        target_effective_support=float(
                            method["target_effective_support"]
                        ),
                        temperature_min=float(method["temperature_min"]),
                        temperature_max=float(method["temperature_max"]),
                        calibration_iterations=int(method["calibration_iterations"]),
                    )
                else:
                    sequence_ids, sampled_codes, behavior_log_probs = (
                        sample_grammar_rollouts(
                            model,
                            prompt_inputs,
                            grammar_ids,
                            float(method["temperature"]),
                        )
                    )
                    action_temperatures = None
                    action_supports = None
                    native_effective_supports = None
                    calibrated_effective_supports = None
                    native_entropies = None
                    controlled_entropies = None
                    top_support_masses = None
                    native_margins = None
                    controlled_native_kls = None
                behavior_action_log_probs = None
                if predicted_evidence_scope_enabled:
                    if controlled_entropies is None or top_support_masses is None or native_margins is None:
                        raise RuntimeError(
                            "PES requires calibrated per-depth evidence diagnostics"
                        )
                    _disable_dropout(model)
                    with torch.no_grad():
                        behavior_action_log_probs = score_sampled_sequences(
                            model,
                            prompt_inputs,
                            sequence_ids,
                            grammar_ids,
                            action_temperatures
                            if action_temperatures is not None
                            else float(method["temperature"]),
                            action_supports,
                            return_action_terms=True,
                        ).detach()
                raw_geometry = (
                    _rollout_geometry_metrics(
                        codec, sample, sampled_codes, accelerator.device
                    )
                    if tail_enabled or boundary_credit_enabled
                    else None
                )
                augmented_sample = None
                augmented_geometry = None
                augmented_native_geometry = None
                if paired_view_enabled:
                    if paired_view_cfg is None:
                        raise RuntimeError("Paired-view settings are unavailable")
                    augmented_sample = build_target_preserving_view_samples(
                        [sample],
                        brightness=float(paired_view_cfg["brightness"]),
                        contrast=float(paired_view_cfg["contrast"]),
                    )[0]
                    augmented_geometry = _rollout_geometry_metrics(
                        codec, augmented_sample, sampled_codes, accelerator.device
                    )
                if boundary_credit_enabled:
                    rewards = (
                        float(method["ciou_weight"]) * raw_geometry[:, 0]
                        + float(method["boundary_iou_weight"]) * raw_geometry[:, 1]
                    )
                elif raw_geometry is not None:
                    rewards = raw_geometry[:, 0]
                else:
                    rewards = _rollout_rewards(
                        codec, sample, sampled_codes, accelerator.device
                    )
                greedy_inputs = build_rollout_prompt_batch(
                    processor, sample, 1, accelerator.device
                )
                native_greedy_codes = greedy_grammar_codes(
                    model, greedy_inputs, grammar_ids
                )
                native_geometry = None
                if (
                    boundary_credit_enabled
                    or pareto_geometry_enabled
                    or rank_pareto_geometry_enabled
                    or native_rank_pareto_enabled
                    or depth_local_geometry_enabled
                    or signed_native_depth_local_enabled
                    or native_rank_local_enabled
                    or native_rank_signed_enabled
                    or soft_native_dominance_enabled
                    or scale_stratified_native_rank_local_enabled
                    or bidirectional_coarse_fine_enabled
                    or uncertainty_native_rank_local_enabled
                    or action_budget_native_rank_local_enabled
                    or boundary_stratified_native_rank_local_enabled
                    or confidence_gated_native_rank_local_enabled
                    or predicted_evidence_scope_enabled
                    or margin_calibrated_native_rank_local_enabled
                    or pareto_prefix_replay_enabled
                ):
                    native_geometry = _rollout_geometry_metrics(
                        codec,
                        sample,
                        [native_greedy_codes],
                        accelerator.device,
                    )[0]
                if paired_view_enabled:
                    augmented_greedy_inputs = build_rollout_prompt_batch(
                        processor, augmented_sample, 1, accelerator.device
                    )
                    augmented_native_codes = greedy_grammar_codes(
                        model, augmented_greedy_inputs, grammar_ids
                    )
                    augmented_native_geometry = _rollout_geometry_metrics(
                        codec,
                        augmented_sample,
                        [augmented_native_codes],
                        accelerator.device,
                    )[0]
                if boundary_credit_enabled:
                    native_greedy_reward = (
                        float(method["ciou_weight"]) * native_geometry[0]
                        + float(method["boundary_iou_weight"]) * native_geometry[1]
                    )
                else:
                    native_greedy_reward = _rollout_rewards(
                        codec, sample, [native_greedy_codes], accelerator.device
                    )[0]
                if paired_view_enabled:
                    clean_pair_reward = 0.5 * (
                        raw_geometry[:, 0] + raw_geometry[:, 1]
                    )
                    augmented_pair_reward = 0.5 * (
                        augmented_geometry[:, 0] + augmented_geometry[:, 1]
                    )
                    rewards = torch.sqrt(
                        (clean_pair_reward.clamp_min(0.0)
                         * augmented_pair_reward.clamp_min(0.0)).clamp_min(0.0)
                    )
                    native_greedy_reward = torch.sqrt(
                        (0.5 * (native_geometry[0] + native_geometry[1])
                         * 0.5 * (augmented_native_geometry[0] + augmented_native_geometry[1])).clamp_min(0.0)
                    )
                improved_over_greedy = rewards > native_greedy_reward + 1e-4
                replay_sequence = None
                replay_action_temperatures = None
                replay_action_supports = None
                replay_action_mask = None
                replay_active = False
                replay_gain = 0.0
                if verified_replay_enabled or verified_prefix_replay_enabled or pareto_prefix_replay_enabled:
                    if pareto_prefix_replay_enabled:
                        if raw_geometry is None or native_geometry is None:
                            raise RuntimeError("Pareto prefix replay requires geometry metrics")
                        # Select only trajectories that improve both geometry
                        # criteria before ranking them. Ranking cIoU first and
                        # checking boundary afterward can miss a valid Pareto
                        # candidate when the cIoU winner regresses boundary.
                        minimum_improvement = float(method["minimum_improvement"])
                        ciou_gain = raw_geometry[:, 0] - native_geometry[0]
                        boundary_gain = raw_geometry[:, 1] - native_geometry[1]
                        pareto_valid = (ciou_gain > minimum_improvement) & (
                            boundary_gain > minimum_improvement
                        )
                        if bool(pareto_valid.any().item()):
                            geometry_score = torch.sqrt(
                                ciou_gain.clamp_min(0.0) * boundary_gain.clamp_min(0.0)
                            )
                            geometry_score = geometry_score.masked_fill(~pareto_valid, -float("inf"))
                            best_index = int(geometry_score.argmax().item())
                        else:
                            best_index = int(rewards.argmax().item())
                    else:
                        best_index = int(rewards.argmax().item())
                    replay_gain = float(
                        (rewards[best_index] - native_greedy_reward).detach().item()
                    )
                    replay_active = replay_gain > float(method["minimum_improvement"])
                    if pareto_prefix_replay_enabled:
                        boundary_gain = float(
                            (raw_geometry[best_index, 1] - native_geometry[1]).detach().item()
                        )
                        replay_active = replay_active and boundary_gain > float(
                            method["minimum_improvement"]
                        )
                    if replay_active:
                        replay_sequence = sequence_ids[best_index : best_index + 1].detach()
                        if action_temperatures is not None:
                            replay_action_temperatures = action_temperatures[
                                best_index : best_index + 1
                            ]
                        if action_supports is not None:
                            replay_action_supports = action_supports[best_index : best_index + 1]
                        if verified_prefix_replay_enabled or pareto_prefix_replay_enabled:
                            native_sequence = grammar_sequence_from_codes(
                                native_greedy_codes, grammar_ids, accelerator.device
                            )
                            replay_action_mask = (
                                sequence_ids[best_index : best_index + 1, 1:-1]
                                != native_sequence[:, 1:-1]
                            ).float()
                preference_best_sequence = None
                preference_greedy_sequence = None
                preference_old_best_log_prob = None
                preference_old_log_odds = None
                preference_active = False
                preference_reward_gain = 0.0
                if greedy_preference_enabled:
                    best_index = int(rewards.argmax().item())
                    preference_best_sequence = sequence_ids[
                        best_index : best_index + 1
                    ].detach()
                    preference_greedy_sequence = grammar_sequence_from_codes(
                        native_greedy_codes, grammar_ids, accelerator.device
                    )
                    preference_reward_gain = float(
                        (rewards[best_index] - native_greedy_reward).detach().item()
                    )
                    preference_active = preference_reward_gain > float(
                        method["minimum_improvement"]
                    )
                    # The rollout policy and all update-time rescoring disable
                    # dropout. Keep the frozen native preference reference in
                    # exactly the same model state so epoch-zero ratios are 1.
                    model.eval()
                    with torch.no_grad():
                        preference_old_best_log_prob = score_sampled_sequences(
                            model,
                            greedy_inputs,
                            preference_best_sequence,
                            grammar_ids,
                            float(method["native_scoring_temperature"]),
                        ).detach()
                        preference_old_greedy_log_prob = score_sampled_sequences(
                            model,
                            greedy_inputs,
                            preference_greedy_sequence,
                            grammar_ids,
                            float(method["native_scoring_temperature"]),
                        ).detach()
                    preference_old_log_odds = (
                        preference_old_best_log_prob
                        - preference_old_greedy_log_prob
                    ).detach()
                    _disable_dropout(model)
                if sign_balanced_enabled:
                    advantages = sign_balanced_greedy_advantages(
                        rewards,
                        native_greedy_reward,
                        float(method["advantage_epsilon"]),
                    )
                elif greedy_relative_enabled:
                    advantages = greedy_relative_advantages(
                        rewards,
                        native_greedy_reward,
                        float(method["advantage_epsilon"]),
                    )
                elif pareto_geometry_enabled:
                    if native_geometry is None or raw_geometry is None:
                        raise RuntimeError("Pareto geometry metrics were not computed")
                    advantages = pareto_geometry_improvement_advantages(
                        raw_geometry,
                        native_geometry,
                        minimum_improvement=float(method["minimum_improvement"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif rank_pareto_geometry_enabled:
                    if raw_geometry is None:
                        raise RuntimeError("Rank-Pareto geometry metrics were not computed")
                    advantages = rank_pareto_geometry_advantages(
                        raw_geometry,
                        eps=float(method["advantage_epsilon"]),
                    )
                elif native_rank_pareto_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Native-anchored rank-Pareto geometry metrics are unavailable"
                        )
                    advantages = native_anchored_rank_pareto_advantages(
                        raw_geometry,
                        native_geometry,
                        minimum_improvement=float(method["minimum_improvement"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif depth_local_shuffle_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Shuffled depth-local geometry metrics are unavailable"
                        )
                    advantages = shuffled_depth_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        rarity_weight=float(method["depth_local_rarity_weight"]),
                        shuffle_seed=int(method["depth_local_shuffle_seed"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif depth_local_geometry_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Depth-local geometry metrics are unavailable"
                        )
                    advantages = depth_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        rarity_weight=(
                            0.0
                            if depth_local_rarity_free_enabled
                            else float(method["depth_local_rarity_weight"])
                        ),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif signed_native_depth_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Signed native-relative depth-local geometry metrics are unavailable"
                        )
                    advantages = asymmetric_signed_native_relative_depth_local_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        beta=float(method["depth_local_beta"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif paired_view_enabled:
                    if (
                        raw_geometry is None
                        or native_geometry is None
                        or augmented_geometry is None
                        or augmented_native_geometry is None
                    ):
                        raise RuntimeError("Paired-view geometry metrics are unavailable")
                    paired_advantage_fn = (
                        boundary_bottleneck_paired_view_geometry_advantages
                        if boundary_bottleneck_paired_view_enabled
                        else paired_view_native_rank_local_geometry_advantages
                    )
                    advantages = paired_advantage_fn(
                        raw_geometry,
                        augmented_geometry,
                        native_geometry,
                        augmented_native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        augmented_native_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError("Native rank-local geometry metrics are unavailable")
                    advantages = native_anchored_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif native_rank_signed_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError("Native signed rank geometry metrics are unavailable")
                    advantages = native_rank_signed_depth_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif soft_native_dominance_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError("Soft native dominance geometry metrics are unavailable")
                    advantages = soft_native_dominance_depth_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        temperature=float(method["soft_dominance_temperature"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif uncertainty_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Uncertainty native rank-local geometry metrics are unavailable"
                        )
                    if controlled_entropies is not None:
                        if top_support_masses is None:
                            raise RuntimeError(
                                "Uncertainty credit requires calibrated support diagnostics"
                            )
                        rollout_uncertainty = calibrated_rollout_uncertainty(
                            controlled_entropies,
                            top_support_masses,
                            support_size=int(method["support_size"]),
                        )
                    else:
                        raise RuntimeError(
                            "Uncertainty credit requires per-prefix calibrated exploration"
                        )
                    advantages = uncertainty_calibrated_native_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        rollout_uncertainty,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        confidence_floor=float(method["uncertainty_confidence_floor"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif margin_calibrated_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError("Margin-calibrated native geometry metrics are unavailable")
                    advantages = margin_calibrated_native_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        margin_power=float(method["margin_power"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif confidence_gated_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Confidence-gated native geometry metrics are unavailable"
                        )
                    if controlled_entropies is None or top_support_masses is None:
                        raise RuntimeError(
                            "Confidence-gated credit requires per-prefix calibrated exploration"
                        )
                    rollout_uncertainty = calibrated_rollout_uncertainty(
                        controlled_entropies,
                        top_support_masses,
                        support_size=int(method["support_size"]),
                    )
                    advantages = confidence_gated_native_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        rollout_uncertainty,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        confidence_threshold=float(method["confidence_threshold"]),
                        confidence_floor=float(method["confidence_floor"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif predicted_evidence_scope_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError("PES native geometry metrics are unavailable")
                    if controlled_entropies is None or top_support_masses is None:
                        raise RuntimeError("PES evidence diagnostics are unavailable")
                    advantages = native_anchored_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                    pes_scope_mask, pes_evidence_state = predicted_evidence_scope_masks(
                        controlled_entropies,
                        top_support_masses,
                        sampled_codes,
                        native_greedy_codes,
                        native_margins=native_margins,
                        support_size=int(method["support_size"]),
                        confident_entropy=float(method["pes_confident_entropy"]),
                        ambiguous_entropy=float(method["pes_ambiguous_entropy"]),
                        confident_margin=float(method["pes_confident_margin"]),
                        ambiguous_margin=float(method["pes_ambiguous_margin"]),
                        shuffle_seed=(
                            int(method["pes_evidence_shuffle_seed"])
                            if method.get("pes_evidence_shuffle") is True
                            else None
                        ),
                    )
                elif scale_stratified_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Scale-stratified native geometry metrics are unavailable"
                        )
                    if tail_data is None:
                        raise RuntimeError("Scale-stratified credit requires the geometry registry")
                    area_stratum = str(
                        tail_data["registry"]["records"][str(sample["pair_id"])]
                        ["area_stratum"]
                    )
                    configured_weights = method["area_rank_weights"][area_stratum]
                    advantages = scale_stratified_native_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        area_stratum,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        axis_weights=(float(configured_weights[0]), float(configured_weights[1])),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif action_budget_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Action-budget native geometry metrics are unavailable"
                        )
                    advantages = action_budget_native_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        action_budget=int(method["action_budget"]),
                        excess_penalty=float(method["action_budget_excess_penalty"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif boundary_stratified_native_rank_local_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Boundary-stratified native geometry metrics are unavailable"
                        )
                    # Sampling strata alter which examples supply the fixed
                    # R18 credit; the credit transform itself remains frozen.
                    advantages = native_anchored_rank_local_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif bidirectional_coarse_fine_enabled:
                    if raw_geometry is None or native_geometry is None:
                        raise RuntimeError(
                            "Bidirectional coarse/fine geometry metrics are unavailable"
                        )
                    advantages = bidirectional_coarse_fine_native_geometry_advantages(
                        raw_geometry,
                        native_geometry,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["depth_local_decay"]),
                        coarse_weight=float(method["coarse_depth_weight"]),
                        fine_weight=float(method["fine_depth_weight"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif hierarchical_prefix_enabled:
                    advantages = hierarchical_prefix_credit_advantages(
                        rewards,
                        native_greedy_reward,
                        sampled_codes,
                        native_greedy_codes,
                        minimum_improvement=float(method["minimum_improvement"]),
                        depth_decay=float(method["prefix_depth_decay"]),
                        novelty_weight=float(method["prefix_novelty_weight"]),
                        eps=float(method["advantage_epsilon"]),
                    )
                elif improvement_only_enabled or tail_positive_only_enabled:
                    improvement_advantages = F.relu(
                        rewards
                        - native_greedy_reward
                        - float(method["minimum_improvement"])
                    )
                    active = improvement_advantages > 0
                    if bool(active.any().item()):
                        advantages = improvement_advantages / improvement_advantages[
                            active
                        ].mean().clamp_min(float(method["advantage_epsilon"]))
                    else:
                        advantages = improvement_advantages
                else:
                    advantages = group_standardized_advantages(rewards)
                evidence_gate = torch.ones_like(advantages)
                if evidence_gate_enabled:
                    evidence_gap = view_drop_evidence_gap(
                        model,
                        prompt_inputs,
                        sequence_ids,
                        grammar_ids,
                        float(method["temperature"]),
                        float(evidence_gate_cfg["noise_std"]),
                    )
                    evidence_gate = detached_group_gate(
                        evidence_gap,
                        mode=str(evidence_gate_cfg["mode"]),
                        scale=float(evidence_gate_cfg["scale"]),
                        clip_min=float(evidence_gate_cfg["clip_min"]),
                        clip_max=float(evidence_gate_cfg["clip_max"]),
                    )
                    advantages = advantages * evidence_gate
                action_change_count = (
                    torch.tensor(
                        [
                            sum(
                                code != native
                                for code, native in zip(codes, native_greedy_codes)
                            )
                            for codes in sampled_codes
                        ],
                        dtype=torch.float32,
                        device=accelerator.device,
                    )
                    if action_budget_native_rank_local_enabled
                    else None
                )
                reward_std = rewards.std(unbiased=False)
                if not torch.isfinite(reward_std):
                    raise FloatingPointError("Non-finite rollout reward standard deviation")
                rollout_groups.append(
                    {
                        "sample_id": sample["id"],
                        "pair_id": sample["pair_id"],
                        "prompt_inputs": prompt_inputs,
                        "sequence_ids": sequence_ids,
                        "behavior_log_probs": behavior_log_probs,
                        "behavior_action_log_probs": behavior_action_log_probs,
                        "rewards": rewards,
                        "raw_geometry": raw_geometry,
                        "paired_view_geometry": augmented_geometry,
                        "native_geometry": native_geometry,
                        "paired_view_native_geometry": augmented_native_geometry,
                        "paired_view_native_codes": (
                            list(augmented_native_codes)
                            if paired_view_enabled
                            else None
                        ),
                        "advantages": advantages,
                        "evidence_gate": evidence_gate,
                        "pes_scope_mask": pes_scope_mask if predicted_evidence_scope_enabled else None,
                        "pes_evidence_state": pes_evidence_state if predicted_evidence_scope_enabled else None,
                        "reward_std": reward_std,
                        "reward_range": rewards.max() - rewards.min(),
                        "nonconstant": bool(
                            float(reward_std.detach().item())
                            > float(method["reward_std_epsilon"])
                        ),
                        "unique_trajectories": len(
                            {tuple(row) for row in sequence_ids.detach().cpu().tolist()}
                        ),
                        "action_temperatures": action_temperatures,
                        "action_supports": action_supports,
                        "native_effective_supports": native_effective_supports,
                        "calibrated_effective_supports": calibrated_effective_supports,
                        "native_entropies": native_entropies,
                        "controlled_entropies": controlled_entropies,
                        "top_support_masses": top_support_masses,
                        "controlled_native_kls": controlled_native_kls,
                        "rollout_uncertainty": (
                            rollout_uncertainty
                            if (
                                uncertainty_native_rank_local_enabled
                                or confidence_gated_native_rank_local_enabled
                            )
                            else None
                        ),
                        "action_change_count": action_change_count,
                        "native_greedy_reward": native_greedy_reward,
                        "improved_over_greedy": improved_over_greedy,
                        "replay_sequence": replay_sequence,
                        "replay_action_temperatures": replay_action_temperatures,
                        "replay_action_supports": replay_action_supports,
                        "replay_action_mask": replay_action_mask,
                        "replay_active": replay_active,
                        "replay_gain": replay_gain,
                        "improvement_advantage_active_count": int(
                            (advantages > 0).sum().item()
                        ),
                        "negative_advantage_active_count": int(
                            (advantages < 0).sum().item()
                        ),
                        "sampled_codes": torch.tensor(
                            sampled_codes,
                            dtype=torch.long,
                            device=accelerator.device,
                        ),
                        "greedy_inputs": greedy_inputs,
                        "preference_best_sequence": preference_best_sequence,
                        "preference_greedy_sequence": preference_greedy_sequence,
                        "preference_old_best_log_prob": preference_old_best_log_prob,
                        "preference_old_log_odds": preference_old_log_odds,
                        "preference_active": preference_active,
                        "preference_reward_gain": preference_reward_gain,
                        "preference_weight": float(preference_active),
                    }
                )
            if gain_preference_enabled:
                local_gain = torch.tensor(
                    [
                        sum(
                            float(group["preference_reward_gain"])
                            for group in rollout_groups
                            if bool(group["preference_active"])
                        ),
                        sum(bool(group["preference_active"]) for group in rollout_groups),
                    ],
                    dtype=torch.float32,
                    device=accelerator.device,
                )
                global_gain = accelerator.gather(local_gain).reshape(-1, 2).sum(0)
                if float(global_gain[1].item()) > 0:
                    active_gain_mean = float(
                        global_gain[0].item() / global_gain[1].item()
                    )
                    if not math.isfinite(active_gain_mean) or active_gain_mean <= 0:
                        raise FloatingPointError("Invalid distributed active-gain mean")
                    for group in rollout_groups:
                        group["preference_weight"] = (
                            float(group["preference_reward_gain"]) / active_gain_mean
                            if bool(group["preference_active"])
                            else 0.0
                        )
            tail_queue_trace: list[dict[str, Any]] | None = None
            if tail_enabled:
                if (
                    tail_ciou_queue is None
                    or tail_boundary_queue is None
                    or tail_hard_flags is None
                ):
                    raise RuntimeError("TB-GPPO queue state is unavailable")
                local_raw = torch.stack(
                    [group["raw_geometry"] for group in rollout_groups]
                )
                global_raw = accelerator.gather(local_raw).reshape(-1, 4, 2)
                local_flags = torch.tensor(
                    [tail_hard_flags[group["pair_id"]] for group in rollout_groups],
                    dtype=torch.int64,
                    device=accelerator.device,
                )
                global_flags = accelerator.gather(local_flags).bool().cpu().tolist()
                transformed: list[list[float]] = []
                tail_queue_trace = []
                for group_index, values in enumerate(global_raw.detach().cpu().tolist()):
                    pre_ciou_hash = tail_ciou_queue.sha256()
                    pre_boundary_hash = tail_boundary_queue.sha256()
                    ciou_ranks = [tail_ciou_queue.midrank(row[0]) for row in values]
                    boundary_ranks = [
                        tail_boundary_queue.midrank(row[1]) for row in values
                    ]
                    if method.get("tail_reward_mode") == "raw_ciou":
                        transformed.append([float(row[0]) for row in values])
                        ciou_weight = 1.0
                        boundary_weight = 0.0
                    elif method["arm"] == "plain_rank":
                        ciou_weight = float(method["plain_ciou_weight"])
                        boundary_weight = float(method["plain_boundary_weight"])
                    elif bool(global_flags[group_index]):
                        ciou_weight = float(method["hard_ciou_weight"])
                        boundary_weight = float(method["hard_boundary_weight"])
                    else:
                        ciou_weight = float(method["ordinary_ciou_weight"])
                        boundary_weight = float(method["ordinary_boundary_weight"])
                    if method.get("tail_reward_mode") != "raw_ciou":
                        transformed.append(
                            [
                                ciou_weight * ciou_rank
                                + boundary_weight * boundary_rank
                                for ciou_rank, boundary_rank in zip(
                                    ciou_ranks, boundary_ranks
                                )
                            ]
                        )
                    tail_ciou_queue.append_group([row[0] for row in values])
                    tail_boundary_queue.append_group([row[1] for row in values])
                    tail_queue_trace.append(
                        {
                            "global_group_index": group_index,
                            "hard_flag": bool(global_flags[group_index]),
                            "ciou_weight": ciou_weight,
                            "boundary_weight": boundary_weight,
                            "pre_ciou_sha256": pre_ciou_hash,
                            "pre_boundary_sha256": pre_boundary_hash,
                            "post_ciou_sha256": tail_ciou_queue.sha256(),
                            "post_boundary_sha256": tail_boundary_queue.sha256(),
                            "queue_length": len(tail_ciou_queue.values),
                        }
                    )
                transformed_tensor = torch.tensor(
                    transformed, dtype=torch.float32, device=accelerator.device
                )
                local_start = accelerator.process_index * len(rollout_groups)
                local_rewards = transformed_tensor[
                    local_start : local_start + len(rollout_groups)
                ]
                for group, ranked_rewards in zip(rollout_groups, local_rewards):
                    group["rewards"] = ranked_rewards
                    if not tail_positive_only_enabled:
                        group["advantages"] = group_standardized_advantages(
                            ranked_rewards
                        )
                    group["reward_std"] = ranked_rewards.std(unbiased=False)
                    group["reward_range"] = (
                        ranked_rewards.max() - ranked_rewards.min()
                    )
                    group["nonconstant"] = bool(
                        float(group["reward_std"].detach().item())
                        > float(method["reward_std_epsilon"])
                    )
            nonconstant_reward_groups += sum(
                1 for group in rollout_groups if bool(group["nonconstant"])
            )
            total_rollout_groups += len(rollout_groups)
            multitrajectory_groups += sum(
                group["unique_trajectories"] >= 2 for group in rollout_groups
            )
            improved_over_greedy_rollouts += sum(
                int(group["improved_over_greedy"].sum().item())
                for group in rollout_groups
            )
            improved_over_greedy_groups += sum(
                bool(group["improved_over_greedy"].any().item())
                for group in rollout_groups
            )
            if method.get("exploration") == "per_prefix_topm_collision_support":
                for group in rollout_groups:
                    supports = group["calibrated_effective_supports"]
                    effective_support_hits += int(
                        (
                            supports
                            >= float(method["target_effective_support"])
                            - float(method["effective_support_tolerance"])
                        )
                        .sum()
                        .item()
                    )
                    effective_support_decisions += supports.numel()
            local_rollout_stats = torch.tensor(
                [
                    float(len(rollout_groups)),
                    float(sum(1 for group in rollout_groups if group["nonconstant"])),
                    float(sum(group["reward_std"].detach().item() for group in rollout_groups)),
                    float(sum(group["reward_range"].detach().item() for group in rollout_groups)),
                    float(
                        sum(
                            group["advantages"].detach().abs().mean().item()
                            for group in rollout_groups
                        )
                    ),
                    float(sum(group["unique_trajectories"] for group in rollout_groups)),
                    float(len(rollout_groups) * int(method["rollouts_per_prompt"])),
                    float(sum(group["unique_trajectories"] >= 2 for group in rollout_groups)),
                    float(
                        sum(
                            group["improved_over_greedy"].sum().item()
                            for group in rollout_groups
                        )
                    ),
                    float(
                        sum(group["evidence_gate"].mean().item() for group in rollout_groups)
                    ),
                ],
                dtype=torch.float32,
                device=accelerator.device,
            )
            global_rollout_stats = accelerator.gather(local_rollout_stats).reshape(
                -1, local_rollout_stats.numel()
            ).sum(dim=0)
            global_group_count = max(float(global_rollout_stats[0].item()), 1.0)
            rollout_summary = {
                "rollout_group_count": int(global_rollout_stats[0].item()),
                "nonconstant_reward_group_count": int(global_rollout_stats[1].item()),
                "reward_std_mean": float(global_rollout_stats[2].item() / global_group_count),
                "reward_range_mean": float(global_rollout_stats[3].item() / global_group_count),
                "advantage_abs_mean": float(global_rollout_stats[4].item() / global_group_count),
                "unique_trajectory_mean": float(global_rollout_stats[5].item() / global_group_count),
                "grammar_valid_trajectory_fraction": 1.0,
                "behavior_logprob_detached": 1,
                "multitrajectory_group_count": int(global_rollout_stats[7].item()),
                "improved_over_greedy_rollout_count": int(
                    global_rollout_stats[8].item()
                ),
                "evidence_gate_mean": float(
                    global_rollout_stats[9].item() / global_group_count
                ),
            }
            if (
                improvement_only_enabled
                or greedy_relative_enabled
                or hierarchical_prefix_enabled
                or pareto_geometry_enabled
                or rank_pareto_geometry_enabled
                or native_rank_pareto_enabled
                or depth_local_geometry_enabled
                or signed_native_depth_local_enabled
                or native_rank_local_enabled
                or scale_stratified_native_rank_local_enabled
                or bidirectional_coarse_fine_enabled
                or uncertainty_native_rank_local_enabled
                or action_budget_native_rank_local_enabled
                or boundary_stratified_native_rank_local_enabled
                or confidence_gated_native_rank_local_enabled
                or margin_calibrated_native_rank_local_enabled
            ):
                local_active = torch.tensor(
                    [
                        sum(
                            group["improvement_advantage_active_count"]
                            for group in rollout_groups
                        ),
                        sum(
                            group["negative_advantage_active_count"]
                            for group in rollout_groups
                        ),
                        len(rollout_groups) * int(method["rollouts_per_prompt"]),
                    ],
                    dtype=torch.float32,
                    device=accelerator.device,
                )
                global_active = accelerator.gather(local_active).reshape(-1, 3).sum(0)
                rollout_summary["improvement_advantage_active_fraction"] = float(
                    global_active[0].item() / max(global_active[2].item(), 1.0)
                )
                if hierarchical_prefix_enabled:
                    rollout_summary["hierarchical_prefix_credit"] = True
                if pareto_geometry_enabled:
                    rollout_summary["pareto_geometry_credit"] = True
                if rank_pareto_geometry_enabled:
                    rollout_summary["rank_pareto_geometry_credit"] = True
                if native_rank_pareto_enabled:
                    rollout_summary["native_anchored_rank_pareto_credit"] = True
                if depth_local_geometry_enabled:
                    rollout_summary["depth_local_geometry_credit"] = True
                if depth_local_shuffle_enabled:
                    rollout_summary["shuffled_depth_local_geometry_credit"] = True
                if depth_local_rarity_free_enabled:
                    rollout_summary["rarity_free_depth_local_geometry_credit"] = True
                if signed_native_depth_local_enabled:
                    rollout_summary["asymmetric_signed_native_relative_depth_local_credit"] = True
                if native_rank_local_enabled:
                    rollout_summary["native_anchored_rank_local_credit"] = True
                if scale_stratified_native_rank_local_enabled:
                    rollout_summary["scale_stratified_native_rank_local_credit"] = True
                if bidirectional_coarse_fine_enabled:
                    rollout_summary["bidirectional_coarse_fine_native_geometry_credit"] = True
                if uncertainty_native_rank_local_enabled:
                    global_uncertainty = accelerator.gather(
                        torch.stack(
                            [group["rollout_uncertainty"] for group in rollout_groups]
                        )
                    )
                    rollout_summary.update(
                        {
                            "uncertainty_calibrated_native_rank_local_credit": True,
                            "rollout_uncertainty_mean": float(global_uncertainty.mean().item()),
                            "rollout_uncertainty_p95": float(
                                torch.quantile(global_uncertainty.float(), 0.95).item()
                            ),
                            "rollout_confidence_floor": float(
                                method["uncertainty_confidence_floor"]
                            ),
                        }
                    )
                if action_budget_native_rank_local_enabled:
                    global_action_changes = accelerator.gather(
                        torch.stack(
                            [group["action_change_count"] for group in rollout_groups]
                        )
                    )
                    rollout_summary.update(
                        {
                            "action_budget_native_rank_local_credit": True,
                            "action_change_count_mean": float(
                                global_action_changes.mean().item()
                            ),
                            "action_change_count_p95": float(
                                torch.quantile(global_action_changes.float(), 0.95).item()
                            ),
                            "action_budget": int(method["action_budget"]),
                            "action_budget_excess_penalty": float(
                                method["action_budget_excess_penalty"]
                            ),
                        }
                    )
                if boundary_stratified_native_rank_local_enabled:
                    rollout_summary["boundary_stratified_native_rank_local_credit"] = True
                if confidence_gated_native_rank_local_enabled:
                    global_uncertainty = accelerator.gather(
                        torch.stack(
                            [group["rollout_uncertainty"] for group in rollout_groups]
                        )
                    )
                    rollout_summary.update(
                        {
                            "confidence_gated_native_rank_local_credit": True,
                            "rollout_uncertainty_mean": float(global_uncertainty.mean().item()),
                            "rollout_uncertainty_p95": float(
                                torch.quantile(global_uncertainty.float(), 0.95).item()
                            ),
                            "rollout_confidence_threshold": float(
                                method["confidence_threshold"]
                            ),
                            "rollout_confidence_floor": float(method["confidence_floor"]),
                        }
                    )
                if margin_calibrated_native_rank_local_enabled:
                    rollout_summary["margin_calibrated_native_rank_local_credit"] = True
                if greedy_relative_enabled:
                    rollout_summary["greedy_relative_positive_fraction"] = float(
                        global_active[0].item() / max(global_active[2].item(), 1.0)
                    )
                    rollout_summary["greedy_relative_negative_fraction"] = float(
                        global_active[1].item() / max(global_active[2].item(), 1.0)
                    )
            if greedy_preference_enabled:
                local_preference = torch.tensor(
                    [
                        sum(bool(group["preference_active"]) for group in rollout_groups),
                        len(rollout_groups),
                        sum(
                            float(group["preference_reward_gain"])
                            for group in rollout_groups
                            if bool(group["preference_active"])
                        ),
                    ],
                    dtype=torch.float32,
                    device=accelerator.device,
                )
                global_preference = accelerator.gather(local_preference).reshape(
                    -1, 3
                ).sum(0)
                active_groups = max(global_preference[0].item(), 1.0)
                rollout_summary.update(
                    {
                        "preference_active_group_count": int(
                            global_preference[0].item()
                        ),
                        "preference_active_group_fraction": float(
                            global_preference[0].item()
                            / max(global_preference[1].item(), 1.0)
                        ),
                        "preference_active_reward_gain_mean": float(
                            global_preference[2].item() / active_groups
                        ),
                    }
                )
                local_weights = torch.tensor(
                    [
                        sum(
                            float(group["preference_weight"])
                            for group in rollout_groups
                            if bool(group["preference_active"])
                        ),
                        sum(bool(group["preference_active"]) for group in rollout_groups),
                        max(
                            (
                                float(group["preference_weight"])
                                for group in rollout_groups
                                if bool(group["preference_active"])
                            ),
                            default=0.0,
                        ),
                    ],
                    dtype=torch.float32,
                    device=accelerator.device,
                )
                global_weights = accelerator.gather(local_weights).reshape(-1, 3)
                weight_sum = float(global_weights[:, 0].sum().item())
                weight_count = float(global_weights[:, 1].sum().item())
                rollout_summary.update(
                    {
                        "preference_active_weight_mean": weight_sum
                        / max(weight_count, 1.0),
                        "preference_active_weight_max": float(
                            global_weights[:, 2].max().item()
                        ),
                    }
                )
            if any(group["raw_geometry"] is not None for group in rollout_groups):
                global_raw_geometry = accelerator.gather(
                    torch.stack(
                        [group["raw_geometry"] for group in rollout_groups]
                    )
                ).reshape(-1, 4, 2)
                rollout_summary.update(
                    {
                        "raw_ciou_mean": float(
                            global_raw_geometry[:, :, 0].mean().item()
                        ),
                        "raw_boundary_iou_mean": float(
                            global_raw_geometry[:, :, 1].mean().item()
                        ),
                    }
                )
                if tail_queue_trace is not None:
                    rollout_summary.update(
                        {
                            "tail_queue_trace": tail_queue_trace,
                            "tail_ciou_queue_sha256": tail_ciou_queue.sha256(),
                            "tail_boundary_queue_sha256": tail_boundary_queue.sha256(),
                        }
                    )
                if paired_view_enabled:
                    if any(
                        group["paired_view_geometry"] is None
                        or group["native_geometry"] is None
                        or group["paired_view_native_geometry"] is None
                        for group in rollout_groups
                    ):
                        raise RuntimeError("Paired-view summary references missing geometry")
                    global_augmented_geometry = accelerator.gather(
                        torch.stack(
                            [group["paired_view_geometry"] for group in rollout_groups]
                        )
                    ).reshape(-1, 4, 2)
                    global_native_geometry = accelerator.gather(
                        torch.stack([group["native_geometry"] for group in rollout_groups])
                    ).reshape(-1, 2)
                    global_augmented_native_geometry = accelerator.gather(
                        torch.stack(
                            [group["paired_view_native_geometry"] for group in rollout_groups]
                        )
                    ).reshape(-1, 2)
                    clean_reward = global_raw_geometry.mean(dim=(1, 2))
                    augmented_reward = global_augmented_geometry.mean(dim=(1, 2))
                    clean_centered = clean_reward - clean_reward.mean()
                    augmented_centered = augmented_reward - augmented_reward.mean()
                    covariance = (clean_centered * augmented_centered).mean()
                    correlation_denominator = (
                        clean_centered.square().mean().sqrt()
                        * augmented_centered.square().mean().sqrt()
                    )
                    if float(correlation_denominator.item()) <= 1e-12:
                        paired_correlation = 0.0
                        paired_correlation_finite = False
                    else:
                        paired_correlation = float(
                            (covariance / correlation_denominator).item()
                        )
                        paired_correlation_finite = math.isfinite(paired_correlation)
                    minimum_improvement = float(method["minimum_improvement"])
                    clean_joint = (
                        global_raw_geometry - global_native_geometry[:, None, :]
                        > minimum_improvement
                    ).all(dim=2)
                    augmented_joint = (
                        global_augmented_geometry
                        - global_augmented_native_geometry[:, None, :]
                        > minimum_improvement
                    ).all(dim=2)
                    joint_positive_fraction = float(
                        (clean_joint & augmented_joint).float().mean().item()
                    )
                    rollout_summary.update(
                        {
                            "paired_view_clean_reward_mean": float(clean_reward.mean().item()),
                            "paired_view_augmented_reward_mean": float(
                                augmented_reward.mean().item()
                            ),
                            "paired_view_reward_correlation": paired_correlation,
                            "paired_view_reward_correlation_finite": paired_correlation_finite,
                            "paired_view_joint_positive_fraction": joint_positive_fraction,
                            "paired_view_summary_gate_passed": bool(
                                paired_correlation_finite and joint_positive_fraction >= 0.20
                            ),
                        }
                    )
            if method.get("exploration") == "per_prefix_topm_collision_support":
                global_temperatures = accelerator.gather(
                    torch.stack([group["action_temperatures"] for group in rollout_groups])
                )
                global_supports = accelerator.gather(
                    torch.stack([group["action_supports"] for group in rollout_groups])
                )
                global_native_supports = accelerator.gather(
                    torch.stack(
                        [group["native_effective_supports"] for group in rollout_groups]
                    )
                )
                global_calibrated_supports = accelerator.gather(
                    torch.stack(
                        [
                            group["calibrated_effective_supports"]
                            for group in rollout_groups
                        ]
                    )
                )
                global_sampled_codes = accelerator.gather(
                    torch.stack([group["sampled_codes"] for group in rollout_groups])
                )
                global_native_entropies = accelerator.gather(
                    torch.stack([group["native_entropies"] for group in rollout_groups])
                )
                global_controlled_entropies = accelerator.gather(
                    torch.stack([group["controlled_entropies"] for group in rollout_groups])
                )
                global_top_masses = accelerator.gather(
                    torch.stack([group["top_support_masses"] for group in rollout_groups])
                )
                global_kls = accelerator.gather(
                    torch.stack([group["controlled_native_kls"] for group in rollout_groups])
                )
                effective_support_target_reached_fraction = float(
                    (
                        global_calibrated_supports
                        >= float(method["target_effective_support"])
                        - float(method["effective_support_tolerance"])
                    )
                    .float()
                    .mean()
                    .item()
                )
                rollout_summary.update(
                    {
                        "action_temperatures": global_temperatures.detach().cpu().tolist(),
                        "frozen_action_supports": global_supports.detach().cpu().tolist(),
                        "native_effective_supports": global_native_supports.detach().cpu().tolist(),
                        "calibrated_effective_supports": global_calibrated_supports.detach().cpu().tolist(),
                        "sampled_code_trajectories": global_sampled_codes.detach()
                        .cpu()
                        .tolist(),
                        "temperature_mean": float(global_temperatures.mean().item()),
                        "temperature_min_observed": float(global_temperatures.min().item()),
                        "temperature_max_observed": float(global_temperatures.max().item()),
                        "native_effective_support_mean": float(global_native_supports.mean().item()),
                        "calibrated_effective_support_mean": float(global_calibrated_supports.mean().item()),
                        "native_shannon_entropy_mean": float(global_native_entropies.mean().item()),
                        "controlled_shannon_entropy_mean": float(global_controlled_entropies.mean().item()),
                        "top8_native_mass_mean": float(global_top_masses.mean().item()),
                        "controlled_to_native_kl_mean": float(global_kls.mean().item()),
                        "target_effective_support_reached_fraction": effective_support_target_reached_fraction,
                    }
                )

            if predicted_evidence_scope_enabled:
                local_states = torch.cat(
                    [group["pes_evidence_state"] for group in rollout_groups]
                )
                local_scope = torch.cat(
                    [group["pes_scope_mask"] for group in rollout_groups], dim=0
                )
                global_states = accelerator.gather(local_states)
                global_scope = accelerator.gather(local_scope)
                pes_state_counts += torch.stack(
                    [(local_states == state).sum() for state in range(3)]
                ).to(pes_state_counts.dtype)
                pes_state_observations += int(local_states.numel())
                rollout_summary.update(
                    {
                        "pes_confident_fraction": float((global_states == 0).float().mean().item()),
                        "pes_ambiguous_fraction": float((global_states == 1).float().mean().item()),
                        "pes_unsupported_fraction": float((global_states == 2).float().mean().item()),
                        "pes_scope_active_fraction": float((global_scope > 0).float().mean().item()),
                        "pes_scope_length_mean": float(global_scope.sum(dim=1).float().mean().item()),
                    }
                )

            for policy_epoch in range(policy_epochs):
                _disable_dropout(model)
                optimizer.zero_grad(set_to_none=True)
                policy_losses: list[torch.Tensor] = []
                replay_losses: list[torch.Tensor] = []
                ratios: list[torch.Tensor] = []
                clip_fractions: list[torch.Tensor] = []
                anchor_kl_value = torch.zeros((), dtype=torch.float32, device=accelerator.device)
                policy_loss_anchor = torch.zeros_like(anchor_kl_value)
                grounded_interface_loss = torch.zeros_like(anchor_kl_value)
                # DDP permits local gradient accumulation under no_sync.  The
                # subsequent paired-null backward synchronizes the sum before
                # the single optimizer update.
                with accelerator.no_sync(model):
                    for group in rollout_groups:
                        if greedy_preference_enabled:
                            current_best_log_prob = score_sampled_sequences(
                                model,
                                group["greedy_inputs"],
                                group["preference_best_sequence"],
                                grammar_ids,
                                float(method["native_scoring_temperature"]),
                            )
                            current_greedy_log_prob = score_sampled_sequences(
                                model,
                                group["greedy_inputs"],
                                group["preference_greedy_sequence"],
                                grammar_ids,
                                float(method["native_scoring_temperature"]),
                            )
                            group_loss, _log_odds_shift, ratio = (
                                greedy_crossing_preference_loss(
                                    current_best_log_prob,
                                    current_greedy_log_prob,
                                    group["preference_old_best_log_prob"],
                                    group["preference_old_log_odds"],
                                    bool(group["preference_active"]),
                                )
                            )
                            group_loss = (
                                float(group["preference_weight"]) * group_loss
                            )
                            clip_fraction = (
                                (ratio - 1.0).abs()
                                > float(method["clip_epsilon"])
                            ).float().mean()
                        else:
                            current_log_probs = score_sampled_sequences(
                                model,
                                group["prompt_inputs"],
                                group["sequence_ids"],
                                grammar_ids,
                                group["action_temperatures"]
                                if group["action_temperatures"] is not None
                                else float(method["temperature"]),
                                group["action_supports"],
                            )
                            if predicted_evidence_scope_enabled:
                                current_action_log_probs = score_sampled_sequences(
                                    model,
                                    group["prompt_inputs"],
                                    group["sequence_ids"],
                                    grammar_ids,
                                    group["action_temperatures"]
                                    if group["action_temperatures"] is not None
                                    else float(method["temperature"]),
                                    group["action_supports"],
                                    return_action_terms=True,
                                )
                                group_loss, ratio, clip_fraction = clipped_scope_policy_loss(
                                    current_action_log_probs,
                                    group["behavior_action_log_probs"],
                                    group["advantages"],
                                    group["pes_scope_mask"],
                                    float(method["clip_epsilon"]),
                                )
                            else:
                                group_loss, ratio, clip_fraction = clipped_policy_loss(
                                    current_log_probs,
                                    group["behavior_log_probs"],
                                    group["advantages"],
                                    float(method["clip_epsilon"]),
                                )
                        policy_losses.append(group_loss)
                        if (verified_replay_enabled or verified_prefix_replay_enabled or pareto_prefix_replay_enabled) and bool(group["replay_active"]):
                            replay_log_prob = score_sampled_sequences(
                                model,
                                group["greedy_inputs"],
                                group["replay_sequence"],
                                grammar_ids,
                                group["replay_action_temperatures"]
                                if group["replay_action_temperatures"] is not None
                                else float(method["temperature"]),
                                group["replay_action_supports"],
                                group["replay_action_mask"],
                            )
                            replay_losses.append(-replay_log_prob.mean())
                        ratios.append(ratio)
                        clip_fractions.append(clip_fraction)
                    if greedy_preference_enabled and policy_epoch == 0:
                        epoch_zero_deviation = float(
                            (torch.cat(ratios).detach() - 1.0).abs().max().item()
                        )
                        if epoch_zero_deviation > float(
                            method["max_epoch0_ratio_deviation"]
                        ):
                            raise RuntimeError(
                                "Greedy-preference frozen reference mismatch: "
                                f"epoch-zero max ratio deviation={epoch_zero_deviation}"
                            )
                    policy_loss = torch.stack(policy_losses).mean()
                    if not torch.isfinite(policy_loss):
                        raise FloatingPointError("Non-finite GR-CPPO policy loss")
                    replay_loss = (
                        torch.stack(replay_losses).mean()
                        if replay_losses
                        else policy_loss * 0.0
                    )
                    if not torch.isfinite(replay_loss):
                        raise FloatingPointError("Non-finite verified replay loss")
                    if anchor_kl_enabled:
                        if len(anchor_kl_samples) != len(anchor_kl_logits):
                            raise RuntimeError("Anchor-KL cache is unavailable for this worker")
                        # Measure the fixed buffer without retaining 64 model
                        # graphs.  If the hinge activates, replay each row and
                        # accumulate its scaled gradient under no_sync.
                        current_values: list[torch.Tensor] = []
                        with _without_gradient_checkpointing(model):
                            with torch.no_grad():
                                for sample, cached_codes in zip(anchor_kl_samples, anchor_kl_codes):
                                    current_logits, _ = collect_anchor_policy_logits(
                                        model,
                                        processor,
                                        sample,
                                        grammar_ids,
                                        accelerator.device,
                                        native_codes=cached_codes,
                                        no_grad=True,
                                    )
                                    current_values.append(
                                        anchor_categorical_kl(
                                            current_logits,
                                            [value.to(accelerator.device) for value in anchor_kl_logits[len(current_values)]],
                                        )
                                    )
                        if not current_values:
                            raise RuntimeError("Anchor-KL buffer unexpectedly empty")
                        anchor_kl_value = torch.stack(current_values).mean()
                        if not torch.isfinite(anchor_kl_value):
                            raise FloatingPointError("Anchor KL is non-finite")
                        if outer_step == 0 and policy_epoch == 0 and anchor_kl_value.detach().item() > 1e-6:
                            raise RuntimeError(
                                "Anchor-KL zero-effect initialization check failed: "
                                f"{anchor_kl_value.detach().item()}"
                            )
                        policy_loss_anchor = float(method["anchor_kl_lambda"]) * F.relu(
                            anchor_kl_value - float(method["anchor_kl_epsilon"])
                        )
                        if policy_loss_anchor.detach().item() > 0.0:
                            scale = float(method["anchor_kl_lambda"]) / len(anchor_kl_samples)
                            with _without_gradient_checkpointing(model):
                                for sample, cached_codes, cached_logits in zip(
                                    anchor_kl_samples, anchor_kl_codes, anchor_kl_logits
                                ):
                                    current_logits, _ = collect_anchor_policy_logits(
                                        model,
                                        processor,
                                        sample,
                                        grammar_ids,
                                        accelerator.device,
                                        native_codes=cached_codes,
                                    )
                                    sample_kl = anchor_categorical_kl(
                                        current_logits,
                                        [value.to(accelerator.device) for value in cached_logits],
                                    )
                                    accelerator.backward(scale * sample_kl)
                        anchor_kl_observations += 1
                        anchor_kl_history.append(float(anchor_kl_value.detach().item()))
                        anchor_kl_active_observations += int(
                            policy_loss_anchor.detach().item() > 0.0
                        )
                    if grounded_interface_enabled:
                        if grounded_interface_cfg is None:
                            raise RuntimeError("Grounded-interface settings are unavailable")
                        view_samples = build_target_preserving_view_samples(
                            samples,
                            brightness=float(grounded_interface_cfg["brightness"]),
                            contrast=float(grounded_interface_cfg["contrast"]),
                        )
                        view_inputs, view_answer_mask = build_supervised_inputs(
                            processor, view_samples
                        )
                        view_inputs = move_tensors(view_inputs, accelerator.device)
                        view_answer_mask = view_answer_mask.to(accelerator.device)
                        view_outputs = model(**view_inputs, use_cache=False)
                        grounded_interface_loss = answer_token_cross_entropy(
                            view_outputs.logits,
                            view_inputs["input_ids"],
                            view_inputs["attention_mask"],
                            view_answer_mask,
                        )
                        if not torch.isfinite(grounded_interface_loss):
                            raise FloatingPointError("Non-finite grounded-interface CE")
                    accelerator.backward(
                        float(method["policy_weight"]) * policy_loss
                        + float(method.get("verified_replay_weight", 0.0)) * replay_loss
                        + float(
                            grounded_interface_cfg["lambda_sup"]
                            if grounded_interface_enabled and grounded_interface_cfg is not None
                            else 0.0
                        )
                        * grounded_interface_loss
                    )
                positive_grad_norm = torch.sqrt(
                    torch.stack(
                        [
                            parameter.grad.detach().float().square().sum()
                            for parameter in parameters
                            if parameter.grad is not None
                        ]
                    ).sum()
                )
                if not torch.isfinite(positive_grad_norm):
                    raise FloatingPointError("Non-finite positive-only policy gradient norm")
                positive_policy_grad_observations += int(
                    float(positive_grad_norm.item())
                    > float(method.get("min_positive_policy_grad_norm", 0.0))
                )
                visual_grad_terms = [
                    parameter.grad.detach().float().square().sum()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and ".visual" in name
                    and parameter.grad is not None
                ]
                if grounded_interface_enabled and not visual_grad_terms:
                    raise RuntimeError("Grounded-interface produced no visual LoRA gradients")
                visual_gradient_norm = (
                    torch.sqrt(torch.stack(visual_grad_terms).sum())
                    if visual_grad_terms
                    else positive_grad_norm * 0.0
                )
                visual_gradient_norm_value = float(visual_gradient_norm.item())
                visual_gradient_norms.append(visual_gradient_norm_value)
                visual_gradient_norm_observations += int(
                    grounded_interface_enabled
                    and visual_gradient_norm_value
                    > float(grounded_interface_cfg["visual_gradient_threshold"])
                )

                null_ce_active = True
                margin_active = True
                sentinel_null_ce_value: float | None = None
                sentinel_margin_min_value: float | None = None
                tail_sentinel_penalty = torch.zeros(
                    (), dtype=torch.float32, device=accelerator.device
                )
                tail_margin_delta_q10: float | None = None
                tail_margin_violation_rate: float | None = None
                primal_dual_excess = 0.0
                if unified_sentinel_enabled:
                    if (
                        sentinel_inputs is None
                        or sentinel_answer_mask is None
                        or sentinel_no_target is None
                        or active_set_state is None
                        or tail_anchor_margins is None
                        or tail_sentinel_samples is None
                    ):
                        raise RuntimeError("Unified sentinel risk state is unavailable")
                    # One differentiable forward serves both constraints.  A
                    # fixed-shape payload (CE plus each local margin) is the
                    # only sentinel collective in this update, so all ranks
                    # execute the same protocol even when no repair is needed.
                    sentinel_outputs = model(**sentinel_inputs, use_cache=False)
                    null_ce, margins = canonical_null_ce_and_first_action_margin(
                        sentinel_outputs.logits,
                        sentinel_inputs["input_ids"],
                        sentinel_inputs["attention_mask"],
                        sentinel_answer_mask,
                        sentinel_no_target,
                        processor.tokenizer,
                        grammar_ids[0],
                    )
                    local_stats = torch.cat(
                        (null_ce.detach().float().reshape(1), margins.detach().float())
                    )
                    global_stats = accelerator.gather(local_stats)
                    local_sentinel_count = len(tail_sentinel_samples)
                    expected_stats = accelerator.num_processes * (local_sentinel_count + 1)
                    if global_stats.numel() != expected_stats:
                        raise RuntimeError(
                            "Unified sentinel gather returned an unexpected shape"
                        )
                    rank_stats = global_stats.reshape(
                        accelerator.num_processes, local_sentinel_count + 1
                    )
                    global_current_ce = rank_stats[:, 0].mean()
                    global_current_margins = rank_stats[:, 1:].reshape(-1)
                    sentinel_null_ce_value = float(global_current_ce.item())
                    sentinel_margin_min_value = float(global_current_margins.min().item())
                    null_ce_active, margin_active = active_set_flags(
                        sentinel_null_ce_value,
                        sentinel_margin_min_value,
                        null_ce_budget=float(active_set_state["null_ce_budget"]),
                        margin_budget=float(active_set_state["margin_budget"]),
                    )
                    margin_penalty = F.relu(
                        float(active_set_state["margin_budget"]) - margins
                    ).mean()
                    # Select the globally worst q10 rows from the detached
                    # measurement, while retaining gradient through the local
                    # margins in the repair loss.
                    local_count = len(tail_sentinel_samples)
                    local_start = accelerator.process_index * local_count
                    local_anchor = tail_anchor_margins[
                        local_start : local_start + local_count
                    ]
                    local_delta = margins.detach().float() - local_anchor
                    margin_delta = global_current_margins - tail_anchor_margins
                    quantile = torch.quantile(
                        margin_delta,
                        float(method["sentinel_tail_quantile"]),
                        interpolation="linear",
                    )
                    tail_margin_delta_q10 = float(quantile.item())
                    budget = float(method["sentinel_degradation_budget"])
                    tail_margin_violation_rate = float(
                        (margin_delta < -budget).float().mean().item()
                    )
                    tail_margin_violation_rates.append(tail_margin_violation_rate)
                    if primal_dual_null_risk_enabled:
                        primal_dual_excess = max(
                            0.0,
                            -tail_margin_delta_q10
                            - float(method["sentinel_degradation_budget"]),
                        ) / max(float(method["sentinel_degradation_budget"]), 1e-6)
                    selected = local_delta <= quantile
                    global_selected = int(
                        (margin_delta <= quantile).sum().item()
                    )
                    if global_selected:
                        tail_sentinel_penalty = F.relu(
                            local_anchor[selected] - margins[selected] - budget
                        ).sum() * accelerator.num_processes / global_selected
                    else:
                        tail_sentinel_penalty = null_ce * 0.0
                    if accelerator.is_main_process:
                        active_set_observations += 1
                        null_ce_active_observations += int(null_ce_active)
                        margin_active_observations += int(margin_active)
                elif active_set_enabled:
                    if (
                        sentinel_inputs is None
                        or sentinel_answer_mask is None
                        or sentinel_no_target is None
                        or active_set_state is None
                    ):
                        raise RuntimeError("Active-set risk state is unavailable")
                    sentinel_outputs = model(**sentinel_inputs, use_cache=False)
                    null_ce, margins = canonical_null_ce_and_first_action_margin(
                        sentinel_outputs.logits,
                        sentinel_inputs["input_ids"],
                        sentinel_inputs["attention_mask"],
                        sentinel_answer_mask,
                        sentinel_no_target,
                        processor.tokenizer,
                        grammar_ids[0],
                    )
                    sentinel_null_ce_value = float(
                        accelerator.gather(null_ce.detach().float().reshape(1))
                        .mean()
                        .item()
                    )
                    sentinel_margin_min_value = float(
                        accelerator.gather(margins.detach().float()).min().item()
                    )
                    null_ce_active, margin_active = active_set_flags(
                        sentinel_null_ce_value,
                        sentinel_margin_min_value,
                        null_ce_budget=float(active_set_state["null_ce_budget"]),
                        margin_budget=float(active_set_state["margin_budget"]),
                    )
                    margin_penalty = F.relu(
                        float(active_set_state["margin_budget"]) - margins
                    ).mean()
                    if accelerator.is_main_process:
                        active_set_observations += 1
                        null_ce_active_observations += int(null_ce_active)
                        margin_active_observations += int(margin_active)
                else:
                    supervised_inputs, answer_mask = build_supervised_inputs(
                        processor, samples
                    )
                    supervised_inputs = move_tensors(
                        supervised_inputs, accelerator.device
                    )
                    answer_mask = answer_mask.to(accelerator.device)
                    no_target = torch.tensor(
                        [bool(sample["no_target"]) for sample in samples],
                        dtype=torch.bool,
                        device=accelerator.device,
                    )
                    supervised_outputs = model(**supervised_inputs, use_cache=False)
                    null_ce, margins = canonical_null_ce_and_first_action_margin(
                        supervised_outputs.logits,
                        supervised_inputs["input_ids"],
                        supervised_inputs["attention_mask"],
                        answer_mask,
                        no_target,
                        processor.tokenizer,
                        grammar_ids[0],
                    )
                    margin_penalty = F.relu(
                        float(method["margin_target"]) - margins
                    ).mean()
                if tail_enabled and not unified_sentinel_enabled:
                    if (
                        tail_sentinel_samples is None
                        or tail_anchor_margins is None
                    ):
                        raise RuntimeError("TB-GPPO sentinel state is unavailable")
                    with torch.no_grad():
                        local_current_margins = evaluate_null_margins(
                            model,
                            processor,
                            tail_sentinel_samples,
                            grammar_ids[0],
                            accelerator.device,
                            microbatch=int(method["sentinel_microbatch"]),
                        ).float()
                    global_current_margins = accelerator.gather(
                        local_current_margins
                    )
                    margin_delta = global_current_margins - tail_anchor_margins
                    quantile = torch.quantile(
                        margin_delta,
                        float(method["sentinel_tail_quantile"]),
                        interpolation="linear",
                    )
                    tail_margin_delta_q10 = float(quantile.item())
                    budget = float(method["sentinel_degradation_budget"])
                    tail_margin_violation_rate = float(
                        (margin_delta < -budget).float().mean().item()
                    )
                    tail_margin_violation_rates.append(tail_margin_violation_rate)
                    local_count = len(tail_sentinel_samples)
                    local_start = accelerator.process_index * local_count
                    local_anchor = tail_anchor_margins[
                        local_start : local_start + local_count
                    ]
                    local_delta = local_current_margins - local_anchor
                    selected = local_delta <= quantile
                    global_selected = int(
                        accelerator.gather(selected.sum().reshape(1))
                        .sum()
                        .item()
                    )
                    selected_samples = [
                        sample
                        for sample, keep in zip(
                            tail_sentinel_samples, selected.cpu().tolist()
                        )
                        if keep
                    ]
                    if selected_samples:
                        selected_current = evaluate_null_margins(
                            model,
                            processor,
                            selected_samples,
                            grammar_ids[0],
                            accelerator.device,
                            microbatch=int(method["sentinel_microbatch"]),
                        )
                        selected_anchor = local_anchor[selected]
                        tail_sentinel_penalty = F.relu(
                            selected_anchor - selected_current - budget
                        ).sum() * accelerator.num_processes / max(global_selected, 1)
                    else:
                        tail_sentinel_penalty = null_ce * 0.0
                constraint_loss = (
                    float(method["null_ce_weight"])
                    * float(null_ce_active)
                    * null_ce
                    + float(method["margin_weight"])
                    * float(margin_active)
                    * margin_penalty
                    + float(method.get("sentinel_tail_weight", 0.0))
                    * tail_sentinel_penalty
                )
                if primal_dual_null_risk_enabled:
                    # The multiplier is detached state; gradients remain only
                    # through the differentiable sentinel repair penalty.
                    constraint_loss = constraint_loss - float(
                        method.get("sentinel_tail_weight", 0.0)
                    ) * tail_sentinel_penalty + primal_dual_lambda * tail_sentinel_penalty
                loss_value = (
                    float(method["policy_weight"]) * policy_loss.detach()
                    + float(method.get("verified_replay_weight", 0.0))
                    * replay_loss.detach()
                    + float(
                        grounded_interface_cfg["lambda_sup"]
                        if grounded_interface_enabled and grounded_interface_cfg is not None
                        else 0.0
                    )
                    * grounded_interface_loss.detach()
                    + policy_loss_anchor.detach()
                    + constraint_loss.detach()
                )
                if not torch.isfinite(constraint_loss) or not torch.isfinite(loss_value):
                    raise FloatingPointError(
                        f"Non-finite GR-CPPO loss at outer step {outer_step}, epoch {policy_epoch}"
                    )
                accelerator.backward(constraint_loss)
                accelerator.clip_grad_norm_(parameters, float(optimizer_config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                if primal_dual_null_risk_enabled:
                    primal_dual_observations += 1
                    if primal_dual_excess > 0.0:
                        primal_dual_active_observations += 1
                    primal_dual_lambda = min(
                        float(method["primal_dual_lambda_cap"]),
                        max(
                            0.0,
                            primal_dual_lambda
                            + float(method["primal_dual_eta"]) * primal_dual_excess,
                        ),
                    )
                    primal_dual_lambda_history.append(primal_dual_lambda)

                all_ratios = torch.cat(ratios)
                reward_values = torch.cat([group["rewards"] for group in rollout_groups])
                scalar_values = torch.stack(
                    (
                        loss_value.float(),
                        policy_loss.detach().float(),
                        null_ce.detach().float(),
                        margin_penalty.detach().float(),
                        margins.detach().mean().float(),
                        reward_values.detach().mean().float(),
                        all_ratios.detach().mean().float(),
                        (all_ratios.detach() - 1.0).abs().mean().float(),
                        torch.stack(clip_fractions).mean().detach().float(),
                        positive_grad_norm.detach().float(),
                        (all_ratios.detach() - 1.0).abs().median().float(),
                        grounded_interface_loss.detach().float(),
                        visual_gradient_norm.detach().float(),
                    )
                )
                gathered = accelerator.gather(scalar_values).reshape(
                    -1, scalar_values.numel()
                ).mean(dim=0)
                values = [float(value.item()) for value in gathered]
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError("Non-finite gathered GR-CPPO metrics")
                item: dict[str, Any] = {
                    "policy_epoch": policy_epoch,
                    "loss": values[0],
                    "policy_loss": values[1],
                    "null_ce": values[2],
                    "margin_penalty": values[3],
                    "first_action_margin": values[4],
                    "reward_mean": values[5],
                    "ratio_mean": values[6],
                    "ratio_abs_deviation": values[7],
                    "clip_fraction": values[8],
                    "positive_policy_grad_norm": values[9],
                    "ratio_abs_deviation_median": values[10],
                    "grounded_interface_ce": values[11],
                    "visual_gradient_norm": values[12],
                    **rollout_summary,
                }
                if anchor_kl_enabled:
                    global_anchor_kl = accelerator.gather(
                        torch.stack((anchor_kl_value.detach().float(), policy_loss_anchor.detach().float()))
                    ).reshape(-1, 2).mean(dim=0)
                    item.update(
                        {
                            "anchor_kl_nats": float(global_anchor_kl[0].item()),
                            "anchor_kl_hinge_loss": float(global_anchor_kl[1].item()),
                            "anchor_kl_epsilon": float(method["anchor_kl_epsilon"]),
                            "anchor_kl_lambda": float(method["anchor_kl_lambda"]),
                            "anchor_kl_buffer_rows": int(method["anchor_buffer_rows"]),
                        }
                    )
                if verified_replay_enabled or verified_prefix_replay_enabled or pareto_prefix_replay_enabled:
                    local_replay_stats = torch.tensor(
                        [
                            float(replay_loss.detach().item()),
                            float(sum(bool(group["replay_active"]) for group in rollout_groups)),
                            float(len(rollout_groups)),
                        ],
                        dtype=torch.float32,
                        device=accelerator.device,
                    )
                    global_replay_stats = accelerator.gather(local_replay_stats).reshape(-1, 3)
                    item.update(
                        {
                            "verified_replay_loss": float(global_replay_stats[:, 0].mean().item()),
                            "verified_replay_active_fraction": float(
                                global_replay_stats[:, 1].sum().item()
                                / max(global_replay_stats[:, 2].sum().item(), 1.0)
                            ),
                            "verified_replay_weight": float(method["verified_replay_weight"]),
                            "verified_replay_mode": method["verified_replay_mode"],
                        }
                    )
                if active_set_enabled:
                    item.update(
                        {
                            "sentinel_null_ce": sentinel_null_ce_value,
                            "sentinel_margin_min": sentinel_margin_min_value,
                            "sentinel_null_ce_budget": active_set_state["null_ce_budget"],
                            "sentinel_margin_budget": active_set_state["margin_budget"],
                            "null_ce_active": int(null_ce_active),
                            "margin_active": int(margin_active),
                            "active_set_any": int(null_ce_active or margin_active),
                        }
                    )
                if tail_enabled:
                    item.update(
                        {
                            "tail_sentinel_penalty": float(
                                tail_sentinel_penalty.detach().item()
                                if unified_sentinel_enabled
                                else accelerator.gather(
                                    tail_sentinel_penalty.detach().reshape(1)
                                ).mean().item()
                            ),
                            "tail_margin_delta_q10": tail_margin_delta_q10,
                            "tail_margin_violation_rate": tail_margin_violation_rate,
                        }
                    )
                if primal_dual_null_risk_enabled:
                    item.update(
                        {
                            "primal_dual_lambda": primal_dual_lambda,
                            "primal_dual_excess": primal_dual_excess,
                        }
                    )
                local_ratio_changed = int(
                    policy_epoch == 1
                    and values[7] > float(method["min_epoch2_ratio_abs_deviation"])
                )
                if accelerator.is_main_process:
                    ratio_change_observations += local_ratio_changed
                item["epoch2_ratio_changed"] = local_ratio_changed if policy_epoch == 1 else 0
                if accelerator.is_main_process:
                    metrics["steps"].append({"outer_step": outer_step, **item})
                    write_json_atomic(output_dir / "metrics.json", metrics)
                    if outer_step % int(config["logging"]["log_every"]) == 0:
                        print(json.dumps(metrics["steps"][-1], sort_keys=True), flush=True)
            outer_step += 1
        if len(dataloader) == 0:
            raise RuntimeError("Paired GR-CPPO dataloader is empty")

    active_set_risk_gate_passed = True
    final_sentinel_null_ce: float | None = None
    final_sentinel_margin_min: float | None = None
    if active_set_enabled and not unified_sentinel_enabled:
        if (
            sentinel_inputs is None
            or sentinel_answer_mask is None
            or sentinel_no_target is None
            or active_set_state is None
        ):
            raise RuntimeError("Active-set final risk state is unavailable")
        _disable_dropout(model)
        with torch.no_grad():
            final_sentinel_outputs = model(**sentinel_inputs, use_cache=False)
            final_null_ce, final_margins = canonical_null_ce_and_first_action_margin(
                final_sentinel_outputs.logits,
                sentinel_inputs["input_ids"],
                sentinel_inputs["attention_mask"],
                sentinel_answer_mask,
                sentinel_no_target,
                processor.tokenizer,
                grammar_ids[0],
            )
        final_sentinel_null_ce = float(
            accelerator.gather(final_null_ce.detach().float().reshape(1)).mean().item()
        )
        final_sentinel_margin_min = float(
            accelerator.gather(final_margins.detach().float()).min().item()
        )
        final_ce_violation, final_margin_violation = active_set_flags(
            final_sentinel_null_ce,
            final_sentinel_margin_min,
            null_ce_budget=float(active_set_state["null_ce_budget"]),
            margin_budget=float(active_set_state["margin_budget"]),
        )
        active_set_risk_gate_passed = not (
            final_ce_violation or final_margin_violation
        )
        if accelerator.is_main_process:
            metrics["active_set"].update(
                {
                    "observations": active_set_observations,
                    "null_ce_active_observations": null_ce_active_observations,
                    "margin_active_observations": margin_active_observations,
                    "null_ce_active_fraction": (
                        null_ce_active_observations / active_set_observations
                        if active_set_observations
                        else 0.0
                    ),
                    "margin_active_fraction": (
                        margin_active_observations / active_set_observations
                        if active_set_observations
                        else 0.0
                    ),
                    "final_sentinel_null_ce": final_sentinel_null_ce,
                    "final_sentinel_margin_min": final_sentinel_margin_min,
                    "final_risk_gate_passed": active_set_risk_gate_passed,
                }
            )

    final_tail_margin_delta_q10: float | None = None
    final_tail_margin_violation_rate: float | None = None
    if tail_enabled and not unified_sentinel_enabled:
        if tail_sentinel_samples is None or tail_anchor_margins is None:
            raise RuntimeError("TB-GPPO final sentinel state is unavailable")
        _disable_dropout(model)
        with torch.no_grad():
            local_final_tail_margins = evaluate_null_margins(
                model,
                processor,
                tail_sentinel_samples,
                grammar_ids[0],
                accelerator.device,
                microbatch=int(method["sentinel_microbatch"]),
            ).float()
        global_final_tail_margins = accelerator.gather(local_final_tail_margins)
        final_tail_delta = global_final_tail_margins - tail_anchor_margins
        final_tail_margin_delta_q10 = float(
            torch.quantile(
                final_tail_delta,
                float(method["sentinel_tail_quantile"]),
                interpolation="linear",
            ).item()
        )
        final_tail_margin_violation_rate = float(
            (
                final_tail_delta
                < -float(method["sentinel_degradation_budget"])
            )
            .float()
            .mean()
            .item()
        )
        if accelerator.is_main_process:
            metrics["tail_gppo"].update(
                {
                    "final_tail_margin_delta_q10": final_tail_margin_delta_q10,
                    "final_tail_margin_violation_rate": (
                        final_tail_margin_violation_rate
                    ),
                }
            )

    if unified_sentinel_enabled:
        if (
            sentinel_inputs is None
            or sentinel_answer_mask is None
            or sentinel_no_target is None
            or active_set_state is None
            or tail_anchor_margins is None
            or tail_sentinel_samples is None
        ):
            raise RuntimeError("Unified sentinel final state is unavailable")
        _disable_dropout(model)
        with torch.no_grad():
            final_outputs = model(**sentinel_inputs, use_cache=False)
            final_null_ce, final_margins = canonical_null_ce_and_first_action_margin(
                final_outputs.logits,
                sentinel_inputs["input_ids"],
                sentinel_inputs["attention_mask"],
                sentinel_answer_mask,
                sentinel_no_target,
                processor.tokenizer,
                grammar_ids[0],
            )
        local_final_stats = torch.cat(
            (final_null_ce.detach().float().reshape(1), final_margins.detach().float())
        )
        global_final_stats = accelerator.gather(local_final_stats)
        local_sentinel_count = len(tail_sentinel_samples)
        expected_stats = accelerator.num_processes * (local_sentinel_count + 1)
        if global_final_stats.numel() != expected_stats:
            raise RuntimeError("Unified final sentinel gather returned an unexpected shape")
        final_rank_stats = global_final_stats.reshape(
            accelerator.num_processes, local_sentinel_count + 1
        )
        final_sentinel_null_ce = float(final_rank_stats[:, 0].mean().item())
        final_global_margins = final_rank_stats[:, 1:].reshape(-1)
        final_sentinel_margin_min = float(final_global_margins.min().item())
        final_ce_violation, final_margin_violation = active_set_flags(
            final_sentinel_null_ce,
            final_sentinel_margin_min,
            null_ce_budget=float(active_set_state["null_ce_budget"]),
            margin_budget=float(active_set_state["margin_budget"]),
        )
        active_set_risk_gate_passed = not (final_ce_violation or final_margin_violation)
        final_tail_delta = final_global_margins - tail_anchor_margins
        final_tail_margin_delta_q10 = float(
            torch.quantile(
                final_tail_delta,
                float(method["sentinel_tail_quantile"]),
                interpolation="linear",
            ).item()
        )
        final_tail_margin_violation_rate = float(
            (final_tail_delta < -float(method["sentinel_degradation_budget"]))
            .float()
            .mean()
            .item()
        )
        if accelerator.is_main_process:
            metrics["active_set"].update(
                {
                    "final_sentinel_null_ce": final_sentinel_null_ce,
                    "final_sentinel_margin_min": final_sentinel_margin_min,
                    "final_risk_gate_passed": active_set_risk_gate_passed,
                    "sentinel_protocol": "single_gather_ce_plus_margins",
                }
            )
            metrics["tail_gppo"].update(
                {
                    "final_tail_margin_delta_q10": final_tail_margin_delta_q10,
                    "final_tail_margin_violation_rate": final_tail_margin_violation_rate,
                    "sentinel_protocol": "single_gather_ce_plus_margins",
                }
            )

    full_data_schedule_enabled = bool(
        tail_enabled and method.get("full_data_schedule") is True
    )
    consumed_pair_count = 0
    consumed_row_count = 0
    if full_data_schedule_enabled:
        # Each rank writes its observed IDs; the shared output directory lets
        # the main process compute an exact distributed union after the final
        # dataloader step instead of trusting manifest size or step count.
        rank_ids_path = output_dir / f"consumed_pair_ids_rank{accelerator.process_index}.json"
        write_json_atomic(rank_ids_path, sorted(consumed_pair_ids))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            all_ids: set[str] = set()
            for rank in range(accelerator.num_processes):
                path = output_dir / f"consumed_pair_ids_rank{rank}.json"
                if path.is_file():
                    all_ids.update(str(value) for value in json.loads(path.read_text(encoding="utf-8")))
            consumed_pair_count = len(all_ids)
            consumed_row_count = consumed_pair_count * 2
    coverage_gate_passed = True
    if full_data_schedule_enabled:
        coverage_gate_passed = bool(
            consumed_pair_count >= int(method.get("minimum_consumed_pairs", 2560))
            and consumed_row_count >= int(method.get("minimum_consumed_rows", 5120))
        )
        if accelerator.is_main_process:
            metrics["tail_gppo"].update(
                {
                    "consumed_pair_count": consumed_pair_count,
                    "consumed_row_count": consumed_row_count,
                    "minimum_consumed_pairs": int(method.get("minimum_consumed_pairs", 2560)),
                    "minimum_consumed_rows": int(method.get("minimum_consumed_rows", 5120)),
                    "full_data_coverage_gate_passed": coverage_gate_passed,
                }
            )

    gate_values = torch.tensor(
        [
            float(nonconstant_reward_groups),
            float(ratio_change_observations),
            float(positive_policy_grad_observations),
            float(total_rollout_groups),
            float(multitrajectory_groups),
            float(improved_over_greedy_rollouts),
            float(improved_over_greedy_groups),
            float(effective_support_hits),
            float(effective_support_decisions),
        ],
        dtype=torch.float32,
        device=accelerator.device,
    )
    gate_values = accelerator.gather(gate_values).reshape(-1, 9).sum(dim=0)
    total_nonconstant_groups = int(gate_values[0].item())
    total_ratio_change_observations = int(gate_values[1].item())
    total_positive_policy_grad_observations = int(gate_values[2].item())
    global_total_groups = int(gate_values[3].item())
    global_multitrajectory_groups = int(gate_values[4].item())
    global_improved_rollouts = int(gate_values[5].item())
    global_improved_groups = int(gate_values[6].item())
    global_support_hits = int(gate_values[7].item())
    global_support_decisions = int(gate_values[8].item())
    anchor_kl_gate_passed = True
    anchor_kl_p95: float | None = None
    anchor_kl_max: float | None = None
    global_anchor_kl_active_observations = anchor_kl_active_observations
    global_anchor_kl_observations = anchor_kl_observations
    if anchor_kl_enabled:
        local_history = torch.tensor(
            anchor_kl_history,
            dtype=torch.float32,
            device=accelerator.device,
        )
        global_history = accelerator.gather(local_history).reshape(-1)
        if global_history.numel() != max_steps * policy_epochs * accelerator.num_processes:
            raise RuntimeError("Anchor-KL history has an unexpected distributed shape")
        anchor_kl_p95 = float(torch.quantile(global_history, 0.95).item())
        anchor_kl_max = float(global_history.max().item())
        anchor_counts = accelerator.gather(
            torch.tensor(
                [anchor_kl_active_observations, anchor_kl_observations],
                dtype=torch.float32,
                device=accelerator.device,
            )
        ).reshape(-1, 2).sum(dim=0)
        global_anchor_kl_active_observations = int(anchor_counts[0].item())
        global_anchor_kl_observations = int(anchor_counts[1].item())
        anchor_kl_gate_passed = bool(
            math.isfinite(anchor_kl_p95)
            and math.isfinite(anchor_kl_max)
            and anchor_kl_p95 <= 0.20
            # The first measurement must be zero at initialization, but a
            # valid trust-region screen must activate its hinge after updates.
            and global_anchor_kl_active_observations > 0
        )
    is_effective_support = (
        method.get("exploration") == "per_prefix_topm_collision_support"
    )
    support_gate_passed = (
        not is_effective_support
        or (
            global_support_decisions > 0
            and global_support_hits / global_support_decisions
            >= float(method["min_target_support_reached_fraction"])
        )
    )
    if is_effective_support and max_steps == 1:
        gate_passed = (
            global_total_groups
            == max_steps
            * int(config["data"]["pairs_per_device_batch"])
            * accelerator.num_processes
            and total_nonconstant_groups
            >= int(method["min_nonconstant_reward_groups"])
            and global_multitrajectory_groups
            >= int(method["min_multitrajectory_groups"])
            and global_improved_rollouts
            >= int(method["min_improved_over_greedy_rollouts"])
            and total_positive_policy_grad_observations > 0
            and total_ratio_change_observations > 0
            and support_gate_passed
        )
    elif is_effective_support:
        gate_passed = (
            global_total_groups
            == max_steps
            * int(config["data"]["pairs_per_device_batch"])
            * accelerator.num_processes
            and total_nonconstant_groups / max(global_total_groups, 1) >= 0.25
            and global_multitrajectory_groups / max(global_total_groups, 1) >= 0.25
            and global_improved_groups / max(global_total_groups, 1) >= 0.10
            and total_positive_policy_grad_observations > 0
            and total_ratio_change_observations > 0
            and support_gate_passed
        )
    else:
        gate_passed = (
            total_nonconstant_groups > 0
            and total_ratio_change_observations > 0
            and support_gate_passed
        )
    tail_risk_gate_passed = True
    tail_mean_violation_rate: float | None = None
    if tail_enabled:
        tail_mean_violation_rate = sum(tail_margin_violation_rates) / max(
            len(tail_margin_violation_rates), 1
        )
        tail_risk_gate_passed = (
            len(tail_margin_violation_rates) == max_steps * policy_epochs
            and math.isfinite(tail_mean_violation_rate)
            and final_tail_margin_delta_q10 is not None
            and math.isfinite(final_tail_margin_delta_q10)
            and final_tail_margin_delta_q10
            >= -float(method["sentinel_degradation_budget"])
            and final_tail_margin_violation_rate is not None
            and math.isfinite(final_tail_margin_violation_rate)
            and final_tail_margin_violation_rate < 0.05
            and tail_ciou_queue is not None
            and tail_boundary_queue is not None
            and len(tail_ciou_queue.values) == int(method["fifo_capacity"])
            and len(tail_boundary_queue.values) == int(method["fifo_capacity"])
        )
    primal_dual_gate_passed = True
    primal_dual_lambda_max: float | None = None
    primal_dual_lambda_active_fraction: float | None = None
    if primal_dual_null_risk_enabled:
        primal_dual_lambda_max = max(primal_dual_lambda_history, default=float("nan"))
        primal_dual_lambda_active_fraction = (
            primal_dual_active_observations / primal_dual_observations
            if primal_dual_observations
            else float("nan")
        )
        primal_dual_gate_passed = bool(
            primal_dual_observations == max_steps * policy_epochs
            and primal_dual_lambda_history
            and all(math.isfinite(value) for value in primal_dual_lambda_history)
            and math.isfinite(primal_dual_lambda_max)
            and primal_dual_lambda_max < float(method["primal_dual_lambda_cap"])
        )
        if accelerator.is_main_process:
            metrics["tail_gppo"].update(
                {
                    "primal_dual_lambda_init": float(method["primal_dual_lambda_init"]),
                    "primal_dual_eta": float(method["primal_dual_eta"]),
                    "primal_dual_lambda_cap": float(method["primal_dual_lambda_cap"]),
                    "primal_dual_observations": primal_dual_observations,
                    "primal_dual_active_observations": primal_dual_active_observations,
                    "primal_dual_active_fraction": primal_dual_lambda_active_fraction,
                    "primal_dual_lambda_final": primal_dual_lambda,
                    "primal_dual_lambda_max": primal_dual_lambda_max,
                    "primal_dual_gate_passed": primal_dual_gate_passed,
                }
            )
    pes_coverage_gate_passed = True
    pes_state_fractions: list[float] | None = None
    if predicted_evidence_scope_enabled:
        global_pes_counts = accelerator.reduce(pes_state_counts, reduction="sum")
        total_states = int(global_pes_counts.sum().item())
        pes_state_fractions = [
            float(value.item()) / max(total_states, 1)
            for value in global_pes_counts
        ]
        pes_coverage_gate_passed = bool(
            total_states > 0
            and all(fraction >= 0.20 for fraction in pes_state_fractions if fraction > 0.0)
            and sum(fraction > 0.0 for fraction in pes_state_fractions) >= 2
        )
    representation_gate_passed = True
    final_representation_summary = None
    if representation_summary is not None:
        final_representation_summary = visual_projector_adapter_summary(
            accelerator.unwrap_model(model)
        )
        peak_bytes = torch.tensor(
            [torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0],
            dtype=torch.long,
            device=accelerator.device,
        )
        peak_bytes = int(accelerator.gather(peak_bytes).max().item())
        final_representation_summary.update(representation_summary)
        final_representation_summary["peak_memory_bytes_max_rank"] = peak_bytes
        visual_threshold = (
            float(grounded_interface_cfg["visual_gradient_threshold"])
            if grounded_interface_enabled and grounded_interface_cfg is not None
            else 0.0
        )
        visual_fraction = (
            visual_gradient_norm_observations / max(len(visual_gradient_norms), 1)
            if grounded_interface_enabled
            else float(total_positive_policy_grad_observations > 0)
        )
        final_representation_summary["visual_gradient_threshold"] = visual_threshold
        final_representation_summary["visual_gradient_norm_observations"] = len(
            visual_gradient_norms
        )
        final_representation_summary["visual_gradient_above_threshold_fraction"] = visual_fraction
        final_representation_summary["visual_gradient_gate"] = (
            visual_fraction
            >= (
                float(grounded_interface_cfg["visual_gradient_min_fraction"])
                if grounded_interface_enabled and grounded_interface_cfg is not None
                else 0.0
            )
        )
        if grounded_interface_enabled:
            probe_model = accelerator.unwrap_model(model)
            if "probe_inputs" not in locals():
                raise RuntimeError("Grounded-interface visual probe inputs were not retained")
            probe_model.eval()
            probe_model.set_adapter("anchor")
            with torch.no_grad():
                post_anchor_logits = probe_model(**probe_inputs, use_cache=False).logits.float()
            activate_visual_projector_adapters(probe_model)
            with torch.no_grad():
                post_visual_logits = probe_model(**probe_inputs, use_cache=False).logits.float()
            post_delta = float((post_visual_logits - post_anchor_logits).abs().max().item())
            final_representation_summary["postupdate_anchor_max_abs_logit_delta"] = post_delta
            final_representation_summary["postupdate_logit_effect_gate"] = post_delta > 1e-8
        representation_gate_passed = bool(
            final_representation_summary["visual_gradient_gate"]
            and final_representation_summary["anchor_trainable_tensors"] == 0
        )
        if grounded_interface_enabled:
            representation_gate_passed = bool(
                representation_gate_passed
                and final_representation_summary["postupdate_logit_effect_gate"]
            )
        metrics["representation_adapter"] = final_representation_summary
    gate_passed = (
        gate_passed
        and active_set_risk_gate_passed
        and tail_risk_gate_passed
        and primal_dual_gate_passed
        and representation_gate_passed
        and anchor_kl_gate_passed
        and pes_coverage_gate_passed
        and coverage_gate_passed
    )
    if accelerator.is_main_process:
        metrics["validity_gate"] = {
            "nonconstant_reward_groups": total_nonconstant_groups,
            "epoch2_ratio_change_observations": total_ratio_change_observations,
            "require_nonconstant_rewards": bool(method["require_nonconstant_rewards"]),
            "require_epoch2_ratio_change": bool(method["require_epoch2_ratio_change"]),
            "positive_policy_grad_observations": total_positive_policy_grad_observations,
            "rollout_group_count": global_total_groups,
            "multitrajectory_group_count": global_multitrajectory_groups,
            "improved_over_greedy_rollout_count": global_improved_rollouts,
            "improved_over_greedy_group_count": global_improved_groups,
            "target_effective_support_reached_fraction": (
                global_support_hits / global_support_decisions
                if global_support_decisions
                else effective_support_target_reached_fraction
            ),
            "min_target_support_reached_fraction": method.get(
                "min_target_support_reached_fraction"
            ),
            "effective_support_gate_passed": support_gate_passed,
            "active_set_risk_gate_passed": active_set_risk_gate_passed,
            "final_sentinel_null_ce": final_sentinel_null_ce,
            "final_sentinel_margin_min": final_sentinel_margin_min,
            "tail_risk_gate_passed": tail_risk_gate_passed,
            "representation_gate_passed": representation_gate_passed,
            "anchor_kl_gate_passed": anchor_kl_gate_passed,
            "primal_dual_gate_passed": primal_dual_gate_passed,
            "primal_dual_lambda_final": primal_dual_lambda if primal_dual_null_risk_enabled else None,
            "primal_dual_lambda_max": primal_dual_lambda_max,
            "primal_dual_active_fraction": primal_dual_lambda_active_fraction,
            "anchor_kl_active_observations": global_anchor_kl_active_observations,
            "anchor_kl_observations": global_anchor_kl_observations,
            "anchor_kl_active_fraction": (
                global_anchor_kl_active_observations / global_anchor_kl_observations
                if global_anchor_kl_observations
                else None
            ),
            "anchor_kl_p95_nats": anchor_kl_p95,
            "anchor_kl_max_nats": anchor_kl_max,
            "pes_coverage_gate_passed": pes_coverage_gate_passed,
            "pes_state_fractions": pes_state_fractions,
            "full_data_coverage_gate_passed": coverage_gate_passed,
            "consumed_pair_count": consumed_pair_count,
            "consumed_row_count": consumed_row_count,
            "tail_mean_margin_violation_rate": tail_mean_violation_rate,
            "final_tail_margin_delta_q10": final_tail_margin_delta_q10,
            "final_tail_margin_violation_rate": final_tail_margin_violation_rate,
            "passed": gate_passed,
        }
        if not gate_passed:
            metrics["status"] = "failed_validity_gate"
            write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()
    if not gate_passed:
        raise RuntimeError(
            "GR-CPPO validity gate failed: need at least one nonconstant K=4 reward "
            "group, useful diverse trajectories, a positive-only gradient, one changed "
            "epoch-2 importance ratio, a valid effective-support controller, and all "
            "registered selective-risk gates"
        )

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        adapter_dir = output_dir / "adapter"
        adapter_artifact = save_trainable_lora_adapter(
            unwrapped,
            adapter_dir,
            representation_mode=representation_mode_enabled,
        )
        processor.save_pretrained(adapter_dir)
        metrics["adapter_artifact"] = str(adapter_artifact)
        metrics["status"] = "finished"
        metrics["steps_completed"] = outer_step
        metrics["optimizer_updates_completed"] = outer_step * policy_epochs
        write_json_atomic(output_dir / "metrics.json", metrics)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
