#!/usr/bin/env python3
"""Run the conditional A-PES contract probe entirely on CPU."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SA2VA_ROOT = ROOT / "Sa2VA"
for path in (SA2VA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projects.samtok_selective.tail_gppo_contract import validate_tail_gppo_config

from apes_probe import probability_gap_scope_masks


def _load_pes_config() -> dict:
    os.environ.setdefault("SAMTOK_BASE_CHECKPOINT", str(ROOT / "third_party/SAMTok"))
    os.environ.pop("SAMTOK_STANDALONE_ADAPTER", None)
    module = importlib.import_module(
        "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu"
    )
    return module.config


def main() -> None:
    config = _load_pes_config()
    validate_tail_gppo_config(config)
    method = config["tail_gppo"]
    manifest = ROOT / "data/fepo_existence/egfepo_train_5120.jsonl"
    rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
    assert rows >= 5000, rows
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] >= 10
    assert method["rollouts_per_prompt"] == 4
    assert method["pes_evidence_shuffle_seed"] == 1907

    # Four rows exercise confident, ambiguous, unsupported, and empty-change
    # handling.  Values are probabilities under the calibrated support.
    entropy = torch.tensor(
        [[0.10, 0.20, 0.30], [0.60, 0.70, 0.80], [2.00, 2.00, 2.00], [0.20, 0.30, 0.40]],
        dtype=torch.float32,
    )
    p_native = torch.tensor(
        [[0.80, 0.80, 0.80], [0.70, 0.70, 0.70], [0.60, 0.60, 0.60], [0.90, 0.90, 0.90]],
        dtype=torch.float32,
        requires_grad=True,
    )
    p_sampled = torch.tensor(
        [[0.75, 0.75, 0.75], [0.50, 0.50, 0.50], [0.00, 0.00, 0.00], [0.70, 0.70, 0.70]],
        dtype=torch.float32,
        requires_grad=True,
    )
    sampled_codes = [[0, 9, 2], [8, 1, 7], [0, 1, 3], [0, 1, 2]]
    native_codes = [0, 1, 2]
    scope, states, gap = probability_gap_scope_masks(
        entropy,
        p_native,
        p_sampled,
        sampled_codes,
        native_codes,
        confident_gap=0.10,
        ambiguous_gap=0.40,
    )
    # The fourth row is ambiguous by gap but has no code change, so its scope
    # remains empty despite state 1.
    assert states.tolist() == [0, 1, 2, 1]
    assert torch.equal(scope, torch.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32))
    assert torch.allclose(gap, p_native.detach() - p_sampled.detach())
    assert not scope.requires_grad and not states.requires_grad and not gap.requires_grad

    shuffled_scope, shuffled_states, _ = probability_gap_scope_masks(
        entropy,
        p_native,
        p_sampled,
        sampled_codes,
        native_codes,
        confident_gap=0.10,
        ambiguous_gap=0.40,
        shuffle_seed=1907,
    )
    generator = torch.Generator()
    generator.manual_seed(1907)
    permutation = torch.randperm(4, generator=generator)
    assert torch.equal(shuffled_states, states[permutation])
    assert torch.isfinite(shuffled_scope).all()
    print(json.dumps({
        "variant": "A-PES",
        "manifest_rows": rows,
        "steps": config["optimizer"]["max_steps"],
        "rollouts_per_prompt": method["rollouts_per_prompt"],
        "states": states.tolist(),
        "shuffled_permutation_seed": 1907,
        "shuffled_states": shuffled_states.tolist(),
        "scope_nonzero": int(scope.sum().item()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
