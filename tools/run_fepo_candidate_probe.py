#!/usr/bin/env python3
"""Offline contract and credit probe for the pending FEPO candidate ladder.

This does not train or evaluate a model.  It catches configuration drift while
the dnacoding queue is unavailable, and exercises each registered credit rule
on the same deterministic K=4 geometry/code group.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import torch

# Keep the offline probe runnable from the repository root without requiring
# callers to remember the worker-only PYTHONPATH setup used by rjob jobs.
ROOT = Path(__file__).resolve().parents[1]
SA2VA_ROOT = ROOT / "Sa2VA"
if str(SA2VA_ROOT) not in sys.path:
    sys.path.insert(0, str(SA2VA_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.samtok_selective.fepo_gr_cppo_trainer import (
    bidirectional_coarse_fine_native_geometry_advantages,
    confidence_gated_native_rank_local_geometry_advantages,
    margin_calibrated_native_rank_local_geometry_advantages,
    native_anchored_rank_local_geometry_advantages,
    paired_view_native_rank_local_geometry_advantages,
    boundary_bottleneck_paired_view_geometry_advantages,
    action_budget_native_rank_local_geometry_advantages,
    scale_stratified_native_rank_local_geometry_advantages,
    uncertainty_calibrated_native_rank_local_geometry_advantages,
    predicted_evidence_scope_masks,
    clipped_scope_policy_loss,
)
from projects.samtok_selective.tail_gppo_contract import validate_tail_gppo_config


CONFIGS = {
    "R21": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_native_rank_local_10step_2gpu",
    "R22": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_scale_stratified_native_rank_local_10step_2gpu",
    "R23": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_bidirectional_coarse_fine_10step_2gpu",
    "R24": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_anchor_kl_10step_2gpu",
    "R25": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_uncertainty_native_rank_local_10step_2gpu",
    "R26": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_conservative_null_tail_10step_2gpu",
    "R27": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_confidence_gated_native_rank_local_10step_2gpu",
    "R28": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_margin_calibrated_native_rank_local_10step_2gpu",
    "R29": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_primal_dual_null_risk_10step_2gpu",
    "R30": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_grounded_interface_10step_2gpu",
    "R35": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu",
    "PV": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu",
    "BA": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu",
    "AB": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_action_budget_native_rank_local_10step_2gpu",
    "BS": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu",
    "PES": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu",
    "PES_SHUFFLED": "projects.samtok_selective.configs.fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_shuffled_10step_2gpu",
}

SUBMITTERS = {
    "R21": "scripts/submit_samtok_tb_gppo_native_rank_local.sh",
    "R22": "scripts/submit_samtok_tb_gppo_scale_stratified_native_rank_local.sh",
    "R23": "scripts/submit_samtok_tb_gppo_bidirectional_coarse_fine.sh",
    "R24": "scripts/submit_samtok_tb_gppo_anchor_kl.sh",
    "R25": "scripts/submit_samtok_tb_gppo_uncertainty_native_rank_local.sh",
    "R26": "scripts/submit_samtok_tb_gppo_conservative_null_tail.sh",
    "R27": "scripts/submit_samtok_tb_gppo_confidence_gated_native_rank_local.sh",
    "R28": "scripts/submit_samtok_tb_gppo_margin_calibrated_native_rank_local.sh",
    "R29": "scripts/submit_samtok_tb_gppo_primal_dual_null_risk.sh",
    "R30": "scripts/submit_samtok_tb_gppo_grounded_interface.sh",
    "R35": "scripts/submit_samtok_tb_gppo_safe_visual_interface.sh",
    "PV": "scripts/submit_samtok_tb_gppo_paired_view.sh",
    "BA": "scripts/submit_samtok_tb_gppo_boundary_bottleneck_paired_view.sh",
    "AB": "scripts/submit_samtok_tb_gppo_action_budget_native_rank_local.sh",
    "BS": "scripts/submit_samtok_tb_gppo_boundary_stratified_native_rank_local.sh",
    "PES": "scripts/submit_samtok_tb_gppo_predicted_evidence_scope.sh",
    "PES_SHUFFLED": "scripts/submit_samtok_tb_gppo_predicted_evidence_scope_shuffled.sh",
}


def _config(module_name: str) -> dict:
    # Config modules intentionally read these at import time; use a harmless
    # placeholder so the probe never points at a PixVL checkpoint.
    os.environ.setdefault("SAMTOK_BASE_CHECKPOINT", str(ROOT / "third_party/SAMTok"))
    os.environ.pop("SAMTOK_STANDALONE_ADAPTER", None)
    return importlib.import_module(module_name).config


def main() -> None:
    manifest = ROOT / "data/fepo_existence/egfepo_train_5120.jsonl"
    rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
    if rows < 5000:
        raise AssertionError(f"training manifest has only {rows} rows")

    contracts = {}
    base_submitter = (ROOT / "scripts/submit_samtok_tb_gppo.sh").read_text(encoding="utf-8")
    for required in ("dna-", "ailab-dnacoding", "rjob_tags.txt", "--positive-tags"):
        assert required in base_submitter, f"base submitter missing {required}"
    # The approved SAMTok weights live under the historical PixVL_ailab mount;
    # that path is allowed, while importing PixVL implementation modules is not.
    assert "projects.pixvl_" not in base_submitter.lower()
    for label, module_name in CONFIGS.items():
        cfg = _config(module_name)
        validate_tail_gppo_config(cfg)
        method = cfg["tail_gppo"]
        assert cfg["data"]["expected_rows"] >= 5120
        assert cfg["optimizer"]["max_steps"] >= 10
        assert int(method["rollouts_per_prompt"]) >= 4
        # The standalone configs expose the base under ``model`` and the
        # frozen adapter under ``checkpoint``; both must remain SAMTok-only.
        assert "samtok" in str(cfg["model"]["base_checkpoint"]).lower()
        # The approved SAMTok release is physically stored under the historical
        # PixVL_ailab mount; only the model identity must be SAMTok, while no
        # PixVL trainer/module may enter the execution path.
        base_path = str(cfg["model"]["base_checkpoint"]).lower()
        assert "samtok" in base_path
        assert "/projects/pixvl" not in base_path
        contracts[label] = {
            "stage": cfg["stage"],
            "rows": cfg["data"]["expected_rows"],
            "steps": cfg["optimizer"]["max_steps"],
            "rollouts": method["rollouts_per_prompt"],
        }
        submitter = (ROOT / SUBMITTERS[label]).read_text(encoding="utf-8")
        assert "dna-" in submitter, f"{label} submitter must preserve dna naming"
        assert "projects.pixvl_" not in submitter.lower()
        if label == "R29":
            assert method["primal_dual_risk"] == "lower10_current_minus_anchor_margin_excess"
            assert method["primal_dual_lambda_init"] == 1.0
            assert method["primal_dual_eta"] == 0.20
            assert method["primal_dual_lambda_cap"] == 4.0
        if label == "R30":
            grounded = method.get("grounded_interface")
            assert isinstance(grounded, dict)
            assert grounded["target_source"] == "same_row_ground_truth_mask_codes"
            assert grounded["lambda_sup"] == 0.10
            assert grounded["uses_pixvl_teacher"] is False
            assert grounded["uses_opd"] is False
        if label == "R35":
            grounded = method.get("grounded_interface")
            assert isinstance(grounded, dict)
            assert method["null_ce_weight"] == 2.0
            assert method["margin_weight"] == 1.0
            assert grounded["target_source"] == "same_row_ground_truth_mask_codes"
            assert grounded["uses_pixvl_teacher"] is False
            assert grounded["uses_opd"] is False
        if label == "PV":
            paired = method.get("paired_view_geometry")
            assert isinstance(paired, dict)
            assert paired["aggregation"] == "geometric_mean"
            assert paired["target_source"] == "same_row_ground_truth_mask_geometry"
            assert paired["uses_pixvl_teacher"] is False
        if label == "BA":
            paired = method.get("paired_view_geometry")
            assert isinstance(paired, dict)
            assert paired["aggregation"] == "boundary_bottleneck_min"
            assert paired["mode"] == "gt_verified_boundary_bottleneck_paired_view_reward"
            assert paired["uses_pixvl_teacher"] is False
        if label == "AB":
            assert method["pareto_credit_mode"] == "action_budget_native_rank_local"
            assert method["action_budget"] == 2
            assert method["action_budget_excess_penalty"] == 0.10
            assert method["depth_local_credit_policy"] == (
                "native_reference_action_budget_first_divergence"
            )
        if label == "BS":
            assert method["pareto_credit_mode"] == "boundary_stratified_native_rank_local"
            assert method["boundary_stratified_schedule"] is True
            assert method["boundary_sampling_mix"] == {
                "ordinary": 0.50,
                "thin": 0.25,
                "boundary_hard": 0.25,
            }
        if label == "PES":
            assert method["pareto_credit_mode"] == "predicted_evidence_scope"
            assert method["depth_local_credit_policy"] == (
                "native_reference_predicted_evidence_scope_first_divergence"
            )
            assert method["pes_confident_entropy"] == 0.35
            assert "pes_confident_mass" not in method
            assert "pes_ambiguous_mass" not in method
            assert method["pes_confident_margin"] == 1.0
            assert method["pes_ambiguous_margin"] == 0.25
        if label == "PES_SHUFFLED":
            assert method["pes_evidence_shuffle"] is True
            assert method["pes_evidence_shuffle_seed"] == 1907

    raw = torch.tensor(
        [[0.73, 0.31], [0.80, 0.37], [0.77, 0.43], [0.84, 0.39]],
        dtype=torch.float32,
    )
    native = torch.tensor([0.72, 0.30], dtype=torch.float32)
    codes = [[0, 1, 2], [0, 1, 3], [0, 4, 2], [5, 1, 2]]
    native_codes = [0, 1, 2]
    uncertainty = torch.tensor([0.10, 0.20, 0.55, 0.05], dtype=torch.float32)
    augmented = torch.tensor(
        [[0.72, 0.30], [0.79, 0.36], [0.76, 0.42], [0.83, 0.38]],
        dtype=torch.float32,
    )
    augmented_native = torch.tensor([0.71, 0.29], dtype=torch.float32)
    outputs = {
        "R21": native_anchored_rank_local_geometry_advantages(raw, native, codes, native_codes),
        "R22": scale_stratified_native_rank_local_geometry_advantages(
            raw, native, codes, native_codes, "small"
        ),
        "R23": bidirectional_coarse_fine_native_geometry_advantages(raw, native, codes, native_codes),
        "R25": uncertainty_calibrated_native_rank_local_geometry_advantages(
            raw, native, codes, native_codes, uncertainty
        ),
        "R27": confidence_gated_native_rank_local_geometry_advantages(
            raw, native, codes, native_codes, uncertainty
        ),
        "R28": margin_calibrated_native_rank_local_geometry_advantages(raw, native, codes, native_codes),
        # R30 changes only the supervised representation auxiliary loss; its
        # clean-view RL credit remains the R18 native rank-local rule.
        "R30": native_anchored_rank_local_geometry_advantages(raw, native, codes, native_codes),
        "R35": native_anchored_rank_local_geometry_advantages(raw, native, codes, native_codes),
        "PV": paired_view_native_rank_local_geometry_advantages(
            raw,
            augmented,
            native,
            augmented_native,
            codes,
            native_codes,
            native_codes,
        ),
        "BA": boundary_bottleneck_paired_view_geometry_advantages(
            raw,
            augmented,
            native,
            augmented_native,
            codes,
            native_codes,
            native_codes,
        ),
        "AB": action_budget_native_rank_local_geometry_advantages(
            raw,
            native,
            codes,
            native_codes,
        ),
        # BS changes only the deterministic training mixture; its local credit
        # remains the frozen R18 native-relative rule.
        "BS": native_anchored_rank_local_geometry_advantages(
            raw, native, codes, native_codes
        ),
        "PES": native_anchored_rank_local_geometry_advantages(
            raw, native, codes, native_codes
        ),
    }
    evidence_entropy = torch.tensor(
        [[0.10, 0.20, 0.30], [0.60, 0.70, 0.80], [2.00, 2.00, 2.00], [0.20, 0.30, 0.40]],
        dtype=torch.float32,
    )
    evidence_mass = torch.tensor(
        [[0.90, 0.90, 0.90], [0.30, 0.30, 0.30], [0.05, 0.05, 0.05], [0.80, 0.80, 0.80]],
        dtype=torch.float32,
    )
    scope, states = predicted_evidence_scope_masks(
        evidence_entropy,
        evidence_mass,
        codes,
        native_codes,
        native_margins=torch.tensor(
            [[1.2, 1.2, 1.2], [0.4, 0.4, 0.4], [0.1, 0.1, 0.1], [1.1, 1.1, 1.1]],
            dtype=torch.float32,
        ),
    )
    assert scope.shape == evidence_entropy.shape
    assert states.tolist() == [0, 1, 2, 0]
    behavior = torch.zeros_like(scope)
    current = torch.zeros_like(scope, requires_grad=True)
    scoped_loss, _, _ = clipped_scope_policy_loss(
        current, behavior, outputs["PES"], scope, 0.2
    )
    assert torch.isfinite(scoped_loss)
    credit_probe = {}
    for label, values in outputs.items():
        assert values.shape == (4,)
        assert torch.isfinite(values).all()
        assert (values >= 0).all()
        credit_probe[label] = {
            "nonzero": int((values > 0).sum()),
            "max": float(values.max()),
            "first_depth_nonzero": bool(values[0] > 0),
        }

    report = {"manifest_rows": rows, "contracts": contracts, "credit_probe": credit_probe}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
