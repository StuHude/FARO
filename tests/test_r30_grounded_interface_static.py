from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.tail_gppo_contract import (
    UNIFIED_GROUNDED_INTERFACE_STAGE,
    validate_tail_gppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    FARO_ROOT
    / "Sa2VA/projects/samtok_selective/configs/"
    / "fepo_tb_gppo_plain_rank_unified_grounded_interface_10step_2gpu.py"
)


def test_r30_grounded_interface_contract(monkeypatch, tmp_path):
    anchor = FARO_ROOT / "outputs/samtok_selective/continued_sft_to500/adapter"
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.setenv("SAMTOK_STANDALONE_ADAPTER", str(anchor))
    config = runpy.run_path(str(CONFIG))["config"]
    # Contract validation checks the registered path; model hashes are outside
    # this local static probe.
    config["checkpoint"]["adapter_init"] = str(anchor)
    config["checkpoint"]["output_dir"] = str(
        FARO_ROOT / "outputs/samtok_selective" / UNIFIED_GROUNDED_INTERFACE_STAGE
    )
    config["provenance"]["manifest_path"] = str(
        Path(config["checkpoint"]["output_dir"]) / "provenance_manifest.json"
    )
    validate_tail_gppo_config(config, FARO_ROOT)
    assert config["stage"] == UNIFIED_GROUNDED_INTERFACE_STAGE
    assert config["data"]["expected_rows"] == 5120
    assert config["optimizer"]["max_steps"] == 10
    assert config["model"]["adapter_mode"] == "frozen_anchor_plus_visual_projector"
    grounded = config["tail_gppo"]["grounded_interface"]
    assert grounded["target_source"] == "same_row_ground_truth_mask_codes"
    assert grounded["lambda_sup"] == 0.10
    assert grounded["uses_pixvl_teacher"] is False
    assert grounded["uses_opd"] is False


def test_r30_trainer_has_supervised_dual_view_and_visual_effect_gates():
    source = (
        FARO_ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"
    ).read_text(encoding="utf-8")
    assert "build_target_preserving_view_samples" in source
    assert "answer_token_cross_entropy" in source
    assert "visual_gradient_above_threshold_fraction" in source
    assert "postupdate_anchor_max_abs_logit_delta" in source
    assert "self_supervised_loop" in source


def test_r30_submission_wrapper_has_resource_and_data_guards():
    submit = (
        FARO_ROOT / "scripts/submit_samtok_tb_gppo_grounded_interface.sh"
    ).read_text(encoding="utf-8")
    assert "rows >= 5000" in submit
    assert "dna-fepo-grounded-interface-10step-2g" in submit
    assert "submit_samtok_tb_gppo.sh" in submit


def test_candidate_probe_includes_r30_contract_and_credit_screen():
    probe = (FARO_ROOT / "tools/run_fepo_candidate_probe.py").read_text(
        encoding="utf-8"
    )
    assert '"R30": "projects.samtok_selective.configs.' in probe
    assert '"R30": "scripts/submit_samtok_tb_gppo_grounded_interface.sh"' in probe
    assert '"R30": native_anchored_rank_local_geometry_advantages' in probe


def test_monitors_accept_nested_visual_representation_adapter():
    main_monitor = (FARO_ROOT / "scripts/monitor_fepo_screens.sh").read_text(
        encoding="utf-8"
    )
    late_monitor = (FARO_ROOT / "scripts/monitor_fepo_late_screens.sh").read_text(
        encoding="utf-8"
    )
    for source in (main_monitor, late_monitor):
        assert 'adapter/visual/adapter_config.json' in source
        assert 'adapter/visual/adapter_model.safetensors' in source
