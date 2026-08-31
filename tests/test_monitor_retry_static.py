from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eval_monitors_backoff_failed_control_plane_submissions():
    for name in ("monitor_fepo_screens.sh", "monitor_fepo_late_screens.sh"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "RETRY_SECONDS=${RETRY_SECONDS:-300}" in source
        assert "eval_retry_after" in source
        assert "retry_backoff" in source
        assert 'output="$OUT_ROOT/${name}_holdout512"' in source
        assert 'heartbeat+=" ${name}=eval_finished"' in source
        assert "PENDING_CONTROL_PLANE" in source
        assert "printf '%s\\n' SUBMITTED > \"$marker\"" in source
        assert "candidate=%s eval_submit_failed" in source
        assert "date +%s) + RETRY_SECONDS" in source
    late = (ROOT / "scripts" / "monitor_fepo_late_screens.sh").read_text(encoding="utf-8")
    assert 'pv_training_gate.json' in late
    assert 'closed_training_gate' in late


def test_adaptive_evaluator_keeps_requested_gpu_ladder():
    source = (ROOT / "scripts" / "submit_fepo_eval_adaptive.sh").read_text(
        encoding="utf-8"
    )
    assert "GPU_LEVELS=(8 6 4 2 1)" in source
    assert "POLL_SECONDS=${POLL_SECONDS:-300}" in source
    assert "(( gpu == 1 )) && exit 0" in source


def test_all_eval_fallbacks_fail_closed_on_status_query_errors():
    scripts = (
        "submit_fepo_text_calibration_adaptive.sh",
        "submit_nc_fepo_probe_adaptive.sh",
        "submit_nc_fepo_verifier_adaptive.sh",
        "submit_official_grefcoco_adaptive.sh",
        "submit_official_refcoco_adaptive.sh",
    )
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "GPU_LEVELS=(8 6 4 2 1)" in source or 'GPU_LEVELS_OVERRIDE=${GPU_LEVELS_OVERRIDE:-"8 6 4 2 1"}' in source
        assert "POLL_SECONDS=${POLL_SECONDS:-300}" in source
        assert "STATUS_QUERY_UNAVAILABLE" in source
        assert "status_rc" in source
        assert "STARTING, +gpu-" in source


def test_pes_shuffled_control_requires_valid_normal_training():
    source = (ROOT / "scripts" / "submit_pes_shuffled_after_pes_completion.sh").read_text(
        encoding="utf-8"
    )
    assert "d.get('status') == 'finished'" in source
    assert "gate.get('passed') is True" in source
    assert "gate.get('effective_support_gate_passed') is True" in source
    assert "gate.get('tail_risk_gate_passed') is True" in source
    assert "gate.get('pes_coverage_gate_passed') is True" in source
    assert "failed_validity_gate" not in source
    late = (ROOT / "scripts" / "monitor_fepo_late_screens.sh").read_text(
        encoding="utf-8"
    )
    assert "gate.get(\"pes_coverage_gate_passed\") is True" in late


def test_pes_transition_markers_are_recoverable_after_shell_restart():
    for name, script in (
        ("submit_pes_after_ab_rejection.sh", "submit_pes_after_ab_rejection.sh"),
        ("submit_pes_shuffled_after_pes_completion.sh", "submit_pes_shuffled_after_pes_completion.sh"),
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'printf \'%s\\n\' "$$" > "$STATE/pid"' in source
        assert "trap cleanup_pid EXIT" in source
        assert "trap 'exit 143' TERM" in source
    late = (ROOT / "scripts" / "monitor_fepo_late_screens.sh").read_text(
        encoding="utf-8"
    )
    assert "runner_active()" in late
    assert "kill -0 \"$pid\"" in late
    assert "submit_pes_after_ab_rejection.sh" in late


def test_pes_retry_refreshes_internal_proxy_after_control_plane_failure():
    for name in (
        "submit_pes_after_ab_rejection.sh",
        "submit_pes_shuffled_after_pes_completion.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "PROXY_SETUP_URL=${PROXY_SETUP_URL:-http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh}" in source
        assert "refresh_proxy_best_effort()" in source
        assert 'curl -fsSL --max-time 20 "$PROXY_SETUP_URL"' in source
        assert 'source /dev/stdin <<<"$setup"' in source
        assert "refresh_proxy_best_effort" in source.split("control_plane_unavailable", 1)[1]


def test_finalizer_accepts_closed_pv_training_gate_without_holdout():
    source = (ROOT / "scripts" / "monitor_finalize_matched_sft_pv.sh").read_text(
        encoding="utf-8"
    )
    assert "pv_decision_closed()" in source
    assert "decision\") == \"closed_training_gate\"" in source
    assert "pv_ready()" in source
    assert "! pv_ready" in source
    assert 'STATE="$STATE/r35_submit"' in source


def test_ab_transition_is_locked_after_rejected_bs_holdout():
    source = (ROOT / "scripts" / "submit_ab_after_screen_closure.sh").read_text(
        encoding="utf-8"
    )
    assert "flock -n 9" in source
    assert "fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu" in source
    assert "closed_metrics" in source
    assert "submit_samtok_tb_gppo_action_budget_native_rank_local.sh" in source
    assert "INTERVAL=${INTERVAL:-300}" in source
    late = (ROOT / "scripts" / "monitor_fepo_late_screens.sh").read_text(
        encoding="utf-8"
    )
    assert "action_budget_native_rank_local" in late
    assert "submit_ab_after_screen_closure.sh" in late


def test_active_eval_outputs_are_forced_under_faro():
    scripts = (
        "submit_fepo_eval.sh",
        "submit_fepo_eval_sharded.sh",
        "submit_fepo_text_calibration_adaptive.sh",
        "submit_nc_fepo_probe_adaptive.sh",
        "submit_nc_fepo_verifier_adaptive.sh",
        "submit_official_refcoco_adaptive.sh",
        "submit_official_grefcoco_adaptive.sh",
    )
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'OUTPUT=$(realpath -m "$OUTPUT")' in source, name
        assert 'OUTPUT must be under FARO_ROOT' in source, name


def test_pes_finalizer_enforces_full_holdout_and_bootstrap_contract():
    source = (ROOT / "scripts" / "finalize_pes_eval.sh").read_text(encoding="utf-8")
    assert "expected 512 rows" in source
    assert "expected 256 positive rows" in source
    assert "invalid output" in source
    assert "boundary_iou" in source
    assert "REPEATS=${REPEATS:-20000}" in source
    assert "run_pair pes_vs_r18" in source
    assert "run_pair pes_vs_matched_sft" in source
    assert "run_pair shuffled_pes_vs_r18" in source
    assert "run_pair shuffled_pes_vs_matched_sft" in source
    assert "run_pair pes_vs_shuffled" in source
    assert "decision.json" in source
    late = (ROOT / "scripts" / "monitor_fepo_late_screens.sh").read_text(encoding="utf-8")
    assert "finalize_pes_eval.sh" in late
    assert "PES_FINALIZE_STATE/running" in late


def test_pes_finalizer_uses_ci_corrected_gate_for_canonical_promotion():
    source = (ROOT / "scripts" / "finalize_pes_eval.sh").read_text(encoding="utf-8")
    assert '"legacy_promotion_gate"' in source
    assert '"promotion_gate": reports["pes_vs_r18"]["ci_corrected_promotion_gate"]' in source


def test_shared_eval_entrypoint_fails_closed_for_closed_branches():
    source = (ROOT / "scripts" / "submit_samtok_standalone_eval_adaptive.sh").read_text(
        encoding="utf-8"
    )
    assert "paired_view_holdout512" in source
    assert "closed_training_gate" in source
    assert 'value.get("decision") != "open"' in source
    assert "boundary_bottleneck_paired_view_vs_matched_sft_bootstrap20k.json" in source
    assert "boundary_stratified_native_rank_local_vs_matched_sft_bootstrap20k.json" in source
    assert "action_budget_native_rank_local_vs_matched_sft_bootstrap20k.json" in source
    assert "predicted_evidence_scope_shuffled_holdout512" in source
    assert "pes_coverage_gate_passed" in source
    assert "logs/eval_submit_guard" in source
    assert "flock -n 8" in source
    assert "now + 300" in source
