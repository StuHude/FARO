from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ba_transition_requires_r35_worker_failure_and_is_idempotent():
    source = (ROOT / "scripts/submit_ba_after_r35_failure.sh").read_text()
    assert 'status") != "failed_validity_gate"' in source
    assert 'active_set_risk_gate_passed") is not False' in source
    assert 'len(value.get("steps") or []) < 10' in source
    assert 'flock -n 9' in source
    assert "rjob list --namespace=ailab-dnacoding" in source
    assert "dna-fepo-boundary-bottleneck-paired-view-10step-2g" in source
    assert "submit_samtok_tb_gppo_boundary_bottleneck_paired_view.sh" in source


def test_ba_transition_uses_registered_training_source():
    source = (ROOT / "scripts/submit_samtok_tb_gppo_boundary_bottleneck_paired_view.sh").read_text()
    assert "egfepo_train_5120.jsonl" in source or "grefcoco_selective_train_256.jsonl" not in source
    assert "rows >= 5000" in source
    assert "dna-" in source


def test_late_monitor_starts_ba_only_after_r35_validity_failure():
    source = (ROOT / "scripts/monitor_fepo_late_screens.sh").read_text()
    assert "failed_validity_gate" in source
    assert "active_set_risk_gate_passed" in source
    assert "tail_risk_gate_passed" in source
    assert "submit_ba_after_r35_failure.sh" in source
    assert "runner_started" in source
