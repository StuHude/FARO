import pytest

from tools.decide_pv_training_gate import decide


def test_pv_training_gate_closes_low_joint_support_without_holdout():
    payload = {
        "stage": "pv",
        "status": "finished",
        "steps_completed": 10,
        "optimizer_updates_completed": 20,
        "steps": [
            {
                "paired_view_joint_positive_fraction": 0.10,
                "paired_view_reward_correlation": 0.99,
            }
        ]
        * 10,
    }
    result = decide(payload)
    assert result["decision"] == "closed_training_gate"
    assert result["holdout_used"] is False
    assert result["joint_positive_fraction_mean"] == pytest.approx(0.10)


def test_pv_training_gate_keeps_strong_support_open():
    payload = {
        "stage": "pv",
        "status": "finished",
        "steps_completed": 10,
        "optimizer_updates_completed": 20,
        "steps": [
            {
                "paired_view_joint_positive_fraction": 0.25,
                "paired_view_reward_correlation": 0.99,
            }
        ]
        * 10,
    }
    assert decide(payload)["decision"] == "open"
