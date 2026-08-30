import pytest

torch = pytest.importorskip("torch")

from projects.samtok_selective.evidence_gate import (
    detached_group_gate,
    validate_evidence_gate_config,
)


def test_evidence_gate_is_clipped_and_detached():
    validate_evidence_gate_config(
        {"mode": "view_drop", "scale": 0.25, "clip_min": 0.25, "clip_max": 1.75, "noise_std": 0.01}
    )
    evidence = torch.tensor([-2.0, 0.0, 1.0, 5.0], requires_grad=True)
    gate = detached_group_gate(
        evidence, mode="view_drop", scale=0.25, clip_min=0.25, clip_max=1.75
    )
    assert torch.isfinite(gate).all()
    assert float(gate.min()) >= 0.25
    assert float(gate.max()) <= 1.75
    assert not gate.requires_grad


def test_shuffled_gate_preserves_group_shape():
    gate = detached_group_gate(
        torch.arange(4, dtype=torch.float32),
        mode="shuffled",
        scale=0.25,
        clip_min=0.25,
        clip_max=1.75,
        generator=torch.Generator().manual_seed(7),
    )
    assert gate.shape == (4,)
    assert torch.isfinite(gate).all()
