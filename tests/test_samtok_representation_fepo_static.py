from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.config import REPO_ROOT
from projects.samtok_selective.representation_fepo_contract import (
    ADAPTER_MODE,
    ONE_STEP_STAGE,
    TWENTY_STEP_STAGE,
    validate_representation_fepo_config,
)

try:
    import torch
    from torch import nn

    from projects.samtok_selective.modeling import (
        discover_visual_projector_lora_targets,
        visual_projector_adapter_summary,
    )
except ModuleNotFoundError:
    torch = None
    nn = None


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = FARO_ROOT / (
    "Sa2VA/projects/samtok_selective/configs/"
    "fepo_projector_plastic_one_step_2gpu.py"
)


if nn is not None:
    class Merger(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = nn.Linear(4, 4)
            self.linear_fc2 = nn.Linear(4, 4)


    class VisualInventory(nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = Merger()
            self.deepstack_merger_list = nn.ModuleList([Merger() for _ in range(3)])
            self.blocks = nn.ModuleList([Merger()])


    class InventoryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = VisualInventory()


    class AdapterWeights(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict(
                {"anchor": nn.Linear(2, 2, bias=False), "visual": nn.Linear(2, 2, bias=False)}
            )


    class AdapterAuditModel(nn.Module):
        active_adapters = ["anchor", "visual"]

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = nn.Module()
            self.model.visual.merger = AdapterWeights()
            for name, parameter in self.named_parameters():
                # Match the PEFT adapter key, not the parent Qwen visual
                # module path (which also appears in anchor parameter names).
                parameter.requires_grad_(name.split(".")[-2] == "visual")


@pytest.mark.skipif(torch is None, reason="torch is available in the registered rjob runtime")
def test_visual_inventory_is_exactly_four_mergers_and_no_blocks():
    targets = discover_visual_projector_lora_targets(InventoryModel())
    assert len(targets) == 8
    assert all("visual" in target for target in targets)
    assert all("merger" in target for target in targets)
    assert not any("blocks" in target for target in targets)


@pytest.mark.skipif(torch is None, reason="torch is available in the registered rjob runtime")
def test_visual_adapter_summary_rejects_trainable_anchor():
    model = AdapterAuditModel()
    summary = visual_projector_adapter_summary(model)
    assert summary["anchor_trainable_tensors"] == 0
    assert summary["visual_trainable_tensors"] == 1
    next(parameter for name, parameter in model.named_parameters() if ".anchor." in name).requires_grad_(True)
    with pytest.raises(RuntimeError, match="anchor=1"):
        visual_projector_adapter_summary(model)


@pytest.mark.parametrize(
    ("filename", "stage", "steps"),
    [
        ("fepo_projector_plastic_one_step_2gpu.py", ONE_STEP_STAGE, 1),
        ("fepo_projector_plastic_20step_2gpu.py", TWENTY_STEP_STAGE, 20),
    ],
)
def test_representation_configs_are_frozen(monkeypatch, tmp_path, filename, stage, steps):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(str(CONFIG.with_name(filename)))["config"]
    validate_representation_fepo_config(config, REPO_ROOT)
    assert config["stage"] == stage
    assert config["optimizer"]["max_steps"] == steps
    assert config["model"]["adapter_mode"] == ADAPTER_MODE
    assert config["lora"]["visual_r"] == 16
    assert config["representation"]["expected_target_linears"] == 8


def test_representation_submit_uses_only_repository_positive_tags():
    script = (FARO_ROOT / "scripts/submit_samtok_projector_plastic_one_step.sh").read_text(
        encoding="utf-8"
    )
    assert "JOB_NAME must start with dna-" in script
    assert "--namespace=ailab-dnacoding" in script
    assert "--gpu=2" in script
    assert "--nproc_per_node=2" in script
    assert 'TAGS_FILE=$FARO_ROOT/rjob_tags.txt' in script
    assert '--positive-tags="$POSITIVE_TAGS"' in script
    assert "rjob_tags_16gpu_partition" not in script
    assert "projects.pixvl_" not in script.lower()
    for wrapper in (
        "submit_samtok_projector_plastic_20step.sh",
        "submit_samtok_projector_plastic_sft_control40.sh",
    ):
        text = (FARO_ROOT / "scripts" / wrapper).read_text(encoding="utf-8")
        assert "dna-samtok" in text
        assert "rjob_tags_16gpu_partition" not in text


def test_projector_sft_control_matches_batches_and_optimizer_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    path = CONFIG.with_name("projector_plastic_sft_control40.py")
    config = runpy.run_path(str(path))["config"]
    assert config["model"]["adapter_mode"] == ADAPTER_MODE
    assert config["data"]["pairs_per_device_batch"] == 4
    assert config["optimizer"]["max_steps"] == 40
    assert config["optimizer"]["updates_per_batch"] == 2
    assert config["representation"]["matched_rl_outer_batches"] == 20
