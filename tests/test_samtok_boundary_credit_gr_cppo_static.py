from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.boundary_credit_gr_cppo_contract import (
    STAGE,
    TWENTY_STEP_STAGE,
    validate_boundary_credit_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_boundary_credit_configs_are_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for stage, steps in ((STAGE, 1), (TWENTY_STEP_STAGE, 20)):
        path = CONFIG_ROOT / f"{stage}.py"
        config = runpy.run_path(str(path))["config"]
        validate_boundary_credit_gr_cppo_config(config)
        method = config["boundary_entropy_gr_cppo"]
        assert config["optimizer"]["max_steps"] == steps
        assert method["ciou_weight"] == 0.5
        assert method["boundary_iou_weight"] == 0.5
        assert method["boundary_width"] == 2
        assert "tail_gppo" not in config
        assert "active_set_entropy_gr_cppo" not in config
