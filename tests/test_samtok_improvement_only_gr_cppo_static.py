from __future__ import annotations

import runpy
from pathlib import Path

from projects.samtok_selective.improvement_only_gr_cppo_contract import (
    STAGE,
    TWENTY_STEP_STAGE,
    validate_improvement_only_gr_cppo_config,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = FARO_ROOT / "Sa2VA/projects/samtok_selective/configs"


def test_improvement_only_configs_are_frozen(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    for stage, steps in ((STAGE, 1), (TWENTY_STEP_STAGE, 20)):
        config = runpy.run_path(str(CONFIG_ROOT / f"{stage}.py"))["config"]
        validate_improvement_only_gr_cppo_config(config)
        method = config["improvement_entropy_gr_cppo"]
        assert config["optimizer"]["max_steps"] == steps
        assert method["positive_reward"] == "plain_ciou"
        assert method["minimum_improvement"] == 1e-4
        assert method["advantage_epsilon"] == 1e-6
        assert "tail_gppo" not in config
        assert "boundary_entropy_gr_cppo" not in config
