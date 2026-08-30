from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from projects.samtok_selective.config import REPO_ROOT, validate_config
from projects.samtok_selective.gr_cppo_contract import (
    FROZEN_ANCHOR_MODEL_SHA256,
    FROZEN_ANCHOR_TOTAL_STEPS,
    expected_frozen_anchor,
)
from projects.samtok_selective.sft_schedule import (
    batch_update_indices,
    validate_updates_per_batch,
)


FARO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    FARO_ROOT
    / "Sa2VA/projects/samtok_selective/configs/continued_sft_es_control40.py"
)


@pytest.mark.parametrize("value", (0, -1, 1.5, True, "2", None))
def test_updates_per_batch_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_updates_per_batch(value)
    assert validate_updates_per_batch(1) == 1
    assert validate_updates_per_batch(2) == 2


def test_two_updates_reuse_each_batch_and_stop_at_40():
    step = 0
    batch_id = 0
    updates: list[tuple[int, int, int]] = []
    while step < 40:
        for batch_update in batch_update_indices(step, 40, 2):
            updates.append((batch_id, batch_update, step))
            step += 1
        batch_id += 1

    assert step == 40
    assert batch_id == 20
    assert [item[1] for item in updates] == [0, 1] * 20
    assert all(
        [item[0] for item in updates].count(batch) == 2 for batch in range(20)
    )
    assert list(batch_update_indices(39, 40, 2)) == [0]
    trainer = (
        FARO_ROOT / "Sa2VA/projects/samtok_selective/sft_trainer.py"
    ).read_text(encoding="utf-8")
    assert "validate_updates_per_batch(" in trainer
    assert "for batch_update in batch_update_indices(" in trainer


def test_control40_config_is_locked_to_frozen_anchor(monkeypatch, tmp_path):
    monkeypatch.setenv("SAMTOK_BASE_CHECKPOINT", str(tmp_path / "SAMTok-base"))
    monkeypatch.delenv("SAMTOK_STANDALONE_ADAPTER", raising=False)
    config = runpy.run_path(str(CONFIG_PATH))["config"]
    validate_config(config)

    assert config["stage"] == "continued_sft_es_control40"
    assert Path(config["checkpoint"]["adapter_init"]).resolve() == (
        expected_frozen_anchor(REPO_ROOT)
    )
    assert Path(config["checkpoint"]["output_dir"]).resolve() == (
        REPO_ROOT / "outputs/samtok_selective/continued_sft_es_control40"
    )
    assert Path(config["checkpoint"]["output_dir"]).name not in {
        "continued_sft_control20",
        "fepo_entropy_gr_cppo_one_step_2gpu",
        "fepo_entropy_gr_cppo_20step_2gpu",
    }
    assert config["optimizer"]["max_steps"] == 40
    assert config["optimizer"]["updates_per_batch"] == 2
    assert config["optimizer"]["lr"] == 5e-7
    assert config["data"]["pairs_per_device_batch"] == 4
    assert FROZEN_ANCHOR_TOTAL_STEPS == 500
    assert FROZEN_ANCHOR_MODEL_SHA256 == (
        "7b409c9f2bc3cf2da61adb9c86270dcda6a1991082d5dca4bb1c8a593ea4dfed"
    )


def test_control40_submit_wrapper_reuses_dnacoding_sft_contract():
    wrapper = (FARO_ROOT / "scripts/submit_samtok_es_control40.sh").read_text(
        encoding="utf-8"
    )
    common = (FARO_ROOT / "scripts/submit_samtok_standalone_sft.sh").read_text(
        encoding="utf-8"
    )

    assert "continued_sft_es_control40.py" in wrapper
    assert "continued_sft_to500/adapter" in wrapper
    assert "dna-samtok-standalone-es-control40-2g" in wrapper
    assert "TAGS_FILE=$FARO_ROOT/rjob_tags.txt" in wrapper
    assert "GPU_COUNT=2" in wrapper
    assert 'exec bash "$SCRIPT_DIR/submit_samtok_standalone_sft.sh"' in wrapper
    assert "--namespace=ailab-dnacoding" in common
    assert '--positive-tags="$POSITIVE_TAGS"' in common
    assert "JOB_NAME must start with dna-" in common
    assert "--gpu=2" in common
    assert "--nproc_per_node=2" in common
