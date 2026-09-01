from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_budget_validator():
    spec = importlib.util.spec_from_file_location(
        "faro_validate_training_budget", ROOT / "tools/validate_training_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_samtok_training_submitters_guard_minimum_rows_and_smokes():
    scripts = (
        "submit_samtok_tb_gppo.sh",
        "submit_samtok_es_gr_cppo_one_step.sh",
        "submit_samtok_gr_cppo.sh",
        "submit_samtok_active_set_es_gr_cppo_one_step.sh",
        "submit_samtok_ampcpo_smoke.sh",
        "submit_samtok_boundary_credit_gr_cppo.sh",
        "submit_samtok_projector_plastic_one_step.sh",
        "submit_samtok_standalone_sft.sh",
    )
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'rows >= 5000' in source, name
        assert 'Training requires at least 5000 rows' in source, name
        assert 'one-step' in source.lower() or 'one_step' in source.lower(), name
        assert 'validate_training_budget.py' in source, name


def test_eval_submitters_are_not_mistaken_for_training_budget_checks():
    source = (ROOT / "scripts/submit_fepo_eval_sharded.sh").read_text(
        encoding="utf-8"
    )
    assert "--namespace=ailab-dnacoding" in source
    assert '--positive-tags="$POSITIVE_TAGS"' in source
    assert "GPU_COUNT" in source


def test_full_data_budget_checks_actual_rows_and_distributed_coverage(tmp_path):
    validator = _load_budget_validator()
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n" * 5120, encoding="utf-8")
    config = tmp_path / "config.py"
    config.write_text(
        "config = {\n"
        "  'data': {'expected_rows': 5120, 'pairs_per_device_batch': 4},\n"
        "  'runtime': {'expected_world_size': 2},\n"
        "  'optimizer': {'max_steps': 320},\n"
        "  'tail_gppo': {'full_data_schedule': True,\n"
        "                 'minimum_consumed_pairs': 2560,\n"
        "                 'minimum_consumed_rows': 5120}\n"
        "}\n",
        encoding="utf-8",
    )
    result = validator.validate(config, data)
    assert result["full_data_required_local_steps"] == 320
    assert result["full_data_minimum_rows"] == 5120

    config.write_text(config.read_text(encoding="utf-8").replace("'max_steps': 320", "'max_steps': 319"), encoding="utf-8")
    with pytest.raises(ValueError, match="full-data schedule max_steps"):
        validator.validate(config, data)


def test_legacy_pixvl_training_entrypoint_is_disabled():
    source = (ROOT / "scripts/submit_samtok_sft.sh").read_text(encoding="utf-8")
    assert "Legacy PixVL trainer submission is disabled" in source
    assert "exit 2" in source


def test_legacy_pixvl_schema_training_entrypoint_is_disabled():
    source = (ROOT / "scripts/submit_fepo_schema_train_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "Legacy PixVL routed-OPD trainer submission is disabled" in source
    assert "exit 2" in source
