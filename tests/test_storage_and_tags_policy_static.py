from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DISABLED = {
    "submit_samtok_sft.sh",
    "submit_fepo_schema_train_smoke.sh",
}


def test_rjob_submitters_use_the_registered_positive_tag_file():
    for path in SCRIPTS.glob("submit_*.sh"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "rjob submit" not in source or path.name in DISABLED:
            continue
        assert "rjob_tags.txt" in source, path.name
        assert any(
            token in source
            for token in (
                '--positive-tags="$POSITIVE_TAGS"',
                '--positive-tags="$tags"',
                '--positive-tags="$positive_tags"',
            )
        ), path.name


def test_no_active_submitter_writes_root_logs_or_pixvl_outputs():
    forbidden = ("mkdir -p '$ROOT/logs'", "tee '$ROOT/logs", "tee -a '$ROOT/logs", "OUTPUT=${OUTPUT:-$ROOT/outputs")
    for path in SCRIPTS.glob("submit_*.sh"):
        if path.name in DISABLED:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "rjob submit" not in source:
            continue
        for token in forbidden:
            assert token not in source, f"{path.name}: {token}"
