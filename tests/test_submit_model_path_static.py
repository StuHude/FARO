from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tb_gppo_submission_uses_worker_visible_samtok_path():
    script = (ROOT / "scripts" / "submit_samtok_tb_gppo.sh").read_text(
        encoding="utf-8"
    )
    approved = (
        "/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/"
        "Nemotrontiaozheng/PixVL_ailab/checkpoints/SAMTok/"
        "Qwen3-VL-4B-SAMTok-co"
    )
    assert f"APPROVED_MODEL={approved}" in script
    assert 'MODEL=${SAMTOK_BASE_CHECKPOINT:-$APPROVED_MODEL}' in script
    assert 'MODEL=$(realpath "$MODEL")' in script
    assert 'must be the worker-visible approved path' in script


def test_tb_gppo_submission_does_not_default_to_submit_host_pfs():
    script = (ROOT / "scripts" / "submit_samtok_tb_gppo.sh").read_text(
        encoding="utf-8"
    )
    assert "MODEL=/mnt/pfs" not in script
