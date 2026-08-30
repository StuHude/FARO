from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "Sa2VA/projects/samtok_selective/fepo_gr_cppo_trainer.py"


def test_anchor_kl_auxiliary_pass_disables_checkpointing_and_restores_it():
    source = TRAINER.read_text(encoding="utf-8")
    assert "def _without_gradient_checkpointing" in source
    assert 'enable(gradient_checkpointing_kwargs={"use_reentrant": False})' in source
    # The frozen, diagnostic, and differentiable anchor replays all use the
    # isolated context; ordinary policy forwards remain checkpointed.
    assert source.count("with _without_gradient_checkpointing(model):") >= 3


def test_r24_screen_disables_checkpointing_for_the_full_mixed_update():
    # Config construction intentionally requires an explicitly approved model
    # path; provide a synthetic path because this is a static contract test.
    import os
    os.environ.setdefault("SAMTOK_BASE_CHECKPOINT", str(ROOT / "third_party/SAMTok"))
    config = runpy.run_path(
        str(ROOT / "Sa2VA/projects/samtok_selective/configs/"
            "fepo_tb_gppo_plain_rank_unified_anchor_kl_10step_2gpu.py")
    )["config"]
    assert config["runtime"]["gradient_checkpointing"] is False
