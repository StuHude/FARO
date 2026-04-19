# JOINT-OVERLAY-OPD-RL

`projects/pixvl_idea1` 是基于 `Sa2VA + SAMTok` 的第 1 版 joint pixel-level MLLM 实现。

目标：

- joint `text -> mask` 与 `mask -> text`
- same-resolution overlay privileged teacher
- failed-only on-policy distillation
- segmentation-first RL

实现策略：

1. 复用 `Qwen3-VL-4B-SAMTok-co` 作为默认 student 起点。
2. 复用 `VQ_SAM2` 作为唯一 mask tokenizer / decoder。
3. 使用统一 JSONL schema 管理 RefCOCO / DAM / GAR。
4. overlay 图只在 dataloader 中在线构造，不离线存盘。

训练入口：

- `projects/pixvl_idea1/trainers/joint_sft_trainer.py`
- `projects/pixvl_idea1/trainers/joint_opd_trainer.py`
- `projects/pixvl_idea1/trainers/joint_opd_rl_trainer.py`

数据准备入口：

- `projects/pixvl_idea1/scripts/prepare_refseg_data.py`
- `projects/pixvl_idea1/scripts/prepare_dam_data.py`
- `projects/pixvl_idea1/scripts/prepare_gar_data.py`

