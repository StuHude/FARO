# JOINT-OVERLAY-OPD-RL Training Plan

## 路径约定

- 工作区：`/mnt/pfs/xiaoyicheng/BRIDGE-OPD`
- 主代码库：`/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA`
- 数据根目录：`/mnt/pfs/xiaoyicheng/data/pixvl_idea1`
- 模型缓存：`/mnt/pfs/xiaoyicheng/models`
- Hugging Face 缓存：`/mnt/pfs/xiaoyicheng/.cache/huggingface`
- 输出目录：`/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1`

## 阶段执行

### Stage 0

- 导出统一 schema
- 导出 smoke split: `32 / 128 / 512`
- 可视化 3 张 overlay 样本
- 跑最小 forward / generate / mask encode / mask decode 自检

### Stage 1

- joint SFT
- 任务配比：`refseg 60% / maskcap 40%`
- 产出：
  - `ckpt_stage1_joint_sft`
  - `metrics_stage1.json`

### Stage 2

- overlay teacher
- failed-only OPD
- 对照：
  - all-sample OPD
  - failed-only OPD
- 产出：
  - `ckpt_stage2_joint_opd`
  - `metrics_stage2.json`

### Stage 3

- 保留 Stage 2 OPD
- 加 `segmentation-first RL`
- 默认弱 caption RL，可随时关闭
- 产出：
  - `ckpt_stage3_joint_opd_rl`
  - `metrics_stage3.json`

## 运行顺序

1. `bash scripts/setup_env.sh`
2. `bash scripts/download_public_data.sh --smoke`
3. `python scripts/check_data_integrity.py --root /mnt/pfs/xiaoyicheng/data/pixvl_idea1`
4. `python Sa2VA/projects/pixvl_idea1/scripts/prepare_refseg_data.py ...`
5. `python Sa2VA/projects/pixvl_idea1/scripts/prepare_dam_data.py ...`
6. `python Sa2VA/projects/pixvl_idea1/scripts/prepare_gar_data.py ...`
7. `python Sa2VA/projects/pixvl_idea1/scripts/export_smoke_splits.py ...`
8. `bash Sa2VA/projects/pixvl_idea1/scripts/train_stage1_joint_sft.sh`
9. `bash Sa2VA/projects/pixvl_idea1/scripts/train_stage2_joint_opd.sh`
10. `bash Sa2VA/projects/pixvl_idea1/scripts/train_stage3_joint_opd_rl.sh`
11. `bash Sa2VA/projects/pixvl_idea1/scripts/run_all_eval.sh`
