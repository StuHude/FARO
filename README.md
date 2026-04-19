# BRIDGE-OPD

`BRIDGE-OPD` is the working repository for the `pixvl_idea1` pipeline built on top of `Sa2VA`.

The core implementation lives under:

- `Sa2VA/projects/pixvl_idea1`

This repository is organized so that:

- code stays in the repo
- datasets stay under `/mnt/pfs/xiaoyicheng/data`
- model weights stay under `/mnt/pfs/xiaoyicheng/models`
- training outputs stay under `/mnt/pfs/xiaoyicheng/outputs`

No large datasets or model checkpoints should be committed to git.

## Layout

- `Sa2VA/projects/pixvl_idea1`: idea1 code, configs, trainers, eval scripts
- `scripts/setup_env.sh`: environment setup
- `scripts/download_public_data.sh`: public data download helpers
- `training_plan.md`: execution plan
- `idea1_joint_overlay_teacher_opd_rl_codex_spec.md`: design spec

## Environment

Create the environment with:

```bash
bash scripts/setup_env.sh
```

After setup, the main environment used in this project is:

```bash
/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa
```

## Data Preparation

Download and prepare the public subsets with:

```bash
bash scripts/download_public_data.sh
python scripts/check_data_integrity.py --root /mnt/pfs/xiaoyicheng/data/pixvl_idea1
python Sa2VA/projects/pixvl_idea1/scripts/prepare_refseg_data.py
python Sa2VA/projects/pixvl_idea1/scripts/prepare_dam_data.py --local-only --dataset-root /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam --subset-names COCOStuff LVIS PACO --export-dlc-bench --dlc-local-only
python Sa2VA/projects/pixvl_idea1/scripts/prepare_gar_data.py --local-only --dataset-root /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/gar
```

## Training Commands

All training commands are run from:

```bash
cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
```

### Baseline

The baseline initialization used by idea1 is:

```text
/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co
```

This baseline is referenced in:

- `Sa2VA/projects/pixvl_idea1/configs/idea1_joint_sft.py`

### Stage 1

Stage 1 starts from the baseline model above and performs joint SFT:

```bash
./projects/pixvl_idea1/scripts/train_stage1_joint_sft.sh
```

The output adapter is written to:

```text
/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage1_joint_sft_maxmem/adapter
```

### Stage 2

Stage 2 loads the Stage 1 adapter as both:

- `student_init.adapter_path`
- `teacher.adapter_path`

These are configured in:

- `Sa2VA/projects/pixvl_idea1/configs/idea1_joint_opd.py`

Run Stage 2 with:

```bash
./projects/pixvl_idea1/scripts/train_stage2_joint_opd.sh
```

The output adapter is written to:

```text
/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage2_joint_opd/adapter
```

### Stage 3

Stage 3 loads:

- student init from Stage 2
- teacher/reference according to `idea1_joint_opd_rl.py`

Run Stage 3 with:

```bash
./projects/pixvl_idea1/scripts/train_stage3_joint_opd_rl.sh
```

The output adapter is written to:

```text
/mnt/pfs/xiaoyicheng/outputs/pixvl_idea1/stage3_joint_opd_rl/adapter
```

## Evaluation

Project-level eval entry:

```bash
./projects/pixvl_idea1/scripts/run_all_eval.sh
```

Official SAMTok evaluation scripts used during verification are under:

```text
Sa2VA/projects/samtok/evaluation/qwen3vl
```

## Notes

- Do not commit anything under `/mnt/pfs/xiaoyicheng/data`
- Do not commit anything under `/mnt/pfs/xiaoyicheng/models`
- Do not commit anything under `/mnt/pfs/xiaoyicheng/outputs`
- Do not commit temporary benchmark caches under `Sa2VA/godx7` or `Sa2VA/temp_save`
