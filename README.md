# FARO / Idea 3

`idea3` is the current working project for **Failure-Routed Policy Optimization for Pixel-Level MLLMs** on top of **Sa2VA + Qwen3-VL + SAMTok**.

The core question of this project is:

> Can we improve pixel-level grounding by routing different failure types to different reward decompositions, correction strengths, and post-training stages?

This repository only stores:
- code
- configs
- training / evaluation scripts
- lightweight docs

This repository does **not** store:
- datasets
- model weights
- training outputs
- local caches

Those stay outside git under `/mnt/pfs/xiaoyicheng/`.

## Project Scope

Current `idea3` experiments cover:

- `unified_rl`
- `unified_opd_rl`
- `routed_rl`
- `routed_opd_rl`
- `semantic coverage / calibration` (`semcovcal`) stages
- `caption calibration SFT`
- `sample-conditioned / atom-conditioned` routing
- `triage` two-stage routing
- corrected self-distill OPD variants
- `GT-only / overlay / GT+overlay` privileged-input ablations

The active implementation lives under:

- `Sa2VA/projects/pixvl_idea3`

Supporting shared code lives under:

- `Sa2VA/projects/pixvl_idea1`

In practice:
- `pixvl_idea1` provides the common dataset / reward / trainer utilities
- `pixvl_idea3` provides the routed / semcov / triage / evaluation logic for this project

## Repository Layout

- `Sa2VA/projects/pixvl_idea3/configs`
  - all `idea3` training and evaluation configs
- `Sa2VA/projects/pixvl_idea3/trainers`
  - routed trainer and related launch logic
- `Sa2VA/projects/pixvl_idea3/scripts`
  - training scripts
  - `RefCOCO / split500 / DLC-Bench` evaluation scripts
- `Sa2VA/projects/pixvl_idea3/eval`
  - slice summarization helpers
- `Sa2VA/projects/pixvl_idea3/routing.py`
  - atom-conditioned routing
  - semantic / relation / geometry reward decomposition
- `Sa2VA/projects/pixvl_idea1/datasets`
  - unified region dataset
  - overlay generation
- `Sa2VA/projects/pixvl_idea1/rewards`
  - base segmentation / caption rewards
- `scripts/setup_env.sh`
  - environment bootstrap

## External Paths

The code assumes the following external layout:

- datasets:
  - `/mnt/pfs/xiaoyicheng/data`
- model weights:
  - `/mnt/pfs/xiaoyicheng/models`
- training / eval outputs:
  - `/mnt/pfs/xiaoyicheng/outputs`
- main workspace:
  - `/mnt/pfs/xiaoyicheng/BRIDGE-OPD`

If you move these, update the corresponding config paths.

## Environment

Bootstrap the environment with:

```bash
bash scripts/setup_env.sh
```

Main Python environment used by training / eval:

```bash
/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa
```

Typical runtime setup:

```bash
cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
```

For some evals we use the local `/dev/shm` copy of the environment / model for stability and speed.

## Data

`idea3` uses the routed schemas and evaluation subsets already prepared outside the repo.

Key training schemas:

- `refseg_train_routed.jsonl`
- `maskcap_train_routed.jsonl`

Key evaluation subsets:

- `eval_subsets_formal_500/semantic_500.jsonl`
- `eval_subsets_formal_500/relation_500.jsonl`
- `eval_subsets_formal_500/geometry_500.jsonl`
- `eval_subsets_formal_500/refseg_val_500.jsonl`
- `eval_subsets_formal_500/dlc_eval_100.jsonl`

`DLC-Bench` generation uses:

- `/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/DLC-bench.json`
- `/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/godx7/DLC-Bench/images`

## Main Training Families

### 1. Old MVP family

Representative configs:

- `idea3_mvp_scale100k_2gpu_unified_opd_rl.py`
- `idea3_mvp_scale100k_3gpu_routed_rl.py`
- `idea3_mvp_scale100k_3gpu_routed_opd_rl.py`

These are the original `idea3` MVP lines:

- unified reward / distillation
- routed reward only
- routed reward + OPD

### 2. Semcovcal family

Representative configs:

- `idea3_semcovcal_routed_opd_rl_8gpu_500.py`
- `idea3_semcovcal_routed_opd_rl_8gpu_100_from_selfdist1500.py`

These replace the semantic caption reward with:

- recognition
- coverage
- calibration

### 3. Caption-calibration / conservative-prompt family

Representative configs:

- `idea3_caption_calibration_sft.py`
- `idea3_atom_semcovcal_routed_opd_rl_8gpu_100_from_ckpt1000_mix55_calprompt.py`
- `idea3_atom_semcovcal_routed_opd_rl_8gpu_200_from_ckpt100_mix55_calprompt_continue.py`

These emphasize more conservative localized captioning.

### 4. Corrected self-distill family

Representative configs:

- `idea3_mvp_scale100k_2gpu_unified_opd_rl_selfdist1500.py`
- `idea3_mvp_scale100k_3gpu_routed_opd_rl_selfdist1500.py`

These switch OPD teacher mode to:

- `self_privileged_rollout`

### 5. Triage family

Representative configs:

- `idea3_atom_triage_noevidence_routed_opd_rl_8gpu_100_from_autoroute200.py`

This adds a first-stage triage gate:

- `clean`
- `suspicious`
- `corrupted`

Current code gates OPD by triage label.

### 6. GT-only privilege ablation family

Representative configs:

- `idea3_mvp_scale100k_4gpu_unified_opd_rl_selfdist1000_gtonly.py`
- `idea3_mvp_scale100k_4gpu_routed_opd_rl_selfdist1000_gtonly.py`

These remove overlay as privileged teacher image input and keep only:

- GT privileged text
- original image

## How Training Works

At a high level, all `idea3` training uses:

1. supervised CE on the answer span
2. on-policy rollout
3. reward computation
4. RL loss on the rollout
5. optional OPD on failed rollouts
6. reference KL regularization

The key project difference is whether reward / thresholds / scales are:

- unified
- routed by failure type
- corrected by semcovcal
- gated by triage

## Main Training Commands

All commands below are run from:

```bash
cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
```

### Old / canonical self-distill commands

```bash
bash projects/pixvl_idea3/scripts/train_selfdist_unified_2gpu_1500.sh
bash projects/pixvl_idea3/scripts/train_selfdist_routed_rl_3gpu_1500.sh
bash projects/pixvl_idea3/scripts/train_selfdist_routed_opd_rl_3gpu_1500.sh
```

### Semcovcal on corrected selfdist parent

```bash
bash projects/pixvl_idea3/scripts/train_semcovcal_routed_opd_rl_8gpu_100_from_selfdist1500.sh
```

### Triage two-stage routing

```bash
bash projects/pixvl_idea3/scripts/train_atom_triage_noevidence_routed_opd_rl_8gpu_100_from_autoroute200.sh
```

### GT-only privileged-input ablations

```bash
bash projects/pixvl_idea3/scripts/train_selfdist_unified_4gpu_1000_gtonly.sh
bash projects/pixvl_idea3/scripts/train_selfdist_routed_opd_rl_4gpu_1000_gtonly.sh
```

## Main Evaluation Commands

### RefCOCO

```bash
bash projects/pixvl_idea3/scripts/eval_generic_refcoco_8gpu.sh <tag> <adapter_path> <out_dir> [model_path] [num_tasks]
```

### 500-sample split eval

```bash
bash projects/pixvl_idea3/scripts/eval_generic_split500_8gpu.sh <tag> <config> <adapter_path> <out_dir> [num_tasks]
```

### DLC-Bench prediction generation

```bash
bash projects/pixvl_idea3/scripts/eval_generic_dlc_legacy_8gpu.sh <tag> <adapter_path> <out_dir> [model_path] [num_tasks]
```

This script generates:

- `raw.json`
- `pred.json`

### DLC-Bench judging

The current stable local judging path uses:

- `Meta-Llama-3.1-8B-Instruct`
- local small-memory `vllm` serve
- `eval_model_outputs.py`

In practice we run:

```bash
python -m vllm.entrypoints.cli.main serve /dev/shm/models/Meta-Llama-3.1-8B-Instruct \
  --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --port 9000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager
```

Then:

```bash
python /tmp/describe-anything/evaluation/eval_model_outputs.py \
  --pred <pred.json> \
  --qa /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/qa.json \
  --class-names /mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench/class_names.json \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --base-url http://127.0.0.1:9000/v1
```

## Notes on Current Implementation

Important implementation facts:

1. Current routed correction is **not** three separate decoder heads.
2. It is a shared-parameter system with:
   - bucket-specific reward definitions
   - bucket-specific failure thresholds
   - bucket-specific `ce / rl / opd` scales
   - task-specific OPD span
3. `semantic-refseg` is **not** yet a fully separate correction branch in code.
4. `triage` currently gates OPD only; it does not yet implement full correction-before-OPD.

## Specs / Docs

Project specs and progress notes:

- `idea1_joint_overlay_teacher_opd_rl_codex_spec.md`
- `idea1_joint_overlay_teacher_opd_rl_codex_spec_upgraded.md`

## Git / Storage Policy

Do **not** commit:

- `/mnt/pfs/xiaoyicheng/data`
- `/mnt/pfs/xiaoyicheng/models`
- `/mnt/pfs/xiaoyicheng/outputs`
- local caches
- checkpoints
- safetensors / `.bin` model files

This repository should remain code-and-doc only.
