# R13: Native-anchored rank-Pareto geometry credit

## Hypothesis

R11/R12 provide within-group geometry ordering but neither establishes
positive cIoU gain. Their weakness is that the ordering is not anchored to the
model's native greedy decision. R13 inserts that native greedy mask as an
explicit reference point on both geometry axes. A sampled trajectory receives
positive policy credit only if it beats native greedy on both cIoU and boundary
IoU; mixed or regressive trajectories are forced to the non-positive side.
Jointly better candidates are ranked by tie-aware midranks against the native
reference and standardized within K=4.

This tests the Fine-R1 verifiable-improvement principle and the DR2Seg/Qwen3VL-
Seg emphasis on continuous region and boundary quality, while retaining one
OpenWorldSAM-style mask-or-null interface. No PixVL checkpoint, self-supervised
cycle, counterfactual view, extra router expert, or target-code supervision is
introduced.

## Fixed protocol

- Original SAMTok continued-SFT adapter initialization.
- `egfepo_train_5120.jsonl` (5,120 rows), 10 outer steps, 2 GPUs.
- Effective-support K=4 grammar rollouts and existing tail schedule.
- Unified 32-row no-target sentinel and null-risk repair.
- Native greedy is computed on the same clean image/prompt as sampled masks.
- Only detached PPO advantages change; holdout remains untouched.

## Promotion and falsification

The complete 512-row holdout is required after a valid screen. Stop on any
grammar, effective-support, nonconstant-reward, ratio, sentinel, or tail-risk
gate failure. Promote only with zero invalid outputs, positive-mask rate 1.0,
positive cIoU and utility non-inferior to the frozen anchor, and no-target
recall non-inferior. A paper-level geometry claim requires paired positive-cIoU
95% CI lower bound above zero; calibration-only gains are reported as an
ablation. No threshold or weight sweep is permitted.

Stage: `fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu`.

Config: `Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu.py`.

Submission wrapper (not submitted):
`scripts/submit_samtok_tb_gppo_native_rank_pareto.sh`.
