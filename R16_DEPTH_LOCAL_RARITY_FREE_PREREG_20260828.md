# R16: rarity-free depth-local geometry credit

## Hypothesis

R14/R15 establish a local-credit family but do not identify whether the
prefix-rarity bonus is responsible for their behavior. R16 is a matched,
rarity-free ablation: it keeps the earliest SAMTok-code divergence and the
joint cIoU/boundary-IoU geometric-mean gain, while removing the frequency
weight entirely. This isolates locality from exploration-frequency shaping and
tests whether verifier/local-credit ideas help through credit assignment alone.

The method is one SAMTok policy with K=4 effective-support grammar rollouts.
It uses no PixVL weights, trainer, OPD, counterfactual view, extra expert, or
self-supervised loop. The decoded mask remains the only geometry signal and
the rarity-free transformation is a detached PPO advantage only.

## Fixed protocol

- Frozen `continued_sft_to500/adapter` SAMTok initialization.
- `egfepo_train_5120.jsonl`, exactly 5,120 rows and 10 outer optimizer steps.
- Two GPUs, effective-support K=4 rollout groups, FIFO-16 geometry ranks.
- Shared 32-row no-target sentinel and existing null-risk/margin gates.
- For every jointly improved sampled mask, credit is assigned at the earliest
  code depth differing from native greedy, with depth decay `0.85` and
  `depth_local_rarity_weight=0.0`.
- Complete 512-row holdout with 20,000 paired bootstrap resamples is required
  after all training gates pass; holdout data is inaccessible during training.

Stage: `fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu`.

Config:
`Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu.py`.

Submission wrapper (not submitted):
`scripts/submit_samtok_tb_gppo_depth_local_rarity_free.sh`.

## Falsification and reporting

The screen closes on any grammar, effective-support, nonconstant-reward,
epoch-two-ratio, sentinel, or tail-risk gate failure. Promotion requires zero
invalid outputs, positive-mask rate `1.0`, and non-inferiority to the frozen
anchor on positive cIoU, utility, and no-target recall. A geometry claim
requires the paired positive-cIoU 95% interval to have a strictly positive
lower bound. Utility or no-target calibration gains with a cIoU interval that
crosses zero are retained only as an ablation; no rarity/depth weight sweep is
authorized.
