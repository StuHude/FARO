# R14: earliest-divergence depth-local geometry credit

## Hypothesis

R11--R13 assigned one sequence-level geometry advantage to a complete sampled
mask. Their utility/null-calibration gains did not produce a reliable positive
cIoU effect, suggesting that credit is diluted across already-correct and
incorrect SAMTok code decisions. R14 tests whether localizing a jointly better
decoded mask's credit to the earliest code depth that diverges from native
greedy makes the geometry signal more actionable.

This is a single-policy SAMTok RL update. It does not use PixVL weights,
training code, OPD, counterfactual views, extra experts, or a self-supervised
loop. The decoded mask and cIoU/boundary-IoU metrics remain the only geometry
supervision; depth localization is a detached PPO-advantage transformation.

## Registered mechanism

Stage: `fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu`.

Config: `Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu.py`.

Submit wrapper: `scripts/submit_samtok_tb_gppo_depth_local.sh`.

Each K=4 rollout group is compared with the native greedy SAMTok code
trajectory. A sampled mask receives positive credit only when both cIoU and
boundary IoU exceed native by `1e-4`. Its joint gain is the geometric mean of
the two improvements. The gain is multiplied by `0.85**d`, where `d` is the
earliest differing code depth, and by `1 + 0.5*(1-frequency)` for the sampled
prefix at that depth. Credits are normalized over active samples; all-zero
groups produce zero advantages. No reward or decoder parameter is changed.

The frozen shared sentinel and effective-support exploration are unchanged:
32 registered no-target rows, K=4, FIFO-16 geometry ranks, and the canonical
no-target CE/margin feasibility constraint. Training uses exactly 5,120 rows
and 10 optimizer steps from the frozen `continued_sft_to500` SAMTok adapter.

## Gates and analysis

The training gate requires finite rewards, at least two nonconstant groups,
one improved rollout, nonzero positive policy gradients, grammar-valid
trajectories, epoch-2 ratio movement, and a passed unified sentinel risk gate.
Only after all gates pass may the full 512-row paired holdout be evaluated,
with 20,000 paired bootstrap resamples. The primary geometry claim requires a
nonnegative paired 95% CI for positive cIoU and no-target recall non-inferior to
the frozen anchor. A utility-only or null-recall improvement with a cIoU CI
crossing zero is reported as calibration evidence and does not support a
geometry claim. If the screen fails the gates or geometry interval, R14 is
closed without a weight sweep.
