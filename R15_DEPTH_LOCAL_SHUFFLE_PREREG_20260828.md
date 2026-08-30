# R15: shuffled-depth localization control

## Hypothesis

R14 tests whether localizing a jointly better decoded mask to its earliest
SAMTok code divergence improves geometry credit. R15 is the matched control:
it keeps R14's rollout groups, decoded cIoU/boundary gains, positive-only joint
criterion, and prefix-rarity calculation, but cyclically permutes the depth
used for the locality decay. Any R14 geometry effect that depends on the
earliest-divergence interpretation should disappear under this control.

## Registered mechanism

Stage: `fepo_tb_gppo_plain_rank_unified_depth_local_shuffle_10step_2gpu`.

Config: `Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_depth_local_shuffle_10step_2gpu.py`.

Submit wrapper: `scripts/submit_samtok_tb_gppo_depth_local_shuffle.sh`.

The detached joint gain is exactly the R14 geometric mean of cIoU and
boundary-IoU improvements over native greedy. The same sampled trajectories
remain active, and rarity is computed at the original earliest-divergence
prefix. Only the depth-decay lookup uses a deterministic cyclic permutation
with seed `20260827` (a nonzero offset for every depth greater than one).
There are no PixVL weights, OPD, counterfactual views, extra experts, or
self-supervised loops.

The run uses exactly 5,120 rows, 10 optimizer steps, two GPUs, the frozen
continued-SFT SAMTok adapter, and the shared 32-row unified sentinel. The
training gate requires finite rewards, nonconstant groups, an improved
rollout, nonzero policy gradients, grammar-valid trajectories, epoch-2 ratio
movement, and a passed sentinel risk gate. Only after all gates pass may the
complete 512-row paired holdout and 20,000 paired bootstrap resamples run.

R15 is a localization control, not an independent geometry claim. If its
positive-cIoU interval is indistinguishable from R14 or the training gate
fails, the shuffled mechanism is closed without a weight sweep.
