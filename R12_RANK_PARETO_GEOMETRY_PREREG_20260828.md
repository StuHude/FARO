# R12: Tie-aware rank-Pareto geometry credit

## Question

Can geometry improvement be recovered by making the within-group credit
relative and dense, rather than requiring an absolute improvement threshold?

## Hypothesis

R7's absolute Pareto verifier may be too sparse: with only four sampled masks,
all candidates can miss the `1e-4` cIoU and boundary-IoU thresholds even when
their ordering contains useful signal. For each positive prompt, R12 computes
tie-aware midranks independently for cIoU and boundary IoU over the four
sampled masks, combines the two ranks by geometric mean, and standardizes the
result within the group. This should transfer a stable two-axis ordering to
the SAMTok policy while preventing cIoU-only winners from dominating.

## What remains fixed

- Original SAMTok checkpoint and continued-SFT adapter initialization.
- One shared mask-or-null policy; no inference router or extra expert.
- Grammar-constrained K=4 rollout and effective-support calibration.
- Tail schedule, FIFO bookkeeping, and 32-row unified no-target sentinel.
- 5,120 training rows and exactly 10 outer steps (20 policy epochs).
- Complete 512-row holdout; no holdout access during training.

Only the detached policy advantage changes. The tail reward remains the
registered plain-rank cIoU/boundary bookkeeping, so this isolates advantage
allocation from data or model changes.

## Falsification and promotion

The two-GPU screen is stopped if grammar, effective-support, nonconstant
reward, epoch-two ratio, sentinel, or tail-risk gates fail. After a valid
screen, evaluate all 512 holdout pairs. Promote only with zero invalid outputs,
positive-mask rate 1.0, positive cIoU and utility non-inferior to the frozen
anchor, and no-target recall non-inferior. A paper-level geometry claim needs a
paired 95% cIoU interval whose lower bound is above zero; a calibration-only
gain is documented as an ablation. R7 remains the matched absolute-Pareto
control, and no threshold or weight sweep is permitted.

## Implementation

Stage: `fepo_tb_gppo_plain_rank_unified_rank_pareto_geometry_10step_2gpu`.

Configuration: `Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_rank_pareto_geometry_10step_2gpu.py`.

Submission wrapper (not submitted in this screen):
`scripts/submit_samtok_tb_gppo_rank_pareto.sh`.
