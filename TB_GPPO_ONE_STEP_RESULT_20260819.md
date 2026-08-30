# TB-GPPO One-Step Result (2026-08-19)

## Decision

Candidate C, Tail-Balanced Geometry PPO with Selective Risk, is closed. It
passed every rollout and optimization validity check but failed the registered
post-update lower-tail null-margin constraint.

## Corrected run

- Job: `dna-samtok-fepo-tb-tail-one-step-r3-2g-17871-48c0f`
- Namespace: `ailab-dnacoding`
- Resources: two dnacoding GPUs with positive tags from `rjob_tags.txt`
- Initialization: frozen total-500-step SAMTok SFT adapter
- Groups: 8 globally, K=4 rollouts per group
- Policy epochs: 2

The prior r2 run evaluated risk only before optimizer updates. The r3 code
retained all registered method constants and added a final sentinel evaluation
after the last `optimizer.step()`.

## Passed checks

- nonconstant reward groups: `8/8`
- multitrajectory groups: `8/8`
- grammar-valid rollouts: `32/32`
- rollouts better than native greedy: `4/32`
- nonzero positive-policy gradient observations: `4`
- epoch-two median absolute ratio deviation: `0.034415`
- effective-support target hit fraction: `1.0`
- invalid or non-finite geometry rewards: `0`
- FIFO capacity after every update: `16`

## Failed risk check

- final q10 current-minus-anchor margin: `-0.25`
- allowed degradation: `-0.05`
- final sentinel violation rate: `0.53125`
- required violation rate: below `0.05`

The tail penalty activated on epoch two, but the final realized policy stayed
well outside the registered risk region. The failure is large enough that it
cannot be attributed to quantile interpolation or float rounding.

## Consequence

Do not submit the plain-rank or shuffled-label one-step controls, a 20-step
TB-GPPO arm, or a hyperparameter repair sweep. The next active evidence is the
complete 512-row evaluation of Candidate A2, Active-Set Selective-Risk
ES-GR-CPPO. Its promotion decision must be made against both the frozen anchor
and matched-data/update SFT control.
