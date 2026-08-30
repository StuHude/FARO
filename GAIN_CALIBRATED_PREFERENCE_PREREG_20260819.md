# Gain-Calibrated Pixel Preference Preregistration (2026-08-19)

## Candidate G hypothesis

Candidate F proved that direct online mask-code preference changes the greedy
policy, but its uniform pair loss produced more positive degradations than
improvements and eight one-way no-target regressions. Candidate G tests one
specific explanation: a sampled mask that beats greedy by a negligible amount
should not receive the same preference strength as a large verified cIoU gain.

For active best-vs-greedy pairs, define within each distributed optimizer batch:

```text
gain_i = best_sampled_cIoU_i - native_greedy_cIoU_i
weight_i = gain_i / mean(gain over active pairs)
loss_i = weight_i * softplus(-native_log_odds_shift_i)
```

Inactive pairs retain zero loss. The mean active weight is exactly one, so the
change redistributes the frozen Candidate F policy strength instead of adding a
new coefficient. Everything else remains fixed: original SAMTok anchor, plain
cIoU, K=4 effective-support exploration, support size 8, target support 4,
`1e-4` activity threshold, native temperature-1 scoring, two epochs, optimizer,
data, seed, canonical-null CE, and first-action margin term.

This is a SAMTok-only online pixel-policy preference objective. It uses no
router, PixVL trainer or weight, cycle training, counterfactual example, FIFO,
boundary mix, hard label, or inference-time component.

## Sequential gate

Run one two-GPU step. Require epoch-zero ratio consistency, at least one active
pair, finite positive weights with distributed active mean one, nonzero positive
gradient, epoch-two ratio change, support hit rate `1.0`, and maximum clip
fraction no greater than `0.5`. Only a passing run receives the fixed 20-step
configuration and complete 512-row evaluation.

Promotion requires noninferiority to both fixed-null ES and matched SFT in
utility and positive cIoU, a no-target CI lower bound above `-0.01` versus
fixed-null ES, mask rate at least `0.99`, invalid rate zero, and exact canonical
response auditing. No weighting exponent, cap, threshold, seed, K, or learning
rate sweep is allowed.

## Candidate G result

Both training gates passed. The 20-step run had 160/160 nonconstant and
multitrajectory groups, 132 sampled improvements, 87/160 active groups, and
distributed active-weight means between `0.9999999` and `1.0000001`. The
complete 512-row result was:

```text
selective utility       0.779807
positive cIoU           0.770551
no-target recall        0.789063
positive mask rate      1.000000
invalid output rate     0.000000
```

Relative to fixed-null ES, utility was `-0.013821` (95% CI
`[-0.024983,-0.004372]`), positive cIoU was `-0.000299`
(`[-0.005839,+0.004515]`), and null recall was `-0.027344`
(`[-0.046875,-0.007812]`). Seven no-target disagreements were all one-way
regressions. Positive changes again split into four improvements, six
degradations, and two serialization-only changes; canonical positive response
rate was `254/256`.

Candidate G is closed. Gain calibration neither preserved selective risk nor
made geometry transfer reliable. No exponent, cap, or threshold variant is
permitted; the online preference branch is closed.
