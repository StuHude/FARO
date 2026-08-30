# Greedy-Relative ES-PPO Preregistration (2026-08-19)

## Candidate H hypothesis

Improvement-Only ES-PPO used the native greedy cIoU baseline but discarded all
negative advantages. It transferred two positive gains, yet lost three
no-target decisions relative to fixed-null ES. Uniform and gain-weighted online
preference then changed more greedy masks but produced more geometry
degradations than improvements and severe one-way null drift.

Candidate H tests whether the missing stabilizer is the signed ReMax-style
control variate rather than another preference loss. For every K=4 sampled
mask trajectory:

```text
delta_i = sampled_cIoU_i - native_greedy_cIoU
advantage_i = delta_i / mean(abs(delta) in the group)
```

This preserves positive credit for sampled improvements and negative credit for
trajectories worse than the actual greedy policy, without group-mean centering.
The normalization has no fitted scale and uses training rollouts only.

All other mechanisms remain exactly fixed-null ES: original SAMTok continued-
SFT anchor, plain cIoU, K=4 effective-support exploration, top-8 support,
target support 4, two clipped PPO epochs, optimizer, data, seed, canonical-null
CE, and first-action margin. There is no router, PixVL training method or
weight, self-supervised cycle, counterfactual example, preference pair, FIFO,
boundary mix, hard slice, or inference module.

## Sequential gate

One two-GPU step must have finite signed advantages with both signs globally,
nonzero positive-policy gradient, at least six multitrajectory groups, at least
two nonconstant groups, epoch-two ratio change, support hit rate `1.0`, and
maximum clip no greater than `0.5`. Only a passing run receives the frozen
20-step configuration and full 512-row evaluation.

Promotion requires utility and positive-cIoU noninferiority to fixed-null ES
and matched SFT, no-target CI lower bound above `-0.01` versus fixed-null ES,
mask rate at least `0.99`, invalid rate zero, and exact canonical auditing. No
normalizer, clipping, K, support, seed, or learning-rate sweep is allowed.

## Candidate H result

The one-step and 20-step training gates passed. Across 160 groups, 129 sampled
trajectories beat greedy and 87 groups contained an improvement. Positive and
negative advantage fractions averaged `0.2016` and `0.6188`; maximum clip was
only `0.0625`. The full 512-row result was:

```text
selective utility       0.792824
positive cIoU           0.769242
no-target recall        0.816406
positive mask rate      1.000000
invalid output rate     0.000000
```

Relative to matched SFT, utility was `+0.012263` (95% CI
`[+0.002911,+0.023208]`) and null recall was `+0.027344`, but positive cIoU was
`-0.002819` (`[-0.006359,-0.000188]`). Relative to fixed-null ES, null decisions
were identical, while positive cIoU was `-0.001608` (`[-0.003832,0]`) and
utility was `-0.000804` (`[-0.001945,0]`). All three changed positive outputs
degraded; exact canonical positive response rate was `250/256`.

Candidate H is closed for scaling. Signed negative advantages restore
selective stability, but their much larger support contracts the policy toward
the old greedy output and erases geometry gains.
