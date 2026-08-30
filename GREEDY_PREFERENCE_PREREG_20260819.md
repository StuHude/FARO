# Greedy-Crossing Pixel Preference Preregistration (2026-08-19)

## Failure evidence

Fixed-null ES-GR-CPPO and Boundary-Credit ES produced many sampled masks above
the native greedy reward but almost no beneficial greedy-output changes.
Improvement-Only ES-PPO transferred two clear positive improvements, but also
two degradations and three one-way no-target regressions. Its full 512-row
utility was `-0.005294` below fixed-null ES and its null-recall delta confidence
interval was `[-0.027344, 0]`. Candidate E is therefore closed.

## Candidate F hypothesis

The remaining positive bottleneck may be an argmax crossing problem rather
than reward availability. Candidate F forms one online preference pair per
positive prompt only when the best of the registered K=4 sampled masks exceeds
the current native greedy cIoU by more than `1e-4`:

```text
winner = argmax(sampled cIoU)
loser = native greedy mask code
old_odds = log pi_old(winner) - log pi_old(loser)
loss = softplus(-((log pi(winner) - log pi(loser)) - old_odds))
```

The log probabilities use the native temperature-1 full depth-specific mask
vocabulary, not the exploration distribution. Groups without an improving
winner contribute zero preference loss. This directly increases the sampled
winner's sequence odds relative to the actual greedy mask code.

Candidate F retains the continued-SFT initialization, plain cIoU reward,
effective-support K=4 sampler, support size 8, target support 4, two policy
epochs, fixed canonical-null CE and first-action margin constraint, optimizer,
data, and seed from fixed-null ES. It has no router, PixVL training component,
cycle training, self-generated label, counterfactual image/prompt, FIFO,
boundary mix, hard-slice label, or inference-time module.

## Sequential gate

Run exactly one two-GPU outer step first. It advances only if all eight groups
remain grammar-valid, at least six are multitrajectory, at least two have
nonconstant cIoU, at least one best-vs-greedy preference is active, the
positive preference gradient is nonzero, epoch two changes the winner
probability ratio, support hit rate is `1.0`, and maximum clip fraction is no
greater than `0.5`.

Only a passing one-step job permits the fixed 20-step configuration. The
20-step adapter must then be evaluated on all 512 registered rows. It advances
only if utility and positive cIoU are noninferior to both fixed-null ES and the
matched SFT control, the no-target recall delta confidence-interval lower
bound versus fixed-null ES exceeds `-0.01`, positive mask rate is at least
`0.99`, invalid rate is zero, and exact canonical response rates are reported.
No loss-temperature, threshold, K, support, seed, or optimizer sweep is
allowed after observing the holdout.

## Invalid r1 implementation run

The first one-step job
`dna-samtok-fepo-greedy-pref-one-step-2g-98104632` is invalid and supplies no
method evidence. Its epoch-zero winner probability ratio mean was `0.931706`
instead of 1. The frozen preference log probability had been measured after
the rollout helper restored `model.train()`, while update-time rescoring used
dropout-disabled mode. This was a reference-state mismatch, not a failed
hypothesis. The fix changes no method constant: old and current native scores
now both disable dropout, and a new runtime gate rejects any epoch-zero maximum
ratio deviation above `0.01`. Candidate F must restart from the frozen anchor
in a fresh one-step r2 job.

## Candidate F result

The corrected r2 one-step and 20-step training gates passed. The 20-step run
had 160/160 nonconstant and multitrajectory groups, 125 sampled trajectories
above native greedy, 84/160 groups with an active preference, mean active
fraction `0.525`, and effective-support hit rate `1.0`. Its complete 512-row
evaluation was:

```text
selective utility       0.778589
positive cIoU           0.772023
no-target recall        0.785156
positive mask rate      1.000000
invalid output rate     0.000000
```

Relative to fixed-null ES, utility changed by `-0.015039` (95% CI
`[-0.026868,-0.005072]`), positive cIoU by `+0.001172`
(`[-0.002227,+0.005041]`), and null recall by `-0.031250`
(`[-0.054688,-0.011719]`). All eight no-target disagreements were one-way
regressions. Twelve positive raw outputs changed: four improved, six degraded,
and two changed serialization only. Exact canonical positive response rate was
`254/256`.

Candidate F is closed and must not receive 100 steps. A uniform online
preference does cross more greedy decision boundaries, but it treats tiny and
large training cIoU improvements equally. The result is neither selective-risk
safe nor a reliable geometry transfer mechanism.
