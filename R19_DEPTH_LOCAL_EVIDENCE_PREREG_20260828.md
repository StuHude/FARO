# R19 preregistration: depth-local geometry with visual evidence gating

R19 tests whether the R18/R16 rarity-free earliest-divergence geometry credit
benefits from a detached sibling-relative visual-support signal. Each sampled
mask action is scored on the clean image and a deterministic one-pixel cyclic
shift with light Gaussian noise. The within-group evidence gap is converted to
a clipped multiplier in `[0.25, 1.75]` and applied only to the detached PPO
advantage. It cannot change the reward sign, the canonical no-target objective,
the native anchor, or the decoded output. This is a SAMTok-only RL experiment;
there is no teacher target, OPD loss, PixVL weight, or self-supervised cycle.

The run uses the frozen continued-SFT anchor, 5,120 rows, K=4 effective-support
rollouts, 10 outer steps, two GPUs, and the unified 32-row sentinel. The first
screen is one complete 512-row paired holdout against the same frozen anchor,
with 20,000 paired bootstrap resamples. It is promoted only if the existing
validity/risk gates pass, utility has a positive 95% interval, and positive
cIoU has a mean gain of at least `0.015` with a nonnegative lower confidence
bound. A shuffled evidence gate is a control, not a tunable fallback; no
threshold sweep is allowed.

The first submission exposed a configuration-only failure before any optimizer
update: the evidence path required the registered action temperature but the
new config omitted that field. It produced no checkpoint or quality metric and
is excluded from all comparisons. The corrected resubmission keeps the same
preregistered method and sets the fixed temperature to `1.0`.
