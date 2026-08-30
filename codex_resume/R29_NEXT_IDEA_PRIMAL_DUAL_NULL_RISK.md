# R29 proposal: primal-dual null-risk FEPO (SAMTok-only)

## Why this is a new test

R18 demonstrates a strong in-domain holdout gain but its complete RefCOCO
transfer is slightly lower than continued-SFT. R26 changes a fixed sentinel
tail weight, and R24 tests an anchor-KL trust region; neither tests whether a
fixed null penalty is itself miscalibrated over training. R29 keeps R18's
native-relative joint cIoU/boundary credit and first-divergence localization,
but replaces the fixed tail-risk coefficient with a scalar Lagrange multiplier
updated from the training-only no-target sentinel.

This is motivated by OpenWorldSAM's explicit open-world absence outcome,
Qwen3VL-Seg's positive/negative separation, Fine-R1's grouped on-policy
optimization, and SenseNova-Vision's post-adaptation retention requirement.
It does not use a router, expert, teacher, OPD target, counterfactual label,
PixVL training loop, or inference-time branch.

## Falsifiable hypothesis

The fixed `sentinel_tail_weight=0.25` can over-regularize safe updates or
under-regularize a transient null-margin failure. A primal-dual update should
keep the realized lower-tail null margin near its frozen-anchor budget while
allowing verified geometry credit when the constraint is inactive. The method
is rejected if the dual variable never activates during a risk excursion, if
it saturates at its cap, or if complete holdout utility/positive cIoU does not
beat the matched R21/R18 control without null-risk regression.

## Minimal fixed experiment (no sweep)

- Initialization: frozen `continued_sft_to500` SAMTok adapter, seed `17`.
- Data: exactly `egfepo_train_5120.jsonl` (5,120 rows, 2,560 no-target rows).
- Training: 10 outer steps, K=4 grammar-valid siblings, two policy epochs,
  two GPUs; all existing effective-support and validity gates remain active.
- Geometry credit: exactly R18 native-relative joint cIoU/boundary rank-local
  credit (`minimum_improvement=1e-4`, depth decay `0.85`).
- Null risk: use the existing 32-row training sentinel and lower-10% current
  minus anchor first-action margin risk. Define normalized excess
  `e_t=max(0, anchor_q10 - slack - current_q10)/max(slack,1e-6)`.
- Fixed dual update after each optimizer step:
  `lambda_{t+1}=clip(lambda_t + 0.20*e_t, 0, 4)`, with `lambda_0=1`.
  Add `lambda_t * e_t` to the differentiable sentinel risk term; do not
  backpropagate through the detached q10 used to update the multiplier.
- Report dual activation fraction, maximum lambda, q10 margin trajectory,
  geometry-credit activity, ratio/support diagnostics, and all existing null
  and invalid-output metrics. No holdout-derived tuning is permitted.

## Local synthetic check

For anchor q10 `0.80`, slack `0.05`, and observed q10 sequence
`[0.72,0.73,0.76,0.78,0.74,0.76,0.71,0.77]`, the fixed update gives
lambda sequence `[1.12,1.20,1.20,1.20,1.24,1.24,1.40,1.40]`. It is finite,
activates only below the registered budget (`0.75`), and remains far below
the cap; an all-safe sequence leaves lambda at `1.0`, while a persistent
violation reaches the cap and is rejected by the activation/saturation gate.
This verifies the update's sign, clipping, and inactive behavior without
requiring torch or a GPU.

## Promotion and queue decision

After training, require all standard validity/support/sentinel gates, then
evaluate all 512 paired holdout rows with 20,000 bootstrap resamples and the
enhanced boundary/slice evaluator. Promote only if utility and positive cIoU
have nonnegative CI lower bounds and at least `+0.010` mean improvement over
the matched R21/R18 control, no-target recall CI lower bound is at least
`-0.01`, invalid rate is zero, positive-mask rate is at least `0.95`, and no
registered geometry slice drops by more than `0.01`. Only then run complete
RefCOCO/GRefCOCO transfer and capability-retention checks.

**Queue recommendation: conditional, not now.** R29 is worth one 2-GPU
screen if R24/R21--R28 fail or show sentinel-risk drift, because it tests a
different causal mechanism and uses the same paper-facing unified policy.
Do not submit it while seven existing screens plus the terminal baseline
evaluation occupy the queue; submit only after a slot/result is available.
