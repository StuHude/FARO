# EG-FEPO Three-Arm Result (2026-08-20)

All three arms used the frozen 500-step standalone SAMTok adapter, 5,120
training rows, 10 optimizer steps, K=4 grammar rollouts, and the same
effective-support controller. The only changed field was the detached group
multiplier: `view_drop`, `shuffled`, or `none`. All runs passed the training
validity gate (`20/20` nonconstant reward groups, `1.0` target-support
fraction, and positive-policy gradients).

The evaluation used the complete 512-row image-disjoint paired holdout (256
positive and 256 no-target rows). The frozen anchor is the matched
`standalone_continued_sft_to500_512` result.

| arm | positive cIoU | no-target recall | positive canonical | negative canonical | utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchor | 0.770612 | 0.796875 | n/a | n/a | 0.783743 |
| none | 0.770047 | 0.812500 | 0.976563 | 0.812500 | 0.791273 |
| shuffled | 0.772044 | 0.808594 | 0.988281 | 0.808594 | 0.790319 |
| view_drop | 0.768529 | 0.812500 | 0.980469 | 0.812500 | 0.790514 |

Paired bootstrap intervals use 20,000 resamples over the 256 positive/no-target
pairs. Deltas are arm minus anchor:

| arm | utility delta (95% CI) | positive cIoU delta (95% CI) | no-target delta (95% CI) |
| --- | --- | --- | --- |
| none | +0.007530 `[+0.000580,+0.016300]` | -0.000565 `[-0.005241,+0.004894]` | +0.015625 `[+0.003906,+0.031250]` |
| shuffled | +0.006575 `[+0.000696,+0.014388]` | +0.001432 `[-0.001374,+0.005946]` | +0.011719 `[0,+0.027344]` |
| view_drop | +0.006771 `[+0.000054,+0.015123]` | -0.002083 `[-0.005270,+0.000023]` | +0.015625 `[+0.003906,+0.031250]` |

The direct paired comparison shuffled minus view-drop is `+0.003515`
positive cIoU (95% CI `[0,+0.008795]`) and `-0.003906` no-target recall
(95% CI `[-0.011719,0]`). Therefore the result does not identify a causal
benefit from view-drop evidence: the randomized-label control is at least as
good on utility and better on positive geometry. The evidence-gated branch is
closed at 10 steps and must not be expanded to 20/100 steps as a claimed
evidence effect.

The common RL protocol itself is viable: all arms have valid diverse
rollouts, stable grammar decoding, zero invalid outputs, and a measurable
selective-utility gain over the anchor. The next experiment must change the
credit-transfer hypothesis rather than sweep the gate scale or tune on this
holdout. Any new minimum-cost run must still use at least 5,000 training rows
and 10 steps.

## 20-step confirmation

The clean `none` arm was extended once to the fixed 20-step configuration,
without changing the reward, optimizer, support controller, or data. It again
used 5,120 training rows and passed every training validity gate. On the same
512-row holdout it reached positive cIoU `0.768416`, no-target recall
`0.820313`, positive canonical rate `0.976563`, and negative canonical rate
`0.820313` (utility `0.794364`). Against the frozen anchor, 20,000 paired
bootstrap resamples give:

| metric | delta (95% CI) |
| --- | --- |
| utility | `+0.010621` `[+0.002137,+0.020987]` |
| positive cIoU | `-0.002196` `[-0.006049,+0.001034]` |
| no-target recall | `+0.023438` `[+0.007813,+0.042969]` |

This confirms the short-run pattern rather than a mode-specific failure:
effective-support GR-CPPO is a valid SAMTok-only selective-calibration method,
but its measurable benefit is abstention/null-risk calibration, not pixel
geometry. It fails the registered nonnegative positive-cIoU promotion gate, so
no 100-step run or official GRefCOCO/RefCOCO claim is justified. The honest
paper direction is a negative/diagnostic result: a strong SFT anchor plus
geometry-verifiable RL can improve no-target behavior while leaving (and here
slightly reducing) mask geometry, and the visual evidence gate does not cause
that gain because its shuffled and none controls match or outperform it.
