# Matched continued-SFT control audit (2026-08-29)

## Contract

`continued_sft_r18_matched_200` is the forward-count-matched supervised
control for FEPO-R18. It uses the same SAMTok anchor family, 5,120 training
rows, and 200 completed optimizer steps. The evaluator contains exactly 512
image-disjoint rows with 256 target-present and 256 no-target examples.

## Paired comparison against R18

The comparison uses 20,000 paired bootstrap resamples and the frozen enhanced
R18 holdout. Candidate-minus-R18 results are:

| Metric | Delta | 95% CI |
| --- | ---: | ---: |
| positive cIoU | `+0.005641` | `[+0.000070,+0.013302]` |
| selective utility | `+0.006727` | `[+0.001306,+0.013945]` |
| no-target explicit recall | `+0.007812` | `[0,+0.019531]` |
| invalid output rate | `0.000000` | fixed |

The corrected overall promotion gate is true. This establishes that ordinary
continued SFT is a meaningful compute control; it prevents attributing the
overall short-horizon gain to RL alone.

## Geometry-tail qualification

The fixed slice analyzer reports four of six slices as non-inferior. The
boundary-hard and thin slices have boundary-IoU delta means `-0.001711`, with
95% intervals `[-0.010771,+0.006595]` and `[-0.010808,+0.006362]`, respectively.
Their lower bounds cross the registered `-0.01` slice limit. The control is
therefore not promoted as the final FEPO method: it is a supervised baseline
with an overall gain and a documented boundary-tail trade-off.

No holdout statistic is used to tune PV-FEPO, R35, or BA-FEPO. Those branches
remain isolated and follow the preregistered decision order.
