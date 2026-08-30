# R27 preregistration: confidence-gated native rank-local geometry

## Hypothesis

OpenWorldSAM and V-Zero motivate explicit confidence-aware positive/null
separation: uncertain predictions should abstain from aggressive mask updates.
R27 tests a deliberately bounded variant in the existing SAMTok FEPO policy.
It preserves R18/R21 native-relative, joint cIoU/boundary geometry credit and
first-divergence code localization, but gates that detached credit on the
existing per-prefix calibration uncertainty.  Confidence is
`1 - (entropy + missing top-support mass)/2`; values are clipped to a fixed
floor of `0.25`, and only confidence >= `0.60` receives positive geometry
credit.  There is no new model head, teacher, router, OPD target,
counterfactual label, or inference-time behavior.

## Fixed screen

- SAMTok-only continued-SFT anchor, seed `17`.
- `egfepo_train_5120.jsonl`: exactly 5,120 rows, including 2,560 no-target rows.
- Ten outer steps, two policy epochs, K=`4` sibling rollouts, two GPUs.
- Unified 32-row training sentinel, support calibration, PPO and validity gates
  remain unchanged.
- Fixed `confidence_threshold=0.60`, `confidence_floor=0.25`; no sweep.

The matched control is R21 native rank-local credit under the same seed,
schedule, and optimizer budget.  Record confidence-gate activation,
per-stratum active-credit rates, geometry diversity, support reach, epoch-two
ratio movement, sentinel risk, invalid outputs, and trainable-parameter
identity.

## Evaluation and closure

After all training gates pass, evaluate every one of the 512 paired holdout
rows with 20,000 paired bootstrap resamples against R21.  Promote only if
positive cIoU and selective utility each improve by at least `+0.010` with
nonnegative paired-bootstrap lower bounds, no-target recall is non-inferior,
invalid-output rate is `0`, positive-mask rate is at least `0.95`, and no
registered geometry slice drops by more than `0.01`.  Otherwise close R27
without changing the threshold, floor, seed, or gate definition.

Complete RefCOCO/GRefCOCO transfer evaluations are allowed only after this
holdout gate passes.  Outputs remain under `Faro_ailab` and below 700G.  Any
future rjob must use namespace `ailab-dnacoding`, a `dna-` name, and all
positive tags from `rjob_tags.txt`; this preregistration submits no job.
