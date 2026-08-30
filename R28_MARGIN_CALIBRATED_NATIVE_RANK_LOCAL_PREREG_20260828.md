# R28 preregistration: margin-calibrated native rank-local geometry

R28 tests whether the native rank ordering from R21 becomes more reliable when
credit is calibrated by the magnitude of the joint geometry improvement.
Each sibling rollout must improve both cIoU and boundary IoU by `1e-4`; its
credit is the native tie-aware rank gain multiplied by
`(cIoU_gain * boundary_gain)^0.5`, then localized to the first changed SAMTok
code depth with decay `0.85`. The margin is detached rollout geometry, so this
is a single SAMTok policy with no router, expert, OPD target, PixVL loop,
counterfactual label, or inference-time change.

## Fixed screen

- SAMTok continued-SFT anchor, seed `17`; 5,120 rows (2,560 no-target).
- Ten outer steps, two policy epochs, K=`4`, two GPUs.
- Unified 32-row sentinel, effective-support calibration, PPO and validity
  gates unchanged from R21.
- Fixed `margin_power=0.5`; no sweep.

The matched control is R21 native rank-local credit under identical data,
schedule, seed, and optimizer budget. Report active-margin fractions, joint
gain distributions, reward diversity, support reach, epoch-two ratio movement,
sentinel risk, invalid outputs, and trainable-parameter identity.

## Evaluation and closure

After training gates pass, evaluate all 512 paired holdout rows with 20,000
paired bootstrap resamples. Promote only with positive cIoU and utility gains
of at least `+0.010` and nonnegative lower bounds, non-inferior no-target
recall, zero invalid outputs, positive-mask rate at least `0.95`, and no
registered geometry slice dropping more than `0.01`. Otherwise close R28
without changing the margin exponent, threshold, seed, or credit rule.

All outputs remain under `Faro_ailab` below 700G. Future jobs require
`ailab-dnacoding`, a `dna-` name, and every positive tag in `rjob_tags.txt`;
this preregistration submits no job.
