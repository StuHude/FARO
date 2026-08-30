# R25 preregistration: calibrated uncertainty native rank-local FEPO

## Hypothesis

R18 gives positive credit only to sibling masks that improve both cIoU and
boundary IoU over the native greedy SAMTok trajectory, localized at the first
changed code. R25 tests whether that credit is more reliable when it is
calibrated by the rollout's own grammar uncertainty. Following the calibration
and uncertainty signals used in OpenWorldSAM, V-Zero, SenseNova-Vision and
Fine-R1, each rollout receives a detached score combining normalized calibrated
entropy and missing top-support mass. Geometry still decides which rollouts are
eligible; uncertainty only rescales eligible credit with a fixed confidence
floor. No teacher, OPD target, counterfactual label, extra expert/router,
PixVL weight, or self-supervised loop is introduced.

## Minimal screen

- Exact continued-SFT SAMTok anchor; seed `17`.
- `egfepo_train_5120.jsonl`, exactly 5,120 rows (2,560 no-target rows).
- Ten outer steps, two policy epochs, K=4 sibling rollouts, two GPUs.
- Existing per-prefix top-m support calibration, shared 32-row no-target
  sentinel, FIFO geometry registry, native greedy reference, and R18 validity
  gates.
- Fixed uncertainty source
  `calibrated_entropy_plus_missing_top_support_mass` and confidence floor
  `0.25`; no threshold, temperature, or seed sweep.
- The run must report finite uncertainty mean/p95, active positive-credit
  fraction, support-target reach, and complete trainable-parameter identity.
  Uncertainty is training-only and is computed from detached rollout logits.

## Evaluation and closure

Evaluate every one of the 512 paired holdout rows with 20,000 paired bootstrap
resamples. Relative to R18, promote only if positive cIoU and utility are
non-inferior (paired lower CI >= -0.005), no-target recall lower CI >= -0.01,
invalid rate is zero, positive-mask rate >= 0.99, and RefCOCO cIoU improves by
>= 0.003 or has a strictly positive paired interval without AP50 regression
versus continued-SFT. At least two registered small/thin/boundary slices must
not drop. If the uncertainty diagnostics are non-finite, the support target is
not reached, or no positive credit is observed, close R25 without tuning the
confidence floor or introducing a new uncertainty source.

All jobs use `dna-` names, namespace `ailab-dnacoding`, every positive tag in
`rjob_tags.txt`, and outputs under `Faro_ailab` below the 700G limit. The
candidate is not submitted automatically; this preregistration is the static
screen for comparison with R18 and R21-R24.
