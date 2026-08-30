# R21 preregistration: anchor-constrained native geometry FEPO

## Motivation and falsifiable hypothesis

R18's native-relative local geometry credit improves the 512-row GRefCOCO
holdout, but the complete RefCOCO transfer is slightly below the continued-SFT
anchor.  R19's view-drop evidence multiplier and R20's signed negative credit
did not recover this cross-domain gap.  The remaining concrete hypothesis is
policy drift: PPO's old-policy clip is local to each outer update, so repeated
updates can move the SAMTok mask-code distribution away from the broad-domain
continued-SFT behavior.

**H_R21.** A frozen-initialization trust region over SAMTok mask-code actions
will retain R18's in-domain geometry gain while reducing cross-domain drift.
The method is a single SAMTok policy; no inference router, visual projector
plasticity, OPD target, PixVL checkpoint, self-supervised cycle, or
counterfactual label is introduced.

This is independently testable against R18: if the trust region is inactive or
does not improve RefCOCO without sacrificing holdout geometry, R21 is closed.

## Mechanism contract

Initialize from the exact frozen continued-SFT anchor used by R18.  Keep R18's
K=4 effective-support rollouts, two policy epochs, 5,120 rows, 10 outer steps,
native-relative joint cIoU/boundary-IoU improvement, first changed SAMTok code
depth credit with decay `0.85`, and the unified 32-row no-target sentinel.

Before the first optimizer update, run a fixed, training-only 64-row
target-present anchor buffer (stratified by registered area/boundary bins;
IDs are committed in the manifest).  Cache the frozen anchor logits over each
depth's SAMTok codebook.  During every policy epoch, score the current model on
the same rows and compute a detached-anchor categorical KL at code actions:

```
  kl_anchor = mean_depth mean_rows KL(pi_theta(code_d) || pi_anchor(code_d))
  loss_anchor = lambda * relu(kl_anchor - epsilon)
```

The KL is evaluated only over depth-specific grammar candidates, with the
anchor distribution detached.  Use one fixed contract point, `epsilon=0.02`
nats and `lambda=0.5`; no holdout or weight sweep is permitted.  The hinge is
zero while the policy remains close to initialization, so R21 does not alter
R18's gradient in the small-drift regime.  A per-depth diagnostic is logged;
the penalty is the mean over depths.  The existing no-target CE, first-action
margin, and lower-tail sentinel losses remain unchanged.

The anchor buffer is not an additional training objective or teacher target:
it only limits cumulative policy drift.  Report its forward count and memory
cost separately from the R18 baseline.  The old-policy PPO ratio and effective
support remain frozen exactly as in R18.

## Minimal screen (R21-S)

Run one 2-GPU `dna-` training job with all positive tags, `ailab-dnacoding`,
exactly 5,120 rows and 10 outer steps.  Use the same seed (17), pair IDs,
learning rate, and schedule as R18.  The only changed operation is the anchor
KL hinge.  No projector adapter or visual-backbone parameter is trainable.

Training validity must pass R18's grammar/effective-support/reward-diversity,
epoch-2 ratio, unified-sentinel, and LoRA trainability gates.  Additionally:

* all 64 anchor rows and per-depth codebook probabilities are finite;
* anchor KL is zero-effect before update (within `1e-6`) and has finite mean,
  p95, and maximum after every epoch;
* at least one update has a nonzero hinge when drift occurs, while runs with
  no active hinge are marked mechanism-inactive rather than promoted;
* anchor-buffer IDs are disjoint from the 512-row evaluation holdout.

Evaluate all 512 paired holdout rows with 20,000 paired bootstrap resamples,
using the same metrics as R18.  Compare directly against the existing R18
seed-17 records; do not tune on the holdout.

## Promotion and close gates

R21-S is promoted to a frozen 20/100-step paper run only if, versus R18:

1. positive cIoU mean is non-inferior (paired CI lower bound above `-0.005`),
   selective utility is non-inferior (lower bound above `-0.005`), no-target
   recall lower bound is above `-0.01`, invalid output rate is zero, positive
   mask rate is at least `0.99`, and canonical response rate does not fall;
2. the complete RefCOCO transfer set is scored (not a tiny subset), and cIoU
   improves by at least `+0.003` or has a paired CI lower bound above zero,
   while AP50 is non-inferior to the continued-SFT anchor (lower bound above
   `-0.005`);
3. at least two registered small/thin/boundary slices improve by `>=0.01`
   over R18 and no slice drops by more than `0.01`;
4. the KL hinge is active on at least `5%` of update/epoch measurements and
   has no p95 value above `0.20` nats (otherwise the mechanism is either
   inactive or destabilizing).

Close without a sweep if any R18 safety gate fails, in-domain positive cIoU
drops by more than `0.01`, RefCOCO remains a regression without a slice gain,
or the hinge is inactive on every update.  A candidate that only improves
holdout utility while failing the transfer criterion is reported as a null
result for the cross-domain hypothesis, not promoted.

## Paper boundary and follow-up

Fine-R1 motivates retaining an SFT anchor before grouped RL; SenseNova-Vision
motivates capability-retention checks after specialization.  Qwen3VL-Seg and
OpenWorldSAM motivate explicit negative/OOD and small-boundary slices.  EVP and
the latent-denoising paper motivate diagnosing representation/shift robustness
without adding their features or denoising objective.  DR2Seg motivates the
continuous cIoU/boundary reward; V-Zero motivates conservative behavior under
uncertain visual evidence.  These references justify the test, not a claim of
their architectures.

If R21 closes, the next independent candidate should change the reward scale
(object-size-normalized geometry) or use a true moderate-view mask-consistency
reward; do not stack both with this trust region or revive projector-only
plasticity.
