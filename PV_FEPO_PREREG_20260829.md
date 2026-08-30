# PV-FEPO preregistration (2026-08-29)

## Hypothesis

R18 improves the image-disjoint SAMTok holdout but is slightly below
continued-SFT on the complete RefCOCO transfer set. R21--R34 show that further
scalar rank, margin, uncertainty, or null-tail shaping does not produce a
reliable geometry gain. PV-FEPO tests a different question: a mask-code action
should receive credit only when its ground-truth geometry improvement survives
a fixed, target-preserving appearance view. This is paired-view robust relative
RL, not a teacher, OPD, EMA, counterfactual, or PixVL cycle.

## Fixed screen

- Exact continued-SFT-to-500 SAMTok anchor, seed 17.
- `egfepo_train_5120.jsonl`: exactly 5,120 rows and 2,560 no-target rows.
- Ten outer steps, two policy epochs, K=4 grammar-valid siblings, two GPUs.
- Existing effective-support controller, FIFO/native reference, first changed
  SAMTok depth, shared 32-row no-target sentinel, and all R18 gates.
- Language LoRA only; visual merger, SAMTok decoder, and base model stay frozen.
- One deterministic geometry-preserving appearance view per row: brightness
  `1.03`, contrast `0.97`; no transform or coefficient sweep.

For each sampled code trajectory, compute the existing native-reference
midrank joint cIoU/boundary-IoU credit independently on clean and transformed
views. A trajectory receives `sqrt(max(c_clean,0)*max(c_view,0))`; therefore a
clean-only improvement cannot dominate. Native greedy references are computed
on both views. The same sampled mask-code trajectory is scored against the
same row's GT mask on the transformed view. Only the clean policy is sampled
and optimized; the second view supplies a detached, GT-verified reward
observation. No evaluator or holdout artifact is read.

## Controls and gates

The screen must be compared with clean-view R18 at the same rows, steps,
rollouts, and with a forward-count-matched clean-reward control. A matched SFT
control is retained to distinguish RL from ordinary supervised/compute effects.

Evaluate all 512 paired holdout rows with 20,000 paired bootstrap repetitions.
Promote only if, relative to R18, positive cIoU mean is at least `+0.005` with
95% CI lower bound above zero, utility CI lower bound is nonnegative, no-target
recall CI lower bound is at least `-0.01`, invalid output rate is zero,
positive-mask rate is at least `0.95`, and no fixed small/thin/boundary/area
slice falls by more than `0.01`. The candidate must also be non-inferior to
the clean-reward control and to matched SFT. Require finite clean/view reward
correlation, joint-positive fraction at least `0.20`, and valid transformed
grammar scoring. Only a survivor receives the complete 5,000-row RefCOCO and
14,229-row GRefCOCO transfer evaluations; RefCOCO cIoU must improve by at
least `+0.003` versus continued-SFT with no AP50 drop larger than `0.002`.

If the paired reward fails to beat clean-only, or the joint-positive fraction
is below `0.20`, close PV-FEPO without changing the transform, aggregation, or
threshold. All artifacts stay under `Faro_ailab`; no PixVL weight or training
cycle is used.

## Implementation amendment (pre-runtime)

The paired credit helper now receives a separate `augmented_native_codes`
trajectory. Clean and transformed views therefore each localize credit against
their own greedy native policy; using clean native codes for both views would
misassign first-divergence depth when photometric changes alter greedy
decoding. The runtime trainer passes transformed-view greedy codes explicitly.
This is a correctness fix, not a changed hyperparameter or an additional
candidate.

The offline contract probe passes with 5,120 manifest rows, ten steps, K=4,
and all registered R21--R30/PV contracts. No rjob was submitted from the head
node because the rjob API was unreachable; runtime validation remains a
worker-image requirement.
