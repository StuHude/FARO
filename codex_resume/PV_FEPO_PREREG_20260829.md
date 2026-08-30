# PV-FEPO preregistration (2026-08-29)

## Hypothesis

The selected R18 policy receives clean-view reward only.  A code trajectory
that improves clean cIoU/boundary IoU but fails under a fixed
target-preserving appearance change may be image-specific.  PV-FEPO therefore
keeps R18's single SAMTok mask-or-null policy and first-divergence scope, but
credits a trajectory only when it is jointly better than the native greedy
trajectory on both views.  The two native-relative rank gains are combined by
a fixed geometric mean.

## Fixed recipe

- SAMTok-only initialization from the approved frozen anchor.
- 5,120 training rows (2,560 target-present and 2,560 no-target), 10 outer
  steps, two policy epochs, K=4 grouped rollouts, two GPUs.
- One deterministic photometric view per row: brightness `1.03`, contrast
  `0.97`; no geometric transform and no label change.
- Clean and augmented cIoU/boundary IoU are both computed from the same
  sampled mask codes and same-row ground-truth mask.  R18's canonical null
  sentinel and margin constraint are unchanged.
- Language LoRA only.  No PixVL weights/trainer/cycle, OPD teacher, EMA,
  counterfactual labels, or inference router.

## Falsification and controls

The complete 512-row enhanced holdout and 20,000 paired bootstrap are required.
Promotion requires positive cIoU delta >= `+0.005` with CI lower bound > 0,
utility CI lower bound >= 0, null-recall lower bound >= `-0.01`, zero invalid
outputs, positive-mask rate >= `0.95`, and no fixed slice drop > `0.01` versus
R18.  It must also be non-inferior to the clean-only R18 control and to the
matched-budget continued-SFT control.  If the joint-positive fraction is
below `0.20`, or only the SFT control is competitive, the robust-RL claim is
closed rather than tuned.

No transform strength, threshold, seed, or aggregation sweep is allowed.
Only a passing screen may receive official RefCOCO/GRefCOCO transfer tests.
