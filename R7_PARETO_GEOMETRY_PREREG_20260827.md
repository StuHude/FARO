# R7 Pareto Geometry Credit (2026-08-27)

## Hypothesis

R5 and R6 transfer positive sampled cIoU gains mainly as selective-risk
calibration; their positive-cIoU intervals still overlap zero. R7 tests whether
the missing signal is a noisy single geometry scalar. A sampled complete
SAMTok mask receives policy credit only when it improves both clean-view cIoU
and boundary IoU over the same prompt's native greedy mask. The advantage is
the geometric mean of the two positive gains, normalized over active samples
in the rollout group. No trajectory that is worse on either axis receives
credit.

This is a single SAMTok mask-or-null policy. It uses no PixVL training,
self-supervised cycle, counterfactual image, inference router, or extra model.
The 32-row unified no-target sentinel and all existing grammar, effective
support, and risk gates remain unchanged. cIoU and boundary IoU are measured
only on training rollouts; the 512-row holdout is untouched until evaluation.

## Paper connections

DR2Seg motivates retaining a continuous region-quality signal while making
boundary quality explicit. Qwen3VL-Seg and OpenWorldSAM motivate preserving a
single unified mask interface whose spatial output, rather than a separate
classifier, is optimized. Fine-R1 motivates crediting only verifiable
improvements, and V-Zero motivates trajectory-level gating; R7 uses these
ideas without importing OPD or a teacher-view loop.

## Registered screen

- initialization: frozen `continued_sft_to500` SAMTok adapter;
- training: `egfepo_train_5120.jsonl`, exactly 10 outer steps (20 optimizer
  updates), two GPUs;
- rollout: existing effective-support K=4 grammar sampler;
- reward: raw cIoU and raw boundary IoU, no ranks or learned weights;
- credit: `sqrt(relu(ciou-ciou_g-1e-4) * relu(boundary-boundary_g-1e-4))`,
  group-normalized over active trajectories;
- negative objective: shared sentinel-only null CE/margin repair;
- validity: all existing grammar, nonconstant reward, epoch-two ratio,
  effective-support, active-set, and tail-risk gates must pass, with at least
  one Pareto-active trajectory.

The full 512-row holdout is required after the screen. Promotion requires
positive-cIoU mean delta at least zero with a 95% CI lower bound above `-0.005`,
utility mean delta at least zero, no-target recall CI lower bound above `-0.01`,
positive mask rate at least `0.99`, zero invalid outputs, and no canonical
response regression. A geometry claim requires positive-cIoU CI lower bound
above zero or mean delta at least `+0.005`; otherwise R7 is closed as an
informative multi-objective calibration ablation.
