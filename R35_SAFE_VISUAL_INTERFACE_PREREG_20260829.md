# R35 preregistration: safe visual-interface FEPO

## Motivation

R30 showed a verified visual-merger gradient and a post-update logit effect,
but failed the training sentinel risk gate: the final no-target first-action
margin fell to `-4.75`. This separates representation plasticity from
abstention safety. EVP motivates testing the visual interface as a bottleneck;
OpenWorldSAM motivates treating missing targets as a protected tail risk.

## Hypothesis

Allowing only the visual merger/deepstack LoRA to update can improve the
geometry policy without sacrificing null behavior when the fixed null CE and
first-action margin terms are strengthened before training. R35 changes only
those two objective weights relative to R30 (`null_ce_weight=2.0`,
`margin_weight=1.0`). The geometry reward, native-relative first-divergence
credit, data order, seed, optimizer, and photometric supervised view remain
unchanged.

This is one SAMTok mask-or-null policy. It has no PixVL checkpoint or trainer,
OPD teacher, EMA, counterfactual/synthetic labels, inference router, or
self-supervised cycle.

## Fixed protocol

- Initialization: frozen `continued_sft_to500` SAMTok adapter.
- Training: exactly 5,120 rows, 10 outer steps, K=4 grammar-valid rollouts.
- Trainable scope: visual merger and deepstack merger LoRA only; anchor adapter
  frozen and required to be logit-equivalent before the first update.
- View CE: same-row target-preserving photometric view, brightness `1.03`,
  contrast `0.97`, coefficient `0.10`.
- Null protection: unified 32-row sentinel, active-set budgets unchanged from
  R18, with the fixed weights above.
- Evaluation: exactly 512 holdout rows and 20,000 paired bootstrap repeats,
  including positive cIoU, boundary IoU, utility, no-target recall,
  canonical/invalid rates, and fixed slices.

## Gates and decision

The run must pass grammar, effective-support, finite-ratio, positive-gradient,
visual-gradient, post-update visual-effect, active-set, and tail-risk gates.
Only then may its complete holdout be evaluated. Promotion requires utility and
positive-cIoU non-inferiority to R18 and matched-SFT, no-target non-inferiority,
zero invalid outputs, canonical response non-regression, and no fixed-slice
drop over `0.01`. A failed gate closes R35 without changing either weight or
using holdout outcomes to tune it.

R35 is prepared but not submitted while the PV-FEPO and matched-SFT jobs are
queued; at most one new 2-GPU training screen may be active under the 24-GPU
budget.
