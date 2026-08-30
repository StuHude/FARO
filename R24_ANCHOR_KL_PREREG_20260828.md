# R24 preregistration: anchor-constrained native geometry FEPO

## Hypothesis

R18 improves the selective holdout but slightly regresses on RefCOCO. R24
tests whether cumulative policy drift from the continued-SFT initialization
causes that transfer loss. It keeps R18 native-relative rank-local geometry
credit and adds a frozen-anchor categorical KL hinge on a deterministic,
training-only 64-row target-present buffer.

For each SAMTok code depth, cache the initial anchor's grammar-constrained
categorical distribution before any update. During each policy epoch compute
the mean KL from the current policy to that cache and add
`0.5 * relu(KL - 0.02)` to the shared PPO loss. The buffer is selected from
the training registry only and is disjoint from the 512-row holdout. No KL
term is applied to holdout rows, no anchor labels are generated, and no
additional policy, expert, visual adapter, inference router, PixVL weight,
OPD target, or self-supervised cycle is introduced.

## Minimal screen

- Exact continued-SFT SAMTok anchor; seed `17`.
- `egfepo_train_5120.jsonl`, exactly 5,120 rows and 2,560 no-target rows.
- Ten outer steps, two policy epochs, K=4 sibling rollouts, two GPUs.
- Existing effective-support calibration, unified 32-row no-target sentinel,
  native-reference rank-local geometry credit, and all R18 validity gates.
- Fixed `anchor_buffer_rows=64`, `anchor_kl_epsilon=0.02`,
  `anchor_kl_lambda=0.5`; no sweep.

The run must report buffer IDs/hash, finite KL at every epoch, initial KL near
zero, p95/max KL, hinge activation fraction, and trainable-parameter identity.
The KL mechanism is considered inactive if its hinge never activates after the
initial zero check.

## Evaluation and closure

Evaluate all 512 paired holdout rows with 20,000 paired bootstrap resamples.
Relative to R18, promote only if positive cIoU and utility are non-inferior
(paired lower CI >= -0.005), no-target recall lower CI >= -0.01, invalid rate
is zero, positive-mask rate >= 0.99, and RefCOCO cIoU improves by >= 0.003 or
has a strictly positive paired interval without AP50 regression versus
continued-SFT. At least two registered small/thin/boundary slices must not
drop. A failed or inactive hinge closes R24 without epsilon/lambda/seed sweep.

All jobs use `dna-` names, namespace `ailab-dnacoding`, every positive tag in
`rjob_tags.txt`, and outputs under `Faro_ailab` below the 700G limit.
