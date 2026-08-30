# R22 preregistration: scale-stratified native rank-local geometry credit

R18 established the most reproducible signal in this workspace: a single
SAMTok policy with native-relative, joint cIoU/boundary improvement credited at
the first changed mask-code depth. R19 and R20 did not improve it. Their common
failure mode is that the same geometry scalar and axis trade-off is applied to
small, medium, and large targets, although boundary errors have very different
pixel support across those scales.

## Hypothesis

Training-only target area strata will make the existing native-relative signal
better conditioned without introducing a router at inference. Small targets
will rank cIoU/boundary quality with weights `(0.35, 0.65)`, medium targets use
`(0.50, 0.50)`, and large targets use `(0.65, 0.35)`. The target-area strata
are derived from the training masks only (q25/q75), are written into the
geometry registry, and never enter holdout evaluation or model inputs.

The sampled trajectories remain sibling rollouts (`K=4`). Native greedy is an
explicit reference point. A trajectory receives credit only if both cIoU and
boundary IoU improve by `1e-4`; its tie-aware native rank gain is assigned to
the first changed SAMTok code depth and decayed by `0.85`. Mixed-axis wins,
regressions, unchanged native trajectories, and no-target rows receive zero
geometry credit. The no-target sentinel and canonical null CE/margin losses are
unchanged.

This is a training-time conditioning variable, not an inference router,
additional expert, OPD path, PixVL training loop, or counterfactual label.
Initialization is the exact frozen continued-SFT SAMTok anchor used by R18.

## Minimal screen

Use `egfepo_train_5120.jsonl` (5,120 rows, 2,560 no-target rows), 10 outer
steps, two GPUs, two policy epochs, and the existing effective-support and
unified 32-row sentinel contract. No hyperparameter sweep is registered. The
matched control is R18's frozen native rank-local method under the same data,
schedule, seed, and optimizer budget; a continued-SFT control is reported for
context.

The screen must record registry/schedule hashes, per-stratum sample counts,
per-stratum joint-gain and active-credit fractions, cIoU/boundary reward
diversity, effective support, epoch-2 ratio movement, sentinel tail risk, and
invalid-output count. The full 512-row holdout and 20,000 paired bootstrap
must be run before any promotion decision; no 32-row smoke result is a claim.

## Gates

Training validity, support, sentinel, and representation gates must all pass.
Relative to R18, promote only if positive cIoU improves by at least `+0.010`
and utility by at least `+0.015`, each with a paired-bootstrap lower bound no
less than zero, while no-target recall is non-inferior and no area-stratum
positive-cIoU slice drops by more than `0.01`. Otherwise close R22 without a
weight or seed sweep. A promoted candidate must then run the registered full
RefCOCO and GRefCOCO transfer evaluations, with 512-row minimum validation
sets and the same adaptive 8->6->4->2->1 GPU queue policy.

## Provenance and constraints

Only SAMTok checkpoints and the existing evaluation/data interfaces are used.
No files are written under `PixVL_ailab`; outputs stay under `Faro_ailab` and
must remain below the 700G workspace limit. Any future rjob must use the
`ailab-dnacoding` namespace, a `dna-` name, and every positive tag in
`rjob_tags.txt`; this preregistration does not submit a job.
