# FEPO follow-up hypotheses (2026-08-29)

This document records the next isolated hypotheses after the PV-FEPO training
support gate closed. It does not authorize concurrent submissions while the
single active queue branch is unresolved.
Every candidate remains a single SAMTok mask-or-null policy; PixVL is limited
to the existing evaluator/data interface.

## What the completed screens say

R18 is the only promoted reference. Its native-relative joint cIoU/boundary-
IoU credit gives a clear short-horizon utility and positive-cIoU gain, while
the 100-step confirmation retains utility but has a cIoU interval crossing
zero. R21--R34 all fail the corrected paired-bootstrap promotion gate. The
common failure is important: replacing the credit transform (rank, scale,
uncertainty, null tail, signed credit, or smooth dominance) does not create a
new signal. The next question should therefore concern *where the signal is
measured*, not another scalar reweighting of the same clean rollout.

### PV-FEPO diagnostic update

PV training completed its required 5,120-row/10-step contract. The clean/view
reward correlation is `0.99988` on average, while joint-positive fractions
range from `0.0` to `0.21875` across optimizer records. This is a training
diagnostic only; the preregistered decision still requires the complete
512-row/20k paired holdout and fixed slice gates, so no PV quality claim or
early closure is inferred from these intermediate records.

## Candidate A: Boundary-Agreement FEPO (BA-FEPO)

**Hypothesis.** Qwen3VL-Seg and DR2Seg suggest that boundary errors are a
distinct tail of the pixel-token decision. A sampled trajectory should receive
credit only when its clean-view and a fixed target-preserving photometric view
both improve the native action on the joint geometry score, with the boundary
improvement acting as the bottleneck. This tests whether PV-FEPO's geometric
mean is too forgiving when one view improves only area overlap.

**Training change.** Keep R18's first-divergence scope, K=4 grouped rollouts,
and sentinel contract. Replace the PV aggregation with a fixed
`min(delta_cIoU, delta_boundary_IoU)` eligibility test and a detached
`0.5 * (delta_cIoU + delta_boundary_IoU)` magnitude. No view is used as a
teacher target, and no extra adapter or inference branch is added.

**Controls.** The matched clean R18 arm and the PV geometric-mean arm are
required controls. BA-FEPO is submitted only if PV is closed; it is never
stacked with R35 visual plasticity.

**Falsification.** Require 5,120 rows, 10 steps, finite ratios, complete
512-row holdout, 20,000 paired bootstrap repetitions, positive-cIoU and
utility non-inferiority to R18 and matched SFT, boundary-hard/thin slice
non-regression, null recall lower bound, and canonical-output validity. A
failure closes the view-family line.

**Implementation status.** The branch is implemented as
`boundary_bottleneck_paired_view_geometry_advantages`, registered as a
10-step contract, and covered by static tests. It has not been trained; the
current PV and R35 decisions must finish first.

## Candidate B: Sentinel-Constrained Boundary FEPO (SCB-FEPO)

**Hypothesis.** OpenWorldSAM and SenseNova-Vision motivate treating absence and
capability retention as hard constraints. R18's fixed sentinel is calibrated
on a small training buffer, but a geometry update can still trade rare null
mistakes for positive gains. A fixed, training-only lower-tail constraint
should permit geometry updates only when the sentinel's worst decile remains
inside the R18 anchor margin.

**Training change.** Keep the R18 native-relative geometry credit unchanged.
Add a detached hinge on the sentinel's 10th-percentile first-action margin,
with a fixed coefficient and no adaptive multiplier. The hinge is evaluated
only on the training sentinel and cannot alter inference or create a route.

**Controls.** The existing R26/R29 null-tail screens are negative controls.
SCB-FEPO is considered only if the matched-SFT holdout shows a reproducible
sentinel regression or if PV improves geometry while losing null recall.

**Falsification.** Use the same 5,120-row/10-step and 512-row/20k protocol.
Reject any candidate with a positive-cIoU CI lower than `-0.01`, utility CI
lower than zero, no-target recall CI lower than `-0.01`, invalid outputs above
1%, or canonical/slice regression. If no sentinel drift is observed, this
branch is closed without a job.

## Candidate C: Boundary-slice confirmation (diagnostic, no new objective)

If PV or BA-FEPO survives, the next experiment is a compute-matched longer
horizon with a pre-registered boundary-hard slice as the primary diagnostic.
This is motivated by the explicit boundary reporting in Qwen3VL-Seg/DR2Seg,
not a new reward. It distinguishes a real thin-object gain from a mean-cIoU
artifact before any official transfer claim.

## Execution order

1. PV-FEPO is now closed by its preregistered training support gate
   (`mean joint-positive=0.10625 < 0.20`); no holdout claim is made and no
   transform/threshold is changed.
2. Submit the already-registered R35 safe visual-interface fallback when the
   dnacoding control plane is reachable. Its own 5,120-row/10-step training
   and complete 512-row/20k evaluation remain mandatory.
3. After R35 is closed without promotion, submit at most one BA-FEPO 2-GPU
   screen with all positive tags. SCB-FEPO is submitted only under its
   sentinel-drift trigger; otherwise record it as falsified without a job.
4. Only a promoted survivor receives 100-step confirmation and complete
   GRefCOCO/RefCOCO transfer. No official claim is made from a 512-row screen
   alone.

## Post-R35 isolated variants

These are deliberately kept as a small queue rather than stacked into a new
router. They are only eligible after R35 and BA have each been closed by the
same full holdout protocol.

### D. Action-budget FEPO (AB-FEPO)

OpenWorldSAM and Qwen3VL-Seg both expose a cost of emitting an incorrect mask
versus abstaining. AB-FEPO tests a fixed per-example action budget: a sampled
trajectory receives the ordinary R18 geometry credit, but an auxiliary
detached penalty is applied when it changes more mask-code positions than the
registered native trajectory budget. The budget is fixed before evaluation,
does not select an inference route, and does not use ground-truth features.
The falsification criterion is no utility gain at matched null recall; if it
only reduces token changes without improving cIoU, it is closed as a decoding
regularizer rather than a segmentation method.

### E. Boundary-stratified sampling FEPO (BS-FEPO)

The ten-paper synthesis (especially DR2Seg, Qwen3VL-Seg, and OpenWorldSAM)
suggests a data-mixture bottleneck: thin and high-boundary targets can be
drowned out by ordinary masks even when the reward is already boundary-aware.
BS-FEPO keeps R18's credit completely unchanged and changes only the
registered training mixture to two ordinary, one thin, and one boundary-hard
positive per four-example batch. Thin and boundary-hard pools are made
disjoint by a deterministic tie-break, so the 50/25/25 ratio is auditable.
The holdout remains image-disjoint and unstratified. An overall gain without
thin/boundary-hard slice gain closes BS as a sampling regularizer rather than
a segmentation contribution; no slice weights are tuned after holdout access.
The implementation is registered as
`fepo_tb_gppo_plain_rank_unified_boundary_stratified_native_rank_local_10step_2gpu`
with a SAMTok-only wrapper and offline schedule tests. It is eligible only
after BA-FEPO has a complete 512-row/20k decision.

### F. Null-calibrated visual interface (NCVI-FEPO)

If R35 improves positive geometry but worsens no-target recall, combine its
visual-merger LoRA scope with the *already registered* fixed sentinel hinge,
without changing R18 credit. This conditional arm is allowed only when the
R35 report shows a reproducible null regression; otherwise it is recorded as
falsified without allocating a job. The arm inherits the 5,120-row/10-step,
512-row/20k bootstrap and capability-retention gates.

All four variants remain isolated hypotheses; the first three have now been
closed by their registered gates and AB has completed its negative holdout.
At most one later candidate may be trained at a time, and each requires a
complete contract-valid screen before any longer confirmation or paper claim.

## Offline implementation update (2026-08-29)

AB-FEPO is now wired into `tools/run_fepo_candidate_probe.py`. The probe
loads its registered config and submitter, checks the fixed action budget
(`B=2`, excess penalty `0.10`), and exercises the detached first-divergence
credit on the same deterministic K=4 geometry group used by the other
candidate screens. It passed with 5,120 manifest rows, 10 optimizer steps,
four rollouts, finite non-negative credit, and no PixVL trainer/teacher path.
This is an implementation/contract result only; it is not a training or
holdout result and does not change the R35 -> BA -> BS -> AB submission order.

## BS-FEPO closure and AB readiness (2026-08-29)

BS-FEPO completed its required 512-row holdout and 20,000 paired bootstrap
comparison. It is closed: versus matched continued-SFT, utility changed by
`-0.018033` (95% CI `[-0.029746, -0.007941]`), positive cIoU by `-0.008722`
(`[-0.018182, -0.000980]`), and no-target recall by `-0.027344`
(`[-0.050781, -0.007812]`). The corrected promotion gate is false and the
boundary-hard/thin slices are non-inferior only on the small unchanged slice.
No BS checkpoint is promoted.

The next isolated screen is AB-FEPO. Its static contract and deterministic
credit probe remain valid: 5,120 rows, 10 steps, K=4, fixed action budget
`B=2`, excess penalty `0.10`, SAMTok-only model path, and all positive
dnacoding tags. The submit wrapper is duplicate-safe through the `ab_submit`
marker and must wait for control-plane recovery before creating one job.

## AB-FEPO closure (2026-08-29)

AB completed valid training and the full 512-row/20,000-bootstrap evaluation,
but the fixed action budget did not improve the policy. Against matched
continued-SFT, utility delta was `-0.016677` (CI `[-0.027620,-0.007197]`),
positive cIoU delta `-0.009917` (CI `[-0.019694,-0.001715]`), and no-target
recall delta `-0.023438` (CI `[-0.042969,-0.007812]`). Positive boundary-IoU
delta was `-0.001226` (CI `[-0.004803,0.002259]`), and the thin and
boundary-hard slices regressed. Relative to the R18-100 confirmation, utility
was `-0.009951` and no-target recall `-0.015625`; the corrected promotion gate
is false. This closes action counting as a useful training regularizer; no
penalty or budget tuning is permitted.

## Candidate G: Predicted-Evidence Scope FEPO (PES-FEPO)

The failed BS and AB screens share a diagnosis: they alter the data mixture or
the scalar magnitude after the same native reward, while the R15 shuffle shows
that a fixed depth-decay claim is not causal. PES-FEPO tests a different
question: can a *predicted-only* failure signal decide which SAMTok code depth
is allowed to receive the already-verified R18 credit? At each rollout, use
detached native-vs-sampled logit margin and per-depth entropy to form a
three-bin evidence state (confident, ambiguous, unsupported). The state gates
the local token scope, but never changes the sign or magnitude of the joint
cIoU/boundary-IoU native-relative rank. No ground-truth geometry metadata,
teacher distribution, view perturbation, inference router, or second policy is
introduced.

The minimal screen fixes a deterministic mapping registered before training:
confident updates the first changed depth only, ambiguous updates the first
two changed depths, and unsupported abstains from positive geometry credit.
The same K=4 groups, sentinel, 5,120 rows, 10 steps, and 512/20k holdout are
used. A shuffled-evidence mapping is the paired negative control. Promotion
requires non-inferiority against R18 and matched-SFT plus evidence-state
coverage (at least 20% in each non-empty state); otherwise the branch closes
without tuning thresholds. This is the first post-AB candidate that changes
credit *scope* using deployable student evidence rather than another reward
transform.

PES-FEPO is a design hypothesis only. It must first pass a static/offline
support probe and can be submitted only after the AB decision is recorded and
its artifacts remain under the 700G budget.
