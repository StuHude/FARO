# FEPO routed variants: paper-grounded experiment register (2026-08-20)

### R21 candidate: native-anchored rank-local geometry credit

R21 inserts native greedy into both cIoU and boundary-IoU rank calculations,
keeps only jointly positive candidates, and localizes their continuous rank
gain to the first changed SAMTok code depth. It tests whether R18's absolute
threshold discards useful sibling ordering without changing the decoder,
null sentinel, or inference behavior. The fixed screen and closure gate are
registered in `R21_NATIVE_RANK_LOCAL_PREREG_20260828.md`.

### R22 candidate: scale-stratified native rank-local geometry credit

R22 addresses the remaining cross-domain scale hypothesis exposed by the
official RefCOCO result.  Training-only target area strata (small/medium/large
from q25/q75 mask area) select fixed boundary-heavy, balanced, or cIoU-heavy
axis weights, while keeping native-reference tie-aware rank and first-change
SAMTok depth localization.  The strata are never model inputs and never
computed from holdout rows.  R22 is an independent screen, not a weight sweep;
its complete gate is registered in
`R22_SCALE_STRATIFIED_NATIVE_RANK_LOCAL_PREREG_20260828.md`.

### R23 candidate: bidirectional coarse/fine native geometry credit

R23 tests whether a jointly improved sibling should receive credit at both
ends of its changed SAMTok code interval: the first changed depth represents
coarse extent, while the last changed depth represents fine boundary
refinement. It uses the fixed native-reference midrank gain and a 0.5/0.5
forward/reverse depth-decay mixture. R23 is a single-policy training credit
screen; it adds no inference router, expert, PixVL training, or OPD target.
The preregistered screen and closure gate are in
`R23_BIDIRECTIONAL_COARSE_FINE_PREREG_20260828.md`.

### R24 candidate: anchor-constrained native geometry FEPO

R24 tests the cross-dataset regression hypothesis directly: retain R18's
native-relative geometry credit but penalize categorical KL drift from the
frozen continued-SFT mask-code policy on a disjoint training-only target
buffer. This is a cumulative anchor trust region, not an inference router or
extra policy. The fixed epsilon/lambda screen is registered separately before
submission.

The fixed screen is registered in `R24_ANCHOR_KL_PREREG_20260828.md`; it is
not promoted until the KL mechanism is active and the complete holdout plus
RefCOCO transfer gates pass.

### R25 candidate: uncertainty-calibrated native rank-local geometry

R25 retains R18's native-relative first-divergence geometry credit and
multiplies only eligible positive credit by a detached confidence derived from
calibrated per-prefix entropy and missing top-support mass. The fixed floor is
`0.25`; geometry still determines the sign. This tests whether uncertain
sibling improvements should be down-weighted without adding a teacher, OPD
target, extra policy, inference router, or PixVL training. The screen is
registered in `R25_UNCERTAINTY_NATIVE_RANK_LOCAL_PREREG_20260828.md`.

This register keeps one SAMTok policy and treats routing as a training-time
conditioning signal. There are no inference experts, PixVL weights, OPD
cycles, or self-generated labels. Every screen uses 5,120 training rows, at
least 10 optimizer steps, and the complete 512-row paired holdout.

## Why revisit routing

The original FARO router failed because each failure type received a separate
objective and the router was trained from noisy predicted failure labels. The
new versions share parameters and rewards. A route is a detached scalar gate
on the same policy gradient, with shuffled-gate controls. This preserves the
original idea's question, "which failure signal should receive credit?", while
removing the inference-time mixture that made comparisons hard to interpret.

## Candidate R1: evidence-gated geometry credit

V-Zero motivates a sibling-relative evidence gap between a target-preserving
crop and a matched distractor crop. Fine-R1 and DR2Seg motivate ranking
multiple sampled masks rather than assigning an absolute label. For each
positive row, sample four grammar-valid mask trajectories, compute cIoU and a
detached evidence gap, and multiply the group-relative advantage by
`clip(1 + z(gap), 0.25, 1.75)`. No-target rows retain the frozen null-risk
constraint. The `none` and `shuffled` arms are mandatory controls.

Promotion requires a positive-cIoU improvement over the frozen continued-SFT
adapter, no-target recall non-inferiority, zero invalid outputs, and a gate
effect that beats its shuffled control. A random gate or an evidence gate
with no geometry effect closes R1.

## Candidate R2: failure-axis soft routing

OpenWorldSAM's tie-breaker and Qwen3VL-Seg's positive/negative/OOD separation
suggest three *diagnostic axes*: null-vs-mask decision, mask geometry, and
boundary/instance ambiguity. Compute these from the sampled answer and the
ground-truth training target, but do not create separate heads. Normalize the
three deficits within each sibling group and use a simplex gate to allocate a
single shared advantage:

`A = (w_null A_null + w_geom A_geom + w_boundary A_boundary)`.

The weights are fixed from a disjoint training calibration buffer and are not
fit on the 512-row holdout. A shuffled-axis control tests whether gains come
from routing semantics or merely a changed reward scale. The key prediction is
that geometry and boundary credit can improve positive cIoU without changing
the null decision, unlike the earlier scalar RL arms.

## Candidate R3: moderate-view consistency gate

S2VOPD shows that moderate information reduction supplies a useful visual
asymmetry, while aggressive crops remove task evidence. Use a clean-view
rollout and a downscaled/light-noise diagnostic view only to estimate a
bounded consistency gate; optimize ordinary ground-truth cIoU RL on the clean
view. This is not OPD: the augmented view supplies no target distribution and
no teacher loss. A shuffled-view control and an aggressive-crop negative
control are required. R3 is promoted only if the moderate view improves
geometry and the aggressive view does not.

## Candidate R4: interface plasticity with routed credit

OpenWorldSAM and EVP indicate that frozen visual backbones can still benefit
from a small cross-modal interface. Start from the original SAMTok anchor and
train only the visual merger/deepstack merger LoRA linears. Use the best
surviving R1/R2 gate, with a matched projector-SFT control. This tests whether
the RL bottleneck is visual grounding rather than policy calibration. It is a
single adapter, not a route-specific expert.

## Order and stopping rules

1. Finish the in-flight R4 10-step holdout and compare against anchor and
   projector-SFT with 20,000 paired bootstrap repeats.
2. If R4 fails positive geometry, close interface plasticity and run only the
   R1 three-arm screen (none, shuffled, evidence) at 10 steps.
3. If R1 fails, run R2 once with fixed weights and its shuffled control. Do not
   sweep weights on the holdout.
4. R3 is reserved for a surviving geometry signal; otherwise it is a
   diagnostic negative result, not a paper method.

The paper claim is valid only for a candidate that passes the pre-registered
utility, positive-cIoU, null-recall, canonical-rate, and invalid-output gates.
All failed arms remain documented as evidence against the corresponding
mechanism.

### R14 result: depth-local geometry credit (complete 512-row holdout)

R14 assigns a jointly positive cIoU/boundary gain to the earliest SAMTok code
depth that diverges from native greedy, with a fixed prefix-rarity bonus. The
decoder, rollout support, sentinel, and null objective remain unchanged. The
first eval attempt used the default 128-row PixVL schema and is retained only
as a sample-size audit; it is not a headline result. The formal eval uses the
FARO `grefcoco_selective_holdout_256.jsonl` schema for all 512 paired rows.

Relative to the frozen continued-SFT anchor, R14 obtains:

| metric | delta | paired 20,000-bootstrap 95% CI |
|---|---:|---:|
| positive cIoU | +0.021240 | [+0.006580, +0.036974] |
| no-target explicit recall | +0.152344 | [+0.109375, +0.199219] |
| selective utility | +0.086792 | [+0.063185, +0.111513] |

Positive mask rate is `1.0` and invalid output rate is `0.0`. Against the
matched plain-rank unified 10-step control, positive cIoU is still +0.021168
([+0.006573, +0.036535]), so the geometry result is not explained by the
registered rank reward alone. R14 passes the promotion gate and is the first
variant in this register with a reliable positive-cIoU interval. The next
control is a depth-localization shuffle that preserves the same gain weights
but breaks the earliest-divergence assignment.

### R15 candidate: shuffled-depth localization control

R15 is the matched control for R14. It uses the same single-policy SAMTok
rollouts, native-greedy reference, joint cIoU/boundary-IoU improvement, and
prefix-rarity distribution. The only changed operation is a deterministic
cyclic permutation of the depth used by the `0.85**depth` locality factor
(seed `20260827`); the permutation is applied after the earliest divergence is
identified. Thus R15 breaks the earliest-depth interpretation without changing
which sampled masks are jointly better or how rare their prefixes are.

The registered stage is
`fepo_tb_gppo_plain_rank_unified_depth_local_shuffle_10step_2gpu`; it uses
exactly 5,120 rows, 10 optimizer steps, two GPUs, the frozen continued-SFT
SAMTok anchor, and the shared 32-row unified sentinel. A complete 512-row
paired holdout with 20,000 bootstrap resamples is required after all training
gates pass. R15 is a control for the R14 mechanism: it is not promoted on a
utility/null-recall gain alone, and it is closed without a weight sweep if the
training gate fails or its geometry interval does not distinguish the
earliest-divergence assignment.

### R15 result (2026-08-28)

R15 completed 10/10 outer steps and passed every training gate. On the same
complete 512-row holdout, relative to the frozen anchor it reached positive
cIoU delta `+0.022385` (95% CI `[+0.007089,+0.038360]`), no-target explicit
recall delta `+0.156250` (CI `[+0.113281,+0.203125]`), and selective utility
delta `+0.089318` (CI `[+0.065721,+0.114312]`), with invalid rate `0.0` and
positive mask rate `1.0`. Directly paired against R14, the cIoU delta was only
`+0.001145` (CI `[-0.000670,+0.004098]`) and utility delta `+0.002526` (CI
`[-0.000222,+0.007244]`). The earliest-depth assignment is therefore not the
causal explanation. R14/R15 are consolidated as a joint geometry-positive,
rarity-conditioned local-credit family; R15 remains a localization ablation,
not a separate method claim.

### R16 candidate: rarity-free local geometry credit

R16 removes only the prefix-rarity multiplier while retaining the same joint
cIoU/boundary-IoU improvement set and local depth weighting. It tests whether
the R14/R15 effect comes from selecting jointly better geometry samples or
from the rarity term. The screen remains fixed at 5,120 rows, 10 steps, two
GPUs, and the unified sentinel; no holdout-driven weight sweep is allowed.

## Existing evidence carried into the register

The complete 512-row screens already close several tempting variants:

| arm | positive cIoU | no-target recall | utility | decision |
|---|---:|---:|---:|---|
| active-set selective risk | 0.7684 | 0.8125 | 0.7905 | close: geometry below anchor |
| boundary-credit | 0.7704 | 0.8164 | 0.7934 | close as a standalone method |
| sign-balanced geometry | 0.7699 | 0.8164 | 0.7931 | close as a standalone method |
| greedy-relative ES | 0.7692 | 0.8164 | 0.7928 | close: no geometry gain |

These results motivate a route-conditioned *credit allocation* experiment,
not another null-risk or reward-coefficient sweep.

### R1 screen result (10 steps, 5,120 rows)

The complete holdout confirms the evidence proxy is not causal:

| arm | positive cIoU | no-target recall | utility | utility 95% CI vs anchor |
|---|---:|---:|---:|---:|
| evidence `none` | 0.7700 | 0.8125 | 0.7913 | [+0.0004, +0.0164] |
| evidence `shuffled` | 0.7720 | 0.8086 | 0.7903 | [+0.0007, +0.0144] |
| evidence `view_drop` | 0.7685 | 0.8125 | 0.7905 | not promoted |

The shuffled control is at least as good on positive geometry and nearly as
good on utility. R1 is therefore closed as a mechanism, while the shared
effective-support RL objective remains a useful calibration baseline.

The matched 20-step projector-plastic RL and projector-SFT controls are
behaviorally identical to the frozen anchor on the 512-row holdout
(`positive cIoU=0.7706`, `no-target recall=0.7969`, `utility=0.7837`). The
separate 10-step projector evaluation was attempted twice but remained in
worker initialization with no output; both jobs were stopped to avoid wasting
GPU time. R4 is closed as a positive result because the completed 20-step
control already shows no greedy behavior change.

### R2 screen result (10 steps, 5,120 rows)

The fixed boundary/evidence route is valid but does not separate from its
matched boundary-only control:

| arm | positive cIoU | no-target recall | utility |
|---|---:|---:|---:|
| boundary + evidence | 0.7711 | 0.8125 | 0.7918 |
| boundary + none | 0.7704 | 0.8164 | 0.7934 |

Against the frozen anchor, the evidence arm has utility delta `+0.0081`
(`95% CI [+0.0010,+0.0168]`) but positive cIoU delta only `+0.0005`
(`95% CI [-0.0042,+0.0059]`). Against the matched none control, utility is
lower and null recall is lower. R2 is closed without a longer run.

The prior tail-GPPO route is also closed at the validity screen: its one-step
hard-tail and plain-rank controls violated the frozen null-risk tail gate
(`q10 margin delta` about `-0.24`, violation rates `0.34`--`0.53`). This is
evidence that a hard difficulty route without explicit feasibility restoration
is unsafe, not a reason to relax the gate.

### R3 screen result

The attempted tail-balanced plus active-set restoration screen reached eight
outer updates, then failed with a 600-second NCCL collective timeout while
the two independent sentinel paths were active. It produced no valid final
metrics and is closed as an implementation-boundary failure; the run is not
counted as a quality result. The lesson is to share one sentinel/feasibility
buffer if this family is revisited, rather than stacking two distributed
collective protocols.

### R4 implementation: unified single-sentinel feasibility restoration

R4 is registered as `fepo_tb_gppo_tail_balanced_unified_sentinel_10step_2gpu`.
It uses the 32 no-target rows already selected by the tail schedule as the
only sentinel buffer. Initialization and each optimizer epoch pack one local
null CE scalar plus the fixed local margin vector into a single gather. The
same detached global margins determine the lower-q10 tail set and risk flags;
the local differentiable margins provide both null repair and tail penalty
gradients. The previous active-set and tail sentinel forwards, selected-count
gather, and second sentinel protocol are disabled for this stage. The config
points at `egfepo_train_5120.jsonl` and runs exactly 10 outer steps. No job
has been submitted yet; the static contract screen passes.

The completed R4 screen used the same 512-row holdout for both arms. The
tail-balanced arm reached positive cIoU `0.769996`, no-target recall
`0.816406`, and utility `0.793201`; its paired positive-cIoU delta versus the
frozen anchor was `-0.000616` (95% CI `[-0.00214,+0.00071]`), while utility
delta was `+0.00473` (CI `[+0.00084,+0.00956]`). The matched plain-rank arm
reached positive cIoU `0.770684`, no-target recall `0.820313`, and utility
`0.795499`; its cIoU delta was `+0.000072` (CI `[-0.00418,+0.00520]`). Both
training gates and the final shared-sentinel risk gate passed, but neither
arm establishes a reliable geometry gain, so R4 is closed as a paper
mechanism.

R5 is now submitted as positive-only greedy-improvement credit: a sampled
mask receives policy credit only when its clean-view cIoU exceeds the current
greedy mask by a fixed `1e-4`; the shared sentinel still supplies null CE,
first-action margin, and tail feasibility repair. This isolates the
Fine-R1/DR2Seg-style positive ranking signal from the rejected difficulty
weighting.

### R5 complete 512-row result

The full holdout completed with 512 paired rows and zero invalid outputs. Against
the frozen continued-SFT anchor, R5 reached utility `0.792240` versus
`0.783744` (paired delta `+0.008497`, 20,000-bootstrap 95% CI
`[+0.001714,+0.017249]`), positive cIoU `0.771981` versus `0.770612`
(delta `+0.001369`, CI `[-0.001399,+0.005803]`), and no-target recall
`0.812500` versus `0.796875` (delta `+0.015625`, CI
`[+0.003906,+0.031250]`). Positive mask rate and invalid rate remained `1.0`
and `0.0`. R5 therefore passes the feasibility/promotion screen, but its
positive-cIoU interval still overlaps zero; it is retained as the calibration
and selective-risk control, not yet as the geometry paper claim.

### R6 preregistration: hierarchical geometry-prefix credit

R6 keeps the R5 raw-cIoU positive-only objective and shared 32-row feasibility
sentinel, but assigns each positive greedy improvement to the SAMTok code
prefixes that carried it. Each prefix receives a detached multiplier from its
within-group rarity and whether it diverges from the native greedy prefix;
earlier prefixes receive a geometrically decayed weight. The decoded full-mask
cIoU remains the sole reward, so this is not a PixVL/self-supervised loop or a
second router. The registered stage is
`fepo_tb_gppo_plain_rank_unified_prefix_credit_10step_2gpu` (5,120 rows, 10
outer steps, 2 GPUs), with `depth_decay=0.85`, `novelty_weight=0.5`, and a
fixed `1e-4` improvement threshold. It is promoted only if the complete
512-row holdout improves positive cIoU with a paired bootstrap interval above
zero while utility and no-target recall do not regress; otherwise it is closed
as an informative-credit ablation.

### R7 preregistration: Pareto geometry improvement credit

R7 tests a stricter geometry hypothesis inspired by boundary-aware segmentation
objectives: a sampled mask receives policy credit only when both its raw cIoU
and boundary IoU exceed the native greedy mask by `1e-4`. The positive gains
are combined by a geometric mean, so one metric cannot compensate for a
regression in the other. The stage keeps the SAMTok-only raw-cIoU rollout,
unified 32-row sentinel, and effective-support controller; no counterfactual
view or PixVL training is used. It runs 5,120 rows for 10 outer steps on 2
GPUs. Promotion requires a positive paired 512-row cIoU interval with no
utility or null-recall regression; otherwise the strict Pareto hypothesis is
closed.

Training job `dna-fepo-pareto-geometry-10step-2g-r1-49057591` completed and
passed all validity, effective-support, active-set, and tail-risk gates. The
full holdout job is `dna-r7-pareto-eval-8g-1787846444-45216076`.

The 512-row holdout completed with zero invalid outputs. Relative to the
continued-SFT anchor, positive cIoU was `0.771707` (delta `+0.001095`,
20,000-bootstrap CI `[-0.001790,+0.005692]`), no-target explicit recall was
`0.808594` (delta `+0.011719`, CI `[0,+0.027344]`), and selective utility was
`0.790151` (delta `+0.006407`, CI `[+0.000485,+0.014219]`). Positive mask rate
remained `1.0` and invalid rate `0.0`. R7 therefore improves utility and
selective risk, with a stronger but still statistically inconclusive geometry
point estimate; it remains a promising variant rather than a confirmed
geometry claim.

Training job `dna-fepo-prefix-credit-10step-2g-r1-19600756` completed 10/10
steps and passed all validity, effective-support, active-set, and tail-risk
gates. The full holdout job `dna-r6-prefix-eval-8g-1787241347-48600937`
completed all 512 rows. Relative to the frozen anchor, R6 reached positive
cIoU `0.771263` (delta `+0.000650`, 20,000-bootstrap CI
`[-0.002951,+0.005383]`), no-target explicit recall `0.808594` (delta
`+0.011719`, CI `[0,+0.027344]`), and selective utility `0.789928` (delta
`+0.006185`, CI `[+0.000178,+0.013951]`). Invalid output stayed `0.0` and
positive mask rate stayed `1.0`. Thus R6 passes the utility/risk promotion
screen but does not establish a geometry gain; it is retained as a promising
hierarchical calibration variant and closed as the primary geometry claim.

### R8 representation-evaluation correction

The original PP-FEPO 20-step evaluation reported bitwise-identical metrics to
the anchor. An adapter probe showed that PEFT list composition with the
anchor's rank-128 language LoRA and the visual rank-16 LoRA silently activated
only the first adapter. This was an evaluator defect, not evidence that the
visual branch had no effect. The evaluator now explicitly merges the two
disjoint adapter target sets with PEFT `cat` composition, which is algebraically
the exact additive LoRA update and avoids the unnecessary SVD merge. A fresh
complete 512-row PP-FEPO evaluation is registered as R8; its result will be
compared with the same frozen anchor before deciding whether the visual
representation hypothesis advances to a longer run.

### R8 result (2026-08-28)

The corrected evaluator completed all 512 paired holdout rows after explicitly
composing the disjoint language and visual LoRA targets. Against the frozen
continued-SFT anchor, positive cIoU was `0.770873` (delta `+0.000261`,
20,000-bootstrap CI `[-0.001436,+0.002219]`), selective utility was
`0.783874` (delta `+0.000131`, CI `[-0.000718,+0.001109]`), and no-target
explicit recall was unchanged at `0.796875` (delta `0`). Invalid output stayed
`0.0` and positive mask rate stayed `1.0`. The PEFT composition defect is
therefore fixed, but the visual-projector branch remains behaviorally neutral
at this budget and is closed as a paper mechanism. No longer representation
run is authorized without a new independent hypothesis.

### R9 preregistration: verified positive replay hybrid

R9 keeps the unified sentinel/effective-support contract and SAMTok-only
raw-cIoU rollout. For each four-sample positive group, the highest-cIoU
sampled mask is selected only when it exceeds native greedy cIoU by `1e-4`; its
complete sampled mask sequence then receives a fixed `0.05` on-policy token-CE
replay term alongside clipped GPPO. No pairwise preference, counterfactual
view, or PixVL cycle is used. The run uses 5,120 rows and 10 outer steps, with
promotion based on a complete 512-row holdout and no sentinel-risk regression.

Training job `dna-fepo-verified-replay-10step-2g-r1-36725963` was submitted
with the registered positive tags and is currently starting.

The first training attempt failed before an optimizer update because replay
rescoring passed a `[4,D]` calibrated-temperature tensor to a single-sequence
score. This shape-only implementation failure was corrected by retaining the
best rollout's `[1,D]` temperature/support row and rerunning as
`dna-fepo-verified-replay-10step-2g-r2-81386018`. R2 completed 10/10 steps and
passed all validity, effective-support, active-set, and tail-risk gates; the
verified replay active fraction was `0.375`.

The complete holdout `dna-r9-replay-eval-8g-1787854129-30620963` finished with
zero invalid outputs. Relative to the continued-SFT anchor, positive cIoU was
`0.768590` (delta `-0.002022`, 20,000-bootstrap CI
`[-0.005164,+0.000063]`), no-target recall was `0.804688` (delta `+0.007813`,
CI `[0,+0.019531]`), and utility was `0.786639` (delta `+0.002895`, CI
`[-0.001641,+0.009289]`). R9 is closed: verified replay did not improve
geometry or utility despite passing training validity gates.

The prior cluster artifact for that submission was not retained after the
workspace cleanup. It is being rerun as `dna-r9-verified-replay-10step-2g-*`
with the same frozen SAMTok initialization, 5,120 rows, 10 outer steps, and
unified sentinel. Promotion requires a complete 512-row holdout, utility and
positive-cIoU non-inferiority, no-target-recall non-inferiority, zero invalid
outputs, and a nonzero verified-replay activation fraction. A replay gain that
only changes null calibration without positive geometry will be retained as a
selective-risk control, not presented as a geometry improvement.

### R10 preregistration: verified prefix replay

R9's full-sequence replay improved selective utility only through a small
null-recall shift and reduced positive cIoU. R10 tests whether the failure is
credit leakage into already-correct code decisions. It keeps the same
best-sampled-mask verification (`cIoU > native greedy + 1e-4`) and replay
weight `0.05`, but masks the replay CE to code depths whose sampled token
differs from native greedy. The complete mask remains the only geometry
verifier; no target code, counterfactual view, PixVL weight, or inference
router is introduced.

The registered stage is
`fepo_tb_gppo_plain_rank_unified_verified_prefix_replay_10step_2gpu`: frozen
continued-SFT SAMTok initialization, `egfepo_train_5120.jsonl`, exactly 10
outer steps (20 optimizer updates), effective-support exploration, and the
shared 32-row sentinel. Promotion requires all training validity/risk gates
and a complete 512-row holdout with positive-cIoU and utility non-inferiority;
the method is only a geometry claim if the paired positive-cIoU interval is
strictly above zero. A replay activation fraction below the registered
threshold or unchanged behavior closes the hypothesis.

### R10 result (2026-08-28)

The prefix-local replay run completed 10/10 steps and passed every training
gate. On the complete 512-row holdout, utility improved by `+0.008838`
(`95% CI [+0.001155,+0.018354]`) and no-target recall by `+0.019531`
(`[+0.003906,+0.039063]`), while positive cIoU changed by `-0.001855`
(`[-0.004908,+0.000040]`). Invalid output remained `0.0` and positive mask
rate `1.0`. R10 therefore confirms a useful selective-risk calibration
effect, but it does not establish a geometry improvement and is closed as a
geometry method.

### R11 preregistration: Pareto prefix replay

R11 combines the two surviving pieces without adding a new expert: a sampled
trajectory must improve both cIoU and boundary IoU over native greedy by the
registered threshold before replay activates, and its replay CE is restricted
to code depths that differ from native greedy. This directly tests whether
R10's geometry loss came from replaying cIoU-only winners and whether R7's
Pareto verifier can be made local. The stage retains the SAMTok-only shared
policy, effective-support exploration, 32-row sentinel, 5,120 training rows,
and 10 outer steps. It is promoted only after a complete 512-row paired
holdout; utility/null-risk gains without positive-cIoU improvement remain an
ablation result, not a paper claim.

### R11 result (2026-08-28)

The corrected Pareto-prefix replay run completed 10/10 outer steps and 20/20
policy epochs. It had nonconstant rewards in every rollout group, effective
support reached fraction `1.0`, replay activation fractions between `0.25` and
`0.375`, zero invalid trajectories, and final tail margin violation rate `0.0`.
The replay selector was corrected before this run to filter jointly feasible
trajectories (both cIoU and boundary-IoU gains) before ranking by geometric
gain.

On the complete 512-row holdout, positive cIoU was `0.770859` (delta
`+0.000247`, 20,000-bootstrap CI `[-0.004467,+0.006614]`), no-target explicit
recall was `0.812500` (delta `+0.015625`, CI `[+0.003906,+0.031250]`), and
selective utility was `0.791680` (delta `+0.007936`, CI
`[+0.000714,+0.016859]`). Invalid output stayed `0.0`, positive mask rate
`1.0`, and canonical response rate was `0.900391`. R11 is retained as a
selective-risk calibration ablation; the positive-cIoU interval crosses zero,
so it does not support a geometry claim.

### R12 candidate: tie-aware rank-Pareto geometry credit

R7's absolute Pareto gate requires both cIoU and boundary-IoU gains over the
native greedy mask. That verifier can make the active set sparse when four
sampled masks are all near the greedy baseline. R12 keeps the same single
SAMTok policy, tail schedule, and unified 32-row sentinel, but changes only
the detached advantage: within each K=4 positive rollout group, each geometry
axis is converted to a tie-aware empirical midrank, and the two ranks are
combined by geometric mean and group-standardized. A trajectory consistently
better on both axes therefore receives positive credit even when its absolute
gain is below `1e-4`; no target code, PixVL weight, or extra expert is used.

The registered stage is
`fepo_tb_gppo_plain_rank_unified_rank_pareto_geometry_10step_2gpu`. It uses
`egfepo_train_5120.jsonl`, exactly 10 outer steps (20 policy epochs), the
effective-support K=4 grammar sampler, and shared sentinel risk repair. The
first screen is one two-GPU training run followed by the complete 512-row
holdout. It is promoted only when all validity/risk gates pass, positive-mask
rate remains 1.0, no invalid outputs occur, and positive cIoU is non-inferior
to the frozen anchor. A geometry claim additionally requires a paired 95%
cIoU interval with lower bound above zero; utility or no-target recall gains
with a cIoU interval crossing zero remain a calibration ablation. No weight
sweep or holdout-driven threshold change is allowed.

### R13 candidate: native-anchored rank-Pareto credit

R11/R12's geometry intervals cross zero, suggesting their group-relative
ordering lacks a strong reference. R13 inserts each prompt's native greedy
mask as an explicit reference point in both cIoU and boundary-IoU ranks. A
sampled mask receives positive detached PPO credit only when it improves both
axes over native; mixed or regressive masks are forced non-positive. The
registered screen uses 5,120 rows, 10 outer steps, effective-support K=4, and
the unified 32-row sentinel, with a complete 512-row holdout and no threshold
sweeps. Any utility/null-recall gain whose cIoU interval crosses zero remains
an ablation rather than a geometry claim.

### R12 result (2026-08-28)

The first attempt failed before the first optimizer step because the new
configuration omitted the contract-required `advantage_epsilon`; this was
corrected and retained as an implementation audit. The rerun completed 10/10
outer steps and 20/20 policy epochs and passed grammar, effective-support,
nonconstant-reward, epoch-two-ratio, sentinel, and tail-risk gates.

On the complete 512-row holdout, positive cIoU was `0.771476` (delta
`+0.000864`, 20,000-bootstrap CI `[-0.002133,+0.005419]`), no-target explicit
recall was `0.816406` (delta `+0.019531`, CI `[+0.003906,+0.039063]`), and
selective utility was `0.793941` (delta `+0.010198`, CI
`[+0.002312,+0.019661]`). Invalid output stayed `0.0` and positive mask rate
`1.0`. R12 is the strongest utility/null-calibration arm so far, but its
positive-cIoU interval crosses zero; it is retained as a calibration ablation,
not a geometry claim.

### R13 result (2026-08-28)

The native-anchored rank-Pareto run completed 10/10 outer steps and 20/20
policy epochs with all validity and sentinel-risk gates passing. On the
complete 512-row holdout, positive cIoU changed by `-0.000007` (20,000-bootstrap
CI `[-0.004188,+0.005189]`), no-target recall improved by `+0.011719` (CI
`[0,+0.027344]`), and utility changed by `+0.005856` (CI
`[-0.000259,+0.013604]`). Invalid output remained `0.0` and positive mask rate
`1.0`. The native reference did not produce a reliable geometry or utility
gain, so R13 is closed.

### R14 candidate: earliest-divergence depth-local geometry credit

R11--R13 all retain sequence-level advantages, so a positive decoded mask can
still update every SAMTok code decision equally. R14 keeps the same single
SAMTok policy and K=4 effective-support rollout, but localizes a jointly better
mask's detached credit to the earliest code depth that differs from native
greedy. The joint cIoU/boundary-IoU gain is weighted by a fixed depth decay
(`0.85`) and a prefix-rarity bonus (`0.5`), then normalized over active
trajectories. This borrows hierarchical/local credit intuition from recent
segmentation and verifier work without importing PixVL training or
counterfactual views.

The registered stage is
`fepo_tb_gppo_plain_rank_unified_depth_local_geometry_10step_2gpu`; it uses
exactly 5,120 rows, 10 outer steps, the frozen continued-SFT SAMTok anchor,
and the shared 32-row unified sentinel. A complete 512-row paired holdout and
20,000 bootstrap resamples are required after all training gates pass. The
screen is closed if it fails the gates or if the positive-cIoU interval crosses
zero; utility/null-recall gains alone are reported only as calibration.

### R16 candidate: rarity-free depth-local geometry credit

R14/R15 leave open whether their local-credit behavior depends on the
prefix-rarity bonus. R16 keeps the same single SAMTok policy, K=4 rollout
groups, native-greedy earliest-divergence localization, joint cIoU/boundary
gain, depth decay `0.85`, unified sentinel, 5,120 rows, and 10 outer steps,
but sets `depth_local_rarity_weight=0.0`. This is a preregistered matched
ablation of frequency shaping, not a new router or decoder. It reuses the
existing detached `depth_local_geometry_advantages` transformation through a
distinct contract mode.

The registered stage is
`fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu`, with
config
`Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_depth_local_rarity_free_10step_2gpu.py`
and wrapper
`scripts/submit_samtok_tb_gppo_depth_local_rarity_free.sh`. The complete
512-row holdout and 20,000 paired bootstrap resamples are required after all
training gates pass. A positive-cIoU interval crossing zero remains an
ablation result rather than a geometry claim; no weight sweep is allowed.

### R16 result (2026-08-28)

R16 completed 10/10 outer steps and passed all training gates. On the complete
512-row holdout, relative to the frozen anchor, positive cIoU improved by
`+0.025055` (95% CI `[+0.009137,+0.041787]`), no-target explicit recall by
`+0.156250` (CI `[+0.113281,+0.203125]`), and selective utility by `+0.090653`
(CI `[+0.066704,+0.116130]`). Invalid output remained `0.0` and positive mask
rate `1.0`. Relative to R14, cIoU changed by `+0.003815` (CI
`[-0.000294,+0.010355]`), so removing rarity is non-inferior and simpler.
The consolidated main candidate is therefore rarity-free native-relative joint
geometry credit with local depth weighting; R14/R15 are frequency/localization
ablations.

### R18 preregistration: second-seed replication of R16

R18 repeats the consolidated rarity-free depth-local candidate with global
seed 17. All training data, frozen SAMTok anchor, 5,120 rows, 10 outer steps,
two-GPU world size, rollout support, geometry credit, and unified sentinel are
held fixed. This is the required cross-seed robustness check before selecting
the method for additional benchmarks; it is not a new router or a parameter
sweep. The complete 512-row holdout and 20,000 paired bootstrap resamples are
required after all training gates pass.

### R18 training result and submission audit (2026-08-28)

The first R18 submission was an infrastructure failure: the requested 8B
SAMTok directory lacked the required `processor_config.json`, so both ranks
failed before the first optimizer update. No checkpoint or metric from that
attempt is used. The shared submit wrapper now checks this artifact before
calling `rjob submit`.

The corrected R18 run used the complete 4B-SAMTok-co checkpoint, seed 17, and
the frozen `continued_sft_to500` anchor. It completed 10/10 outer steps and
20/20 policy epochs on 5,120 rows. All validity, effective-support,
non-constant-reward, positive-gradient, representation, and unified
sentinel/tail-risk gates passed. The complete 512-row holdout is queued under
the mandated 8 -> 6 -> 4 -> 2 -> 1 GPU fallback; no holdout metric is claimed
until that job produces its merged output.

### R18 holdout result (2026-08-28)

The one-GPU fallback produced all 512 paired records. Against the frozen
`standalone_continued_sft_to500` anchor, 20,000 paired bootstrap resamples give
positive cIoU `+0.023769` (95% CI `[+0.008231,+0.040535]`), selective utility
`+0.090010` (CI `[+0.066087,+0.115072]`), and explicit no-target recall
`+0.156250` (CI `[+0.113281,+0.203125]`). Candidate positive cIoU is
`0.794381`, utility `0.873753`, no-target recall `0.953125`, invalid output
rate `0.0`, and positive-mask rate `1.0`. The candidate-only no-target
corrections are `40` versus `0` anchor-only corrections. The preregistered
promotion gate passed with the complete holdout, so R18 replicates R16's
geometry and selective-risk gains on a second seed; it is retained as the
cross-seed main result rather than treated as a weight sweep.

The direct R16-to-R18 paired comparison is also stable: positive cIoU changed
by `-0.001286` (95% CI `[-0.004020,+0.000114]`), selective utility by
`-0.000643` (CI `[-0.006499,+0.005201]`), and explicit no-target recall by
`0.000000` (CI `[-0.011719,+0.011719]`). Thus the second seed is a
non-inferior replication of the same operating point, rather than a second
anchor with a materially different behavior.

### R19 submission audit (2026-08-28)

R19's first submission was `dna-fepo-depth-local-evidence-10step-2g-58872990`
and reached the worker before exposing the missing fixed temperature field;
it produced no optimizer update or result. The corrected submission,
`dna-fepo-depth-local-evidence-10step-2g-r2-4-c4755`, passed the SAMTok
artifact, 5,120-row data, frozen-anchor, contract, and complete positive-tag
checks, then ran for two GPUs and 10 outer steps. Its only method change from
R18 is the detached view-drop evidence multiplier described in the R19
preregistration; no PixVL training artifact or OPD path is involved.

### R19 result and closure (2026-08-28)

The corrected R19 run completed 10/10 outer steps and 20/20 policy epochs with
all training validity, support, positive-gradient, unified-sentinel, and
tail-risk gates passing. On the complete 512-row holdout, the evidence-gated
combination reached positive cIoU `0.768732` (delta `-0.001880`, 20,000
bootstrap CI `[-0.004995,+0.000015]`), selective utility `0.786710` (delta
`+0.002966`, CI `[-0.001550,+0.009220]`), and no-target recall `0.804688`
(delta `+0.007813`, CI `[0,+0.019531]`). Invalid output stayed `0.0` and
positive-mask rate `1.0`, but the promotion gate failed. Directly against the
cross-seed R18 main candidate, R19 lost `0.025649` positive cIoU (CI
`[-0.042555,-0.010158]`) and `0.087043` utility (CI
`[-0.112095,-0.064048]`), with 38 anchor-only no-target corrections.

R19 is therefore closed: the V-Zero-inspired view-drop proxy is not
complementary to the successful local geometry credit in this implementation.
The result strengthens the paper's negative-control story and leaves R16/R18's
rarity-free depth-local credit as the selected method; no evidence-gate sweep
or additional training is justified.

### R18 official GRefCOCO transfer evaluation (2026-08-28)

After the second-seed replication passed, the selected R18 adapter was submitted
to the complete official GRefCOCO validation set (7,115 records) using the
approved SAMTok evaluation code. The wrapper was audited to read the shared
FARO positive-tag file and to write logs/output only under `Faro_ailab`; it
does not import PixVL weights or training code. The mandated 8 -> 6 -> 4 -> 2
GPU fallback reached a concrete H200 at 2 GPUs under
`dna-r18-grefcoco-official-2g-1787872020-21331917`. The run completed only
after all shards were merged and scored, producing the official summary below.

The complete run subsequently merged all 14,229 prediction records and
reported `N_acc=77.98`, `T_acc=99.77`, `g_iou=79.38`, and `c_iou=71.38`.
Relative to the continued-SFT SAMTok baseline (`76.23/99.85/78.29/70.85`),
the deltas are `+1.75/-0.08/+1.09/+0.53` percentage points for these four
metrics. Relative to the original SAMTok base (`70.72/99.92/74.81/68.80`),
the deltas are `+7.26/-0.15/+4.57/+2.58`. The merged artifact is
`evals/r18_grefcoco_official/summary.json`; the temporary input shards were
removed only after the 14,229-record merge was verified.

### R18 official RefCOCO transfer evaluation (2026-08-28)

The selected R18 adapter was submitted on the complete 5,000-row official
RefCOCO validation set. An early fallback failed at 3,697 rows because
stopped fallbacks shared a temporary directory; this was infrastructure-only.
The wrapper was corrected to use job-unique temporary directories, and retry
`dna-r18-refcoco-official-r3-1g-1787879451-52835564` completed all 5,000 rows
with the required positive tags. R18 scored `AP50=0.9706` and
`cIoU=0.8625707`; versus continued-SFT, deltas were `-0.0028` and
`-0.0019548` with bootstrap CIs crossing zero. Baseline references remain in
`official_refcoco_base_full` and `official_refcoco_continued_sft_full`.

### R20 pending candidate: asymmetric signed native-relative credit

R20 is preregistered as a single-point follow-up only if the complete R18
official transfers expose harmful-mask regressions. It keeps R18's positive
joint cIoU/boundary-IoU credit and first-divergence depth factor, then assigns
a fixed weak negative credit (`beta=0.25`) only to trajectories that regress
on both axes. Mixed-axis trade-offs and unchanged trajectories stay neutral.
The same 5,120-row, 10-step, K=4 SAMTok-only protocol and 512-row/20k-bootstrap
screen apply; no beta sweep is allowed. See
`R20_SIGNED_NATIVE_RELATIVE_PREREG_20260828.md`. The complete RefCOCO run
finished with `AP50=0.9706`, `cIoU=0.8625707`; versus continued-SFT the
deltas are `-0.0028` and `-0.0019548`, with paired 20,000-bootstrap CIs
crossing zero. This is the preregistered concrete headroom signal, so R20
was submitted as `dna-fepo-signed-native-depth-local-beta025-1-0e587`.

R20 uses no sweep: 5,120 rows, 10 outer steps, K=4 rollouts, fixed
`beta=0.25`, and the shared 32-row sentinel. Its signed rule is now aligned
with the preregistration: both-axis improvements are positive, both-axis
regressions receive weak negative credit, and mixed/unchanged trajectories
are neutral.

The complete R20 holdout finished with positive cIoU `+0.001340` versus R18
(95% CI `[0,+0.004020]`), utility `+0.004576` (CI `[0,+0.011106]`), and
no-target recall `+0.007813` (CI `[0,+0.019531]`). Invalid output was `0.0`
and positive-mask rate `1.0`, but the preregistered improvement thresholds
were not met. R20 is closed without a sweep; R18 remains the selected method.
