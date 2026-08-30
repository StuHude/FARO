# FEPO: Native-Relative Local Credit for SAMTok Pixel Actions

## Abstract

Discrete mask-token policies make segmentation amenable to verifiable
reinforcement learning. We introduce FEPO-R18, a single SAMTok mask-or-null
policy whose sampled trajectories receive native-relative joint geometry
credit only when both cIoU and boundary IoU improve over the native greedy
mask; credit is localized to the first changed SAMTok code depth and a fixed
no-target sentinel constrains abstention risk. Complete mask-code trajectories
are sampled with an effective-support controller so grouped policy updates
remain non-degenerate. On 5,120 training rows and a fixed 512-row
image-disjoint holdout with 20,000 paired bootstrap repetitions, R18 improves
positive cIoU by `+0.023769` (95% CI `[+0.008231,+0.040535]`) and selective
utility by `+0.090010` (`[+0.066087,+0.115072]`) over the frozen SAMTok SFT
anchor. A compute-matched continued-SFT control is stronger on aggregate
positive cIoU and utility but crosses the registered thin/boundary tail
constraint, so these results do not establish RL superiority over SFT. A
second seed, shuffled-depth and multiple recent-paper-inspired controls
establish the claim boundary; none of the evidence, uncertainty, scale,
margin, null-tail, or trust-region variants is promoted. All runs use
the original SAMTok base, no router, no PixVL checkpoint, no OPD teacher, and
no self-supervised cycle.

## 1. Problem and Scope

We consider referring-expression segmentation with an explicit absent-target
case. The policy emits either a grammar-valid two-depth SAMTok mask code or
the canonical response `No target.`. The central question is whether online
pixel reward can improve this policy without introducing routers, verifier
models, counterfactual rollouts, external visual features, or PixVL's cyclic
self-training.

## 2. Method

The frozen initialization is a 500-step standalone SAMTok SFT adapter. For
each positive training example, the policy samples `K=4` complete mask-code
trajectories. At each code depth, the top-8 native code tokens are calibrated
to effective support 4 before sampling. The reward is plain mask cIoU and the
advantage is group-standardized. A paired no-target row receives canonical
null cross-entropy and a first-action null-versus-mask margin term. PPO uses
two epochs over the same detached rollout behavior probabilities. The
effective-support controller is a training mechanism, not a holdout-tuned
hyperparameter.

The proposed EG-FEPO gate computes a sibling-group visual evidence gap by
masking the image view, then detachedly rescales the group advantage. We test:

* `view_drop`: the proposed image-evidence gap;
* `shuffled`: the same gap after a random sibling permutation;
* `none`: no evidence multiplier, the clean support-calibrated RL control.

No model router is used. The gate cannot backpropagate through evidence.

## 3. Experimental Protocol

Every minimum-cost training run uses 5,120 rows and at least 10 optimizer
steps. The three-arm screen uses 10 steps; the clean control is then extended
once to 20 steps without changing any method constant. Evaluation always uses
all 512 image-disjoint holdout rows: 256 target-present and 256 no-target.
Each reported confidence interval is a 20,000-repeat paired bootstrap over
the 256 positive/no-target pairs. Training jobs use `dna-` names,
`ailab-dnacoding`, and repository positive tags. Evaluation starts at 8 GPUs
and downgrades after five minutes only if still queued.

## 4. Results

### 4.1 Ten-step evidence ablation

| arm | positive cIoU | no-target recall | utility |
| --- | ---: | ---: | ---: |
| frozen anchor | 0.770612 | 0.796875 | 0.783743 |
| none | 0.770047 | 0.812500 | 0.791273 |
| shuffled | 0.772044 | 0.808594 | 0.790319 |
| view_drop | 0.768529 | 0.812500 | 0.790514 |

Relative to the anchor, utility deltas are `+0.00753`, `+0.00658`, and
`+0.00677` for none, shuffled, and view-drop. Positive cIoU deltas are
`-0.00057`, `+0.00143`, and `-0.00208`, respectively. The direct shuffled
minus view-drop comparison is `+0.00352` cIoU, so the proposed evidence
signal is not the source of the gain.

### 4.2 Twenty-step confirmation

The clean none arm obtains positive cIoU `0.768416`, no-target recall
`0.820313`, and utility `0.794364`. Relative to the frozen anchor:

| metric | delta (95% CI) |
| --- | --- |
| utility | `+0.010621` `[+0.002137,+0.020987]` |
| positive cIoU | `-0.002196` `[-0.006049,+0.001034]` |
| no-target recall | `+0.023438` `[+0.007813,+0.042969]` |

All training validity gates pass: 40 rollout groups are nonconstant, grammar
validity is 1.0, effective-support target fraction is 1.0, and positive
policy gradients are observed. The geometry promotion gate nevertheless
fails because the positive cIoU point delta is negative.

## 5. Analysis

The result separates two capabilities often conflated in pixel RL. The
paired null objective and online reward reliably move the policy toward safer
abstention, as shown by the recall gain. They do not move the greedy mask
policy toward better geometry. Randomizing the evidence assignment preserves
the utility gain, which rules out a causal interpretation of view-drop
evidence. Earlier SAMTok-only branches using improvement-only, boundary,
active-set, preference, and visual-projector variants show the same
geometry bottleneck; they are retained as preregistered controls rather than
stacked into a method.

## 6. Relation to Recent Work

Qwen3VL-Seg motivates explicit negative and boundary slices; OpenWorldSAM and
V-Zero ("Answer-Label-Free On-Policy Distillation with Contrastive Evidence
Gating", arXiv:2606.25319) motivate treating absence and visual uncertainty as
first-class outputs. Fine-R1 supports SFT stabilization before group-relative RL. DR²Seg
and related work motivate continuous geometry rewards, while EVP motivates
checking whether representation limits dominate boundary quality. SenseNova-
Vision motivates staged capability-retention evaluation. S2VOPD
(arXiv:2608.14144; the earlier latent-denoising reference is
arXiv:2604.21343) motivates robustness and corruption tests only after a
clean geometry gain. We explicitly do not import S2VOPD's asymmetric-view
distillation, teacher distribution, or self-supervised objective. PixVL
contributes only evaluation/data interfaces here; its cyclic self-supervision
and checkpoints are deliberately excluded.

## 7. Limitations and Next Test

The study does not claim that no possible visual-representation update can
improve SAMTok geometry. It shows that the tested single-policy, frozen
visual representation and detached advantage gates do not. A justified next
experiment must change that bottleneck directly, for example by a
pre-registered visual-merger plasticity objective with an equal-compute SFT
control. It must continue to use at least 5,000 training rows, at least 10
steps, the same 512-row holdout, and the same canonical-format and null-risk
gates. Reward, gate-scale, rank, and learning-rate sweeps on this holdout are
not justified.

## 8. Revision addendum (2026-08-28)

The subsequent tail-GPPO study replaces the earlier evidence-only framing with
a cleaner unified claim: **native-relative local geometry credit for SAMTok
pixel actions**. A sampled mask receives positive credit only when both cIoU
and boundary IoU improve over the frozen native-greedy reference. That credit
is assigned to the first changed mask-code depth, while a shared canonical
no-target sentinel constrains null risk. There is one policy, one reward
contract, and no inference-time router or extra expert. Prefix rarity was
removed after its matched ablation; the resulting R16 arm is the selected
method, with the seed-17 R18 replication serving as the current selected
checkpoint for the registered follow-up screens.

R16 improves positive cIoU by `+0.025055` and utility by `+0.090653` on the
complete 512-row holdout, with 20,000-bootstrap lower bounds `+0.009137` and
`+0.066704`. R18 repeats the same method at seed 17: cIoU `+0.023769` (CI
`[+0.008231,+0.040535]`) and utility `+0.090010` (CI
`[+0.066087,+0.115072]`). The direct R16/R18 comparison is non-inferior, so
the geometry result is not a single-seed artifact. Both runs preserve invalid
output rate `0.0` and positive-mask rate `1.0`.

R19 tests a V-Zero-inspired sibling view-drop evidence multiplier on top of
R18. Although its training validity gates pass, it reduces positive cIoU by
`0.001880` versus the anchor and utility gains are not significant; versus R18
it loses `0.025649` cIoU and `0.087043` utility. This interaction is closed
without tuning. The final paper should present evidence gating as a negative
control and focus the novelty on native-relative, local, geometry-aware credit
with explicit null-risk restoration.

On the complete official GRefCOCO transfer, R18 scores `N_acc=77.98`,
`T_acc=99.77`, `g_iou=79.38`, and `c_iou=71.38` over 14,229 merged records.
Compared with continued-SFT SAMTok, this is `+1.75` N_acc, `-0.08` T_acc,
`+1.09` gIoU, and `+0.53` cIoU percentage points. On complete RefCOCO,
R18 scores `AP50=0.9706` and `cIoU=0.8625707`; versus continued-SFT the
deltas are `-0.0028` and `-0.0019548`, with paired 20,000-bootstrap intervals
crossing zero. This non-significant transfer regression motivated the single
pre-registered R20 test: weak signed negative credit only for jointly
regressive masks (`beta=0.25`), with mixed-axis samples neutral. R20 improved
R18 by only `+0.001340` positive cIoU and `+0.004576` utility on the complete
512-row holdout (both below thresholds), so it is closed without a sweep.

### 8.1 Registered follow-up screens

To test transferability without reintroducing the original failure-type
router, eight fixed screens were registered. R21 inserts native greedy as an
explicit tie-aware sibling-rank reference. R22 conditions the two geometry
axes on training-only target-area strata: small targets receive
boundary-heavy weight, large targets receive cIoU-heavy weight, and medium
targets remain balanced. R23 distributes the same native-relative rank gain
between the first and last changed SAMTok code depths, testing coarse extent
against fine boundary refinement. R24 adds a fixed categorical KL hinge to
the frozen continued-SFT policy on a disjoint 64-row training-only buffer,
testing whether cumulative policy drift explains the RefCOCO regression. R25
keeps the R18 credit scope and applies a fixed detached confidence calibration
from rollout entropy and missing top-support mass, testing whether uncertain
sibling improvements should receive less update weight.

All eight screens use one SAMTok policy, 5,120 training rows, ten outer steps,
K=4 sibling rollouts, the unified no-target sentinel, and complete
512-row/20,000-bootstrap evaluation. They have no PixVL training, OPD target,
visual expert, inference router, counterfactual label, or self-supervised
cycle. A failed screen is retained as a mechanism-level negative result and
is not rescued by holdout tuning.

R26 adds a conservative no-target tail repair, R27 adds a fixed detached
confidence floor on already-verified geometry gains, and R28 applies a fixed
continuous joint cIoU/boundary margin calibration. These screens are
independent, non-stacked tests; none introduces an inference router, PixVL
training component, OPD target, or self-supervised cycle.

The R21--R28 jobs subsequently produced valid 5,120-row/10-step contracts and
complete 512-row diagnostics, but none passed the strict promotion gate (see
`RESULTS_LEDGER_20260828.md`). They remain closed mechanism-level controls.
The enhanced R18 512-row boundary/slice diagnostic is the frozen reference for
historical comparisons. R18 remains the provisional RL reference; the later
paired-view control was closed by its training support gate without a holdout
claim. The compute-matched
continued-SFT control is complete: it improves overall positive cIoU by
`+0.005641` (95% CI `[+0.000070,+0.013302]`) and utility by `+0.006727`
(CI `[+0.001306,+0.013945]`) versus R18, with zero invalid outputs. Its
boundary-hard/thin boundary-IoU slices cross the registered `-0.01`
non-inferiority limit, so this is a supervised compute control with a
geometry-tail trade-off, not an RL replacement.

The complete-result table and claim boundary are maintained in
`RESULTS_LEDGER_20260828.md`. It records only 512-row, 20,000-bootstrap
comparisons and explicitly separates promoted evidence from closed negative
controls.

## 9. Final Claim Boundary and Pending Controls (2026-08-29)

The paper's method claim is now **FEPO-R18**: a single SAMTok mask-or-null
policy, native-relative joint cIoU/boundary-IoU verification, first-divergence
local credit, and a fixed training-only no-target sentinel. This is an RL
credit-assignment result, not OPD: there is no teacher distribution, detached
teacher target, PixVL checkpoint, cyclic self-supervision, inference router, or
extra expert. The OPD/evidence-inspired arms are explicitly negative controls.

The enhanced screen table contains 512-row paired evaluations with 20,000
bootstrap repetitions. R21--R29 all fail the preregistered promotion gate:
utility and positive-cIoU intervals cross zero (or point deltas are negative),
despite valid training contracts. R30 is excluded at worker validity. These
results support a narrow conclusion: after strong SAMTok SFT, changing rank,
scale, view evidence, uncertainty, null-tail weight, confidence, margin, or
primal-dual calibration alone does not improve geometry over R18.

The paired-view FEPO screen completed its 5,120-row/10-step training contract,
but its mean joint-positive rollout fraction was `0.10625`, below the fixed
pre-registered support threshold `0.20`. It was therefore closed by a
training-only falsification gate; no 512-row holdout quality result is
claimed, and the photometric transform was not tuned. The decision is recorded
in `evals/pv_training_gate.json` with `holdout_used=false`. R18's GRefCOCO gain
and its non-significant RefCOCO regression remain the only transfer claims.
The queue state and retry history are recorded separately in
`codex_resume/STATUS_20260829_CONTINUATION.md` and are never treated as a
quality result.

After auditing the failed R30 representation screen, we preregistered R35 as
one further isolated test rather than silently tuning R30. R35 keeps the same
visual-merger-only trainable scope and geometry credit, but fixes
`null_ce_weight=2.0` and `margin_weight=1.0` to address the observed sentinel
margin collapse. Its submission is prepared but currently blocked before rjob
creation by the dnacoding control-plane DNS outage; no R35 training or quality
result is claimed here.

## 10. Historical Submission Plan (Superseded)

The manuscript is not marked complete until the pending representation
falsification is resolved. The registered order is:

1. Run R35 once through the single `dna-` submitter after the control plane
   returns, using the fixed visual-merger/deepstack scope and null protection.
2. Require its finished 5,120-row/10-step provenance and the full 512-row,
   20,000-bootstrap holdout before any interpretation.
3. If R35 closes without promotion, run the independent BA-FEPO boundary-
   bottleneck screen; never stack R35 and BA objectives.
4. Only a promoted survivor receives 100-step confirmation, official
   RefCOCO/GRefCOCO transfer, capability retention, and ablations.

The reserved AB-FEPO, BS-FEPO, and NCVI-FEPO variants are conditional
falsification tests, not unreported hyperparameter searches. They remain
unsubmitted until the preceding arm is closed and inherit the same data,
step, holdout, bootstrap, null-risk, canonical-output, and resource gates.
This ordering preserves the paper's single-policy contribution and prevents
the ten referenced papers from becoming a collection of post-hoc routers.

The training-only geometry audit also constrains interpretation of the tail
analysis: among 2,560 target-present pairs, `small`, `thin`, and
`boundary-hard` each contain 640 pairs, but 150 pairs satisfy all three and
490 satisfy `thin` plus `boundary-hard`. These strata are therefore correlated
descriptors, not independent task routes. The fixed 5,120-row existence mix is
balanced 50/50, and query/path shortcut checks show no obvious existence
leakage; these diagnostics are reported to prevent overclaiming a slice-only
mechanism.

Reproduction requires the original SAMTok checkpoint, the frozen continued-
SFT anchor, the versioned manifests and configs under `Sa2VA/projects/
samtok_selective`, and the exact evaluator/data interfaces copied read-only
from PixVL. No PixVL weights, trainer, OPD teacher, cyclic self-supervision,
EMA, counterfactual labels, or inference-time router are part of the claim.

## 11. Authoritative continuation update (2026-08-30)

This section supersedes the earlier queue language in Sections 9 and 10.
The R35 safe-visual-interface screen and the conditional BA, BS, and AB
screens have been closed under their registered worker or complete-holdout
gates. They are retained as falsification evidence and are not pending
experiments. The only open method-level experiment is **PES-FEPO**: predicted
evidence selects a detached one-depth or two-depth token-credit scope while
the native-relative joint geometry reward and fixed null sentinel remain
unchanged. Its shuffled-evidence mapping is a mandatory negative control.

PES normal has not yet produced a rjob, checkpoint, worker metrics, holdout,
or promotion result. The dnacoding control plane currently fails before job
creation because `h.pjlab.org.cn` cannot be resolved from this workspace;
the lock-protected submitter records five-minute retries and prevents
duplicate jobs. Therefore the paper's final method claim remains provisional
FEPO-R18, and no PES quality or causal evidence claim is made. Completion
still requires a valid 5,120-row/10-step PES worker, its full 512-row
image-disjoint evaluation, the shuffled control, and paired 20,000-bootstrap
comparisons before any survivor can receive official transfer evaluation.
