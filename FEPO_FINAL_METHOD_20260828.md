# FEPO provisional method specification (2026-08-28; refreshed 2026-08-29)

## Method

The provisional holdout-selected method is a single-policy SAMTok RL objective with native-relative
local geometry credit. For each target-present prompt, the policy samples a
K=4 sibling group under effective-support calibration. Each decoded mask is
scored by cIoU and boundary IoU against the training mask. A trajectory is
eligible for positive geometry credit only when both metrics exceed the frozen
native-greedy mask by `1e-4`. Its detached credit is assigned to the first
SAMTok mask-code depth that differs from native greedy and scaled by a fixed
depth decay `0.85`. No prefix-rarity term remains.

The same grammar-constrained policy handles target-present and no-target rows.
A fixed 32-row no-target sentinel is gathered once per update and enforces the
registered null CE, first-action margin, and lower-tail risk budgets. This is a
training-time credit assignment mechanism, not an inference router: there are
no route-specific experts, decoder branches, PixVL checkpoints, OPD targets,
self-supervised cycles, or counterfactual labels.

## Why this is one idea

The original FARO branches independently routed semantic, relation, and
geometry failures. The unified formulation treats a failure as a policy action
whose *geometry credit scope* is too broad. Native-relative joint geometry
verifies that a sampled mask is genuinely better; earliest changed-code credit
limits the update to the decision that introduced the change; the shared null
sentinel prevents geometry gains from being purchased by hallucinated masks.
Thus routing is implicit credit localization inside one policy and one risk
contract, rather than separate objectives for each failure type.

## Evidence

R14 with prefix rarity improved positive cIoU, but R15's shuffled-depth control
showed that earliest depth was not independently causal. R16 removed rarity and
retained the simpler local geometry family: positive cIoU `+0.025055` and
utility `+0.090653` versus the frozen continued-SFT anchor on all 512 holdout
rows. R18 repeats R16 at seed 17 with cIoU `+0.023769` (95% CI
`[+0.008231,+0.040535]`) and utility `+0.090010` (CI
`[+0.066087,+0.115072]`). The direct R16/R18 comparison is non-inferior.

R19 added a V-Zero-inspired detached view-drop evidence multiplier to R18. It
passed all training validity gates but lost positive cIoU and utility; the
complete 512-row promotion gate failed. This is retained as a pre-registered
interaction negative control, not tuned into the method.

The complete official GRefCOCO transfer (14,229 merged records) reports
`N_acc=77.98`, `T_acc=99.77`, `g_iou=79.38`, and `c_iou=71.38`. Relative to
continued-SFT SAMTok, the gains are `+1.75`, `-0.08`, `+1.09`, and `+0.53`
percentage points, respectively. The complete official RefCOCO transfer
reports `AP50=0.9706` and `cIoU=86.2571`. Relative to continued-SFT
(`0.9734/0.8645`), deltas are `-0.28/-0.1955` percentage points; both paired
20,000-bootstrap intervals cross zero (`AP50 [-0.92,+0.38]`, cIoU
`[-0.7575,+0.3626]`). This is a non-significant transfer regression, so R18
is retained as the holdout-led method while its cross-dataset limitation is
explicit.

Because this exposed concrete harmful-mask headroom, the preregistered R20
single-point follow-up was run. It kept R18 positive credit and applied fixed
`beta=0.25` weak negative credit only to jointly regressive cIoU and
boundary-IoU samples, leaving mixed/unchanged trajectories neutral. R20
improved R18 by only `+0.00134` positive cIoU and `+0.00458` utility on the
complete holdout, below its preregistered thresholds, and is closed without a
sweep.

The forward-count-matched continued-SFT control is now complete. On the same
512-row holdout it improves R18 positive cIoU by `+0.005641` (95% CI
`[+0.000070,+0.013302]`) and utility by `+0.006727` (CI
`[+0.001306,+0.013945]`), with zero invalid outputs. Its boundary-hard and
thin boundary-IoU slices cross the registered non-inferiority limit, so this
is reported as a supervised compute control with a geometry-tail trade-off;
it does not establish an RL advantage or replace the provisional R18 claim.

## Literature boundary

The explicit ten-paper mechanism-to-hypothesis mapping and exclusion list is
maintained in `LITERATURE_TO_FEPO_HYPOTHESES_20260828.md`; it is part of the
method audit and prevents importing PixVL/S2VOPD training machinery.

Qwen3VL-Seg and OpenWorldSAM motivate positive/negative/OOD slices and
interface-level grounding checks. Fine-R1 motivates sibling-relative grouped
policy optimization; DR2Seg motivates ranked continuous geometry rewards;
SenseNova-Vision motivates post-adaptation capability checks. EVP and latent
denoising motivate representation robustness diagnostics. V-Zero and S2VOPD
motivate the R19 evidence/augmentation controls, whose failure prevents a
central evidence-gating claim. PixVL is used only for approved evaluator/data
interfaces. None of these papers' weights, self-supervised cycles, or decoder
architectures enter FEPO training.

## Closed scale-aware screens and pending representation test

R21 and R22 were registered follow-up screens, not promoted method claims.
R21 uses native-anchored tie-aware sibling ranks localized to the first changed
SAMTok code depth. R22 additionally conditions the fixed rank-axis mixture on
training-only target-area strata (small/medium/large). Both preserved the
SAMTok-only initialization, 5,120-row/10-step budget, unified sentinel, and
complete holdout requirement. Their complete holdout analyses failed the
corrected promotion gate; the evidence and negative decisions are in
`RESULTS_LEDGER_20260828.md`. R18 remains the selected method.

R23 was a third independent screen: it split native-relative geometry credit
between the first and last changed SAMTok code depths using a fixed 0.5/0.5
mixture. It tests coarse extent versus fine boundary refinement without
adding an inference router, expert, PixVL training, or OPD target. Its valid
training contract and complete holdout analysis failed promotion; no R23 claim
is made.

R24 was a design follow-up: an anchor KL trust region on a disjoint
training-only target buffer, intended to test whether cumulative policy drift
causes the RefCOCO transfer regression. It used the same SAMTok-only contract
and failed the complete holdout promotion gate; no R24 claim is made.

R25 is a fifth independent screen. It keeps R18's native-relative first-
divergence geometry credit and multiplies eligible credit by a detached,
calibrated uncertainty confidence derived from per-prefix entropy and missing
top-support mass. Geometry still determines the sign; uncertainty cannot
create credit, a negative label, an OPD target, or an inference route. The
confidence floor is fixed at `0.25` with no sweep. R25 completed the same
contract and its 512-row/20k comparison failed the corrected promotion gate
(positive cIoU `+0.001604`, CI `[-0.000742,+0.004892]`; utility `+0.000802`,
CI `[-0.000417,+0.002452]`). It is a closed uncertainty ablation, not a
method claim.

R26 was a sixth independent screen. It kept R18 geometry credit unchanged and
raises only the fixed training-sentinel worst-tail repair weight to `0.50` at
quantile `0.05`, testing conservative null calibration motivated by
OpenWorldSAM/V-Zero. R26 changes no router, expert, reward, teacher, OPD target,
or inference path; its complete holdout analysis failed promotion.

R27 was an independent confidence-gated screen. It kept R18's native-relative
joint geometry credit and applies a fixed detached confidence floor to suppress
low-support sibling improvements. The gate cannot create a positive sign or
change the inference policy; its complete holdout analysis failed promotion.

R28 was an additional independent screen. It kept R21's native tie-aware rank
and first-divergence scope, but multiplies verified joint geometry credit by
the fixed detached margin `sqrt(gain_cIoU * gain_boundary)`. This tests whether
continuous margin calibration suppresses fragile boundary-only wins without
combining the uncertainty, scale, evidence, or confidence variants. Its
complete holdout analysis failed promotion.

PV-FEPO then tested a fixed target-preserving photometric view. Although its
training contract was valid, the mean joint-positive support was `0.10625`,
below the pre-registered `0.20` threshold, so PV was closed without a holdout
claim. The next isolated test is R35, a visual-merger/deepstack LoRA arm with
fixed stronger null protection; its rjob submission is currently blocked by
control-plane DNS.

Historical screen jobs (all `Inqueue/STARTING` at the earlier snapshot; their
completed results are recorded in the ledger):

- R21 `dna-fepo-native-rank-local-10step-2g-73872300`
- R22 `dna-fepo-scale-stratified-native-rank-local-6039d`
- R23 `dna-fepo-bidirectional-coarse-fine-10step-2g-b5466`
- R24 `dna-fepo-anchor-kl-10step-2g-1787892498-99977985`
- R25 `dna-fepo-uncertainty-native-rank-local-10ste-3215c`
- R26 `dna-fepo-conservative-null-tail-10step-2g-30508209`
- R27 `dna-fepo-confidence-gated-native-rank-local-554e6`
- R28 `dna-fepo-margin-calibrated-native-rank-local-c4628`

## Required report

Every method/ablation uses at least 5,120 training rows and 10 outer steps.
The selective screen reports all 512 paired holdout rows with 20,000 paired
bootstrap resamples, positive cIoU, selective utility, no-target recall,
invalid output rate, positive-mask rate, canonical response rate, and boundary
IoU. Official RefCOCO/GRefCOCO scores are reported only after their complete
datasets are merged and scored. Every job uses an `dna-` name, namespace
`ailab-dnacoding`, all tags in `rjob_tags.txt`, and the 8 -> 6 -> 4 -> 2 -> 1
GPU fallback for evaluation.

## 10. Authoritative continuation refresh (2026-08-29)

The preceding R35 paragraph is historical. R35 subsequently ran and failed its
worker active-set validity gate (`final_sentinel_margin_min=-4.8125` versus
the registered `-4.6125` budget), so it received no holdout claim. The
conditional BA-FEPO and BS-FEPO screens then completed valid 5,120-row/10-step
training and full 512-row/20,000-bootstrap evaluation, and both were rejected:
BA utility delta `-0.018013` and positive-cIoU delta `-0.008683`; BS utility
delta `-0.018033` and positive-cIoU delta `-0.008722`, each with no-target
recall regression. AB-FEPO also completed the same protocol and was rejected
with utility delta `-0.016677` and positive-cIoU delta `-0.009917`. These are
closed controls, not tunable components.

PES-FEPO is the sole pending screen. It keeps the FEPO-R18 native-relative
joint geometry reward and sentinel, but gates selected token scope using
detached student evidence: mean normalized entropy plus the native-versus-
sampled logit gap. Confident trajectories update the first changed depth,
ambiguous trajectories the first two, and unsupported trajectories receive an
empty positive scope. A seed-1907 shuffled-evidence arm is the mandatory
negative control. The normal and shuffled jobs have not yet been created
because the dnacoding control plane cannot resolve `h.pjlab.org.cn`; no PES
quality claim exists. Both require the same 5,120-row/10-step/K=4 worker gate,
complete 512-row holdout, zero invalid outputs, and 20,000 paired bootstrap
before any promotion or official transfer evaluation.
