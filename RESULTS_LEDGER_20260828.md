# FEPO results ledger (2026-08-28)

This ledger contains only completed paired evaluations with 512 rows and
20,000 bootstrap repetitions. A queued training job is never treated as a
quality result. Deltas use candidate minus the named frozen reference.

## Selected method and controls

| Run | Mechanism | Positive cIoU delta (95% CI) | Utility delta (95% CI) | Gate | Interpretation |
| --- | --- | ---: | ---: | --- | --- |
| R16 | rarity-free first-divergence joint geometry | +0.025055 [+0.009137,+0.041787] | +0.090653 [+0.066704,+0.116130] | pass | selected first-seed method |
| R18 | same rule, seed 17 | +0.023769 [+0.008231,+0.040535] | +0.090010 [+0.066087,+0.115072] | pass | selected reference; cross-seed replication |
| R18-100 | same R18 rule, 100-step confirmation | +0.004530 [-0.001545,+0.012268] | +0.017890 [+0.007580,+0.029930] | strict utility/null pass; geometry CI crosses zero | long-horizon selective utility and null calibration gain; no significant cIoU claim |
| R17 | uniform joint geometry | +0.020251 [+0.004228,+0.036980] | +0.088250 [+0.064325,+0.113853] | pass | geometry signal survives without rarity |
| R16 vs R18 | direct replication comparison | -0.001286 [-0.004020,+0.000114] | -0.000643 [-0.006499,+0.005201] | non-inferior | no seed-specific advantage |
| R15 | shuffled depth control | +0.022385 [+0.007089,+0.038360] | +0.089318 [+0.065721,+0.114312] | pass | earliest-depth causality is not established |
| R15 vs R16 | rarity ablation comparison | +0.002670 [-0.000091,+0.007968] | +0.001335 [-0.004521,+0.007297] | fail | rarity removed from final method |

## Closed alternatives

| Run | Mechanism | Positive cIoU delta (95% CI) | Utility delta (95% CI) | Decision |
| --- | --- | ---: | ---: | --- |
| R19 | detached view-drop/evidence multiplier | -0.001880 [-0.004995,+0.000015] | +0.002966 [-0.001550,+0.009219] | reject |
| R18 vs R19 | direct evidence ablation | -0.025649 [-0.042555,-0.010158] | -0.087043 [-0.112094,-0.064048] | reject; negative control |
| R20 | fixed weak signed negative credit, beta=0.25 | +0.001340 [+0.000000,+0.004020] | +0.004576 [+0.000000,+0.011105] | close; below preregistered thresholds |
| R13 | native rank-pareto | -0.000007 [-0.004188,+0.005189] | +0.005856 [-0.000259,+0.013604] | reject |
| R12 | rank-pareto geometry | +0.000864 [-0.002133,+0.005419] | +0.010198 [+0.002313,+0.019661] | reject; cIoU threshold not met |
| R10 | verified prefix replay | -0.001855 [-0.004908,+0.000040] | +0.008838 [+0.001155,+0.018354] | reject |
| R34 | soft native-dominance depth-local | +0.000078 [-0.001664,+0.001819] | +0.000039 [-0.000832,+0.000922] | reject; strict CI gate failed |

## Claim boundary

The robust provisional holdout claim supported by completed evidence is a single SAMTok policy
trained with native-relative, jointly verified cIoU/boundary-IoU credit
localized at the first changed mask-code depth, together with a fixed
no-target sentinel. Prefix rarity, detached view evidence, weak signed
negative credit, and replay are not part of the promoted method. The R15
shuffled-depth result means the method should be described as *local credit
scope aligned with the mask-token interface*, not as a proven causal benefit
of the earliest depth itself.

R21--R34 are closed screens or registered follow-ups. The completed R21--R29
screens have valid 5,120-row/10-step contracts and complete 512-row diagnostic
evaluations with 20,000 paired bootstrap gates; R30 failed its worker validity
gate. Official transfer claims remain limited to the completed R18
GRefCOCO/RefCOCO reports and their documented RefCOCO non-significant
regression. PV, matched-SFT, R35, and BA remain in the dated status section
below and are not promoted from queue or training evidence alone.

## R21-R29 enhanced screens (512 rows, 20,000 paired bootstrap)

These runs use the frozen enhanced R18 reference and the corrected CI gate.
None is promoted: every positive mean below has a confidence interval crossing
zero, while the negative means fail the same gate in the other direction.

| Run | Utility delta (95% CI) | Positive cIoU delta (95% CI) | Decision |
| --- | ---: | ---: | --- |
| R21 native rank-local | -0.001610 [-0.006149,+0.001025] | +0.000687 [-0.001037,+0.002473] | reject |
| R22 scale-stratified | -0.001171 [-0.005945,+0.001923] | +0.001565 [-0.000794,+0.004852] | reject |
| R23 bidirectional coarse/fine | -0.000779 [-0.003173,+0.000836] | -0.001558 [-0.006321,+0.001672] | reject |
| R25 uncertainty-calibrated | +0.000802 [-0.000417,+0.002452] | +0.001604 [-0.000742,+0.004892] | reject |
| R26 conservative null tail | -0.000688 [-0.003095,+0.000897] | -0.001377 [-0.006140,+0.001807] | reject |
| R27 confidence-gated | -0.000807 [-0.003191,+0.000768] | -0.001614 [-0.006352,+0.001512] | reject |
| R28 margin-calibrated | +0.000656 [-0.000611,+0.002320] | +0.001313 [-0.001270,+0.004639] | reject |
| R29 primal-dual null-risk | -0.000181 [-0.002806,+0.002020] | -0.000362 [-0.005508,+0.003998] | reject |

## R35 worker-gate closure (2026-08-29)

R35 (`safe_visual_interface`) completed its registered 5,120-row/10-step
training loop with valid effective support, grammar, representation, and
positive-gradient diagnostics, but failed the pre-holdout validity gate:
`active_set_risk_gate_passed=false`, with final sentinel first-action margin
`-4.8125` and final tail margin violation rate `0.375`. It therefore has no
512-row quality evaluation, no cIoU/utility claim, and is closed as a training
validity failure. The result is not evidence that the visual-interface
hypothesis improves or harms holdout segmentation quality.

BA-FEPO is now the next isolated candidate under the preregistered order. Its
submission state machine requires this exact R35 failure status, the same
5,120-row/10-step contract, and a fresh lock-protected `dna-` job before any
evaluation is allowed. At the current refresh it is eligible but not yet
created because the submit host has intermittent control-plane DNS.

R30 grounded-interface failed its worker validity gate and is not a model
result. The matched-budget continued-SFT control has completed training; its
rjob `dna-fepo-r18-matched-sft-200-2g-17928944` is externally reported as
Succeeded, but the complete enhanced holdout has no valid output: prior
8/6/4/2-GPU submissions failed at the control-plane DNS layer and the
terminal 1G queue state cannot currently be queried. Its stale local marker
was removed for a future retry. PV-FEPO training is complete and valid; it
has no holdout or promotion result yet. The queue wording in this historical
paragraph is retained only as submission provenance.

## PV and matched-control status (2026-08-29 13:10 HKT)

PV-FEPO training is complete and contract-valid (5,120 rows, 10 steps, K=4,
effective-support and tail-risk gates passed), but its 512-row holdout has not
been generated. The matched continued-SFT control now has a complete 512-row
holdout and 20,000 paired bootstrap comparison. It improves R18 overall
positive cIoU by `+0.005641` (95% CI `[+0.000070,+0.013302]`) and utility by
`+0.006727` (CI `[+0.001306,+0.013945]`), with no-target recall delta
`+0.007812` (CI `[0,+0.019531]`) and zero invalid outputs. Its fixed
boundary-hard/thin boundary-IoU slice intervals cross the `-0.01`
non-inferiority limit, so it is a compute-matched control with a documented
geometry-tail trade-off, not the promoted method. PV evaluation submission
remains blocked before rjob creation by control-plane DNS failure. Its
training diagnostics are not used as a quality decision: joint-positive
fractions range from `0.0` to `0.21875` across optimizer records, and the
complete 512-row/20k paired holdout remains required. R18 remains provisional
until PV is closed by that registered evaluation; only then may the R35
fallback be considered.

## PV training-gate closure (2026-08-29 14:33 HKT)

The PV training artifact was rechecked against its preregistered support
criterion before any holdout inference. Across 20 optimizer records, clean
and transformed reward correlations were finite, but the joint-positive
fraction had mean `0.10625` (min `0.0`, max `0.21875`), below the fixed `0.20`
threshold. PV-FEPO is closed by this training-only gate. The machine-readable
decision is `evals/pv_training_gate.json`; it explicitly records
`holdout_used=false` and no quality or transfer claim. The paired-view adapter
and all diagnostics remain available for audit.

The registered R35 safe visual-interface fallback is now the sole next arm.
Its first submission attempt reached the positive-tag SAMTok submitter but was
blocked before rjob creation by unresolved `h.pjlab.org.cn`; no GPU task or
checkpoint exists. BA-FEPO remains conditional on R35 closure.

## R35 worker closure (2026-08-29 16:48 HKT)

R35 later ran to completion under the registered 5,120-row/10-step/K=4
contract. It failed the worker validity gate, not an evaluation gate:

| Run | Worker result | Key diagnostics | Decision |
| --- | --- | --- | --- |
| R35 safe visual interface | `failed_validity_gate` | active-set risk `false`; tail-risk `true`; final sentinel margin `-4.8125` vs budget `-4.6125`; 80 rollout groups; 20 optimizer updates | close without holdout |

The visual-merger/deepstack adapter is retained for audit, but no R35 quality,
transfer, or paper claim is made. This failure unlocks only the preregistered
BA-FEPO screen. The first BA submission attempt passed local contract,
SAMTok-anchor, and positive-tag checks but failed during control-plane rjob
creation; no BA task exists yet.

## BA-FEPO holdout closure (2026-08-29 17:30 HKT)

BA-FEPO completed the full evaluation contract: exactly 512 unique rows (256
positive and 256 no-target), zero invalid outputs, positive-mask rate 1.0,
mean cIoU 0.7903469, positive cIoU 0.7721001, and no-target recall 0.8085938.
The required 20,000 paired bootstrap comparison against matched continued-SFT
rejected the candidate: utility delta -0.0180131 (95% CI
[-0.02968,-0.00792]), positive cIoU delta -0.0086825 (CI
[-0.01817,-0.00091]), and no-target recall delta -0.0273438 (CI
[-0.05078,-0.00781]). BA is closed without promotion or transfer claims.
The next permitted arm is the isolated BS-FEPO boundary-stratified sampling
screen; it must use the same 5120-row/10-step training and 512-row/20k
evaluation contract.

## BS-FEPO holdout closure (2026-08-29 17:54 HKT)

BS-FEPO completed the registered 5,120-row/10-step training contract and
passed all worker validity gates. Its holdout contained exactly 512 rows,
with 256 positive and 256 no-target examples and zero invalid-output rate.
The candidate produced positive cIoU `0.7720604`, mean selective utility
`0.7903271`, and no-target explicit recall `0.8085938`.

The required 20,000 paired bootstrap rejected the sampling hypothesis against
matched continued-SFT: utility delta `-0.0180329` (95% CI
`[-0.0297463,-0.0079414]`), positive cIoU delta `-0.0087221` (CI
`[-0.0181823,-0.0009796]`), and no-target recall delta `-0.0273438` (CI
`[-0.0507813,-0.0078125]`). Thin and boundary-hard slices also regressed.
Against the R18-100 reference, utility delta was `-0.0113063` (CI
`[-0.0210481,-0.0032315]`); therefore BS is closed without promotion,
confirmation, or transfer claims. The fixed 50/25/25 mixture is not retained
as part of FEPO.

AB-FEPO is now the next isolated registered hypothesis. Its fixed action
budget (`B=2`, excess penalty `0.10`) and 5,120-row/10-step contract pass
local checks, but no rjob has been created while the dnacoding control-plane
DNS is unavailable. No AB quality claim exists until its complete holdout and
20,000 paired bootstrap are produced.

## AB-FEPO training closure and evaluation queue (2026-08-29 18:10 HKT)

AB-FEPO's unique SAMTok training job was created as
`dna-fepo-action-budget-native-rank-local-10s-77de4` after the R35, BA, and BS
closures. It completed the registered 5,120-row/10-step/K=4 contract and
passed every worker validity gate (effective support, nonconstant groups,
positive gradients, sentinel tail, representation, and anchor-KL checks).
The fixed action-count diagnostic was mean `1.25--1.38` with p95 `2.0` in the
final records, so the budget penalty was intentionally left unswept.

The 512-row evaluation has not yet been submitted: the first 8-GPU attempt
failed before rjob creation because `h.pjlab.org.cn` was unavailable. A
lock-protected retry session remains active with the required 8 -> 6 -> 4 ->
2 -> 1 fallback and 300-second waits. No AB holdout, bootstrap, promotion,
or transfer claim is made until that session creates a task and the complete
20,000-repetition analysis finishes.

## AB-FEPO holdout closure (2026-08-29 18:16 HKT)

AB-FEPO produced a complete 512-row holdout (256 positive and 256 no-target,
zero invalid outputs) after its valid 5,120-row/10-step/K=4 training. The
candidate's positive cIoU was `0.7708651`, selective utility `0.7916825`, and
no-target explicit recall `0.8125`.

The 20,000 paired bootstrap rejected AB against matched continued-SFT:
utility delta `-0.0166775` (95% CI `[-0.0276201,-0.0071972]`), positive cIoU
delta `-0.0099175` (CI `[-0.0196944,-0.0017152]`), and no-target recall delta
`-0.0234375` (CI `[-0.0429688,-0.0078125]`). Against R18-100, utility delta
was `-0.0099509` (CI `[-0.0189127,-0.0026120]`) and the positive-cIoU CI
crossed the registered non-inferiority boundary. Thin and boundary-hard
slices regressed while small-object slices were unchanged. AB is closed as a
non-beneficial action regularizer; no 100-step confirmation, official
transfer, or paper claim is allowed.

## PES-FEPO implementation gate (2026-08-29)

PES-FEPO is the next isolated hypothesis. The SAMTok trainer now uses
detached per-depth predicted-evidence scope masks with a scoped PPO objective;
normal and shuffled-evidence control configs both validate at 5,120 rows,
10 steps, and K=4. The offline probe and direct finite-loss unit check pass.
Training and the required 512-row/20k evaluation are pending because the
dnacoding control plane currently fails DNS resolution. A lock-protected
positive-tagged `dna-` submitter retries every 300 seconds; no quality claim
or promotion decision exists before the full holdout.

The pre-submission implementation audit subsequently corrected three issues:
the rollout margin now uses the sampled action (`max(native_logits) -
sampled_logit`), unused mass-threshold fields were removed, and evidence-state
counts accumulate locally before one final distributed reduction. These
changes do not alter the registered reward, thresholds, data, or any completed
result. The control plane remains unavailable, so no PES job or result is added
until a real worker artifact exists.
