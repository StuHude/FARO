# FEPO paper-readiness audit

This is an evidence checklist, not a claim of completion. A candidate is
paper-eligible only when every required artifact below exists and is tied to
the same SAMTok anchor and holdout IDs.

## Evidence ledger

The initial rows are the 2026-08-28 queue snapshot retained for audit; the
later live refresh supersedes those historical queue labels.

| Item | Current evidence | Decision |
| --- | --- | --- |
| SAMTok-only initialization/training | Config and provenance contracts for R18--R28; no PixVL trainer or weight path in those stages | Verified for submitted screens; recheck manifests after completion |
| Minimum training budget | Registered configs pin 5,120 rows and 10 outer steps | Required gate for every output |
| R18 geometry result | Complete 512-row holdout and 20,000 paired bootstrap; positive cIoU and utility gains | Provisional holdout-selected method |
| R18 official transfer | Complete GRefCOCO and RefCOCO reports; RefCOCO point estimates are slightly lower with CIs crossing zero | Report limitation, no transfer-superiority claim |
| R18 boundary/slice baseline | Complete 512-row diagnostic with boundary-IoU, pair IDs, target geometry and fixed small/thin/boundary/area metadata; 20,000-bootstrap self-pair slice sanity passed | Frozen reference for pending comparisons |
| matched continued-SFT control | Complete 512-row holdout and 20,000 paired bootstrap; overall cIoU/utility improve over R18, while boundary-hard/thin boundary-IoU cross the registered tail limit | Keep as compute control; no RL-superiority claim |
| PV-FEPO | Valid 5,120-row/10-step SAMTok-only training; training-only support gate closed at joint-positive mean `0.10625 < 0.20`; no holdout claim | Closed by preregistered training gate |
| R21 native rank-local | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R22 scale-stratified rank-local | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R23 coarse/fine credit | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R24 anchor-KL trust region | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R25 uncertainty credit | Complete 5,120-row/10-step contract and 512-row/20k evaluator; positive cIoU `+0.001604` CI `[-0.000742,+0.004892]`, utility `+0.000802` CI `[-0.000417,+0.002452]` | Reject; no promotion |
| R26 conservative null tail | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R27 confidence-gated native rank-local | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R28 margin-calibrated native rank-local | Complete 5,120-row/10-step training and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R29 primal-dual null risk | Complete valid screen and 512-row/20k paired analysis; corrected promotion gate failed | Reject; no claim |
| R30 grounded interface | Worker sentinel-margin validity gate failed before holdout | Closed training gate; no claim |
| R35 safe visual interface | Registered, but no rjob/checkpoint because control-plane DNS fails before creation | Pending; sole next screen |
| BA-FEPO boundary bottleneck | Implemented and registered; submission is conditional on R35 closure | Pending; no claim |

## Candidate promotion contract

After a job finishes, validate `status=finished`, 5,120 training rows, at
least 10 optimizer steps, finite rollout diagnostics, grammar-valid K=4
groups, nonconstant rewards, changed epoch-2 importance ratios, positive
policy gradient, and the registered sentinel/null-risk gates. Then submit the
complete 512-row evaluation. The evaluation must include `boundary_iou` and
fixed training-registry slices; a 128-row smoke is never sufficient.

The paired analyzer must use 20,000 resamples and report overall utility,
positive cIoU, null recall, positive mask rate, invalid/canonical response
rates, and small/thin/boundary/area slices. Only a candidate that passes its
pre-registration may receive official RefCOCO/GRefCOCO transfer evaluation.

## Claim discipline

The unified contribution is native-relative, geometry-aware credit inside one
SAMTok policy with an explicit null-risk constraint. R21--R34 test individual
mechanisms and must not be stacked post hoc. Evidence/view gates are controls
unless they beat their shuffled and no-gate counterparts. A candidate that
improves abstention while losing positive geometry is a calibration result,
not a pixel-RL geometry improvement. A candidate with a holdout gain but no
transfer robustness is reported with that limitation.

## Resource audit

All new artifacts are under `Faro_ailab`; no new files are written to
`PixVL_ailab`. Current usage is approximately 38G, below the 700G ceiling.
Training screens use at most 12 requested GPUs in aggregate, and evaluation
fallback is fixed at 8 -> 6 -> 4 -> 2 -> 1 GPUs with a five-minute wait at
each level. All dnacoding jobs use positive tags from `rjob_tags.txt` and
names beginning with `dna-`.

## Historical live refresh (2026-08-28 16:50 HKT; superseded)

The terminal R18 evaluator has succeeded and the enhanced diagnostic is now
available at `evals/r18_depth_local_rarity_free_seed17_holdout512_diagnostic`.
All eight R21--R28 screens are still queued with `STARTING` replicas and no
assigned node. They remain valid pending experiments; no candidate claim is
made until its 5,120-row/10-step contract, complete 512-row evaluation, and
20,000 paired bootstrap gates are satisfied.

## Historical live refresh (2026-08-29 04:30 HKT; superseded)

R21--R34 are now completed valid screens with full 512-row/20,000-bootstrap
analyses; none passes the corrected promotion gate. R30's visual-interface
branch fails its worker sentinel-margin validity gate and is closed. R18
therefore remains the only promoted method claim.

PV-FEPO training `dna-fepo-paired-view-10step-2g-1787942860-61768072` remains
without a checkpoint. The matched-SFT training control has finished, but its
terminal evaluator has no valid output: earlier 8/6/4/2-GPU submissions failed
at the control-plane DNS layer and the 1-GPU queue state cannot currently be
queried from this workspace. The finalizer is waiting for the complete
matched-SFT holdout before computing paired comparisons.

R35 safe visual interface and BA-FEPO boundary-bottleneck paired-view are
registered conditional hypotheses. BA-FEPO has implementation, submitter,
and independent preregistration, but has not been submitted. The paper is not
complete: the compute-matched SFT comparison, PV decision, and any survivor's
official transfer evaluation are still missing.

## Authoritative continuation refresh (2026-08-29)

The historical rows above are superseded by `RESULTS_LEDGER_20260828.md` and
`codex_resume/STATUS_20260829_RESUME.md`. R35 failed its worker validity gate;
BA, BS, and AB completed full 512-row/20,000-bootstrap screens and were
rejected. PES-FEPO is now the sole pending candidate, with a required shuffled
evidence control. No PES training, holdout, promotion, or official transfer
claim exists while the dnacoding control plane remains unavailable. The paper
is therefore not yet complete.

## Historical live refresh (2026-08-29 04:40 HKT; superseded)

The matched continued-SFT training control has now materialized at
`outputs/samtok_selective/continued_sft_r18_matched_200` with 200 completed
steps, 5,120-row provenance, and a standalone adapter. Its 512-row evaluator
has not produced output: prior 8/6/4/2-GPU submissions failed at the control
plane's DNS layer and the terminal 1-GPU queue state cannot be queried from
this workspace. The stale local evaluator marker was removed for retry. PV
training remains without a checkpoint; no BA/R35 training was submitted.

## Live refresh (2026-08-29 13:10 HKT)

PV-FEPO training has completed with a valid SAMTok-only contract: 10 optimizer
steps, 20 updates, 5,120 rows, K=4 rollouts, effective-support, tail-risk,
and overall validity gates passed. Its paired-view metadata disables PixVL
teacher, OPD, EMA, and counterfactual objectives. The required 512-row
evaluator has not produced output because rjob submission fails at the
unresolved control-plane hostname. The matched-SFT evaluator is complete and
beats R18 on overall paired utility/cIoU, but its boundary-hard/thin
boundary-IoU slices cross the registered non-inferiority limit; it remains a
control, not a promoted FEPO method. PV has no promotion or transfer claim.

## Live refresh (2026-08-29 14:33 HKT)

PV-FEPO is closed by the preregistered training support gate, not by a
holdout result: the mean joint-positive rollout fraction is `0.10625`, below
`0.20`, across 20 finite-correlation optimizer records. The decision artifact
is `evals/pv_training_gate.json` and explicitly uses no holdout. This closes
the PV line without tuning its view transform or aggregation. R35 is the next
isolated SAMTok-only hypothesis; its submission remains blocked before rjob
creation by the dnacoding control-plane DNS outage.

## Authoritative continuation update (2026-08-30)

This update supersedes the earlier queue snapshots above. R35, BA-FEPO,
BS-FEPO, and AB-FEPO have completed their registered worker or holdout
decisions and are closed; they are not pending paper requirements. The sole
open method-level falsification is PES-FEPO and its mandatory shuffled-
evidence control. PES passes local implementation and contract checks, but no
normal or shuffled rjob, checkpoint, worker metrics, 512-row holdout, or
20,000-bootstrap comparison exists because the dnacoding control plane fails
DNS resolution before rjob creation. The final method and official transfer
therefore remain unselected, and no PES quality or causal evidence claim is
permitted until those runtime artifacts exist.
