# FARO goal completion audit (2026-08-31)

This is a live evidence audit, not a completion claim.  It supersedes the
queue snapshot in `GOAL_COMPLETION_AUDIT_20260830.md` where timestamps differ.

| Requirement | Current authoritative evidence | Status |
| --- | --- | --- |
| Ten-paper literature synthesis | `LITERATURE_CLAIM_AUDIT_20260829.md` and `LITERATURE_TO_FEPO_HYPOTHESES_20260828.md` | verified for analysis |
| SAMTok-only training boundary | `Sa2VA/projects/samtok_selective/` contracts and manifests; no PixVL trainer/teacher/checkpoint in the method path | verified for implementation |
| Multiple hypotheses and falsification | R21-R35, PV, BA, BS, AB, and PES preregistrations with recorded closures | verified for registered exploration |
| Minimum training contract | `validate_training_budget.py` reports `actual_rows=5120 configured_rows=5120 configured_steps=10`; configs pin K=4 | verified for PES configuration |
| Positive tags and job policy | Submitters read all entries from `rjob_tags.txt`, require `ailab-dnacoding`, and enforce `dna-` names | verified statically |
| Evaluation fallback | Adaptive evaluators pin `8 -> 6 -> 4 -> 2 -> 1` and 300-second waits; terminal 1-GPU stage is left queued | verified statically |
| A-PES implementation probe | `run_apes_contract_probe.py` passed with detached state/scope and seed `1907` | verified as offline contract only |
| Candidate implementation probe | `run_fepo_candidate_probe.py` exited `0`; 17 contracts and 13 credit variants passed finite/local checks | verified as offline contract only |
| Global training-config guard | 81 configs load; 43 satisfy the current minimum contract, while 38 historical/one-step configs are rejected by submitter guards before `rjob` | verified fail-closed |
| Promoted FEPO quality result | R18 has complete 512-row/20k paired holdout and provisional positive utility/cIoU result | verified provisional reference |
| PES normal training result | No rjob, checkpoint, worker metrics, or valid finished-manifest artifact exists | missing |
| PES shuffled causal control | Locked until normal PES worker validity and holdout gates pass | missing |
| PES 512-row/20k comparisons | No normal or shuffled PES holdout/Bootstrap JSON exists | missing |
| Final survivor official transfer | Existing R18 transfer is documented, but no PES survivor is selected | incomplete |
| Submission-ready final paper | Draft and audits exist, but final method selection depends on the open PES branch | incomplete |

## Current external condition

The lock-protected normal PES retry state remains at `logs/pes_submit/`.
Retries through 21:21 HKT fail before rjob creation because the configured
proxy and direct cluster hostnames cannot be resolved.  The requested proxy
setup endpoint is likewise unresolved.  No alternate endpoint, fabricated
worker artifact, or PixVL write is used.

## Continuation update (2026-08-31 21:53 HKT)

The PES sampler was corrected before job creation: native-vs-sampled margin
collection now uses the depth-local sampled index for offset code vocabularies.
The focused PES/A-PES, policy-isolation, submission-contract, and evaluation
guard suite passes `28 passed`; the 17-candidate contract probe and shell
syntax audit also pass.  These are implementation checks only and do not
substitute for a worker or quality result.

The complete `tests/test_*static.py` suite was also rerun in the worker-like
environment and passed `120 tests`, covering every registered candidate and
the shared submission/evaluation guards.

The worker launch preflight was executed with the approved SAMTok anchor and
PES manifest; both `manifests guard` and `tail_gppo_contract --skip-model-hash`
returned `status: ok`.  This validates launch-time contracts only and does not
constitute a training or quality result.

The lock-protected PES retry is still active.  The latest retry at
`21:48:49+08:00` returned `control_plane_unavailable status=1`; DNS for
`h.pjlab.org.cn` remains unresolved, no `submitted` marker exists, and no PES
checkpoint, 512-row holdout, or 20,000-bootstrap artifact can yet be claimed.

## Next valid evidence transition

When the control plane becomes reachable, reconcile the existing namespace,
submit only the registered normal PES job with all positive tags, and require
the worker validity gates before evaluation.  Then run the complete 512-row
image-disjoint holdout and 20,000 paired bootstrap, submit the shuffled
negative control only after normal PES passes, and select a final survivor
before official transfer.  Until these artifacts exist, R18 remains the only
defensible method claim and the overall goal remains active.

## Current recheck (2026-08-31 22:44 HKT)

The local repository is clean and now includes the corrected SAMTok-only
README and the artifact-backed final results table; the latest local commit is
`94d17d2` (the README itself was recorded in `118a876`).  The 19 completed
rows in `FINAL_RESULTS_TABLE_20260831.md` were machine-checked against their
JSON artifacts with `errors=0`.  The PES training manifest still
has exactly 5,120 rows, and the standalone budget validator reports 10
configured optimizer steps.  With the approved read-only SAMTok anchor, the
manifest guard and tail-GPPO contract both return `status: ok`.

The dnacoding proxy and cluster DNS remain unresolved.  The PES retry lock is
currently held and `submit.log` has a new five-minute
`control_plane_unavailable` heartbeat, confirming that the runner is active in
its own process namespace.  The namespace-local PID cannot be checked with
`kill -0` from this shell; do not launch a second copy.  No PES rjob,
checkpoint, holdout, bootstrap artifact, or final-paper promotion is inferred
from the preflight checks.

## Evidence recheck (2026-08-31 23:20 HKT)

The registered static suite passes `121 tests`, and a machine scan confirms
64 FARO bootstrap artifacts with exactly 512 paired rows and 20,000
repetitions; no unexpected non-official artifact violates the format. The
working tree is clean. PES remains unsubmitted because the control-plane DNS
failure persists, so the final method and transfer claims remain open.

## Current recheck (2026-08-31 22:58 HKT)

The repository remains clean after commit `9b1521a`, which enforces the
complete registered `rjob_tags.txt` allowlist for the adaptive action-margin
diagnostic at every GPU level and removes the obsolete alternate tag file.
The worker-like registered static suite passes `120 tests`; focused PES/A-PES
and submission/monitor checks pass `49 tests`. The PES budget validator and
both SAMTok launch preflight checks remain successful.

The normal PES retry has no submission marker. The latest control-plane
recheck still fails before API access because the configured proxy hostname
cannot be resolved. Therefore there is still no PES checkpoint, worker
metrics, 512-row holdout, shuffled control, or final promotion result.
