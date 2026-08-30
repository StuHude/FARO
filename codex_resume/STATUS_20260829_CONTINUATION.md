# FEPO continuation status (2026-08-29)

## Resume refresh (2026-08-29 16:22 HKT)

- Control-plane probing remains unsuccessful; the R35 lock is still held and
  no new task or artifact is visible locally. No queue status is treated as a
  result.
- Added a submission-readiness section to `EG_FEPO_PAPER_DRAFT.md`, making
  R35/BA ordering, conditional AB/BS/NCVI variants, and the exact claim
  boundary explicit for later paper assembly.
- Revalidated contracts and policy tests (`16 passed` in the focused suite);
  storage remains below the 700G ceiling.

## Resume refresh (2026-08-29 15:58 HKT)

- The dnacoding control plane remains DNS-unreachable from this login node;
  no new job was submitted and the external R35 lock/monitor was preserved.
- Fixed a finalizer state-machine bug: a valid `pv_training_gate.json` with
  `decision=closed_training_gate` now satisfies the PV closure condition even
  when no PV holdout exists, allowing the registered R35 submit path to run
  once the control plane recovers. The existing matched-SFT 512/20k artifact
  is still required.
- Active evaluation, null-probe, official-eval, schema-smoke, and rollout
  diagnostics now force outputs/logs under `FARO_ROOT`; `PixVL_ailab` remains
  read-only for SAMTok/data/evaluator inputs. The historical selective-anchor
  gate is disabled because it targets the retired PixVL output tree.
- Validation: 14 focused static tests pass and all touched shell scripts pass
  `bash -n`. Storage remains about 38G, below the 700G ceiling.
- Added `scripts/watch_finalizer_upgrade.sh` as a non-destructive handoff aid;
  it only starts the corrected finalizer after the existing lock is released.

## Resume refresh (2026-08-29 16:07 HKT)

- The control plane still fails DNS resolution; R35 has no checkpoint or rjob
  output and no new submission was attempted.
- Audited all `submit_*.sh` rjob entry points: active submitters now source
  positive tags from `rjob_tags.txt` (including legacy 2-GPU wrappers), while
  historical disabled paths remain disabled. Added a static policy test for
  this invariant and for accidental `PixVL_ailab` output writes.
- Focused regression suite now passes `16` tests; touched scripts pass shell
  syntax checks. Storage remains below 700G.

## Resume refresh (2026-08-29 16:11 HKT)

- Revalidated the expanded policy suite after normalizing all 2-GPU wrappers
  to the full `rjob_tags.txt` list: `28` tests pass and all shell scripts pass
  `bash -n`.
- No files newer than the current run were created under `PixVL_ailab`; no
  R35 checkpoint/evaluator output exists yet. The monitor heartbeats remain
  queue/control-plane status only.

## Resume refresh (2026-08-29 16:14 HKT)

- Rechecked the queue: control-plane DNS still fails and the registered R35
  lock remains held; no new rjob, checkpoint, or evaluator output is present.

## Resume refresh (2026-08-29 16:40 HKT)

- The offline FEPO candidate probe now includes AB-FEPO (fixed action budget
  `B=2`, excess penalty `0.10`) alongside R21--R30, R35, PV, and BA. All
  registered configs pass the 5,120-row/10-step/K=4 contract and the probe
  reports finite non-negative detached credit for every candidate.
- This is a contract/implementation check only. AB-FEPO remains reserved
  until R35 and BA are closed; no training, holdout, or paper claim is made.
- A fresh `rjob list --namespace=ailab-dnacoding` still fails before API
  access because the cluster endpoint cannot be resolved from this node. No
  new task or output was created; the existing adaptive 8 -> 6 -> 4 -> 2 -> 1
  evaluator policy remains unchanged.
- Full login-node collection cannot import ten Torch-dependent tests because
  this node has no `torch`; the expanded non-Torch/contract suite remains
  green (`28 passed`). Worker-only validation is intentionally deferred to
  the actual rjob image.

## Resume refresh (2026-08-29 16:18 HKT)

- Removed environment overrides from adaptive tag selection: 8/6/4/2/1
  evaluation replicas now all receive exactly the full `rjob_tags.txt` list.
- Re-ran the tag/path policy and monitor tests (`7 passed` in the focused
  subset), all shell syntax checks, and `git diff --check` successfully.
- The control plane and R35 state are unchanged; no training or quality claim
  is inferred from queue heartbeats.

## Resume refresh (2026-08-29 15:43 HKT)

- The current login workspace still cannot resolve/query `h.pjlab.org.cn`; no
  R35 or BA rjob is visible from the authenticated control-plane probe.
- The R35 submit lock remains held outside this PID namespace. It is retained
  deliberately; no lock file or monitor is deleted and no duplicate submit is
  attempted. The registered monitor remains the only permitted R35 submit path.
- All adaptive evaluation entry points now fail closed on a status-query/API
  error. They retry the query before stopping a queued replica, recognize a
  concrete `STARTING` GPU node as having left the queue, and preserve the
  fixed `8 -> 6 -> 4 -> 2 -> 1` / 300-second ladder with 1-GPU terminal
  behavior.
- The historical PixVL routed-OPD schema training submitter is disabled. Any
  future training path must use SAMTok-only initialization and the validated
  5,120-row/10-step contract.
- Focused static, budget, and shell checks pass (`7` pytest tests plus shell
  syntax checks); FARO storage remains approximately `38G`.

**Authority rule:** the newest `Resume refresh` block is authoritative; older
blocks below are retained as timestamped provenance and must not drive job
submission or quality claims.

## Resume refresh (2026-08-29 14:43 HKT)

- A new state audit found no R35 checkpoint, evaluator output, or authenticated
  rjob. `h.pjlab.org.cn` still fails DNS resolution from this workspace.
- PV remains marked `CLOSED_TRAINING_GATE` with no retry marker; the legacy
  monitor heartbeat reports it as submitted only because its idempotent marker
  is non-pending. No new PV submission occurred.
- The workspace remains at approximately 38G. The SAMTok-only R35 submitter,
  finalizer, and adaptive evaluator scripts pass shell syntax and preserve all
  positive-tag, namespace, naming, data, step, and GPU-ladder constraints.

## Resume refresh (2026-08-29 14:41 HKT)

- A fresh control-plane check still fails DNS resolution for
  `h.pjlab.org.cn`. A controlled external attempt to submit the registered
  R35 2-GPU job was also rejected by the execution policy before reaching the
  API; no task was created, stopped, or counted against the 24-GPU budget.
- The R35 submitter remains the only allowed next action. It still enforces
  the approved SAMTok checkpoint, 5,120-row manifest, 10 optimizer steps,
  `dna-` naming, `ailab-dnacoding`, and all positive tags. The adaptive eval
  ladder remains 8 -> 6 -> 4 -> 2 -> 1 with 300 seconds per nonterminal level.
- Broader login-node regression completed: 53 non-Torch tests passed and 2
  environment-dependent tests were skipped. Worker-only Torch coverage is
  not claimable on this CPU-only node.

## Resume refresh (2026-08-29 14:33 HKT)

- Replayed the PV preregistered training-only decision with the completed
  `metrics.json`: 20 optimizer records, finite clean/view correlations, and
  joint-positive fraction mean `0.10625` (min `0.0`, max `0.21875`) versus the
  fixed `0.20` threshold. PV is therefore **closed_training_gate**; the
  decision artifact is `evals/pv_training_gate.json`, records
  `holdout_used=false`, and does not claim a quality result.
- The PV evaluator marker is now `CLOSED_TRAINING_GATE`; its retry backoff was
  removed, so the adaptive ladder will not submit or downgrade a PV evaluator.
  The PV adapter, metrics, provenance, and diagnostics remain intact.
- The updated finalizer accepted the closed PV decision and the complete
  matched-SFT/R18 20,000-bootstrap control, then reached the registered R35
  `dna-` submitter. The attempt failed before rjob creation because
  `h.pjlab.org.cn` remains unresolvable; no R35 GPU was allocated. Its retry
  state remains safe to resume when the control plane returns.
- R35 remains the only next training hypothesis. BA-FEPO stays unsubmitted
  until R35 closes, preserving the one-new-arm and 24-GPU budget. All jobs
  continue to require `ailab-dnacoding`, every positive tag in
  `rjob_tags.txt`, and the 8 -> 6 -> 4 -> 2 -> 1 evaluation ladder.
- Added `tools/decide_pv_training_gate.py` and focused regression coverage;
  the PV/R35/BA/monitor suite passes (`15 passed`). No files were written
  under `PixVL_ailab`; FARO storage remains about 38G.

## Resume refresh (2026-08-29 13:10 HKT)

- PV-FEPO training is finished and contract-valid at
  `outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu`:
  5,120 rows, 10 outer steps/20 optimizer updates, K=4, effective-support,
  tail-risk, and grammar gates all pass. Its 512-row evaluator output is
  still absent because the control-plane DNS currently fails before rjob
  creation; no PV quality or transfer claim is made.
- The matched continued-SFT control now has `evals/r18_matched_sft_holdout512`
  with exactly 512 rows. Against R18, 20,000 paired bootstrap gives positive
  cIoU `+0.005641` CI `[+0.000070,+0.013302]`, utility `+0.006727` CI
  `[+0.001306,+0.013945]`, and no-target recall `+0.007812` CI
  `[0,+0.019531]`, with zero invalid outputs. Boundary-hard/thin
  boundary-IoU slices cross the registered `-0.01` limit, so this remains a
  matched supervised control rather than the promoted RL method.
- R25 uncertainty-calibrated FEPO is closed after its complete 512-row/20k
  result (cIoU CI `[-0.000742,+0.004892]`, utility CI
  `[-0.000417,+0.002452]`). No scalar-credit variant is being stacked.
- The late monitor/finalizer retains the required evaluator ladder
  `8 -> 6 -> 4 -> 2 -> 1`, five minutes per nonterminal level, with all
  submissions using `dna-`, `ailab-dnacoding`, and every positive tag in
  `rjob_tags.txt`. R35 and BA-FEPO remain gated behind the PV decision.
- Workspace usage is approximately 38G (`outputs` 33G, `evals` 4.4G), below
  the 700G ceiling; no new files have been written under `PixVL_ailab`.

## Resume refresh (2026-08-29 13:28 HKT)

- PV diagnostics were rechecked before making any early decision: joint-positive
  fractions span `0.0` to `0.21875` across 20 optimizer records (mean
  `0.10625`), so the intermediate summary is inconclusive and the required
  512-row/20k evaluator remains mandatory.
- A stale PV evaluator marker was removed and the live late monitor retried the
  evaluator at 8 GPUs; the retry again failed at the control-plane DNS before
  rjob creation and recorded the required 300-second backoff. No GPU task was
  created or stopped. The matched-SFT marker is retained because its output is
  complete.
- R35/BA-FEPO are still blocked by the PV decision; no training artifact or
  new data was created. Static shell/Python checks remain clean.

## Resume refresh (2026-08-29 13:42 HKT)

- The control plane still cannot resolve `h.pjlab.org.cn`; PV has no evaluator
  output and no R35/BA job is visible. The late monitor continues pre-submit
  retries, so no GPU task is being downgraded or duplicated.
- With the approved SAMTok base checkpoint and `continued_sft_to500` adapter
  explicitly set, all 12 registered configurations (R21--R30, PV, BA, R35)
  loaded and passed `validate_tail_gppo_config`, including the 5,120-row and
  10-step minimums. This is an offline contract check, not a training result.

## Evidence state

- R18 native-relative first-divergence joint geometry remains the promoted
  SAMTok-only reference. Its 100-step confirmation has a positive utility
  delta of `+0.017890` versus the SFT anchor, 95% CI `[+0.007580,+0.029930]`;
  the positive-cIoU interval crosses zero, so no long-horizon cIoU claim is
  made.
- R21-R34 completed valid screens and did not pass the preregistered strict
  promotion gate. They remain closed, not candidates for mechanism stacking.
- `continued_sft_r18_matched_200` completed 200 steps on 5,120 rows. Its
  enhanced 512-row holdout is still pending because the dnacoding API cannot
  resolve `h.pjlab.org.cn`.
- PV-FEPO is registered as the next isolated hypothesis. No training or
  evaluation result is inferred until its worker contract and full holdout
  are complete.

## Isolated idea queue

1. **PV-FEPO (active):** geometric-mean R18 credit on clean and fixed
   target-preserving photometric views. Tests whether clean-view gains are
   image-specific, while retaining one SAMTok policy and one geometry reward.
2. **MV-FEPO view-family ablation (contingent):** replace the photometric
   transform with a fixed resize/letterbox view and require paired improvement.
   This isolates spatial-interface sensitivity; it is not combined with PV.
3. **Boundary-first FEPO (contingent):** use the existing two-axis geometry
   reward but evaluate a fixed thin/boundary-hard slice as the preregistered
   selection axis, motivated by Qwen3VL-Seg and DR2Seg. No extra router or
   teacher is introduced.
4. **Null-calibrated FEPO (contingent):** only if the matched-SFT control shows
   sentinel drift, test a fixed lower-tail null constraint inspired by
   OpenWorldSAM. R26/R29 already provide negative evidence against changing
   null credit without observed drift, so this branch stays closed unless the
   control reopens it.

5. **R35 safe visual interface (prepared):** an isolated R30 follow-up that
   keeps visual-merger/deepstack LoRA as the only trainable scope and fixes
   `null_ce_weight=2.0`, `margin_weight=1.0` after R30's sentinel-margin
   failure. It is registered in the contract, probe, and late monitor but is
   not submitted while PV and matched-SFT occupy the active queue.

Each branch requires at least 5,120 training rows, at least 10 optimizer
steps, a complete 512-row holdout, 20,000 paired bootstrap repetitions, and
fixed slice/canonical/null gates. At most one branch may advance at a time.

## Queue automation

The PV submitter and late-screen monitor hold their lock files and continue to
append retry logs. Every training/eval submission is `dna-*`, namespace
`ailab-dnacoding`, and includes all positive tags from `rjob_tags.txt`. Eval
fallback remains 8, 6, 4, 2, then terminal 1 GPU, with 300 seconds per
nonterminal level. The control plane recovered intermittently at 03:00. PV-FEPO
training is queued as `dna-fepo-paired-view-10step-2g-1787942860-61768072`.
The matched-SFT evaluation completed the required fallback: 8G, 6G, 4G, and
2G were stopped after five minutes each; terminal 1G job
`dna-fepo-r18-matched-sft-eval-1g-1787944114-3256d` is `Inqueue` and is left
untouched. These are queue states, not quality results.

Once either candidate's evaluator writes a complete output, the long-running
`monitor_finalize_matched_sft_pv.sh` session invokes
`finalize_matched_sft_pv.sh`. That analysis requires exactly 512 paired rows,
20,000 bootstrap repetitions, boundary-aware fixed slices, and both
PV-vs-R18 and PV-vs-matched-SFT comparisons.

## Storage

Workspace usage is approximately 37G. Closed-candidate adapters account for
about 30.5G; metrics, provenance, logs, and holdout/bootstrap artifacts are
kept for audit. A scoped deletion plan is in
`STORAGE_CLEANUP_PLAN_20260829.md`; no irreversible deletion is performed
until the keep list is explicitly confirmed.

## Resume refresh (2026-08-29 03:25 HKT)

- Rechecked queue state through the eval audit: PV training
  `dna-fepo-paired-view-10step-2g-1787942860-61768072` remains `Inqueue`
  (`STARTING`, no node). The matched-budget SFT evaluator remains at the
  terminal one-GPU fallback
  `dna-fepo-r18-matched-sft-eval-1g-1787944114-3256d`, also `Inqueue`.
- Restarted the PV submitter, late-screen evaluator, and finalizer monitors in
  this shell. The PV submitter reconciles the existing job by prefix before
  any retry; no duplicate job was created. The late evaluator still reports
  `paired_view=waiting` and `r18_matched_sft=eval_submitted`.
- Data contracts remain intact: `egfepo_train_5120.jsonl` has exactly 5,120
  rows and `grefcoco_selective_holdout_256.jsonl` has exactly 512 rows.
- Login-node checks passed: nine non-torch static tests, Python compilation
  of the PV/R25 trainer and contracts, and shell syntax checks. The R25 static
  test still requires the worker image's `torch` package and was not run on
  the login node.
- The standalone GR-CPPO static suite was rerun after removing a stale
  exclusion-word match in a trainer comment: 11 relevant standalone/PV tests
  now pass. The R24 checkpoint test still needs its explicit model-path
  environment when run outside the worker launcher; with the approved
  SAMTok path set, both R24 checkpoint tests pass. This is an environment
  setup check, not a training result.
- Workspace usage is 37G (`FARO/outputs` 32G, `FARO/evals` 4.4G), below the
  700G ceiling. Closed-candidate adapter deletion remains deferred until the
  PV/matched-SFT decision so their current audit artifacts cannot be removed
  prematurely.
- The latest monitor heartbeat at 03:30 HKT still shows both active jobs
  waiting without worker output; no new quality claim is entered in the
  ledger.
- R35 safe visual interface is now preregistered and wired into the candidate
  probe and late evaluator. Its CPU contract tests pass (13 tests in the
  latest R30/R35/PV batch); it remains intentionally unsubmitted until one of
  the two active jobs exits so the 24-GPU cap is auditable.
- Added `submit_r35_after_pv_decision.sh` to the finalizer. It waits for both
  PV-vs-R18 and PV-vs-matched-SFT 20k bootstrap files and submits R35 only if
  both promotion gates are false; a promoted PV candidate therefore cannot be
  silently followed by an extra training arm.
- Tightened that trigger to require `num_paired=512`, 20,000 repetitions for
  utility/cIoU/null-recall, and the CI-corrected gate (with legacy fallback
  only for compatibility). The trigger cannot promote R35 from an incomplete
  or legacy-only analysis.
- At the 03:51 HKT refresh, no PV/matched-SFT/R35 output exists; the same
  queue-independent decision remains in force.
- Removed only regenerable Python `__pycache__`/`.pyc` files. Checkpoint,
  metrics, provenance, logs, and evaluation shards were untouched. The
  standalone/PV/R25-adjacent CPU static suite (15 tests) passes; worker-only
  torch tests remain deferred to the rjob image.
- Updated `EG_FEPO_PAPER_DRAFT.md` title and abstract to the evidence-backed
  FEPO-R18 claim. The early evidence-gated arm remains documented as a failed
  ablation rather than the headline method.

## Resume refresh (2026-08-29 04:02 HKT)

- The external late-screen monitor continues to heartbeat. PV-FEPO and the
  matched-SFT evaluator remain waiting/eval-submitted respectively; no
  checkpoint or holdout output has appeared, so no quality result is inferred.
- The finalizer is still waiting for the matched-SFT 512-row evaluator output.
  Its terminal 1-GPU fallback remains untouched, and the adaptive evaluator
  retains the 8 -> 6 -> 4 -> 2 -> 1 policy with 300 seconds at each
  non-terminal level.
- Added `FEPO_NEXT_IDEAS_20260829.md` with two isolated post-PV hypotheses:
  boundary-bottleneck paired-view credit (BA-FEPO) and a conditional
  sentinel-constrained variant (SCB-FEPO). Neither is submitted while the
  active queue branches are unresolved.
- Re-ran the login-node CPU/static suite for PV/R35/paired-view/bootstrap:
  11 tests passed. Worker-only torch tests remain deferred because this node
  does not provide `torch`.
- Storage remains approximately 37G (`FARO/outputs` and `FARO/evals`); no
  writes were made under `PixVL_ailab` and no audit artifacts were deleted.

## Authenticated queue audit (2026-08-29 04:06 HKT)

- Direct authenticated API inspection confirms PV training
  `dna-fepo-paired-view-10step-2g-1787942860-61768072` is `Inqueue` with a
  `STARTING` replica and no assigned node.
- The terminal matched-SFT evaluator
  `dna-fepo-r18-matched-sft-eval-1g-1787944114-3256d` is also `Inqueue` with a
  `STARTING` replica and no assigned node. The 8/6/4/2 stages were already
  downgraded after their 300-second waits; the 1-GPU stage is intentionally
  retained.
- The authenticated API exposes 4,803 rjobs. No PV checkpoint,
  `r18_matched_sft_holdout512`, or paired-view evaluator output exists yet.
- PV submission reconciliation found the existing job by prefix and did not
  create a duplicate. The late monitor therefore has not submitted a PV eval.
- A full metrics-directory audit found 35 FEPO run manifests: 30 finished
  runs satisfy the registered 10-step validity contract, while the remaining
  entries are explicitly recorded failed/running branches. No finished valid
  checkpoint is missing an evaluation marker outside the monitor's candidate
  list.
- BA-FEPO is now implemented but intentionally unsubmitted: its isolated
  trainer branch uses the minimum of clean and photometric paired-view
  R18-local credits, with a 5,120-row/10-step config, positive-tag submitter,
  contract registration, candidate probe, and three static tests. It can only
  be considered after PV is closed, so no GPU is consumed by this preparation.

## Resume refresh (2026-08-29 04:15 HKT)

- Authenticated five-minute monitoring found no queue transition. PV and the
  terminal matched-SFT 1-GPU evaluator remain `Inqueue/STARTING` with no node;
  their API `lastUpdateTime` values remain 2026-08-28 18:47:42Z and
  19:08:35Z, respectively.
- Neither a PV checkpoint nor `r18_matched_sft_holdout512` has appeared. No
  task was submitted or stopped during the monitoring window.
- BA-FEPO implementation verification passed: `14 passed` across BA, PV,
  R35, paired-view, and bootstrap static tests; Python compilation, shell
  syntax, and `git diff --check` also pass. BA output is confirmed absent,
  so this candidate has not been submitted.
- The late evaluator registry now includes BA-FEPO for automatic 512-row
  evaluation if a future gated training submission produces its checkpoint;
  the registry entry does not submit training or consume GPUs by itself.

## Resume refresh (2026-08-29 04:19 HKT)

- The monitor heartbeat remains healthy through 04:19. PV and matched-SFT are
  still waiting/eval-submitted with no new files, so the paired finalizer has
  not advanced.
- Re-ran the BA/PV/R35/paired-view/bootstrap static suite: `14 passed`.
  Storage remains `37G`, and `git diff --check` is clean.

## Resume refresh (2026-08-29 04:20 HKT)

- The authenticated monitor still reports PV/matched-SFT waiting with no
  checkpoint or evaluation directory. The finalizer continues waiting for the
  matched-SFT holdout rather than producing an incomplete comparison.
- BA-FEPO's static suite remains green after its evaluator registry update:
  `14 passed`; workspace storage remains approximately 37G.

## Resume refresh (2026-08-29 04:22 HKT)

- Monitor heartbeats continue through 04:22 with PV and matched-SFT still
  waiting/eval-submitted; no checkpoint or holdout output exists.
- Re-ran the complete BA/PV/R35/paired-view/bootstrap static suite: `14
  passed`. BA remains implementation-only and unsubmitted pending the
  registered PV/R35 decision order.

## Resume refresh (2026-08-29 04:24 HKT)

- The monitor heartbeat continues through 04:24. PV remains waiting and the
  matched-SFT evaluator remains at its terminal 1-GPU submission; neither has
  produced a checkpoint or holdout output.
- Repeated the full static suite after the BA registry checks: `14 passed`.
  The PV submission marker still resolves to the original job, with no
  duplicate training submission; storage remains approximately 37G.

## Resume refresh (2026-08-29 04:26 HKT)

- PV and matched-SFT remain queued with no new checkpoint or evaluation
  output; monitor/finalizer processes continue to run.
- Added the standalone `BA_FEPO_PREREG_20260829.md`, specifying the fixed
  boundary-bottleneck method, matched controls, complete 512/20k evaluation,
  slice/canonical/null gates, and the no-overlap queue policy.

## Resume refresh (2026-08-29 04:28 HKT)

- Monitor heartbeats continue through 04:28 with PV and matched-SFT still
  waiting/eval-submitted. The PV output directory is still absent, and no
  holdout evaluator output has been created.
- No task was submitted or stopped; workspace usage remains approximately
  37G.

## Resume refresh (2026-08-29 04:30 HKT)

- Queue heartbeats remain unchanged through 04:30: PV/matched-SFT are still
  waiting/eval-submitted with no checkpoint or holdout output.
- Refreshed `PAPER_READINESS_AUDIT_20260828.md` to include R21--R34 results,
  R30 closure, PV/matched-SFT queue evidence, R35/BA conditional status, and
  the remaining missing evidence for a complete paper.
- Post-refresh static suite remains green (`14 passed`); storage remains
  approximately 37G.

## Resume refresh (2026-08-29 04:32 HKT)

- Queue heartbeat remains unchanged through 04:32. PV and matched-SFT have no
  checkpoint or holdout output, and the finalizer remains waiting.
- BA's submitter/configuration audit found no forbidden PixVL/OPD/teacher or
  self-supervised component; its wrapper still enforces 5,000+ rows and
  `dna-` naming. No BA task was submitted.

## Resume refresh (2026-08-29 04:40 HKT)

- Filesystem reconciliation found that `continued_sft_r18_matched_200` did
  finish: its metrics report `status=finished` with 200 optimizer steps and
  its provenance records 5,120 training rows. The standalone adapter and
  provenance are retained under `FARO/outputs`.
- The previous matched-SFT evaluator marker was stale: its log contains only
  failed control-plane submissions caused by DNS resolution of
  `h.pjlab.org.cn`, and no evaluator output or bootstrap artifact exists. The
  stale marker was removed so the adaptive evaluator can retry when dnacoding
  networking recovers. The prescribed 8 -> 6 -> 4 -> 2 -> 1 GPU ladder and
  300-second nonterminal waits remain unchanged.
- PV-FEPO still has no checkpoint/output directory and BA remains
  implementation-only. No new training job was submitted, no duplicate was
  created, and no file was written under `PixVL_ailab`.
- The BA/PV/R35/paired-view/bootstrap static suite passes (`14 passed` with
  `PYTHONPATH=FARO/Sa2VA:FARO`). Workspace usage remains approximately 37G,
  below the 700G ceiling.

## Resume refresh (2026-08-29 04:48 HKT)

- A bounded local monitor retry reconfirmed that the matched-SFT adapter is
  valid, but every evaluator submission still fails before job creation because
  `h.pjlab.org.cn` cannot be resolved in this workspace. The retry removed its
  transient marker again; no rjob was stopped or duplicated.
- No PV files or matched-SFT holdout output appeared during the retry. The
  experiment order and terminal 1-GPU fallback remain unchanged.

## Resume refresh (2026-08-29 04:53 HKT)

- The persistent late monitor continues to detect the finished matched-SFT
  control, but evaluator submission still fails before rjob creation at the
  unresolved dnacoding control-plane hostname. PV remains pending with no
  `metrics.json`; no new training arm was started.
- Marked the old router/OPD experiment matrix as historical and added the
  current R18 -> matched-SFT/PV -> R35 -> BA decision matrix. This prevents
  future automation from reviving the abandoned PixVL-style route plan.

## Resume refresh (2026-08-29 12:55 HKT)

- PV-FEPO training has now completed at
  `outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_paired_view_10step_2gpu`.
  The metrics report `status=finished`, `steps_completed=10`,
  `optimizer_updates_completed=20`, K=4 rollouts, 5,120 provenance rows,
  effective-support gate passed, tail-risk gate passed, and overall validity
  passed. The paired-view contract is geometric-mean clean/photometric credit
  with `uses_pixvl_teacher=false`, `uses_opd=false`, and no EMA or
  counterfactual objective.
- No quality claim is made yet: neither `paired_view_holdout512` nor
  `r18_matched_sft_holdout512` exists. Both evaluator submission attempts are
  currently failing before rjob creation because the control-plane hostname
  cannot be resolved from this workspace.
- Hardened both screen monitors to back off failed control-plane submissions
  for 300 seconds and to recognize an already-complete evaluator output before
  retrying. This preserves the requested 8 -> 6 -> 4 -> 2 -> 1 evaluator
  ladder without flooding the API or duplicating a completed evaluation.
- Workspace usage is approximately 33G outputs plus 4.4G evals and 28M logs;
  no data or code was written under `PixVL_ailab`.

## Resume refresh (2026-08-29 13:00 HKT)

- A direct queue query still fails with `Name or service not known` for
  `h.pjlab.org.cn`; no authenticated queue transition can be established from
  this login workspace.
- The two pending evaluator markers now explicitly contain
  `PENDING_CONTROL_PLANE`, preventing the pre-existing monitor process from
  launching repeated submissions. The updated monitor treats this state as
  retryable after 300 seconds and writes `SUBMITTED` only after an adaptive
  evaluator command returns success.
- PV training metrics were independently rechecked against its provenance:
  all 11 contract checks pass. This is training evidence only; holdout and
  bootstrap evidence remain absent.
- The focused CPU/static suite passes (`16 passed`). The full test directory
  still has 10 worker-only collection errors because this login node has no
  `torch`; those tests must run inside the dnacoding worker image once an
  evaluator worker is allocated.

## Resume refresh (2026-08-29 13:50 HKT)

- PV-FEPO remains contract-valid at 5,120 rows, 10 optimizer steps, 20
  updates, K=4, with effective-support, tail-risk, grammar, and validity
  gates passed. This is still training evidence only: the required
  `paired_view_holdout512` output and both 20,000-repetition paired
  comparisons are absent.
- The matched continued-SFT control has a complete 512-row/20k result and is
  retained strictly as a compute control. Its overall cIoU/utility gain over
  R18 coexists with boundary-hard/thin boundary-IoU tail regression, so it is
  not an RL-superiority claim and does not authorize stacking another arm.
- The adaptive evaluator markers for PV and matched-SFT are
  `PENDING_CONTROL_PLANE`; recent submit logs fail before rjob creation while
  resolving `h.pjlab.org.cn`. No evaluator GPU has been allocated, stopped, or
  duplicated. The 8 -> 6 -> 4 -> 2 -> 1 ladder remains configured with
  300-second waits and a terminal 1-GPU queue stage.
- BA-FEPO and R35 remain registered, implementation-checked conditional
  hypotheses. Neither is submitted before the PV decision and the prescribed
  queue order. The ten-paper synthesis continues to motivate only the
  isolated PV -> representation-control -> boundary-bottleneck sequence;
  no PixVL trainer/weights, OPD teacher, inference router, or cyclic
  self-supervised loop has been reintroduced.
- Focused static checks pass (`12 passed` with `PYTHONPATH=FARO/Sa2VA`), shell
  syntax and Python compilation pass. Current usage is approximately 33G
  outputs, 4.4G evals, and 32M logs (about 38G total), below the 700G cap;
  no new data or code was written under `PixVL_ailab`.

## Resume refresh (2026-08-29 13:54 HKT)

- A fresh monitor pair is now active under independent v3 locks. It records
  PV as `retry_backoff` between attempts and leaves the matched-SFT output in
  `eval_finished`; the legacy monitor only observes existing markers and does
  not submit duplicate work. The latest PV retry deadline is five minutes
  after the 13:51 DNS failure.
- No `paired_view_holdout512` file, evaluator shard, or new training output
  has appeared. The next permitted transition is an automatic evaluator retry
  when the control plane resolves; no BA/R35 submission is triggered by this
  transient failure.
- The monitor scripts now default to the v3 lock names used by the active
  corrected pair, so a future resume will not silently attach to the stale
  v1/v2 process state.

## Resume refresh (2026-08-29 13:58 HKT)

- The next PV retry reached the adaptive submitter and failed again before
  rjob creation with the same unresolved `h.pjlab.org.cn` control-plane DNS.
  The retry marker was advanced another 300 seconds; `paired_view_holdout512`
  remains absent. The active v3 monitor is still running and will retry
  automatically.

## Resume refresh (2026-08-29 14:03 HKT)

- Fixed and tested a finalizer bug: the helper's zero exit status after
  processing matched-SFT used to make the monitor stop even when PV was
  missing. It now waits for both
  `paired_view_vs_r18_bootstrap20k.json` and
  `paired_view_vs_matched_sft_bootstrap20k.json` before invoking the R35
  decision script. The corrected finalizer is active under `finalize_v8` and
  currently logs `waiting_for_pv`.
- Focused static suite is green (`13 passed`); no new GPU job, checkpoint, or
  evaluator output was created by this fix.
- The underlying `finalize_matched_sft_pv.sh` helper now also returns waiting
  code `3` when PV is absent, so direct/manual invocation has the same
  semantics as the persistent monitor.

## Resume refresh (2026-08-29 14:15 HKT)

- Tightened the PV-to-R35 decision guard. R35 now requires both 512-row,
  20,000-bootstrap overall PV comparisons plus
  `paired_view_vs_r18_slices20k.json` with `num_paired=512` and
  `slice_gate=true`; missing or partial statistics cannot trigger a training
  submission. The guard is covered by the focused static suite.

## Resume refresh (2026-08-29 14:09 HKT)

- The 14:08 PV retry also failed before rjob creation at control-plane DNS;
  no paired-view evaluator shards or bootstrap files exist. The corrected
  finalizer recorded `waiting_for_pv rc=3` and remains asleep until its next
  five-minute check. No candidate selection or training transition is
  justified by this external failure.

## Resume refresh (2026-08-29 14:10 HKT)

- The subsequent 14:09 retry failed identically before rjob creation. PV has
  no evaluator output, while matched-SFT remains the only complete new
  comparison. Training code compilation, adaptive/finalizer shell syntax,
  focused contract tests (`13 passed`), and storage audit (`38G` total) are
  clean. The active monitors retain the next retry deadline and no new GPU
  allocation has occurred.

## Resume refresh (2026-08-29 14:12 HKT)

- Heartbeats through 14:12 remain healthy. The PV retry marker is still
  `PENDING_CONTROL_PLANE`; no evaluator shard, 512-row output, or paired
  bootstrap file has appeared. The PV training contract remains
  `finished/10 steps/20 updates`, and matched-SFT remains complete only as a
  control. No queue transition is inferred from the unavailable API.

## Resume refresh (2026-08-29 15:14 HKT)

- Rechecked the dnacoding endpoint: `h.pjlab.org.cn` still does not resolve,
  and `rjob list --namespace=ailab-dnacoding` fails before contacting the API.
  R35 has no output directory, checkpoint, or evaluator result; BA-FEPO has
  not been submitted.
- The late monitor continues healthy heartbeats through 15:14. The R35 state
  lock remains held by the existing idempotent submitter, but no GPU task is
  visible from this workspace and no task transition is inferred while the
  API is unavailable.
- The R35 submitter now probes the namespace before invoking its heavyweight
  wrapper. A controlled DNS-failure test records `control_plane_unavailable`
  and confirms that no submit wrapper is called. This avoids repeated
  packaging during the outage without changing the registered experiment.
- The readiness table was synchronized to the current evidence: R21--R29
  are completed rejected screens, R30 is a failed training gate, PV is closed
  by its training support gate, R35 is the sole next screen, and BA remains
  conditional. Focused static tests remain green (`10 passed`); storage is
  approximately 38G and no files were written under `PixVL_ailab`.

## Resume refresh (2026-08-29 15:18 HKT)

- Audited all direct SAMTok training submitters. The shared TB-GPPO, ES-GR-
  CPPO, GR-CPPO, active-set, AM-CPPO, boundary-credit, projector, and
  standalone-SFT entry points now reject fewer than 5,000 data rows and
  one-step/smoke job or config names. This closes a historical bypass of the
  current 5,120-row/10-step rule; it does not alter any registered candidate.
- The guard regression suite passed (`12 passed`) along with shell syntax and
  diff checks. Evaluation-only submitters remain separate and retain the
  fixed 8 -> 6 -> 4 -> 2 -> 1 GPU ladder.

## Resume refresh (2026-08-29 15:23 HKT)

- The control-plane recheck still fails at DNS resolution; late-monitor
  heartbeats continue and neither R35 nor BA has an output directory.
- Direct data audit confirms the registered FEPO training file has exactly
  5,120 nonempty rows, while the historical `grefcoco_selective_train_256`
  file has 512 rows. The new submit guards reject the latter, preventing an
  accidental sub-5k training job.
- R35 and BA configs both load as `expected_rows=5120`, `max_steps=10`, and
  `K=4`. Focused guard/candidate tests pass (`16 passed`); storage remains
  approximately 38G with no PixVL_ailab writes.

## Resume refresh (2026-08-29 15:30 HKT)

- Added a runtime training-budget validator used by every active SAMTok
  training submitter. It counts actual nonempty JSONL rows and loads the
  submitted config to verify `expected_rows >= 5000` and
  `optimizer.max_steps >= 10`; failures occur before any rjob packaging.
- The registered R35 config passes the validator (`actual_rows=5120`,
  `configured_rows=5120`, `configured_steps=10`). Historical 512-row and
  one-step/smoke entries are rejected. Static budget/candidate tests pass
  (`13 passed`), with no change to R35/BA experiment definitions.

## Resume refresh (2026-08-29 15:29 HKT)

- The control-plane check still fails at DNS and no R35/BA output exists.
  Late-monitor heartbeats continue through 15:29 without a GPU allocation.
- A complete script audit found no active `submit_samtok_*.sh` rjob entry
  lacking either the runtime budget validator or the explicit legacy PixVL
  disable. Both R35 and BA pass the validator with 5,120 rows and 10 steps.

## Resume refresh (2026-08-29 16:27 HKT)

- The dnacoding control plane briefly returned a successful authenticated
  response: an exact R35 name query found zero matching jobs. A subsequent
  local query encountered the same DNS resolution failure again, so queue
  reachability is currently intermittent rather than reliably restored.
- The registered `r35_submit/.lock` remains held and its `runner.log` and
  `submit.log` contain no successful submission record. No R35 output or
  checkpoint directory exists. No manual or duplicate submission was made.
- The fixed training-only coverage audit found 5,120 rows balanced 50/50;
  among 2,560 target-present pairs, small/thin/boundary-hard each contain 640,
  with 150 overlapping all three. Query-only existence prediction is 0.6602
  on the fixed 512-row holdout, while path and phrase shortcuts are 0.5.
  This is diagnostic evidence only and does not alter the candidate order.
- Focused contract tests remain green (14 passed with the login-node
  `PYTHONPATH=FARO/Sa2VA` setup); Torch-dependent runtime tests remain
  worker-only. Storage remains approximately 38G and no new writes occurred
  under `PixVL_ailab`.

## Resume refresh (2026-08-29 16:36 HKT)

- The registered R35 submit path was re-entered only through its existing
  `r35_submit/.lock`; it performed control-plane probes and recorded repeated
  `control_plane_unavailable` results. No submit wrapper was invoked and no
  task/checkpoint/output was created.
- An exact queue query from the control-plane audit briefly returned zero R35
  jobs, but the same query is currently back to DNS failure. The finalizer
  lock remains held by the long-lived monitor. No lock file was removed or
  bypassed, and no duplicate job was submitted.
- The current patched finalizer source is ready to transition directly from
  the closed PV training gate to R35 once the stale monitor state is released
  by its owner. BA remains blocked behind R35 as preregistered.

## Resume refresh (2026-08-29 16:42 HKT)

- R35 was successfully created by the existing lock-protected finalizer after
  the control plane recovered. Submission evidence is in
  `logs/screen_monitor/finalize/r35_submit/submitted` and records the platform
  name `dna-fepo-safe-visual-interface-10step-2g-178-557d0` (source name
  `dna-fepo-safe-visual-interface-10step-2g-1787992814`).
- The submit log confirms the runtime budget validator passed with
  `actual_rows=5120`, `configured_rows=5120`, `configured_steps=10`, and the
  SAMTok anchor validation passed. No R35 output/checkpoint has appeared yet;
  no worker validity or queue-state claim is made until an authenticated
  status query succeeds.
- The outer monitor continues emitting only state heartbeats. BA remains
  unsubmitted and conditional on a complete R35 closure.

## Resume refresh (2026-08-29 16:55 HKT)

- R35 is now conclusively closed at the worker gate. The submitted job
  `dna-fepo-safe-visual-interface-10step-2g-178-557d0` failed only because
  `active_set_risk_gate_passed=false`; its visual update moved the sentinel
  margin from `-4.5625` to `-4.8125`, beyond the fixed budget. Effective
  support, grammar, trajectory diversity, gradients, and representation
  checks passed. No promotable checkpoint or holdout evaluation exists.
- The stale finalizer processes were terminated by their owner using SIGTERM;
  no lock files were deleted or bypassed. The patched state machine remains
  the only permitted transition path.
- BA-FEPO submission is now eligible and its lock-protected submitter is
  running. It is currently retrying `rjob list` because the submit host has
  intermittent DNS; no BA task or output has been observed yet. No duplicate
  BA process was started.

## Resume refresh (2026-08-29 16:48 HKT)

- R35 worker output is complete and conclusively closed by its registered
  validity gate: `status=failed_validity_gate`, active-set risk failed,
  tail-risk passed, final sentinel margin `-4.8125` versus budget `-4.6125`.
  No R35 holdout was run, and no quality claim is inferred.
- BA-FEPO is now the sole eligible next training arm. Its first submission
  attempt passed the 5,120-row/10-step validator, SAMTok anchor check, and
  positive-tag preparation, but control-plane DNS failed during rjob
  creation. No BA task or checkpoint was created.
- The existing `submit_ba_after_r35_failure.sh` transition is wired into the
  late monitor. It is lock-protected and retries at 300 seconds; AB/BS/NCVI
  remain locked until BA is conclusively closed.

## Resume refresh (2026-08-29 16:58 HKT)

- BA's lock-protected submitter remains the only active next-candidate path.
  Its log shows repeated `control_plane_unavailable` retries; no BA task,
  checkpoint, or training output is present yet. R35 eligibility remains
  conclusively closed by its worker gate.
- No holdout evaluator was launched for R35 or BA. Storage remains below the
  700G cap, and all new artifacts remain under `Faro_ailab/FARO`.

## Resume refresh (2026-08-29 17:00 HKT)

- BA-FEPO configuration validation succeeds with stage
  `fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu`,
  5,120 expected rows, 10 optimizer steps, K=4 rollouts, and
  `boundary_bottleneck_min` aggregation.
- The BA submitter lock is still held and its log continues to show DNS/API
  retry failures. No BA task or output has appeared, so no evaluation has been
  started and no BA quality claim is made.

## Resume refresh (2026-08-29 17:02 HKT)

- BA remains in the single lock-protected retry path. Repeated queue probes
  from this login context fail at DNS before API contact; there is no BA
  `submitted` marker, task, checkpoint, or output directory.
- BA config and transition gates were revalidated after R35 closure. AB/BS/
  NCVI remain unsubmitted by design; no parallel training is permitted while
  BA is unresolved.

## Resume refresh (2026-08-29 17:04 HKT)

- The BA submitter lock remains held and its retry log continues advancing
  through control-plane DNS failures. This confirms the transition process is
  alive; it does not imply a BA task exists.
- No BA output, checkpoint, or `submitted` marker is present. The next
  permitted action remains a single BA submission after a successful API
  probe, followed by worker-gate validation before any holdout evaluation.

## Resume refresh (2026-08-29 17:07 HKT)

- BA has still not reached the API: its submit log advances through DNS
  failures and no `submitted` marker or output exists. The lock remains held,
  so no second submitter was started.
- For context only, the completed PV diagnostic passed its training gates with
  final sentinel margin `-3.375` and zero final tail violations, while R35
  failed at `-4.8125`. These are training diagnostics and do not predict BA
  quality; BA remains the next independent test rather than a stacked
  visual/null objective.

## Resume refresh (2026-08-29 17:10 HKT)

- BA remains uncreated. The lock-protected submitter has advanced through
  fresh DNS failures up to 17:07, while no BA `submitted` marker, task,
  checkpoint, or output directory exists.
- BA transition tests and config validation remain green. No evaluator was
  launched and no later candidate was submitted while this control-plane
  condition persists.
## Resume refresh (2026-08-29 17:20 HKT)

- BA-FEPO was submitted exactly once after the R35 worker-gate failure:
  requested job `dna-fepo-boundary-bottleneck-paired-view-10step-2g-1787994810`,
  normalized queue name `dna-fepo-boundary-bottleneck-paired-view-10s-c35ed`.
- BA finished its registered 5,120-row/10-step/K=4 training contract with
  `validity_gate.passed=true`, effective support 1.0, positive gradients,
  changed epoch-2 ratios, grammar validity, and both sentinel/tail-risk gates
  passing. Its paired-view joint-positive fraction remains sparse, so this is
  a worker-validity result only; no quality claim is inferred.
- The late evaluator entered the required 8-GPU stage for exactly 512 rows.
  It is protected by the existing 8 -> 6 -> 4 -> 2 -> 1 ladder and 20,000
  paired bootstrap requirement. The first control-plane attempt encountered
  DNS; no partial evaluator output is treated as a result.
- Added the next isolated BS-FEPO hypothesis: fixed 50/25/25 ordinary/thin/
  boundary-hard sampling with unchanged R18 credit. It has a SAMTok-only
  config, positive-tag wrapper, deterministic disjoint schedule, and static
  tests; it remains unsubmitted until BA's full holdout decision.
