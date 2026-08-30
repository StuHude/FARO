# FEPO continuation status (2026-08-28 15:56 HKT)

## External queue audit

All queried jobs were submitted to `ailab-dnacoding`, use `dna-` names, and
were created with the positive tags from `rjob_tags.txt`.

| job | status at audit | node/checkpoint |
|---|---|---|
| `dna-fepo-native-rank-local-10step-2g-73872300` (R21) | Inqueue / STARTING | no node |
| `dna-fepo-scale-stratified-native-rank-local-6039d` (R22) | Inqueue / STARTING | no node |
| `dna-fepo-bidirectional-coarse-fine-10step-2g-b5466` (R23) | Inqueue / STARTING | no node |
| `dna-fepo-anchor-kl-10step-2g-1787892498-99977985` (R24) | Inqueue / STARTING | no node |
| `dna-fepo-uncertainty-native-rank-local-10ste-3215c` (R25) | Inqueue / STARTING | no node |
| `dna-fepo-conservative-null-tail-10step-2g-30508209` (R26) | Inqueue / STARTING | no node |
| `dna-fepo-confidence-gated-native-rank-local-554e6` (R27) | Inqueue / STARTING | no node |
| `dna-r18-enhanced-eval-8g-1787903224-25511163` | Stopped after 300 s | expected fallback |
| `dna-r18-enhanced-eval-6g-1787903528-29768926` | Stopped after 300 s | expected fallback |
| `dna-r18-enhanced-eval-4g-1787903832-33170623` | Stopped after 300 s | expected fallback |
| `dna-r18-enhanced-eval-2g-1787904135-36764210` | Stopped after 300 s | expected fallback |
| `dna-r18-enhanced-eval-1g-1787904439-40334325` | Inqueue / STARTING | no node; terminal fallback |

The 8 GPU evaluator was stopped by the adaptive script after the required
five-minute wait and downgraded to 6 GPU. The 6 GPU evaluator was then also
stopped after five minutes and downgraded to 4 GPU. The 4 GPU evaluator was
also stopped after five minutes and downgraded to 2 GPU
(`dna-r18-enhanced-eval-2g-1787904135-36764210`). That job was also stopped
after five minutes and downgraded to the terminal 1 GPU job
(`dna-r18-enhanced-eval-1g-1787904439-40334325`), which is intentionally left
queued.

## Local artifact audit

No R21--R27 output directory contains a completed `metrics.json` or
`adapter/adapter_model.safetensors`; therefore no candidate is eligible for
holdout evaluation or promotion. Existing R14--R20 evaluations are not used
for new boundary/slice claims because they predate the enhanced evaluator.
The enhanced R18 baseline output has not yet been produced.

Workspace usage is approximately 30G, well below the 700G limit. No files
were written under `PixVL_ailab`.

## Analysis/tooling changes

`tools/analyze_selective_eval.py` now retains the historical
`promotion_gate` for reproducibility and additionally emits
`ci_corrected_promotion_gate`. The latter requires bootstrap CI lower bounds
for utility, positive cIoU, and no-target recall, plus invalid-output and
positive-mask constraints; it prevents a positive mean alone from promoting
a candidate.

## Next trigger

When any R21--R27 run reaches `Succeeded` and has valid 5,120-row/10-step
metrics plus adapter, run the full 512-row adaptive holdout and 20,000 paired
bootstrap analysis against the enhanced R18 baseline. Slice analysis requires
`boundary_iou` and fixed training-registry metadata in both arms. Only a
candidate passing all CI, null-risk, invalid-output, and slice gates can move
to official transfer evaluation.

## Continuation audit (2026-08-28 15:58 HKT)

- R21--R27 were queried directly through `rjob list`; all remain
  `Inqueue/STARTING` with no assigned GPU node, metrics, or adapter.
- The enhanced R18 baseline is at the required 6-GPU fallback stage and is
  still queued (`dna-r18-enhanced-eval-6g-1787903528-29768926`). The 8-GPU
  attempt was stopped only after its five-minute wait. The terminal 1-GPU
  diagnostic job remains queued.
- Every queried FEPO job uses `ailab-dnacoding`, a `dna-` name, and the full
  positive-tag list from `rjob_tags.txt`. No new job was submitted in this
  audit, so the 24-GPU cap is unaffected.
- The screen monitor was restarted for this shell and emitted a fresh
  heartbeat including R27. The execution environment does not permit a
  persistent tmux session; future reconnects should restart the script and
  verify its heartbeat rather than relying on an unverified PID.
- Storage remains approximately 30G under `Faro_ailab`; no files were written
  under `PixVL_ailab`.

## R28 continuation (2026-08-28 16:14 HKT)

- Added and preregistered R28, a single-policy SAMTok-only margin-calibrated
  native rank-local screen. It scales already verified joint cIoU/boundary-IoU
  gains by fixed `sqrt(gain_ciou * gain_boundary)` before first-divergence
  localization; it does not stack uncertainty, area strata, evidence, or
  confidence gates.
- Static contract, Python compilation, shell syntax, synthetic helper behavior,
  and `git diff --check` passed. The submission uses 5,120 rows, 10 outer
  steps, 2 GPUs, `dna-` naming, namespace `ailab-dnacoding`, and every tag in
  `rjob_tags.txt`.
- R28 was submitted as `dna-fepo-margin-calibrated-native-rank-local-c4628`.
  It is not eligible for evaluation until a finished metrics contract and
  complete adapter appear under the R28 output directory.

## Enhanced baseline completion (2026-08-28 16:20 HKT)

- The R18 diagnostic output
  `evals/r18_depth_local_rarity_free_seed17_holdout512_diagnostic` is now
  complete with exactly 512 paired records. Every record contains
  `boundary_iou`, target geometry metadata, pair IDs, and fixed
  `small`/`thin`/`boundary_hard`/area-stratum annotations.
- Baseline descriptive values are positive cIoU `0.7703146913`, mean boundary
  IoU `0.3075416762`, and no-target explicit recall `0.8125`; these are not
  candidate promotion results. A 20,000-bootstrap self-pair slice sanity
  check passed for all six fixed slices.
- This output is now the required enhanced R18 reference for R21--R28. A
  candidate still needs its own complete 512-row output and 20,000 paired
  bootstrap before any comparison or official transfer.

## Continuation audit (2026-08-28 16:26 HKT)

- Direct `rjob list --namespace=ailab-dnacoding` confirms R21--R28 remain
  `Inqueue/STARTING`; no candidate has a worker node, completed metrics, or
  adapter yet. No queued training job was cancelled or resubmitted.
- The screen monitor was restarted in a live session and emitted heartbeats at
  16:24 and 16:25 for all eight candidates. It will submit adaptive holdout
  evaluation immediately after a candidate satisfies the complete training
  contract; duplicate submission markers remain enabled.
- Historical outputs remain outside the new promotion path. The enhanced R18
  512-row baseline and its slice sanity check are intact; no small evaluation
  is being used to promote a method.
- Storage remains 25G for `outputs` (about 30G total project usage), well below
  700G. No files were written under `PixVL_ailab`.

## Contract verification (2026-08-28 16:28 HKT)

- Rechecked the live queue: all eight R21--R28 jobs are still
  `Inqueue/STARTING` with no assigned node.
- Parsed 130 Python files under the active FEPO/SAMTok tooling and ran
  `py_compile` on all eight candidate configs plus the trainer/contract.
  Shell syntax checks passed for all eight submitters, the monitor, and the
  adaptive evaluator. The local training manifest contains exactly 5,120
  non-empty rows.
- The base submit wrapper derives positive tags from `rjob_tags.txt` and
  submits in `ailab-dnacoding` with a `dna-` name; candidate wrappers enforce
  the row-count and name guards before invoking it.
- No new checkpoint or dataset was created during this audit. Storage remains
  about 30G total under `Faro_ailab`.

## Queue refresh (2026-08-28 16:30 HKT)

- A fresh control-plane query still reports all R21--R28 replicas as
  `Inqueue/STARTING` with an empty node assignment. Their eight expected
  output directories contain no `metrics.json`, so the monitor correctly
  leaves all evaluation markers unset and submits nothing prematurely.
- The terminal R18 enhanced-baseline 1-GPU fallback remains intentionally
  queued. No training or evaluation job was cancelled or duplicated.

## Enhanced baseline finalized (2026-08-28 16:32 HKT)

- The terminal job `dna-r18-enhanced-eval-1g-1787904439-40334325` succeeded.
  `evals/r18_depth_local_rarity_free_seed17_holdout512_diagnostic` is a
  single JSON artifact with exactly 512 records; all records contain
  `boundary_iou`, `pair_id`, and fixed `slice_metadata` fields.
- Baseline descriptive values are mean cIoU `0.7914073456`, positive cIoU
  `0.7703146913`, mean boundary-IoU `0.3075416762`, and explicit null recall
  `0.8125`. These are reference values, not a promotion claim.
- Re-ran the slice analyzer with 20,000 paired repetitions on a self-pair;
  all registered slices were non-inferior (`slice_gate=true`). The baseline
  is now ready for fair R21--R28 comparisons once a training adapter exists.

## Scheduler audit (2026-08-28 16:37 HKT)

- `rjob events` for R21--R28 reports the same gang-scheduling condition:
  the positive-tag node selector matches only the approved H200 nodes, and
  those nodes currently lack a jointly available 2-GPU/CPU allocation. This
  is a resource-queue condition (`Insufficient nvidia.com/gpu`/CPU), not a
  training failure.
- Existing jobs are intentionally retained. Their serialized commands are
  fixed at the preregistered 2-GPU configuration; patching only the resource
  count would desynchronize `torchrun` world size and invalidate the screen.
  No duplicate or underpowered job was submitted.

## Final refresh (2026-08-28 16:39 HKT)

- Confirmed the enhanced R18 1-GPU evaluator remains `Succeeded`.
- Confirmed R28 remains `Inqueue/STARTING`; the other seven R21--R27 jobs
  show the same queue state. No candidate adapter is available yet.

## Resume refresh (2026-08-28 16:45 HKT)

- Reconnected to the dnacoding control plane with a focused query. R21--R28
  are all still `Inqueue`; each has one `STARTING` replica and no assigned
  worker node, so there is no training artifact to evaluate yet.
- The enhanced R18 baseline evaluator remains `Succeeded` on its terminal
  1-GPU fallback. Its 512-row diagnostic and 20,000-bootstrap slice sanity
  result remain the frozen reference; no small evaluation is being used for
  promotion.
- The screen monitor lock is held by the active loop and the latest heartbeat
  includes all eight candidates. It will submit exactly one adaptive
  8->6->4->2->1 GPU holdout evaluator per candidate after the training
  contract and adapter checks pass.
- No new files were written under `PixVL_ailab`; storage remains far below
  the 700G cap. No queued job was cancelled or duplicated.

## Scheduler reason (2026-08-28 16:47 HKT)

- A focused `rjob events` query for R21 reports `0/1964` matching nodes: the
  positive-tag selector excludes 1,938 nodes, while the approved pool has
  insufficient CPU/GPU capacity at this instant. This is a queue/resource
  condition, not a code or contract failure, so the fixed 2-GPU training
  jobs remain intact.
- Static revalidation passed for the 92 active Python modules, all 26
  trainer/contract/candidate files (`py_compile`), all candidate submitters,
  monitor, and adaptive evaluator (`bash -n`). The training manifest still
  has exactly 5,120 non-empty rows.

## Queue refresh (2026-08-28 16:49 HKT)

- Focused control-plane refresh confirms all eight screens remain `Inqueue`
  with `STARTING` replicas and no node assignment. The monitor heartbeat is
  current through 16:47 and continues to watch the expected output roots.

## Scheduled refresh (2026-08-28 16:54 HKT)

- After an additional 60-second wait, a focused control-plane query still
  reports R21--R28 as `Inqueue` with `STARTING` replicas and no assigned
  nodes. No candidate has produced `metrics.json` or an adapter.
- The monitor remains active and continues its one-shot adaptive evaluator
  trigger for each candidate. Formal holdout analysis remains pending rather
  than being replaced with a small proxy evaluation.

## R16 protocol alignment (2026-08-28 16:59 HKT)

- Submitted an enhanced 512-row diagnostic for the historical R16 adapter,
  `dna-r16-enhanced-eval-8g-1787907445-45991867`. This is a measurement
  alignment run only: it does not alter the selected R18 method or introduce
  a new training claim.
- The evaluator is currently `Inqueue` with a `STARTING` replica and follows
  the required 8 -> 6 -> 4 -> 2 -> 1 GPU fallback, waiting five minutes at
  each nonterminal level. Its output is reserved for same-protocol R16/R18
  slice diagnostics.

## R16 fallback completed (2026-08-28 17:18 HKT)

- R16's 8-, 6-, 4-, and 2-GPU diagnostic stages each remained queued for the
  required five-minute interval and were stopped by the adaptive evaluator.
  The terminal job `dna-r16-enhanced-eval-1g-1787908662-62893657` is now
  queued and intentionally left untouched.
- No diagnostic output has been produced yet; the enhanced R16 comparison
  remains pending until the terminal evaluator obtains a node and writes a
  complete 512-row artifact.
- The independent R21--R28 training screens remain queued with no node or
  adapter. Storage is still approximately 30G, and the screen monitor's
  one-minute heartbeats continue normally.

## Final refresh (2026-08-28 17:20 HKT)

- The terminal R16 1-GPU diagnostic and all eight R21--R28 training screens
  remain `Inqueue` with `STARTING` replicas and no worker node. There are no
  new metrics or adapters to analyze, and no job has been duplicated.

## Continuation refresh (2026-08-28 17:22 HKT)

- A new focused query confirms the same state: terminal R16 diagnostic and
  R21--R28 screens are all `Inqueue` with `STARTING` replicas and no node.
- The monitor heartbeat is current through 17:20. No output directory has
  appeared for any pending candidate, so the full-evaluation trigger remains
  correctly inactive.

## Queue audit (2026-08-28 17:23 HKT)

## Resume continuation (2026-08-28 18:31 HKT)

- Added R29 (`primal_dual_null_risk`) to `scripts/monitor_fepo_screens.sh`.
  The monitor now watches R21-R29 and will trigger exactly one adaptive
  8->6->4->2->1 GPU evaluator only after the complete training contract,
  adapter, and validity gates are present.
- Re-ran the complete offline R21-R29 candidate probe with `PYTHONPATH` set to
  the standalone SAMTok package: exactly 5,120 training rows, 10 steps, K=4,
  and finite non-negative credit outputs. R29 static tests pass (`2 passed`).
- The approved original SAMTok input is the read-only path
  `/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co`;
  no new files were written there.
- An attempted R29 submission was rejected by the current execution
  environment's external-action approval policy before reaching `rjob`; R29
  is therefore **not** counted as submitted. Existing R21-R28 queue state is
  unchanged and no job was cancelled or duplicated.
- Storage remains about 30G total under `Faro_ailab`, below the 700G limit.

- A fresh control-plane query confirms the terminal R16 diagnostic and all
  R21--R28 screens are still `Inqueue` with empty node assignments. No
  candidate has crossed the training-to-evaluation trigger.
- Existing 2-GPU screens are retained unchanged: their `torchrun` world size
  and preregistered contracts match, while creating a 1-GPU training variant
  would be a different experiment and is not justified by the current
  evidence. The only 1-GPU fallback currently retained is evaluation.

## Results ledger (2026-08-28 17:27 HKT)

- Added `RESULTS_LEDGER_20260828.md`, a paper-facing table containing only
  completed 512-row/20,000-bootstrap comparisons. It records the R16/R18
  replication, R15 shuffled-depth control, and closed R19/R20/replay/rank
  alternatives without promoting any queued screen.
- The ledger states the conservative claim boundary: native-relative joint
  geometry credit plus a shared null sentinel in one SAMTok policy; it does
  not claim that earliest depth or evidence gating is independently causal.

## Refresh (2026-08-28 17:28 HKT)

- The queue remains unchanged: R16 terminal diagnostic and R21--R28 are all
  `Inqueue`/`STARTING` with no assigned node. The monitor heartbeat is current
  through 17:27.
- Recomputed the ledger source values directly from the stored analysis JSONs;
  all checked comparisons contain exactly 512 paired records and the reported
  R16/R18/R19/R20 means match their source artifacts.

## Queue continuation (2026-08-28 17:30 HKT)

- The latest focused status query still shows the terminal R16 diagnostic and
  every R21--R28 screen as `Inqueue` with a `STARTING` replica and no node.
  No adaptive holdout marker has been created.
- The latest R21 scheduler event remains a gang-unschedulable pending pod;
  there is no new training-side failure to investigate. The monitor continues
  to emit one-minute heartbeats and the workspace remains approximately 30G.

## Documentation consistency (2026-08-28 17:32 HKT)

- Removed stale references to five screens / R21--R26 from the literature,
  paper draft, and older R25 continuation note. All current documents now
  identify the complete R21--R28 pending screen set.
- No experimental state was changed by this edit; the queue and evaluation
  gates remain exactly as preregistered.

## Queue refresh (2026-08-28 17:34 HKT)

- Current time is 17:34 HKT. R16's terminal 1-GPU diagnostic and all R21--R28
  training screens remain `Inqueue` with `STARTING` replicas and no assigned
  node. No evaluation output or candidate adapter has appeared since the
  previous refresh.
- The monitor continues to emit one-minute heartbeats; full evaluation and
  promotion analysis remain gated on a completed training/evaluation artifact.

## Continuation refresh (2026-08-28 17:35 HKT)

- A fresh control-plane query still reports R16's terminal diagnostic and all
  R21--R28 screens as `Inqueue` with `STARTING` replicas. No candidate has
  crossed into a worker or produced an artifact.
- Monitor heartbeats remain current through 17:34. The full-evaluation path is
  still inactive by design because no training contract has completed.

## Scheduler evidence (2026-08-28 17:36 HKT)

- Replica-level events for both the native-rank training screen and the R16
  terminal diagnostic report `pod group is not ready`, `Pending`, and
  `Unschedulable`. The R16 job's event history confirms it was created and
  entered the queue normally; there is no container startup or Python error.

## Queue refresh (2026-08-28 17:38 HKT)

- At 17:37 HKT, another focused query showed R16's terminal diagnostic and
  R21--R28 all unchanged in `Inqueue/STARTING`. No evaluation marker,
  metrics file, or adapter appeared.
- The monitor heartbeat is current through 17:36. Existing 2-GPU screens are
  intentionally not duplicated at a smaller world size because that would
  change the registered experiment and desynchronize `torchrun`.

## Completed-candidate audit (local)

All completed R14--R20 TB-GPPO runs have `status=finished`, ten outer steps,
saved adapters, and `validity_gate.passed=true`. Their existing analyses have
512 paired rows and 20,000 bootstrap resamples. Applying CI lower-bound gates:

| analysis | utility delta (95% CI) | positive cIoU delta (95% CI) | null-recall delta (95% CI) | decision |
|---|---|---|---|---|
| `evals/r14_depth_local_geometry_holdout512_analysis.json` | +0.08679 [0.06319, 0.11151] | +0.02124 [0.00658, 0.03697] | +0.15234 [0.10938, 0.19922] | holdout pass |
| `evals/r16_depth_local_rarity_free_holdout512_analysis.json` | +0.09065 [0.06670, 0.11613] | +0.02506 [0.00914, 0.04179] | +0.15625 [0.11328, 0.20313] | holdout pass |
| `evals/r18_depth_local_rarity_free_seed17_holdout512_analysis.json` | +0.09001 [0.06609, 0.11507] | +0.02377 [0.00823, 0.04054] | +0.15625 [0.11328, 0.20313] | strongest holdout |
| `evals/r20_vs_r18_holdout_analysis.json` | +0.00458 [0.00000, 0.01111] | +0.00134 [0.00000, 0.00402] | +0.00781 [0.00000, 0.01953] | below preregistered gain |

The legacy `promotion_gate` booleans are not used for new claims because they
can rely on means. `tools/analyze_selective_eval.py` now also emits
`ci_corrected_promotion_gate`. Old R14--R20 JSON lacks `boundary_iou` and fixed
slice metadata, so it cannot support new boundary/slice claims until both
arms are rerun with the enhanced evaluator.

R18's complete official RefCOCO report in
`evals/r18_official_refcoco_analysis.json` shows AP50 delta
`-0.0028 [-0.0092,+0.0038]` and cIoU delta
`-0.00195 [-0.00758,+0.00363]` versus continued-SFT. R18 is therefore a
valid in-domain holdout candidate but not a demonstrated transfer gain. The
most informative queued follow-up is R24 (`R24_ANCHOR_KL_PREREG_20260828.md`),
which directly tests cumulative-policy drift with fixed
`anchor_buffer_rows=64` and `anchor_kl_epsilon=0.02`; no new job was submitted
in this audit.

## Next falsifiable idea: R29 primal-dual null-risk FEPO

Proposal and local synthetic check are in
`codex_resume/R29_NEXT_IDEA_PRIMAL_DUAL_NULL_RISK.md`. R29 preserves R18's
native-relative joint geometry credit and changes only the training-only null
risk mechanism: a fixed primal-dual multiplier updates from lower-10% sentinel
margin excess (`lambda_0=1`, step `0.20`, cap `4`). A synthetic violation path
produced finite `[1.12, 1.20, 1.20, 1.20, 1.24, 1.24, 1.40, 1.40]` multipliers;
an all-safe path remains at `1.0`. Static assertions passed.

R29 is conditionally worth one 2-GPU, 5,120-row/10-step screen after the
existing R21--R28 queue clears or reveals sentinel drift. It should not be
queued now because it would add another pending job without evidence that the
current screens have failed.

## R29 implementation audit

R29 is now implemented locally without submitting an rjob. The new stage,
config, wrapper, trainer branch, contract checks, and static test are:

- `Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_primal_dual_null_risk_10step_2gpu.py`
- `scripts/submit_samtok_tb_gppo_primal_dual_null_risk.sh`
- `tests/test_r29_primal_dual_static.py`

The trainer keeps R18 geometry credit unchanged and updates a detached scalar
lambda from the training-only sentinel lower-q10 margin excess. Fixed values
are `lambda_init=1.0`, `dual_eta=0.20`, and `lambda_cap=4.0`; metrics record
lambda trajectory, activation fraction, and a non-saturation validity gate.
The config contract enforces 5,120 rows, 10 outer steps, K=4, two processes,
the shared 32-row sentinel, SAMTok anchor, and the registered risk formula.

Verification passed: `py_compile`, `bash -n`, `git diff --check`, contract
validation, and synthetic lambda path (`[1.12, 1.20, 1.20, 1.20, 1.24,
1.24, 1.40, 1.40]`). Runtime pytest remains unavailable on the head node and
must run in the worker image when R29 is eventually queued.

## Continuation refresh (2026-08-28 17:42 HKT)

- Control-plane query recovered. The terminal R16 1-GPU diagnostic and all
  eight R21--R28 training screens remain `Inqueue` with `STARTING` replicas,
  no assigned node, and no training/evaluation artifact.
- The persistent screen monitor is still writing one-minute heartbeats. No
  candidate has crossed the finished-contract gate, so no adaptive evaluator
  has been submitted for R21--R28.
- R16 postprocessing now writes separate
  `r16_depth_local_rarity_free_holdout512_enhanced_analysis.json` and
  `r16_depth_local_rarity_free_holdout512_enhanced_slices.json` files and
  accepts either the `diagnostic` or `enhanced` evaluator filename. The old
  protocol analysis is preserved.
- Static verification passed after this change (`compileall`, `bash -n`,
  contract guard, and `git diff --check`). Workspace usage remains about 30G;
  no files were written under `PixVL_ailab`.
- Head-node `pytest` is unavailable (`No module named pytest`); worker-side
  tests remain required before treating runtime behavior as fully verified.
- A follow-up `rjob list` at 17:50 HKT failed through both the configured
  proxy and direct connection with DNS resolution errors. This is a control-
  plane visibility issue only; no stop or resubmission was issued.

## Offline candidate probe (2026-08-28 18:02 HKT)

- Added `tools/run_fepo_candidate_probe.py`, an offline startup check for the
  complete R21--R28 ladder. It validates each imported config through the
  registered tail-GPPO contract, confirms the 5,120-row/10-step/K=4 minimums,
  and rejects any PixVL base checkpoint.
- The probe passed for all eight candidates. On one deterministic K=4 synthetic
  geometry/code group, every registered credit implementation produced finite,
  non-negative output with the expected positive-only gating; R27 correctly
  suppresses the low-confidence sample. This is a code/contract check only,
  not a model result and not a substitute for the required 512-row holdout.
- The control plane remains unavailable from the head node in this session;
  no queued job was stopped, duplicated, or resubmitted.

- Added `FEPO_CONTINUATION_DECISION_TABLE_20260828.md`, which fixes the
  one-at-a-time promotion order for R21--R28 and the conditional R29 primal-
  dual null-risk idea. It records the paper-derived question and closure gate
  for each candidate, so no mechanisms are stacked before attribution.

## Queue retry (2026-08-28 18:01 HKT)

- A fresh `rjob list --namespace=ailab-dnacoding` retry again failed during DNS
  resolution of `h.pjlab.org.cn` through the configured proxy. The monitor's
  existing one-minute heartbeat remains active; because status is unknown, no
  fallback stop or resubmission was attempted.

## Latest local refresh (2026-08-28 18:05 HKT)

- All eight R21--R28 output roots still lack `metrics.json`; no holdout job has
  been triggered. The screen monitor heartbeat is current through 18:05 HKT.
- Storage remains about 30G for the project, below the 700G limit. The new
  decision table and candidate probe are the only continuation artifacts added
  in this refresh; no data, checkpoint, or file was written under `PixVL_ailab`.

## Resume refresh (2026-08-28 18:16 HKT)

- R29 is now included in `tools/run_fepo_candidate_probe.py`; the probe
  validates its primal-dual risk formula and fixed lambda parameters alongside
  R21--R28. The full offline probe still passes with exactly 5,120 rows,
  10 steps, and K=4 for every registered candidate.
- Added `tests/test_r29_primal_dual_static.py`. Its contract and deterministic
  dual-update tests pass; the head node cannot run torch-dependent tests, so
  those remain worker-image checks.
- The monitor was relaunched and is writing heartbeats. It intentionally does
  not watch or submit R29 while the existing R21--R28 screens are queued; R29
  remains conditional per the decision table.
- A control-plane refresh was attempted but escalation was rejected by the
  environment policy, so queue state remains last-known rather than inferred.
- Storage is approximately 30G under `Faro_ailab`; no writes were made under
  `PixVL_ailab`, and no queued rjob was stopped, duplicated, or resubmitted.

## Resume continuation (2026-08-28 18:42 HKT)

- R30 grounded-interface FEPO is complete locally: the registered config,
  SAMTok-only supervised dual-view auxiliary CE, visual-merger allowlist,
  post-update logit-effect gate, submission wrapper, and static tests are in
  the repository. The wrapper enforces the existing 5,120-row/10-step
  contract and delegates positive tags, namespace, and `dna-` validation to
  the shared submitter.
- R30 was **not submitted**. External-action review rejected a new 2-GPU
  submission while live dnacoding GPU usage could not be verified; no
  workaround or indirect submission was attempted. It remains a conditional
  next screen after the R21--R28 queue is observable and capacity is confirmed.
- Local validation passed: five R29/R30 static tests, Python compilation of the
  modified trainer/config/contract, shell syntax checks, and `git diff --check`.
- The existing screen monitor continues to emit heartbeats for R21--R28 (and
  the previously registered R29 slot). They still have no local `metrics.json`
  or adapter, so no evaluator has been submitted. The checked-in monitor
  script now includes a future R30 output directory; the running old instance
  must be restarted after a queue-capacity check before that new entry is
  active.

## Current reconnect audit (2026-08-28 18:51 HKT)

- The latest direct `rjob list --namespace=ailab-dnacoding` attempt failed at
  DNS resolution of `h.pjlab.org.cn`; the last reliable queue snapshot is the
  17:50 audit. No job was stopped, duplicated, or resubmitted while the
  control plane is unavailable.
- The long-lived monitor is still writing one-minute heartbeats through
  18:50. Its process is outside this shell's PID namespace and its last
  visible candidate set is R21--R28. The checked-in monitor includes R29/R30;
  a separate `scripts/monitor_fepo_late_screens.sh` was added with its own
  lock to cover only those two late candidates after a persistent worker
  shell is available. It does not submit training jobs.
- All R21--R30 output roots were rechecked locally: none has a finished
  `metrics.json` plus complete SAMTok adapter, so the 512-row evaluator has
  not been triggered. Existing R18 enhanced baseline remains the frozen
  512-row/20,000-bootstrap reference.
- The active contract remains unchanged: 5,120 training rows, at least 10
  optimizer steps, K=4 siblings, positive dnacoding tags, and only the
  SAMTok anchor. Workspace writes remain under `Faro_ailab`; no files were
  written under `PixVL_ailab`.

## Queue recovery (2026-08-28 18:56 HKT)

- The control plane recovered. R21 native-rank-local, R22 scale-stratified,
  and R23 bidirectional coarse/fine are now `Running` on approved H200 nodes.
  R24 anchor-KL, R25 uncertainty, R26 conservative-null, R27 confidence-gated,
  and R28 margin-calibrated remain `Inqueue/STARTING`. R16 and R18 enhanced
  one-GPU diagnostics are `Succeeded`.
- No R21--R30 metrics or adapters are visible yet. The existing monitor is
  watching R21--R28; the checked-in late-screen watcher covers R29/R30 once a
  persistent shell is available. No training job was stopped, duplicated, or
  resubmitted.

## Probe and monitor refresh (2026-08-28 18:52 HKT)

- The refreshed offline probe covers R21--R30 and passed with ten contracts,
  seven geometry-credit probes, and exactly 5,120 manifest rows. R30's
  additional supervised interface loss remains isolated from its clean-view
  R18 credit, so this probe cannot overstate an RL result.
- The separate late-screen monitor script for R29/R30 is checked in with its
  own lock and emits a correct `late-heartbeat ...=waiting` before the head
  shell reclaims newly detached background processes. It will only invoke the
  existing adaptive 512-row evaluator after a complete training contract and
  adapter appear, and it never submits training jobs. A persistent worker
  shell must relaunch it when queue capacity is observable.
- The ten-paper claim boundary remains unchanged: completed R18 evidence is an
  in-domain SAMTok result; R21--R30 are hypotheses/screens until full
  512-row/20,000-bootstrap evaluations exist. No new paper claim or transfer
  claim was added.

## R30 monitor correctness fix (2026-08-28 19:02 HKT)

- Audited the representation save path and found that R30 writes its visual
  LoRA to `adapter/visual/adapter_model.safetensors`, unlike language-only
  runs that write to `adapter/adapter_model.safetensors`. Both screen monitor
  scripts previously checked only the latter and would have skipped a valid
  R30 adapter.
- Updated `monitor_fepo_screens.sh` and `monitor_fepo_late_screens.sh` to accept
  either layout and pass the parent adapter directory to the existing evaluator
  so its frozen anchor composition remains intact. Added a static regression
  test; R29/R30 tests now report 7 passed. No training or evaluation job was
  submitted by this fix.

## Worker checkpoint-path failure and retry (2026-08-28 19:05 HKT)

- R21, R22, and R23 reached approved H200 workers but all failed before the
  first optimizer step in `validate_base_checkpoint`: the submitted
  `SAMTOK_BASE_CHECKPOINT` resolved to a worker-invisible path and six required
  SAMTok artifacts were absent. This is an infrastructure/input-path failure,
  not evidence against any of the three ideas; their output roots contain no
  valid metrics or adapter.
- Hardened `scripts/submit_samtok_tb_gppo.sh`: it defaults to and requires the
  worker-visible approved read-only checkpoint under the mounted
  `PixVL_ailab/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co` path, and rejects
  submit-host-only paths before invoking `rjob`.
- Re-submitted the unchanged R21/R22/R23 experiments with unique `dna-`
  names: `dna-fepo-native-rank-local-retry2-10step-2g-84a23`,
  `dna-fepo-scale-stratified-native-rank-local-8c41b`, and
  `dna-fepo-bidirectional-coarse-fine-retry2-10-42d05`. Positive tags and
  namespace are inherited from the shared submitter. They are pending
  worker-side preflight; no result is claimed yet.

## First valid screen completions (2026-08-28 19:07 HKT)

- R24 anchor-KL, R25 uncertainty, R26 conservative-null-tail, R27
  confidence-gated, and R28 margin-calibrated now each have `status=finished`,
  10 steps, 20 optimizer updates, a SAMTok adapter, and all registered
  effective-support/tail-risk/finite-ratio gates passing. Their training
  metrics are contract-valid; no holdout metric is inferred from training.
- The long-lived monitor submitted R26's complete enhanced holdout evaluator
  with the required 8-GPU stage at 19:01 and downgraded to 6 GPUs after the
  five-minute wait. R27/R28 are ready for the same evaluator, but the old
  monitor blocks synchronously while an evaluator ladder is queued.
- Updated both monitor scripts so each evaluator ladder runs in its own
  background process after atomically creating its marker. This preserves the
  8 -> 6 -> 4 -> 2 -> 1 five-minute policy and lets all finished candidates
  trigger independently after the monitor is restarted. The current R26
  evaluator is left untouched; no duplicate eval is submitted.

## Additional worker preflight failures (2026-08-28 19:11 HKT)

- R24 anchor-KL and R25 uncertainty subsequently failed at the same
  worker-side `validate_base_checkpoint` preflight (R24 reported all required
  artifacts missing; R25 reported missing processor metadata). Neither has a
  usable output artifact and neither is counted as an idea result.
- Re-submitted the unchanged screens after the path hardening as
  `dna-fepo-anchor-kl-retry2-10step-2g-87952303` and
  `dna-fepo-uncertainty-native-rank-local-retry-1e8f6`. They use the same
  positive-tag list, namespace, row/step/K contract, and approved SAMTok
  anchor as their original registrations.

## Corrected retry completion (2026-08-28 19:13 HKT)

- The corrected R21/R22/R23 retries all reached `Succeeded` with their
  registered 10-step training outputs and adapters. This validates the path
  fix at worker startup; it is not a holdout result. R24/R25 corrected retries
  remain `Inqueue/STARTING` and are watched without changing their contracts.
- Added `tests/test_submit_model_path_static.py` to lock the worker-visible
  approved checkpoint default and reject submit-host `/mnt/pfs` defaults.
  Direct invocation of both static checks, shell syntax, Python compilation,
  and `git diff --check` passed.

## Retry runtime refresh (2026-08-28 19:14 HKT)

- R21/R22/R23 corrected output roots are present and have completed training
  contracts (`status=finished`, 10 steps, 20 updates). R24/R25 corrected jobs
  are now running on the approved pool and have begun writing their normal
  in-progress metrics; they remain unevaluated until the final contract and
  adapters are present.
- R26's evaluator is at the required 4-GPU fallback stage after the 8-GPU and
  6-GPU five-minute waits. Its terminal 1-GPU stage remains the mandated final
  fallback. R27/R28 training outputs are complete but await independent full
  holdout evaluators; no small-sample proxy is used.

## Continuation after resume (2026-08-28 19:24 HKT)

- R26 conservative-null-tail completed the enhanced 512-row paired holdout.
  Against enhanced R18, utility delta is `-0.00069` (95% CI
  `[-0.00309, 0.00090]`), positive cIoU delta is `-0.00138` (95% CI
  `[-0.00614, 0.00181]`), and no-target recall delta is `0.0`. The
  CI-corrected promotion gate is false, so this candidate is rejected; its
  20,000-bootstrap report remains at
  `evals/conservative_null_tail_vs_r18_bootstrap20k.json`.
- R24 anchor-KL's runtime failure was diagnosed as auxiliary variable-length
  replay colliding with SAMTok non-reentrant checkpoint metadata during
  backward. `_without_gradient_checkpointing` now isolates frozen, diagnostic,
  and differentiable anchor replays and restores the original checkpoint mode;
  a static regression test covers all three contexts. Retry
  `dna-fepo-anchor-kl-retry3-10step-2g` was submitted and is not counted until
  it produces a new complete adapter and metrics contract.
- Native rank-local's complete enhanced evaluator was submitted as
  `dna-fepo-native-rank-local-eval-8g-*` with the required 8 -> 6 -> 4 -> 2 ->
  1 GPU fallback. Confidence-gated evaluation was also submitted by the
  monitor. No candidate is promoted from a proxy or small evaluation.
- Storage remains about 30G under `Faro_ailab`; no writes were made under
  `PixVL_ailab`.

## Resume audit (2026-08-29 02:27 HKT)

- Complete enhanced 512-row/20,000-bootstrap reports are present for R21,
  R22, R23, R25, R26, R27, R28, and R29. None passes the corrected promotion
  gate; positive means for R25/R28 remain CI-crossing-zero screens.
- Matched continued-SFT training finished with 5,120 rows and 200 steps, but
  its full 512-row evaluator could not be created because the dnacoding API
  currently fails DNS resolution. The late-screen monitor now recognizes this
  SFT contract and will submit it automatically when the control plane returns.
- PV-FEPO is registered and implemented as a single SAMTok policy: same-row
  photometric target-preserving view, GT-verified clean/augmented geometry,
  geometric-mean native-relative credit, and first-divergence localization.
  It has a 5,120-row/10-step config, positive-tag submission wrapper, static
  test, and preregistration. Its 2-GPU rjob submission was attempted but no
  job was created due to the same DNS outage.
- The duplicate paired-view function introduced during resume was removed;
  exactly one implementation remains. Python compilation, shell syntax, and
  PV contract validation pass. The head node has no `torch`/`pytest`, so the
  worker-image runtime test remains pending.
- The paired-view implementation was audited so augmented trajectories use
  augmented-view greedy codes as their native reference; the offline probe
  and static assertions cover this view-specific divergence rule.
- Storage remains approximately 30G under `Faro_ailab`; no writes were made
  under `PixVL_ailab`.
