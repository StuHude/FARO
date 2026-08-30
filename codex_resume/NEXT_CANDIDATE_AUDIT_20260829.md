# Next-candidate audit (2026-08-29)

## Status

- No R35 or BA-FEPO training output/checkpoint exists under `outputs/samtok_selective`.
- PV-FEPO is closed by its preregistered training-only support gate (`mean joint-positive=0.10625 < 0.20`); no holdout or quality claim.
- Matched continued-SFT has a complete 512-row/20,000-bootstrap holdout and overall gains over R18, but its boundary-hard/thin boundary-IoU slices cross the registered non-inferiority limit, so it remains a compute control.
- R35 submission attempts reached the SAMTok submitter but failed before rjob creation because `h.pjlab.org.cn` was unresolved. There is no GPU task to monitor or cancel.

## Next candidate decision

R35 remains the sole next candidate: visual-merger/deepstack LoRA only, R18 native-relative geometry credit, fixed null CE weight 2.0 and margin weight 1.0. It is validly registered for exactly 5,120 rows, at least 10 optimizer steps, K=4 grouped rollouts, unified 32-row sentinel and full 512-row/20,000-bootstrap evaluation. No evidence currently supports submitting BA-FEPO before R35 closes.

After R35 is conclusively closed (worker validity failure or complete failed holdout gate), BA-FEPO is the next isolated screen. BA takes the minimum of two independently normalized GT-verified native-relative credits from clean and target-preserving photometric views. Its config and implementation pass static contract tests; it uses no PixVL trainer/weights, OPD, EMA, counterfactual labels, or inference route. It must use the same 5,120-row/10-step/K=4 and 512-row/20,000-bootstrap protocol and should be compared against both R18 and matched-SFT.

## Script/state audit

- `submit_r35_after_pv_decision.sh` correctly requires the matched-SFT paired artifact and either complete PV comparisons or `pv_training_gate.json` with `decision=closed_training_gate` and mean support below 0.20 before trying R35.
- `monitor_fepo_late_screens.sh` has no active BA submission path; BA is evaluation-only after an independently materialized training output. Its `contract_finished` gate prevents evaluating an absent BA run.
- `submit_fepo_eval_adaptive.sh` implements 8 -> 6 -> 4 -> 2 -> 1 at 300 seconds; the 1-GPU stage is terminal. When control-plane status is unavailable it retries status queries instead of stopping a possibly live job, which is appropriate during the DNS outage.
- All SAMTok submitters read `rjob_tags.txt`, pass every positive tag, use namespace `ailab-dnacoding`, and require `dna-` names.

## Verification

Focused contract suite (`BA`, `R35`, `PV`, paired-view, monitor retry and PV
gate tests) -> 17 passed. `bash -n` passes for the R35/BA submitters, late
monitor, finalizer, adaptive evaluator and sharded evaluator.

Both registered output directories are absent (no empty or partial checkpoint
was mistaken for a run):

- `outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_safe_visual_interface_10step_2gpu`
- `outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_boundary_bottleneck_paired_view_10step_2gpu`

Current storage is approximately 38G, well below the 700G cap.

Config validation on the login node succeeds when the approved SAMTok base path
is exported.  Torch-dependent runtime tests remain worker-only.

The status ledger now explicitly marks its newest refresh block as authoritative;
older queue snapshots are provenance only and cannot trigger submissions.

## Latest transition (2026-08-29 16:48 HKT)

R35 has since produced a worker artifact and failed its preregistered
active-set validity gate (`final_sentinel_margin_min=-4.8125`, budget
`-4.6125`; tail-risk gate passed). It is closed without holdout evaluation.
BA-FEPO is therefore unlocked as the single next training arm. The initial
BA submission attempt reached the final `rjob submit` call but failed on
control-plane DNS; no BA job exists. The lock-protected
`submit_ba_after_r35_failure.sh` retry path is wired into the late monitor.
