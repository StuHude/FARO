# PV-FEPO training audit (2026-08-29)

This is a training-contract diagnostic, not a holdout result. It compares the
completed 10-step PV-FEPO run with the completed 10-step R18 seed-17 reference
before any 512-row evaluation is available.

| signal | R18 seed-17 | PV-FEPO | reading |
| --- | ---: | ---: | --- |
| optimizer updates | 20 | 20 | compute matched |
| rollout groups | 80 | 80 | compute matched |
| K rollouts | 4 | 4 | support matched |
| improved-over-greedy rollouts | 52 | 48 | paired-view may reduce exploration support |
| positive policy-grad observations | 32 | 30 | weaker update signal, still valid |
| nonconstant reward groups | 72 | 73 | reward remains non-degenerate |
| final sentinel margin minimum | -3.25 | -3.375 | slightly worse extreme margin |
| tail mean margin violation rate | 0.0265625 | 0.0203125 | lower average tail violation |
| clean/view reward correlation (mean over 20 optimizer records) | n/a | 0.99988 | view supplies little independent ranking signal |
| joint-positive fraction (min/mean/max) | n/a | 0.0 / 0.10625 / 0.21875 | mixed support; holdout gate remains mandatory |
| effective-support gate | pass | pass | no support collapse |
| tail-risk gate | pass | pass | no contract failure |

The PV arm therefore cannot be promoted from training diagnostics. The mixed
support signal motivates a pre-registered evaluation question: does target-preserving
photometric consistency improve boundary/thin slices enough to justify its
slightly weaker exploration, while remaining non-inferior on null recall and
utility? The answer requires the complete 512-row holdout and 20,000 paired
bootstrap comparison against both R18 and matched continued-SFT. If PV fails,
the geometric-mean view family is closed before considering the independent
R35 representation-control branch; BA-FEPO is not stacked onto PV.

All values are read from the two run metrics files and are not used to tune
the holdout, thresholds, or candidate selection.

## Decision refresh (14:33 HKT)

The preregistered `joint-positive fraction >= 0.20` criterion is a training
support gate. The 20-record mean is `0.10625`, so PV-FEPO is closed without
launching a holdout evaluator. This is a falsification of the paired-view
support hypothesis, not evidence that the model is worse on the holdout. The
decision is recorded in `evals/pv_training_gate.json` with
`holdout_used=false`; the fixed photometric transform and aggregation were not
changed. R35 is now the isolated follow-up.
