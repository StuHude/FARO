# FARO Final Results Table (Evidence Snapshot)

This table is a compact, paper-facing view of completed FARO screens.  Every
row below comes from a checked JSON artifact under `evals/`; all paired
comparisons use 512 rows (256 target-present and 256 no-target) and 20,000
paired bootstrap repetitions.  Intervals are percentile 95% CIs for
candidate-minus-reference.  Values are rounded to four decimals only for
display; the JSON files remain authoritative.

## Completed screens

| Run | Reference | Positive cIoU delta [95% CI] | Utility delta [95% CI] | Corrected gate |
| --- | --- | ---: | ---: | --- |
| R15 shuffled depth-local | continued-SFT anchor | +0.0224 [+0.0071,+0.0384] | +0.0893 [+0.0657,+0.1143] | pass (diagnostic) |
| R16 rarity-free depth-local | continued-SFT anchor | +0.0251 [+0.0091,+0.0418] | +0.0907 [+0.0667,+0.1161] | pass |
| R17 uniform joint geometry | continued-SFT anchor | +0.0203 [+0.0042,+0.0370] | +0.0883 [+0.0643,+0.1139] | pass |
| R18 seed-17 replication | continued-SFT anchor | +0.0238 [+0.0082,+0.0405] | +0.0900 [+0.0661,+0.1151] | pass; provisional reference |
| R18-100 confirmation | continued-SFT anchor | +0.0045 [-0.0015,+0.0123] | +0.0179 [+0.0076,+0.0299] | strict pass; cIoU CI crosses zero |
| matched continued-SFT | R18 | +0.0056 [+0.0001,+0.0133] | +0.0067 [+0.0013,+0.0139] | pass; compute control |
| R21 native rank-local | R18 | +0.0007 [-0.0010,+0.0025] | -0.0016 [-0.0061,+0.0010] | reject |
| R22 scale-stratified | R18 | +0.0016 [-0.0008,+0.0049] | -0.0012 [-0.0059,+0.0019] | reject |
| R23 bidirectional coarse/fine | R18 | -0.0016 [-0.0063,+0.0017] | -0.0008 [-0.0032,+0.0008] | reject |
| R24 anchor-KL | R18 | -0.0004 [-0.0055,+0.0040] | -0.0002 [-0.0028,+0.0020] | reject |
| R25 uncertainty-calibrated | R18 | +0.0016 [-0.0007,+0.0049] | +0.0008 [-0.0004,+0.0025] | reject |
| R26 conservative-null tail | R18 | -0.0014 [-0.0061,+0.0018] | -0.0007 [-0.0031,+0.0009] | reject |
| R27 confidence-gated | R18 | -0.0016 [-0.0064,+0.0015] | -0.0008 [-0.0032,+0.0008] | reject |
| R28 margin-calibrated | R18 | +0.0013 [-0.0013,+0.0046] | +0.0007 [-0.0006,+0.0023] | reject |
| R29 primal-dual null risk | R18 | -0.0004 [-0.0055,+0.0040] | -0.0002 [-0.0028,+0.0020] | reject |
| R34 soft native dominance | R18 | +0.0001 [-0.0017,+0.0018] | +0.0000 [-0.0008,+0.0009] | reject |
| BA boundary bottleneck | matched continued-SFT | -0.0087 [-0.0182,-0.0009] | -0.0180 [-0.0297,-0.0079] | reject |
| BS boundary-stratified | matched continued-SFT | -0.0087 [-0.0182,-0.0010] | -0.0180 [-0.0297,-0.0079] | reject |
| AB action-budget | matched continued-SFT | -0.0099 [-0.0197,-0.0017] | -0.0167 [-0.0276,-0.0072] | reject |

## Artifact mapping

- R15-R18 and the R18-100 rows are in the corresponding
  `r15_*`, `r16_*`, `r17_*`, and `r18_*` analysis JSON files.
- R21-R29 and R34 use the named `*_vs_r18_bootstrap20k*.json` artifacts.
- BA, BS, and AB use their named `*_vs_matched_sft_bootstrap20k*.json`
  artifacts.  Their negative deltas are closed controls, not tuned settings.
- The matched-SFT row is `evals/r18_matched_sft_vs_r18_bootstrap20k.json`.

## Open evidence

PES-FEPO and its mandatory seed-1907 shuffled-evidence control have no rjob,
checkpoint, worker metrics, holdout, or bootstrap artifact.  They are not
included as zero-valued rows.  The current defensible method claim therefore
remains FEPO-R18 until the normal PES worker and all preregistered gates are
completed.
