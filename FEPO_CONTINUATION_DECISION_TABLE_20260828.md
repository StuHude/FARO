# FEPO continuation decision table (2026-08-28)

This table is the queue-independent experiment protocol.  It does not change
the registered jobs or use holdout statistics for tuning.

| order | candidate | paper-derived question | change from R18 | keep only if |
| --- | --- | --- | --- | --- |
| 1 | R21 native rank-local | Does DR2Seg-style ranking help beyond raw geometry scale? | Replace raw sibling reward with native-reference tie-aware rank. | 512-row utility and positive-cIoU CI lower bounds pass; null risk non-inferior. |
| 2 | R22 scale-stratified | Does geometry feedback need target-scale calibration (Qwen3VL-Seg)? | Fixed small/medium/large rank-axis weights. | Same gates plus no registered slice regression. |
| 3 | R23 bidirectional coarse/fine | Are coarse extent and late boundary edits both needed? | Split credit between first and last changed mask-code depths. | Same gates; report coarse/fine edit frequencies. |
| 4 | R24 anchor-KL | Is the RefCOCO regression cumulative policy drift? | Training-only KL to a disjoint anchor buffer; geometry credit unchanged. | In-domain non-inferiority and transfer recovery without null regression. |
| 5 | R25 uncertainty-calibrated | Can V-Zero-like detached uncertainty downweight fragile gains? | Continuous confidence multiplier; geometry still sets sign. | Improvement survives a confidence-stratified slice audit. |
| 6 | R26 conservative-null | Does OpenWorldSAM-like abstention protection improve transfer? | Increase only fixed sentinel lower-tail repair. | Null recall non-inferior and positive geometry does not regress. |
| 7 | R27 confidence-gated | Are low-support improvements harmful even when their mean reward is high? | Fixed hard confidence floor over R18 credit. | Gate retains effective support and passes all geometry/null gates. |
| 8 | R28 margin-calibrated | Does continuous joint margin remove near-threshold boundary wins? | Multiply rank credit by fixed geometric-mean gain. | Positive cIoU and boundary slices both improve with no utility loss. |
| 9 | R29 primal-dual null-risk (conditional) | Is a fixed null penalty miscalibrated during policy drift? | Update a training-only lower-tail Lagrange multiplier; geometry credit stays R18. | Queue only after current screens clear or show sentinel drift; then require the same full holdout gates. |
| 10 | R30 grounded interface (conditional) | Is the in-domain/RefCOCO gap partly caused by a brittle visual-to-mask interface? | Add a small allowlisted visual-merger LoRA and fixed target-preserving supervised mask-code replay; clean-view RL credit stays R18. | Require verified visual gradients/logit effect, beat the matched visual-SFT control, and pass all geometry/null/slice gates. |
| 11 | R31 native-rank signed depth-local | Does sparse positive-only credit fail because clearly worse sibling masks are never suppressed? | Keep native two-axis midrank gain, add signed negative credit for jointly-worse masks, and localize both signs to first changed depth. | 512-row utility and positive-cIoU CI lower bounds pass, with null/sentinel and canonical-format non-regression. |
| 12 | R32 persistent signed native-rank | Do the small positive R31/R25 signals require longer accumulation rather than another credit transform? | Keep R31 unchanged and run 20 outer steps on the same 5,120-row contract. | Same full holdout gates; compare against R31 with horizon and compute held-out gain per optimizer update. |
| 13 | R33 persistent signed native-rank seed replication | Is the R32 positive utility signal reproducible across a fixed independent seed? | Same R32 stage and data, seed 18. | Strict CI-corrected gate on the paired holdout, followed by matched 100-step confirmation only if replicated. |
| 14 | R34 soft native-dominance depth-local | Does fixed-temperature continuous geometry dominance avoid K=4 midrank quantization? | Replace native midrank with joint cIoU/boundary tanh gains; retain first-divergence scope and sentinel. | Strict CI-corrected gate and fixed slice/canonical checks. |
| 15 | PV-FEPO paired-view native rank-local | Are clean-only gains image-specific and responsible for the transfer gap? | Keep R18 native-relative first-divergence credit, but require the same sampled codes to improve cIoU and boundary IoU on a fixed target-preserving photometric view. | Complete 512-row/20k gate, non-inferior to R18 and matched-SFT, joint-positive fraction >=0.20, and no slice/null regression. |
| 16 | R35 safe visual interface | Can visual-merger plasticity address the transfer bottleneck when null-risk drift is explicitly penalized? | R30 visual-merger-only adapter with fixed `null_ce_weight=2.0` and `margin_weight=1.0`; geometry credit unchanged. | Valid visual-effect and sentinel gates, then complete 512-row/20k non-inferiority to R18 and matched-SFT; no slice/canonical/null regression. |
| 17 | BA-FEPO boundary-agreement paired view (conditional) | Is PV's geometric mean too forgiving when only area overlap improves? | Same R18 local scope; require clean and photometric views to both improve, with the smaller geometry gain as the eligibility bottleneck. | Consider only after PV and the already-registered R35 fallback close; complete 512-row/20k non-inferiority to R18 and matched-SFT, plus boundary-hard/thin non-regression. |
| 18 | SCB-FEPO sentinel-constrained boundary (conditional) | Can a fixed sentinel lower-tail hinge protect null behavior during geometry RL? | R18 credit unchanged; training-only hinge on the sentinel's 10th-percentile first-action margin. | Queue only after observed sentinel drift; otherwise falsify without a job. Same full holdout/null/canonical gates. |

## Promotion sequence

1. A finished screen must contain at least 5,120 training rows, 10 optimizer
   steps, four rollouts per prompt, a complete SAMTok adapter, finite PPO
   ratios/advantages, grammar validity, effective support, and sentinel-risk
   gates.  Otherwise close it without evaluation.
2. Run exactly the enhanced 512-row paired holdout.  Retain all records and
   compute 20,000 paired bootstrap intervals for utility, positive cIoU,
   boundary IoU, and no-target recall.  Also run the fixed small/thin/
   boundary-hard/area slices and canonical-format checks.
3. Only one survivor at a time advances to a 100-step confirmation against the
   frozen continued-SFT anchor and a forward-pass-matched SFT control.  No
   mechanism is combined with another before its isolated screen passes.
4. Only a 100-step survivor advances to complete GRefCOCO and RefCOCO.  A
   transfer regression is reported as a limitation; it is not tuned away on
   the official test sets.

R18's 100-step confirmation passed the strict utility/null screen and is
awaiting the 200-update continued-SFT control before transfer evaluation.

PV-FEPO is the next non-stacked screen.  Its paired-view control is the clean
R18 policy under the same holdout; matched continued-SFT remains the separate
equal-budget training control.  No PV result is inferred from the offline
credit probe or from a queued/failed submission.

The queue monitor must wait five minutes at each evaluation level and use the
`8 -> 6 -> 4 -> 2 -> 1` GPU fallback.  The terminal one-GPU job remains queued.
All dnacoding submissions use namespace `ailab-dnacoding`, a `dna-` prefix,
and every positive tag in `rjob_tags.txt`.
