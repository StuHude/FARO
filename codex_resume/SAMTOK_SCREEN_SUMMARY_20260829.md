# SAMTok FEPO screen summary (2026-08-29)

The complete 512-row/20,000-bootstrap artifacts show that changing the scalar
credit while keeping the frozen visual interface does not produce a reliable
geometry gain. Representative paired deltas against R18 are:

| screen | positive cIoU delta (95% CI) | utility delta (95% CI) | decision |
| --- | --- | --- | --- |
| native rank-local | +0.00069 [-0.00104, +0.00247] | -0.00161 [-0.00615, +0.00103] | closed |
| scale-stratified | +0.00156 [-0.00079, +0.00485] | -0.00117 [-0.00594, +0.00192] | closed |
| uncertainty-calibrated | +0.00160 [-0.00074, +0.00489] | +0.00080 [-0.00042, +0.00245] | closed |
| margin-calibrated | +0.00131 [-0.00127, +0.00464] | +0.00066 [-0.00061, +0.00232] | closed |
| soft dominance | +0.00008 [-0.00166, +0.00182] | +0.00004 [-0.00083, +0.00092] | closed |
| matched continued SFT | +0.00564 [+0.00007, +0.01330] | +0.00673 [+0.00131, +0.01395] | control; boundary-tail risk |

The screen pattern supports testing the representation bottleneck directly,
not another rank/temperature/uncertainty sweep. R35 is therefore the next
single candidate: visual-merger/deepstack LoRA only, fixed R18 geometry credit,
and stronger fixed null/margin protection. BA-FEPO remains a later isolated
paired-view boundary-bottleneck test, submitted only after R35 is conclusively
closed.

All numbers above are descriptive artifacts; no holdout result is used to tune
R35 or BA. Both candidates retain the 5,120-row/10-step/K=4 contract and the
512-row/20,000-bootstrap promotion protocol.
