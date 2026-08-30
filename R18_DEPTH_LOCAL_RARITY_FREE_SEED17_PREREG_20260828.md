# R18 preregistration: second-seed rarity-free local geometry

R18 is a direct second-seed replication of R16. It keeps the SAMTok-only
initialization, 5,120 training rows, 10 outer steps, two GPUs, four rollouts
per prompt, unified 32-row null sentinel, native-relative joint cIoU and
boundary-IoU positive credit, earliest-divergence depth localization with
decay 0.85, and zero prefix-rarity weight. The only changed stochastic
parameter is the global training seed, from 42 (R16) to 17.

The frozen continued-SFT adapter and all holdout rows remain untouched. The
complete 512-row paired holdout is evaluated once after training. We report
positive cIoU, explicit no-target recall, selective utility, invalid-output
rate, and positive-mask rate with 20,000 paired bootstrap resamples. R18 is a
replication test, not a new method or a weight sweep; a result whose 95% CI
crosses zero is reported as uncertainty rather than promoted.
