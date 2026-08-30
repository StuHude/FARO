# R21 preregistration: native-anchored rank-local geometry credit

R21 tests whether a tie-aware continuous sibling rank retains useful ordering
that R18's absolute native-gain threshold discards. The native greedy mask is
inserted as an explicit reference on cIoU and boundary IoU; only samples
improving both axes by `1e-4` receive rank credit. Credit is localized to the
first changed SAMTok code depth with fixed decay `0.85`.

This remains one SAMTok policy and a training-time credit rule: no router,
expert, PixVL weight, OPD target, self-supervised cycle, or counterfactual
label. The design combines Fine-R1 grouped relative optimization, DR2Seg
continuous geometry ranking, Qwen3VL-Seg/OpenWorldSAM positive-vs-null
separation, and SAMTok's hierarchical code interface.

## Fixed screen

- `5,120` training rows (`2,560` target-present and `2,560` no-target), `10`
  outer steps, `20` policy epochs per step, K=`4` rollouts.
- Two GPUs; frozen continued-SFT SAMTok adapter initialization.
- Shared 32-row no-target sentinel, effective-support calibration, tail-risk,
  canonical, and invalid-output gates unchanged from R18.
- Config: `Sa2VA/projects/samtok_selective/configs/`
  `fepo_tb_gppo_plain_rank_unified_native_rank_local_10step_2gpu.py`.

## Promotion / closure

Evaluate all 512 paired holdout rows with 20,000 paired bootstrap resamples.
Promote only if positive cIoU improves R18 by at least `+0.010` with a
strictly positive lower CI, utility improves by at least `+0.010` with a
strictly positive lower CI, no-target recall is non-inferior (lower CI >=
`-0.01`), invalid output is `0`, positive-mask rate is >= `0.95`, and all
training validity/support/sentinel/tail gates pass. Otherwise close R21
without a rank or decay sweep. A surviving screen must then run complete
RefCOCO and GRefCOCO transfer before any paper claim.
