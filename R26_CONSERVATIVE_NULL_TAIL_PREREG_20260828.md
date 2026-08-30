# R26 preregistration: conservative null-tail native geometry FEPO

## Hypothesis

OpenWorldSAM and V-Zero emphasize explicit positive/null separation and
calibrated abstention under uncertainty. R26 tests the narrowest corresponding
change in the existing SAMTok FEPO framework: preserve R18's native-relative,
first-divergence joint cIoU/boundary credit, but concentrate the existing
training-only no-target sentinel repair on its worst 5% margin tail. The fixed
tail weight is increased from `0.25` to `0.50`; no geometry reward, model,
router, expert, teacher, OPD target, PixVL weight, or inference behavior is
changed.

The sentinel remains a deterministic 32-row training buffer disjoint from the
512-row holdout. The repair is differentiable through current no-target
margins, while the selected tail is detached. Thus R26 tests whether reducing
rare catastrophic null-margin failures improves selective transfer without
buying gains through hallucinated masks.

## Minimal screen

- SAMTok-only continued-SFT anchor, seed `17`.
- `egfepo_train_5120.jsonl`: exactly 5,120 rows, including 2,560 no-target rows.
- Ten outer steps, two policy epochs, K=`4` sibling rollouts, two GPUs.
- Native rank-local geometry credit and all existing support, grammar, PPO,
  sentinel, and tail validity checks unchanged.
- Fixed `sentinel_tail_quantile=0.05`, `sentinel_tail_weight=0.50`; no sweep.

The run must report finite tail penalty/quantile values, final tail-risk gate,
null CE and first-action margin activity, support reach, reward diversity,
epoch-two ratio movement, and complete trainable-parameter identity.

## Evaluation and closure

After all training gates pass, evaluate every one of the 512 paired holdout
rows with 20,000 paired bootstrap resamples against R18
(`evals/r18_depth_local_rarity_free_seed17_holdout512.json`). Promote only if:

- selective utility lower CI is at least `-0.005`;
- positive cIoU lower CI is at least `-0.005`;
- no-target explicit-recall lower CI is at least `-0.01`;
- invalid-output rate is exactly `0` and positive-mask rate is at least `0.99`;
- at least two registered small/thin/boundary slices are non-inferior.

If the tail diagnostics are non-finite, no tail repair is active, or any gate
fails, close R26 without a quantile/weight/seed sweep. Complete RefCOCO and
GRefCOCO transfer is allowed only after this holdout gate passes.

All jobs use namespace `ailab-dnacoding`, a `dna-` name, every positive tag in
`rjob_tags.txt`, and outputs under `Faro_ailab` below the 700G limit.
