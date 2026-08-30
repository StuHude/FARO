# PES offline diagnostic (2026-08-30)

An ephemeral CPU simulation exercised 5,120 synthetic rows for 10 scope
decisions with the registered A-PES probability-gap thresholds and K=4
contract. The implementation remained finite and detached, and the shuffled
state permutation was deterministic for each step.

The synthetic evidence distribution placed nearly every row in the ambiguous
state and produced no unsupported rows. This is a distributional warning only:
the random probabilities are not calibrated SAMTok logits and no model quality
or training conclusion is drawn. The thresholds and registered PES objective
are unchanged. On a real worker, state counts, effective support, and
`pes_coverage_gate_passed` must be inspected before any holdout submission;
coverage failure closes PES rather than authorizing threshold tuning on the
holdout.
