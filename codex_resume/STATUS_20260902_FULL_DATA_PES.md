# Full-data PES-FEPO continuation (2026-09-02)

## Implementation

- Added an explicit full-data registered schedule for PES-FEPO.
- The schedule includes all 2,560 positive/no-target pairs, including the
  32 sentinel pairs that ordinary tail schedules reserve for risk checks.
- It creates 640 deterministic four-pair batches and declares 5,120 covered
  rows.  The new stage uses 640 outer steps so coverage is complete under
  either Accelerate batch dispatch behavior.
- Each distributed rank writes observed pair IDs.  The main process unions
  those IDs and records `consumed_pair_count`/`consumed_row_count`; the
  validity gate requires at least 2,560 pairs and 5,120 rows.
- Added normal and shuffled full-data configs and submit wrappers. Both use
  the approved SAMTok anchor, K=4 rollouts, no PixVL trainer/OPD/EMA or
  self-supervised loop, and all positive tags from `rjob_tags.txt`.

## Queue

- Submitted normal full-data PES with the previously verified command chain,
  after unsetting all proxy variables.
- Normalized rjob name: `dna-fepo-predicted-evidence-scope-full-data-4ca4e`.
- Showname: `dna-fepo-predicted-evidence-scope-full-data-640step-2g-1788293356`.
- Current scheduler state: `Inqueue`; replica is `STARTING` with no worker
  node. Latest event reports 24 insufficient GPUs plus selector/CPU limits.
  This is resource waiting, not a submission or DNS failure.

## Gates and next transition

- Static checks passed: budget (`actual_rows=5120`, `configured_steps=640`),
  full schedule (`2560` unique pairs, `5120` rows), manifest guard, and
  full-data stage contract.
- No shuffled control is submitted until the normal worker finishes and its
  full-data coverage and validity gates pass.
- After a valid normal worker, submit its 512-row standalone eval and the
  shuffled full-data control. Evaluations retain the 8 -> 6 -> 4 -> 2 -> 1
  GPU adaptive ladder with a 300-second wait per rung.
