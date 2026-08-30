# PES-FEPO continuation checkpoint (2026-08-30)

## Current state

PES-FEPO remains the sole open training arm. Its registered objective is a
single SAMTok policy with native-relative joint cIoU/boundary-IoU credit and
detached predicted-evidence scope. The mandatory shuffled-evidence arm remains
locked until the normal worker and holdout gates pass. No PES checkpoint,
holdout, or promotion result exists.

The late monitor is still emitting heartbeats and the PES transition retains
its single-instance lock. The latest retries fail before rjob creation because
the dnacoding control plane cannot resolve `h.pjlab.org.cn`. An authorized
control-plane retry was attempted on this checkpoint and was rejected by the
execution policy; no workaround or indirect submission was used.

## Offline verification

The CPU A-PES contract probe and PES candidate probe pass the registered
configuration checks: 5,120 training rows, at least 10 optimizer steps, K=4,
detached evidence/scope tensors, and the fixed seed-1907 shuffle control.
The local machine has no CUDA device, so no substitute training run is
possible. Existing non-PES branches remain closed by their recorded gates;
their artifacts are retained for reproducibility.

## Recovery order

1. Keep the lock-protected normal PES submitter retrying every 300 seconds.
2. When the control plane returns, reconcile existing `dna-` jobs before one
   normal PES submission, using all positive tags in `rjob_tags.txt`.
3. Require worker validity, effective support, tail-risk, and PES-coverage
   gates, then submit the 512-row normal holdout through the fixed
   `8 -> 6 -> 4 -> 2 -> 1` evaluation ladder.
4. Submit shuffled PES only after the normal worker gates pass, and finalize
   both complete 512-row reports with 20,000 paired bootstrap repetitions.
5. Select at most one registered follow-up (M/A/B/N) only if its preregistered
   trigger fires; do not tune thresholds on the holdout.

Storage remains below the 700G ceiling and all new state is under `Faro_ailab`.

## Historical submit-template reconciliation

The successful AB-FEPO record at 18:04 HKT used the same shared
`submit_samtok_tb_gppo.sh` template now used by PES: `dna-` name,
`ailab-dnacoding`, `--cpu=20 --gpu=2 --memory=240000`, all eight positive tags,
the two gpfs mounts, `wuyucheng:test1`, `brainpp.cn/fuse=1`, and
`--enable-sshd`. The PES wrapper adds only the registered PES config and
5,120-row data path. A lock-protected manual reconciliation was attempted on
this checkpoint and correctly skipped because the live transition already
held `logs/pes_submit/.lock`; no duplicate rjob was created.
