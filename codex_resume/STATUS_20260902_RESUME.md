# FARO resume status (2026-09-02)

## External-state recheck

- The initial DNS diagnosis came from the Codex sandbox, whose network
  namespace could not resolve the cluster host. It was not representative of
  the server's actual control-plane network.
- In an escalated, server-network shell, `curl -Iv https://h.pjlab.org.cn`
  resolved `h.pjlab.org.cn` to `10.68.64.26` and completed TLS with HTTP/2
  200. Python `socket.getaddrinfo` and `rjob list` also succeeded with all
  proxy variables unset, confirming the configured
  `KUBEBRAIN_CLUSTER_ENTRY=https://h.pjlab.org.cn` is the correct rjob
  endpoint.
- The normal PES submitter then created
  `dna-fepo-predicted-evidence-scope-10step-2g-70ca4` (showname
  `dna-fepo-predicted-evidence-scope-10step-2g-1788290427`) using the complete
  positive-tag list. The job is currently `Inqueue` with a `STARTING` replica;
  `logs/pes_submit/submitted` records the requested name.
- A persistent `faro-pes-monitor` tmux session is running with all proxy
  variables unset. It waits for the worker validity contract before submitting
  the full 512-row/20,000-bootstrap normal and shuffled evaluations.

## Local evidence

- The FARO source worktree is clean apart from this status-record update. The
  PES manifest remains 5,120 rows with 2,560 no-target rows and 2,560 pair
  IDs.
- The approved read-only SAMTok anchor, manifest guard, budget validator, and
  tail-GPPO contract had already passed before this external recheck. No new
  checkpoint, worker metric, holdout, bootstrap, or official transfer result
  is inferred from the failed network calls.
- Existing local commits and all experiment outputs remain under
  `Faro_ailab`; no new files were written under `PixVL_ailab`.

## Next transition

After the PES worker leaves the queue, require the full worker validity gates
before the 512-row/20,000-bootstrap evaluation and shuffled control.

## 03:42 HKT recheck

- `rjob list` still reports `dna-fepo-predicted-evidence-scope-10step-2g-70ca4`
  as `Inqueue` with a `STARTING` replica and no assigned node. This is a
  scheduler wait, not a submission or Python failure.
- The local candidate and APES contract probes passed again. No PES checkpoint,
  worker metrics, holdout, or bootstrap result exists yet.
- A read-only audit of completed screens confirms that R18 is the only robust
  positive reference; R18-100 and the matched continued-SFT control define the
  required bar for PES. The conditional mass-aware PES follow-up remains
  unsubmitted until normal PES and its seed-1907 shuffled control have complete
  512-row/20,000-bootstrap evidence.
- Workspace usage remains below 40G. No files were added under `PixVL_ailab`.

## 03:45 HKT scheduler evidence

- `rjob events` reports the replica has remained pending since submission. The
  latest scheduler reason is `24 Insufficient nvidia.com/gpu`, with additional
  CPU/selector constraints across the cluster. The worker has not started, so
  there is no training failure to debug and no reason to resubmit or alter the
  registered two-GPU request.
- The evaluation fallback ladder remains reserved for evaluation jobs only;
  the PES training job stays at its preregistered two GPUs.

## 03:56 HKT continuation recheck

- `rjob get` still reports the same showname `dna-fepo-predicted-evidence-scope-10step-2g-1788290427`
  as `Inqueue`, with replica `dna-fepo-predicted-evidence-scope-10step-2g-70ca4-7t8bb`
  in `STARTING`; no worker node is assigned.
- The latest control-plane event remains gang-unschedulable: `24 Insufficient
  nvidia.com/gpu`, one CPU shortage, one selector mismatch, and 1,940 nodes
  outside the positive-tag/node-label selector. This confirms a resource queue
  wait rather than a submission, DNS, proxy, or training error.
- The late-screen monitor heartbeat is still advancing once per minute. PES
  normal and shuffled evaluation markers remain `waiting`; no metrics,
  checkpoint, holdout, or bootstrap artifact exists, so no result is promoted
  or inferred.
- Static checks reconfirm the PES manifest has 5,120 rows, the approved
  SAMTok-only initialization path, and all eight positive tags. Workspace use
  remains approximately 39G; no files were written under `PixVL_ailab`.
