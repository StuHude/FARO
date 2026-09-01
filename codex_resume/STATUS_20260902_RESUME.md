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
