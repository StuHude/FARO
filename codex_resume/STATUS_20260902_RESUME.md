# FARO resume status (2026-09-02)

## External-state recheck

- The goal was resumed after the previous blocked state. The existing
  lock-protected normal PES retryer remains the only instance; its
  `logs/pes_submit/.lock` is held and `logs/pes_submit/submitted` is absent.
- In a fresh shell with `http_proxy`, `https_proxy`, `HTTP_PROXY`,
  `HTTPS_PROXY`, `all_proxy`, `ALL_PROXY`, `no_proxy`, and `NO_PROXY` all
  unset, `rjob list --namespace=ailab-dnacoding` still fails before API access
  with `Name or service not known` for `h.pjlab.org.cn`.
- The requested `source <(curl -fsSL
  http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)` sequence was
  also attempted after unsetting the variables; the setup endpoint itself
  cannot be resolved. No rjob was created and no duplicate retryer was
  started.

## Local evidence

- The FARO worktree is clean. The PES manifest remains 5,120 rows with 2,560
  no-target rows and 2,560 pair IDs.
- The approved read-only SAMTok anchor, manifest guard, budget validator, and
  tail-GPPO contract had already passed before this external recheck. No new
  checkpoint, worker metric, holdout, bootstrap, or official transfer result
  is inferred from the failed network calls.
- Existing local commits and all experiment outputs remain under
  `Faro_ailab`; no new files were written under `PixVL_ailab`.

## Next transition

When either direct cluster DNS or the internal proxy endpoint becomes
reachable, reconcile the namespace first. Keep the existing single-instance
retry lock, submit normal PES only with the registered positive tags, then
require the full worker validity gates before the 512-row/20,000-bootstrap
evaluation and shuffled control.
