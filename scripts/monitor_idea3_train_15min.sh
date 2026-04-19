#!/usr/bin/env bash

set -euo pipefail

LOG="${1:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/monitor_idea3_train_15min.log}"
mkdir -p "$(dirname "$LOG")"

while true; do
  {
    echo "=== $(date '+%F %T %Z') ==="
    python - <<'PY'
import json, os, subprocess

names = [
    "2gpu_bs4_2000_unified_opd_rl",
    "2gpu_bs4_2000_routed_rl",
    "2gpu_bs4_2000_routed_opd_rl",
]

for name in names:
    p = f"/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/{name}/metrics.json"
    q = f"/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/{name}/metrics.partial.json"
    print("MODEL", name)
    if os.path.exists(p):
        obj = json.load(open(p))
        last = obj.get("steps", [])[-1] if obj.get("steps") else None
        print("STATE", "finished", last)
    elif os.path.exists(q):
        obj = json.load(open(q))
        last = obj.get("steps", [])[-1] if obj.get("steps") else None
        print("STATE", "running", last)
    else:
        print("STATE", "missing_metrics")

print("PROCS")
print(subprocess.check_output(
    "pgrep -a -f 'idea3_mvp_2gpu_bs4_2000_unified_opd_rl.py|idea3_mvp_2gpu_bs4_2000_routed_rl.py|idea3_mvp_2gpu_bs4_2000_routed_opd_rl.py' || true",
    shell=True, text=True
))
print("GPUS")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
    text=True
))
PY
    echo
  } >> "$LOG" 2>&1
  sleep 900
done
