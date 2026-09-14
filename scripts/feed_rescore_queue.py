#!/usr/bin/env python3
"""Auto-feeder for the 3-scorer rescoring workers.

Submits `rescore_batch.sbatch.sh` workers as GPUs free up, then exits (cron
re-runs it). Workers are claim-based, so the count is elastic: submit more when
the cluster is idle, and they cooperatively drain the same work list.

Guard rails:
  * never exceed --max-workers rescore jobs (running + pending) at once;
  * keep at most --max-workers workers, on the `test` partition only, so the
    6000ada pool stays free for the cofolding/docking runs;
  * only submit to a partition that currently has a free GPU (avoids parking
    jobs behind other users and starving the docking queue);
  * stop cleanly when every fragment in the list is scored.

Typical cron line (every 10 min):
  */10 * * * * <repo>/.venv/bin/python \
      <repo>/scripts/feed_rescore_queue.py --once \
      >> <repo>/experiments/ligand_series/rescore_feed.log 2>&1
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SBATCH = REPO / "scripts" / "rescore_batch.sbatch.sh"
DEFAULT_LIST = REPO / "experiments/ligand_series/L01_rescore_list.txt"
STOP = REPO / "experiments/ligand_series/rescore_feed.stop"

# test + the heavy H100 when it is idle. 6000ada stays reserved for the
# per-fragment cofold/dock runs, which are the critical path for coverage.
PARTITIONS = [
    ("heavy", ["--gres=gpu:h100:1"]),
    ("test", ["--gres=gpu:1"]),
]


def _user() -> str:
    return os.environ.get("USER") or subprocess.run(
        ["whoami"], capture_output=True, text=True).stdout.strip()


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def my_jobs() -> tuple[int, int]:
    """(total jobs in my queue, rescore jobs in my queue)."""
    out = run(["squeue", "-u", _user(), "-h", "-r", "-o", "%j"])
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return len(lines), sum(1 for ln in lines if ln.startswith("rescore"))


def free_gpus(partition: str, gres_kind: str | None) -> int:
    """GPUs configured minus allocated, summed over the partition's nodes."""
    nodes = run(["sinfo", "-h", "-p", partition, "-o", "%N"]).strip()
    if not nodes:
        return 0
    total = 0
    for node in run(["scontrol", "show", "hostnames", nodes]).split():
        txt = run(["scontrol", "show", "node", node])
        cfg = _gpu_count(txt, "CfgTRES", gres_kind, node_txt=txt)
        alloc = _gpu_count(txt, "AllocTRES", gres_kind, node_txt=txt)
        total += max(0, cfg - alloc)
    return total


def _gpu_count(txt: str, field: str, gres_kind: str | None, node_txt: str = "") -> int:
    m = re.search(rf"{field}=([^\s]*)", txt)
    if not m:
        return 0
    tres = m.group(1)
    if gres_kind:
        # typed request (e.g. gpu:h100:1): prefer the typed TRES when present
        t = re.search(rf"gres/gpu:{re.escape(gres_kind)}=(\d+)", tres)
        if t:
            return int(t.group(1))
        # node doesn't report typed TRES -> only trust an untyped count when the
        # node actually advertises this gres kind in its Gres= line
        if gres_kind not in node_txt:
            return 0
    g = re.search(r"gres/gpu=(\d+)", tres)
    return int(g.group(1)) if g else 0


def remaining(list_path: Path) -> int:
    n = 0
    for ln in list_path.read_text().splitlines():
        ln = ln.strip()
        if ln and not (Path(ln) / "outputs/scoring/pose_scores.csv").is_file():
            n += 1
    return n


def submit(partition: str, gres: list[str], list_path: Path) -> bool:
    env = dict(os.environ, LIST=str(list_path), DEVICE="cuda")
    cmd = ["sbatch", "-p", partition, *gres, str(SBATCH)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0:
        print(f"[rescore-feed] submitted on {partition} {' '.join(gres)} :: {r.stdout.strip()}", flush=True)
        return True
    print(f"[rescore-feed] sbatch failed ({partition}): {r.stderr.strip()}", flush=True)
    return False


def tick(args) -> int:
    if STOP.exists():
        print("[rescore-feed] stop file present — nothing submitted", flush=True)
        return 0
    left = remaining(args.list)
    if left == 0:
        print("[rescore-feed] all fragments scored — done", flush=True)
        return 0
    total, mine = my_jobs()
    room_workers = args.max_workers - mine
    room_queue = args.queue_cap - total
    room = min(room_workers, room_queue)
    print(f"[rescore-feed] remaining={left} rescoreJobs={mine} totalQueue={total} room={room}", flush=True)
    if room <= 0:
        return 0

    submitted = 0
    for partition, gres in PARTITIONS:
        if submitted >= room:
            break
        kind = None
        m = re.match(r"--gres=gpu:([^:]+):\d+", gres[0])
        if m:
            kind = m.group(1)
        n_free = free_gpus(partition, kind)
        while n_free > 0 and submitted < room:
            if not submit(partition, gres, args.list):
                break
            submitted += 1
            n_free -= 1
            time.sleep(1)
    if submitted == 0:
        print("[rescore-feed] no free GPU right now", flush=True)
    return submitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST)
    ap.add_argument("--max-workers", type=int, default=8,
                    help="max concurrent rescore jobs (running+pending)")
    ap.add_argument("--queue-cap", type=int, default=100000,
                    help="keep my TOTAL queued jobs under this (AssocMaxSubmit is now 100000)")
    ap.add_argument("--once", action="store_true", help="single tick then exit (cron mode)")
    ap.add_argument("--sleep", type=int, default=600)
    args = ap.parse_args()
    if not args.list.is_file():
        sys.exit(f"work list not found: {args.list}")
    while True:
        tick(args)
        if args.once:
            return 0
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
