#!/usr/bin/env python3
"""Throttled submit feeder for the ligand-series holo runs.

The cluster caps submitted jobs at AssocMaxSubmit=100 (and ~8 concurrent under
the ``normal`` QOS), so the 1856 fragment runs cannot be fired as one big array.
This feeder keeps at most ``--cap`` of MY jobs in the queue at a time, submitting
each fragment's own ``run_wrapper.sbatch.sh`` (one job per fragment) and topping
up as jobs finish. Restart-safe: already-submitted wrappers are tracked in a
state log and never resubmitted.

Run detached on the (CPU-only) master node, e.g.:
    nohup setsid uv run python scripts/feed_holo_queue.py > experiments/ligand_series/feed.log 2>&1 &
Stop with:
    touch experiments/ligand_series/holo_feed.stop   # graceful, or
    pkill -f feed_holo_queue.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "ligand_series"
DEFAULT_MANIFESTS = [EXP / "L01" / "holo" / "_manifest.txt", EXP / "L02" / "holo" / "_manifest.txt"]
STATE = EXP / "holo_feed_submitted.log"
STOP = EXP / "holo_feed.stop"


def my_queue_count() -> int:
    # -r expands array tasks; -h no header. Each line = one job counting toward
    # AssocMaxSubmit.
    out = subprocess.run(
        ["squeue", "-u", _user(), "-h", "-r"],
        capture_output=True, text=True,
    )
    return sum(1 for ln in out.stdout.splitlines() if ln.strip())


def _user() -> str:
    import os
    return os.environ.get("USER") or subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()


def load_wrappers(manifests: list[Path]) -> list[str]:
    out: list[str] = []
    for m in manifests:
        if not m.exists():
            print(f"[feed] WARN manifest missing: {m}", flush=True)
            continue
        out.extend(ln.strip() for ln in m.read_text().splitlines() if ln.strip())
    return out


def load_submitted() -> set[str]:
    if not STATE.exists():
        return set()
    return {ln.strip() for ln in STATE.read_text().splitlines() if ln.strip()}


# Fragment runs stay on the partition their wrapper pins (6000ada): that GPU
# pool is reserved for cofolding+docking, while `test` is left to the rescoring
# workers. Kept for reference / manual overrides.
FRAGMENT_PARTITIONS = [
    ("6000ada", ["--gres=gpu:1"]),
    ("heavy", ["--gres=gpu:h100:1"]),
    ("heavy", ["--gres=gpu:6000pro_maxq:1"]),
    ("test", ["--gres=gpu:a5000:1"]),
]


def _free_gpus(partition: str, kind: str | None) -> int:
    """GPUs configured minus allocated across the partition's nodes."""
    import re
    nodes = subprocess.run(["sinfo", "-h", "-p", partition, "-o", "%N"],
                           capture_output=True, text=True).stdout.strip()
    if not nodes:
        return 0
    total = 0
    for node in subprocess.run(["scontrol", "show", "hostnames", nodes],
                               capture_output=True, text=True).stdout.split():
        txt = subprocess.run(["scontrol", "show", "node", node],
                             capture_output=True, text=True).stdout

        def count(field: str) -> int:
            m = re.search(rf"{field}=([^\s]*)", txt)
            if not m:
                return 0
            tres = m.group(1)
            if kind:
                t = re.search(rf"gres/gpu:{re.escape(kind)}=(\d+)", tres)
                if t:
                    return int(t.group(1))
                if kind not in txt:
                    return 0
            g = re.search(r"gres/gpu=(\d+)", tres)
            return int(g.group(1)) if g else 0

        total += max(0, count("CfgTRES") - count("AllocTRES"))
    return total


def pick_partition() -> list[str]:
    """sbatch args for a partition with a free GPU, else [] (plain submit)."""
    import re
    for partition, gres in FRAGMENT_PARTITIONS:
        m = re.match(r"--gres=gpu:([^:]+):\d+", gres[0])
        kind = m.group(1) if m else None
        if _free_gpus(partition, kind) > 0:
            return ["-p", partition, *gres]
    return []


def submit(wrapper: str, extra: list[str] | None = None) -> bool:
    r = subprocess.run(["sbatch", *(extra or []), wrapper], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[feed] submitted {wrapper.split('/holo/')[-1]} :: {r.stdout.strip()}", flush=True)
        with STATE.open("a") as fh:
            fh.write(wrapper + "\n")
        return True
    print(f"[feed] sbatch FAIL rc={r.returncode} :: {r.stderr.strip()} :: {wrapper}", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=90, help="max jobs kept in my queue")
    ap.add_argument("--sleep", type=int, default=120, help="seconds between top-ups")
    ap.add_argument("--manifests", nargs="+", type=Path, default=DEFAULT_MANIFESTS)
    ap.add_argument("--batch", type=int, default=20, help="max submits per top-up cycle")
    ap.add_argument(
        "--once",
        action="store_true",
        help="single top-up (fill queue to cap) then exit — for cron. No sleep loop.",
    )
    args = ap.parse_args()

    wrappers = load_wrappers(args.manifests)
    print(f"[feed] {len(wrappers)} total wrappers; cap={args.cap} sleep={args.sleep}s", flush=True)
    if STOP.exists():
        STOP.unlink()

    while True:
        if STOP.exists():
            print("[feed] stop file present — exiting", flush=True)
            return 0
        submitted = load_submitted()
        todo = [w for w in wrappers if w not in submitted]
        if not todo:
            print(f"[feed] all {len(wrappers)} submitted — done", flush=True)
            return 0
        cur = my_queue_count()
        room = args.cap - cur
        # --once fills all available room to cap; loop mode caps per-cycle at batch.
        per_cycle = room if args.once else min(room, args.batch)
        n = 0
        while room > 0 and todo and n < per_cycle:
            w = todo.pop(0)
            if submit(w):
                room -= 1
                n += 1
            time.sleep(1)  # gentle on the scheduler
        print(f"[feed] cycle: queue={cur} submitted+={n} remaining={len(todo)}", flush=True)
        if args.once:
            return 0
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
