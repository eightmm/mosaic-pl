#!/usr/bin/env python3
"""Re-score the 9 Cat-3 targets that hit the 180-sec per-target timeout
in score_per_metric.py's main loop. These targets have full pose data on
disk but contain big/complex cofactor ligands (FAD, COA, B12, large
fatty-acids, modified sugars) whose RMSD computation exceeded 180s.

Strategy: bump per-target timeout to 1800s (30 min) — that's enough for
even the worst FAD/COA cases (8z15 took 223s for 18,832 poses; the worst
remaining likely needs <10 min).

Output appends to the existing per_pose_scores.csv so resume-aware
analysis scripts pick the new rows up automatically.
"""
from __future__ import annotations

import csv
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "novel2025_test"))

from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

import score_per_metric as spm  # noqa: E402
import evaluate as ev  # noqa: E402

spm.RUNS = REPO / "experiments/msa_e2e_test/runs"
spm.SUBMISSIONS = REPO / "experiments/msa_e2e_test/submissions"
CSV_PATH = REPO / "experiments/msa_e2e_test/per_pose_scores.csv"
WORK = REPO / "experiments/msa_e2e_test/_per_metric_work"
WORK.mkdir(exist_ok=True, parents=True)
spm.WORK = WORK

CAT3 = [
    "9dsv", "9emt", "9ifw", "9mgt", "9mh5",
    "9n1b", "9uo2", "9vjx", "9zno",
]

PER_TARGET_TIMEOUT = 1800  # 30 min


@contextmanager
def hard_timeout(seconds: int):
    def handler(*_):
        raise TimeoutError("hard timeout")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> int:
    targets = ev.load_targets()
    header = [
        "target", "seq_zone", "source", "pose_name",
        "ba_pred_pkd", "prmsd", "lscore",
        "iptm", "ptm", "plddt", "conf",
        "boltz_aff_log10_kd_nM", "boltz_binder_prob",
        "true_rmsd",
    ]
    new_rows = 0
    with CSV_PATH.open("a", newline="") as fh:
        w = csv.writer(fh)
        for t in CAT3:
            if t not in targets:
                print(f"  {t}: not in targets, skipping", flush=True)
                continue
            print(f"  {t}: start (timeout {PER_TARGET_TIMEOUT}s)", flush=True)
            t0 = time.time()
            try:
                with hard_timeout(PER_TARGET_TIMEOUT):
                    rs = spm.evaluate_run(t, targets[t], WORK)
            except TimeoutError:
                print(f"  {t}: HARD TIMEOUT ({PER_TARGET_TIMEOUT}s) — still bad", flush=True)
                continue
            except Exception as e:
                print(f"  {t}: CRASH {e!r}", flush=True)
                continue
            for r in rs:
                w.writerow([
                    r.target, r.seq_zone, r.source, r.pose_name,
                    r.ba_pred_pkd, r.prmsd, r.lscore,
                    r.iptm, r.ptm, r.plddt, r.conf,
                    r.boltz_aff, r.boltz_binder_prob,
                    r.true_rmsd,
                ])
            new_rows += len(rs)
            fh.flush()
            print(f"  {t}: ok {time.time()-t0:.1f}s  +{len(rs)} pose rows", flush=True)
    print(f"appended {new_rows} rows to {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
