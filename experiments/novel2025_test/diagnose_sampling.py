#!/usr/bin/env python3
"""Diagnose the sampling ceiling: for each target compute min true_rmsd overall
/ within cofold / within docking, and which family achieves the best pose.

Answers: are oracle-misses (a) near-miss in the right pocket (finer sampling
helps) or (b) wrong-pocket entirely (need better site/cofold, not more docking)?
And: does cofold or docking own the best pose (where to spend samples)?
"""
from __future__ import annotations

import csv
from collections import defaultdict

import ablate_pose_selection as A

HIT = 2.0
COF = A.COFOLD_FAMS


def main():
    best = {}           # target -> (min_rmsd, fam)
    best_cof = defaultdict(lambda: 1e9)
    best_dock = defaultdict(lambda: 1e9)
    zone = {}
    with open(A.CSV) as f:
        for r in csv.DictReader(f):
            tr = A._f(r["true_rmsd"])
            if tr is None:
                continue
            t = r["target"]
            zone.setdefault(t, r["seq_zone"])
            fam = A._family(r["source"])
            is_cof = fam in COF
            if t not in best or tr < best[t][0]:
                best[t] = (tr, fam)
            if is_cof:
                if tr < best_cof[t]:
                    best_cof[t] = tr
            else:
                if tr < best_dock[t]:
                    best_dock[t] = tr

    targets = list(best)
    n = len(targets)
    print(f"targets: {n}\n")

    # 1) where does the BEST pose come from?
    cof_owns = sum(1 for t in targets if best[t][1] in COF)
    print(f"best pose is a COFOLD pose: {cof_owns}/{n} ({100*cof_owns/n:.0f}%)")
    print(f"best pose is a DOCKING pose: {n-cof_owns}/{n} ({100*(n-cof_owns)/n:.0f}%)\n")

    # 2) oracle-miss regime: distribution of min_rmsd for targets with NO sub-2A pose
    miss = [t for t in targets if best[t][0] >= HIT]
    print(f"oracle-MISS targets (no sub-2A pose anywhere): {len(miss)}/{n} ({100*len(miss)/n:.0f}%)")
    bins = [(2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]
    print("  min_true_rmsd distribution among misses:")
    for lo, hi in bins:
        c = sum(1 for t in miss if lo <= best[t][0] < hi)
        tag = "near-miss (finer sampling in right pocket)" if hi <= 4 else \
              ("partial (pocket roughly right)" if hi <= 6 else "WRONG POCKET (site/cofold problem)")
        print(f"    {lo:>2}-{hi if hi < 1e9 else '∞':>3} A : {c:4d}  {tag}")

    # 3) cofold vs docking: does docking ever beat cofold's best?
    both = [t for t in targets if best_cof[t] < 1e9 and best_dock[t] < 1e9]
    dock_better = sum(1 for t in both if best_dock[t] + 0.5 < best_cof[t])
    dock_rescue = sum(1 for t in both if best_cof[t] >= HIT and best_dock[t] < HIT)
    print(f"\ncofold vs docking (targets with both, n={len(both)}):")
    print(f"  docking beats cofold by >0.5A: {dock_better} ({100*dock_better/len(both):.0f}%)")
    print(f"  docking RESCUES (cofold>=2A but docking<2A): {dock_rescue} ({100*dock_rescue/len(both):.0f}%)")

    # 4) by zone: oracle + cofold-only oracle
    print("\nper-zone oracle (any<2A) vs cofold-only oracle:")
    for z in ("novel", "remote", "related", "ALL"):
        zt = targets if z == "ALL" else [t for t in targets if zone[t] == z]
        orc = 100*sum(best[t][0] < HIT for t in zt)/len(zt)
        cof = 100*sum(best_cof[t] < HIT for t in zt)/len(zt)
        print(f"  {z:8s} oracle={orc:5.1f}%   cofold-only-oracle={cof:5.1f}%   (n={len(zt)})")


if __name__ == "__main__":
    main()
