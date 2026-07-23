#!/usr/bin/env python3
"""Scope Tier B: find 'recoverable' targets (a sub-2A pose exists but the
scalar selector misses it), dedup by cluster, stratify by zone, and verify
staged poses + receptor exist on disk. Picks a sample for a coords-based
(clash + symmetry-consensus) prototype.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import ablate_pose_selection as A

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "novel2025_runs" / "runs"
CLUSTER_CSV = ROOT / "cluster_targets_100.csv"
HIT = 2.0


def main():
    by_target = A.load()
    targets = list(by_target)
    zone = {t: by_target[t][0]["zone"] for t in targets}
    oracle = {t: any(p["true_rmsd"] < HIT for p in by_target[t]) for t in targets}
    rrf_hit = {t: A.rrf_pick(by_target[t])["true_rmsd"] < HIT for t in targets}

    rep = {}
    if CLUSTER_CSV.exists():
        for r in csv.DictReader(open(CLUSTER_CSV)):
            rep[r["target"].replace("_input", "")] = r["cluster_rep"].replace("_input", "")
    for t in targets:
        rep.setdefault(t, t)

    recoverable = [t for t in targets if oracle[t] and not rrf_hit[t]]
    print(f"recoverable (oracle hit, selector miss): {len(recoverable)} / {len(targets)}")

    # dedup by cluster (one per cluster), stratify by zone
    seen_rep = set()
    by_zone = defaultdict(list)
    for t in recoverable:
        if rep[t] in seen_rep:
            continue
        seen_rep.add(rep[t])
        by_zone[zone[t]].append(t)
    print("recoverable, cluster-deduped, by zone:",
          {z: len(v) for z, v in by_zone.items()})

    # disk-presence check helper
    def staged(t):
        d = RUNS / f"{t}_input" / f"{t}_input" / "outputs" / "analysis" / "poses"
        rec = RUNS / f"{t}_input" / f"{t}_input" / "inputs" / "docking" / "receptor.pdb"
        n_sdf = len(list(d.glob("*.sdf"))) if d.is_dir() else 0
        return n_sdf, rec.is_file()

    # pick up to 10 per zone with staged data present
    sample = []
    print("\n=== sample (staged check) ===")
    for z in ("novel", "remote", "related"):
        picked = 0
        for t in by_zone.get(z, []):
            n_sdf, has_rec = staged(t)
            if n_sdf > 0 and has_rec:
                sample.append(t)
                picked += 1
                print(f"  {z:8s} {t:8s} sdf={n_sdf:4d} receptor={has_rec}")
                if picked >= 10:
                    break
        if picked == 0:
            print(f"  {z:8s} (no staged-data recoverable target found)")

    out = ROOT / "tier_b_sample.txt"
    out.write_text("\n".join(sample) + "\n")
    print(f"\nsample size: {len(sample)} → {out}")


if __name__ == "__main__":
    main()
