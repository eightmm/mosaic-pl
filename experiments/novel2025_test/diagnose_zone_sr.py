#!/usr/bin/env python3
"""Diagnose why `related` zone SR < `remote` (counterintuitive).

Hypothesis: the `related` zone is dominated by a few mega-clusters of
near-identical targets (notably the XChem fragment screen on 9s4h, ~130
members per CLAUDE.md) — tiny low-affinity fragments that dock poorly and
drag per-target SR down. Tests: per-zone cluster composition + SR under
per_target / cluster_rep_only / cluster_mean, and SR with the biggest
cluster removed.

CSV-only (scalar), uses rrf_scalar pick from the ablation module.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import ablate_pose_selection as A  # reuse load() + rrf_pick()

ROOT = Path(__file__).resolve().parent
CLUSTER_CSV = ROOT / "cluster_targets_100.csv"
HIT = 2.0


def main():
    by_target = A.load()
    targets = list(by_target)
    zone = {t: by_target[t][0]["zone"] for t in targets}
    oracle = {t: any(p["true_rmsd"] < HIT for p in by_target[t]) for t in targets}
    rrf_hit = {t: A.rrf_pick(by_target[t])["true_rmsd"] < HIT for t in targets}

    # cluster map (strip _input). target may not be in cluster csv → singleton.
    rep = {}
    size = {}
    if CLUSTER_CSV.exists():
        for r in csv.DictReader(open(CLUSTER_CSV)):
            t = r["target"].replace("_input", "")
            rep[t] = r["cluster_rep"].replace("_input", "")
            size[t] = int(r["cluster_size"])
    for t in targets:
        rep.setdefault(t, t)
        size.setdefault(t, 1)

    # ---- per-zone cluster composition ----
    print("=== zone composition ===")
    print(f"{'zone':10s} {'targets':>8s} {'clusters':>9s} {'biggest_cluster(rep:size)':>30s}")
    for z in ("novel", "remote", "related"):
        zt = [t for t in targets if zone[t] == z]
        reps_in = defaultdict(int)
        for t in zt:
            reps_in[rep[t]] += 1
        big = max(reps_in.items(), key=lambda kv: kv[1]) if reps_in else ("-", 0)
        print(f"{z:10s} {len(zt):8d} {len(reps_in):9d} {big[0]+':'+str(big[1]):>30s}")

    # ---- SR modes per zone ----
    def sr_per_target(zt, hitset):
        return 100.0 * sum(hitset[t] for t in zt) / len(zt) if zt else 0.0

    def sr_cluster_rep(zt, hitset):
        # one vote per cluster: use the rep target's hit (rep must be in zt set)
        rs = {rep[t] for t in zt}
        rs = [r for r in rs if r in hitset]
        return 100.0 * sum(hitset[r] for r in rs) / len(rs) if rs else 0.0

    def sr_cluster_mean(zt, hitset):
        # mean of per-cluster hit-rates (each cluster weighted equally)
        buckets = defaultdict(list)
        for t in zt:
            buckets[rep[t]].append(hitset[t])
        rates = [100.0 * sum(v) / len(v) for v in buckets.values()]
        return sum(rates) / len(rates) if rates else 0.0

    print("\n=== SR by zone × aggregation (rrf_scalar pick) ===")
    print(f"{'zone':10s} {'per_target':>11s} {'cluster_rep':>12s} {'cluster_mean':>13s} {'oracle_pt':>10s}")
    for z in ("novel", "remote", "related", "ALL"):
        zt = targets if z == "ALL" else [t for t in targets if zone[t] == z]
        print(f"{z:10s} {sr_per_target(zt, rrf_hit):10.1f}% {sr_cluster_rep(zt, rrf_hit):11.1f}% "
              f"{sr_cluster_mean(zt, rrf_hit):12.1f}% {sr_per_target(zt, oracle):9.1f}%")

    # ---- biggest cluster in `related`: isolate + remove ----
    related = [t for t in targets if zone[t] == "related"]
    reps_in = defaultdict(list)
    for t in related:
        reps_in[rep[t]].append(t)
    big_rep, big_members = max(reps_in.items(), key=lambda kv: len(kv[1]))
    print(f"\n=== biggest `related` cluster: {big_rep}  ({len(big_members)} members) ===")
    print(f"  cluster per-target SR : {sr_per_target(big_members, rrf_hit):.1f}%  "
          f"oracle {sr_per_target(big_members, oracle):.1f}%")
    rest = [t for t in related if t not in set(big_members)]
    print(f"  related WITHOUT it    : per_target {sr_per_target(rest, rrf_hit):.1f}%  "
          f"oracle {sr_per_target(rest, oracle):.1f}%  (n={len(rest)})")
    print(f"  related WITH it       : per_target {sr_per_target(related, rrf_hit):.1f}%  "
          f"oracle {sr_per_target(related, oracle):.1f}%  (n={len(related)})")

    # ligand size proxy: n poses with very small min true_rmsd dispersion? skip — no atom count in CSV.
    # Instead: top-5 biggest related clusters and their SR
    print("\n=== top related clusters by size (SR + oracle) ===")
    ranked = sorted(reps_in.items(), key=lambda kv: -len(kv[1]))[:8]
    print(f"  {'rep':14s} {'n':>4s} {'SR':>7s} {'oracle':>7s}")
    for r, mem in ranked:
        print(f"  {r:14s} {len(mem):4d} {sr_per_target(mem, rrf_hit):6.1f}% {sr_per_target(mem, oracle):6.1f}%")


if __name__ == "__main__":
    main()
