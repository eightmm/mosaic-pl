#!/usr/bin/env python3
"""Tier-A pose-selection ablation on per_pose_scores.csv (scalar signals only).

Measures top-1 success rate (selected pose true_rmsd < 2 Å) for several
selection strategies, vs the oracle ceiling (any pose < 2 Å exists).

LIMITATION: the CSV has no coordinates, so the production selector's
consensus / cluster modifier (needs cross-pose RMSD) is NOT reproduced.
This compares the SCALAR portion of selection only — useful to find which
scalar signals lift SR and to size the oracle gap, not to replace the full
production ranker eval.

Reported: overall + per seq_zone + cluster_rep_only@100% (XChem-cluster
de-weighted headline per CLAUDE.md).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent / "novel2025_runs" / "per_pose_scores.csv"
CLUSTER_CSV = ROOT / "cluster_targets_100.csv"
HIT = 2.0
COFOLD_FAMS = {"cofold_af3", "cofold_boltz2", "cofold_boltz2x", "cofold_protenix"}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _family(src: str) -> str:
    for fam in COFOLD_FAMS:
        if src.startswith(fam):
            return fam
    return src.split("_seed")[0]


def load():
    by_target = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            tr = _f(r["true_rmsd"])
            if tr is None:
                continue
            by_target[r["target"]].append({
                "fam": _family(r["source"]),
                "is_cofold": _family(r["source"]) in COFOLD_FAMS,
                "zone": r["seq_zone"],
                "lscore": _f(r["lscore"]),
                "prmsd": _f(r["prmsd"]),
                "plddt": _f(r["plddt"]),
                "iptm": _f(r["iptm"]),
                "ptm": _f(r["ptm"]),
                "conf": _f(r["conf"]),
                "binder": _f(r["boltz_binder_prob"]),
                "true_rmsd": tr,
            })
    return by_target


def rrf_pick(poses, k=60):
    """Production-style RRF over {lscore, plddt(within cofold-family rank),
    iptm, ptm, conf}; cofold-only scorers rank within cofold pool."""
    score = defaultdict(float)

    def acc(ranked):
        for r, p in enumerate(ranked):
            score[id(p)] += 1.0 / (k + r + 1)

    acc(sorted([p for p in poses if p["lscore"] is not None],
               key=lambda p: -p["lscore"]))
    cof = [p for p in poses if p["is_cofold"]]
    by_fam = defaultdict(list)
    for p in cof:
        if p["plddt"] is not None:
            by_fam[p["fam"]].append(p)
    for ps in by_fam.values():
        acc(sorted(ps, key=lambda p: -p["plddt"]))
    for attr in ("iptm", "ptm", "conf"):
        acc(sorted([p for p in cof if p[attr] is not None],
                   key=lambda p: -p[attr]))
    return max(poses, key=lambda p: score[id(p)])


STRATS = {
    "lscore":            lambda ps: max(ps, key=lambda p: (p["lscore"] if p["lscore"] is not None else -1)),
    "prmsd":             lambda ps: min(ps, key=lambda p: (p["prmsd"] if p["prmsd"] is not None else 1e9)),
    "rrf_scalar":        rrf_pick,
    "cofold_only_lscore": lambda ps: max([p for p in ps if p["is_cofold"]] or ps,
                                         key=lambda p: (p["lscore"] if p["lscore"] is not None else -1)),
    "lscore_plddt_tie":  lambda ps: max(ps, key=lambda p: ((p["lscore"] or -1), (p["plddt"] or -1))),
    "lscore_binder_gate": lambda ps: max(
        [p for p in ps if (p["binder"] or 0) >= 0.5] or ps,
        key=lambda p: (p["lscore"] if p["lscore"] is not None else -1)),
}


def main():
    by_target = load()
    clu = {}
    if CLUSTER_CSV.exists():
        for r in csv.DictReader(open(CLUSTER_CSV)):
            clu[r["target"].replace("_input", "")] = r["cluster_rep"].replace("_input", "")
    reps = set(clu.values())

    targets = list(by_target)
    n = len(targets)
    print(f"targets: {n}  | poses: {sum(len(v) for v in by_target.values()):,}")

    # zone per target (majority/any — all poses of a target share zone)
    zone = {t: by_target[t][0]["zone"] for t in targets}

    # oracle
    oracle = {t: any(p["true_rmsd"] < HIT for p in by_target[t]) for t in targets}

    def sr(hitset, subset):
        sub = [t for t in targets if subset(t)]
        return 100.0 * sum(hitset[t] for t in sub) / len(sub) if sub else 0.0

    subsets = {
        "ALL":              lambda t: True,
        "novel":            lambda t: zone[t] == "novel",
        "remote":           lambda t: zone[t] == "remote",
        "related":          lambda t: zone[t] == "related",
        "cluster_rep@100":  lambda t: (t in reps) or (t not in clu),
    }

    print(f"\n{'strategy':22s} " + " ".join(f"{k:>16s}" for k in subsets))
    # oracle row
    orow = {nm: sr(oracle, fn) for nm, fn in subsets.items()}
    print(f"{'ORACLE(any<2A)':22s} " + " ".join(f"{orow[k]:15.1f}%" for k in subsets))
    print("-" * 120)
    for sname, fn in STRATS.items():
        hits = {}
        for t in targets:
            pick = fn(by_target[t])
            hits[t] = pick["true_rmsd"] < HIT
        row = {nm: sr(hits, ssfn) for nm, ssfn in subsets.items()}
        print(f"{sname:22s} " + " ".join(f"{row[k]:15.1f}%" for k in subsets))


if __name__ == "__main__":
    main()
