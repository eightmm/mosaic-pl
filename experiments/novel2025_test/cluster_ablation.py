"""Cluster-quality ablation: cluster ONCE per target, score by N quality functions.

Per target:
  1. collect_pose_scores
  2. _build_pose_clusters (one expensive call ~3-5s)
  3. for each of 6 quality functions:
       - sort clusters by quality
       - take top-5 representatives
       - look up true_rmsd of each, compute top1<2 / best5<2

Reuses the truth map from per_pose_scores.csv (no need to re-do USalign).
"""
from __future__ import annotations

import csv
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from compute_submission_scores import (  # noqa: E402
    PoseScore,
    collect_pose_scores,
    _build_pose_clusters,
    _cluster_representative,
    _cluster_quality,
    _cq_size,
    _cq_n_fam,
    _cq_max_lscore,
    _cq_sum_lscore,
    _cq_size_x_max_lscore,
    _source_family,
)

CSV_PATH = ROOT / "per_pose_scores.csv"
RUNS = REPO / "experiments" / "runs"

QUALITY_FNS = [
    ("lscore_x_fam",  _cluster_quality),
    ("size",          _cq_size),
    ("n_fam",         _cq_n_fam),
    ("max_lscore",    _cq_max_lscore),
    ("sum_lscore",    _cq_sum_lscore,),
    ("size_x_lscore", _cq_size_x_max_lscore),
]


def load_truth_map() -> tuple[dict, dict]:
    truth: dict[tuple[str, str, str], float] = {}
    zones: dict[str, str] = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            t = r["true_rmsd"]
            if t in ("", "None"):
                continue
            try:
                truth[(r["target"], r["source"], r["pose_name"])] = float(t)
            except ValueError:
                continue
            zones[r["target"]] = r["seq_zone"]
    return truth, zones


def lookup_true_rmsd(truth: dict, target: str, pose: PoseScore) -> float | None:
    key = (target, pose.source, pose.pose_name)
    if key in truth:
        return truth[key]
    for k, v in truth.items():
        if k[0] == target and k[2] == pose.pose_name:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-stride", type=int, default=1)
    args = ap.parse_args()

    truth, zones = load_truth_map()
    targets = sorted({k[0] for k in truth.keys()})
    if args.sample_stride > 1:
        targets = targets[::args.sample_stride]
    if args.limit > 0:
        targets = targets[: args.limit]
    print(f"targets: {len(targets)}")

    # results[quality_id][zone] = [(top1_rmsd, best5_rmsd), ...]
    results = {qid: {} for qid, _ in QUALITY_FNS}
    fails: list[str] = []
    t_start = time.time()

    for i, target in enumerate(targets, 1):
        run_dir = RUNS / f"{target}_input"
        if not run_dir.exists():
            fails.append(target); continue
        try:
            poses = collect_pose_scores(run_dir)
        except Exception as e:
            print(f"  [{target}] collect failed: {e}", flush=True)
            fails.append(target); continue
        if not poses:
            fails.append(target); continue

        # Build clusters ONCE
        try:
            clusters = _build_pose_clusters(poses, threshold=2.0)
        except Exception as e:
            print(f"  [{target}] cluster build failed: {e}", flush=True)
            fails.append(target); continue
        if not clusters:
            fails.append(target); continue

        zone = zones.get(target, "?")
        for qid, fn in QUALITY_FNS:
            try:
                ranked = sorted(clusters, key=fn, reverse=True)
                top5_reps = [_cluster_representative(c) for c in ranked[:5]]
            except Exception as e:
                print(f"  [{target}/{qid}] failed: {e}", flush=True)
                continue
            if not top5_reps:
                continue
            top1_t = lookup_true_rmsd(truth, target, top5_reps[0])
            if top1_t is None:
                continue
            best5 = top1_t
            for p in top5_reps[1:]:
                t = lookup_true_rmsd(truth, target, p)
                if t is not None and t < best5:
                    best5 = t
            results[qid].setdefault(zone, []).append((top1_t, best5))

        if i % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(targets) - i)
            print(f"  [{i}/{len(targets)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nfails: {len(fails)}\n")
    print("=== Cluster quality ablation: top-1 / best-5 SR ===")
    print(f"  {'quality':<16}  {'zone':<8} {'n':>4}  {'top1<2Å':>9}  {'best5<2Å':>10}  {'top1_med':>9}  {'best5_med':>10}")
    print("  " + "-" * 80)
    for qid, _ in QUALITY_FNS:
        zs = results[qid]
        all_t1, all_b5 = [], []
        for zone in ("novel", "remote", "related"):
            d = zs.get(zone)
            if not d:
                continue
            top1 = [t for t, _ in d]
            best5 = [b for _, b in d]
            n = len(top1)
            n2 = sum(1 for v in top1 if v < 2.0)
            n5 = sum(1 for v in best5 if v < 2.0)
            mt = sorted(top1)[n // 2]
            mb = sorted(best5)[n // 2]
            print(f"  {qid:<16}  {zone:<8} {n:>4}  {n2:>4}({100*n2/n:>4.1f}%)  {n5:>5}({100*n5/n:>4.1f}%)  {mt:>9.2f}  {mb:>10.2f}")
            all_t1.extend(top1); all_b5.extend(best5)
        if all_t1:
            n = len(all_t1)
            n2 = sum(1 for v in all_t1 if v < 2.0)
            n5 = sum(1 for v in all_b5 if v < 2.0)
            mt = sorted(all_t1)[n // 2]
            mb = sorted(all_b5)[n // 2]
            print(f"  {qid:<16}  {'TOTAL':<8} {n:>4}  {n2:>4}({100*n2/n:>4.1f}%)  {n5:>5}({100*n5/n:>4.1f}%)  {mt:>9.2f}  {mb:>10.2f}")
        print()


if __name__ == "__main__":
    main()
