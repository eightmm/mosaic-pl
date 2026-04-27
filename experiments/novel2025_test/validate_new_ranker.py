"""Offline validation: compare new RRF+consensus ranker vs lscore baseline.

For each target with a crystal pose:
  1. collect_pose_scores → 새 ranker (select_best_pose / select_diverse_top_k)
  2. legacy ranker (select_best_pose_pRMSD_legacy)
  3. lscore-only baseline (max(lscore))
  4. true_rmsd lookup from per_pose_scores.csv

집계: top-1 < 2 Å, best-of-5 < 2 Å, per-zone.

per_pose_scores.csv 의 (target, source, pose_name) → true_rmsd 매핑이 핵심.
PoseScore 의 (source, pose_name) 와 매칭한다 (collect_pose_scores 의 dedup 후 source 가
다를 수 있어 fallback 도 시도).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import json

from compute_submission_scores import (  # noqa: E402
    PoseScore,
    collect_pose_scores,
    select_best_pose,
    select_best_pose_pRMSD_legacy,
    select_best_pose_cluster,
    select_best_pose_cascaded,
    select_top_k_cluster,
    select_top_k_cluster_by,
    select_top_k_cascaded,
    select_diverse_top_k,
    _load_pose_mol,
    _pose_pair_rmsd,
)

CSV_PATH = ROOT / "per_pose_scores.csv"
RUNS = REPO / "experiments" / "runs"


def load_truth_map() -> tuple[dict, dict]:
    """Return ({(target, source, pose_name): true_rmsd}, {target: zone})."""
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


def lookup_true_rmsd(
    truth: dict, target: str, pose: PoseScore
) -> float | None:
    """Find true_rmsd in CSV for this PoseScore, with source fallbacks."""
    key = (target, pose.source, pose.pose_name)
    if key in truth:
        return truth[key]
    # Source might differ (CSV dedup vs collect_pose_scores dedup ordering).
    # Try all source variants for the same pose_name.
    for k, v in truth.items():
        if k[0] == target and k[2] == pose.pose_name:
            return v
    return None


def lscore_top1(poses: list[PoseScore]) -> PoseScore | None:
    cand = [p for p in poses if p.lscore is not None]
    if not cand:
        return None
    return max(cand, key=lambda p: p.lscore)


def lscore_top5_diverse(
    poses: list[PoseScore], k: int = 5, rmsd_threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[PoseScore]:
    """Fair best-5 baseline: lscore-ordered with ≥ rmsd_threshold heavy-atom diversity."""
    cand = [p for p in poses if p.lscore is not None]
    if not cand:
        return []
    cand.sort(key=lambda p: -p.lscore)
    if mol_cache is None:
        mol_cache = {}
    selected: list[PoseScore] = []
    selected_mols: list = []
    for c in cand:
        m = _load_pose_mol(c, mol_cache)
        if not selected:
            selected.append(c); selected_mols.append(m)
            if len(selected) == k: break
            continue
        if m is None:
            continue
        too_close = False
        for prev in selected_mols:
            if prev is None: continue
            r = _pose_pair_rmsd(m, prev)
            if r is not None and r < rmsd_threshold:
                too_close = True
                break
        if not too_close:
            selected.append(c); selected_mols.append(m)
            if len(selected) == k: break
    return selected


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit number of targets (0 = all)")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="Sample every Nth target (after sorting); 1 = all")
    args = ap.parse_args()

    truth, zones = load_truth_map()
    targets = sorted({k[0] for k in truth.keys()})
    print(f"targets in truth map: {len(targets)}")
    if args.sample_stride > 1:
        targets = targets[::args.sample_stride]
        print(f"after stride={args.sample_stride}: {len(targets)} targets")
    if args.limit > 0:
        targets = targets[: args.limit]
        print(f"limit={args.limit}: {len(targets)} targets")

    # ablation rankers
    def _wrap1(fn):
        return lambda poses: ([fn(poses)] if fn(poses) else [])

    # mol_cache shared per-target across all rankers (saves 3-5x mol load).
    # Each lambda receives (poses, mol_cache); _wrap1 ignores mol_cache.
    # Cluster ranker dropped — already shown to lose in stride=10 ablation
    # (cluster_sum 30.6%/40.8% vs lscore_top5 37.2%/44.2%) and 3x more expensive.
    rankers = [
        ("lscore",          lambda poses, mc: ([lscore_top1(poses)] if lscore_top1(poses) else [])),
        ("lscore_top5",     lambda poses, mc: lscore_top5_diverse(poses, k=5, mol_cache=mc)),
        ("legacy_pRMSD",    lambda poses, mc: ([select_best_pose_pRMSD_legacy(poses)] if select_best_pose_pRMSD_legacy(poses) else [])),
        ("cascaded",        lambda poses, mc: ([select_best_pose_cascaded(poses)] if select_best_pose_cascaded(poses) else [])),
        ("cascaded_top5",   lambda poses, mc: select_top_k_cascaded(poses, k=5, mol_cache=mc)),
    ]

    # results[ranker_id][zone] = [(top1_rmsd, best5_rmsd or None), ...]
    results = {rid: {} for rid, _ in rankers}

    fails: list[str] = []
    t_start = time.time()
    for i, target in enumerate(targets, 1):
        run_dir = RUNS / f"{target}_input"
        if not run_dir.exists():
            fails.append(target)
            continue
        try:
            poses = collect_pose_scores(run_dir)
        except Exception as e:
            print(f"  [{target}] collect failed: {e}")
            fails.append(target)
            continue
        if not poses:
            fails.append(target)
            continue

        zone = zones.get(target, "?")
        # Per-target shared mol cache: pose mols loaded once across rankers
        # (saves 3-5x SDMolSupplier calls when multiple rankers need coords).
        mol_cache: dict = {}

        # First pass: run all rankers, collect picks. We'll only commit to
        # results if EVERY ranker's top-1 has a successful truth lookup —
        # this keeps the denominator identical across rankers so SR is
        # comparable on the same target set. A target where any ranker's
        # top-1 fails lookup is added to ``lookup_fails`` and dropped from
        # all rankers.
        per_ranker_pick: dict[str, tuple[list[PoseScore], float | None]] = {}
        any_lookup_fail = False
        for rid, fn in rankers:
            try:
                picks = fn(poses, mol_cache)
            except Exception as e:
                print(f"  [{target}/{rid}] failed: {e}", flush=True)
                any_lookup_fail = True
                break
            if not picks or picks[0] is None:
                # Some rankers may legitimately return empty (no candidate
                # with required score type). Treat as missing → drop target
                # for fair comparison.
                any_lookup_fail = True
                break
            top1_t = lookup_true_rmsd(truth, target, picks[0])
            if top1_t is None:
                print(f"  [{target}/{rid}] truth lookup miss: src={picks[0].source!r} name={picks[0].pose_name!r}", flush=True)
                any_lookup_fail = True
                break
            per_ranker_pick[rid] = (picks, top1_t)

        if any_lookup_fail:
            fails.append(target)
            continue

        # Commit: every ranker has a valid top-1. Compute best-of-5 best.
        for rid, (picks, top1_t) in per_ranker_pick.items():
            best5 = top1_t  # top-1 is always among the top-5 candidates
            for p in picks[1:]:
                if p is None:
                    continue
                t = lookup_true_rmsd(truth, target, p)
                if t is None:
                    continue
                if t < best5:
                    best5 = t
            results[rid].setdefault(zone, []).append((top1_t, best5))

        # Periodic JSON checkpoint — survives if process dies mid-batch.
        if i % 50 == 0:
            import json as _json
            ckpt = ROOT / "_validate_full489_ckpt.json"
            ROOT.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(_json.dumps({
                "i": i, "n": len(targets),
                "results": {rid: {z: rs for z, rs in zs.items()} for rid, zs in results.items()},
                "fails": fails,
            }))

        if i % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(targets) - i)
            print(f"  [{i}/{len(targets)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\nfails: {len(fails)} (skipped — no run dir or empty)")
    print()

    # Per-zone aggregation
    print("=== Per-ranker top-1 / best-5 success rate ===")
    print(f"  {'ranker':<14}  {'zone':<8}  {'n':>4}  {'top1<2Å':>8}  {'best5<2Å':>9}  {'top1_med':>9}  {'best5_med':>10}")
    print("  " + "-" * 76)
    for rid, _ in rankers:
        zone_stats = results[rid]
        all_top1 = []
        all_best5 = []
        for zone in ("novel", "remote", "related"):
            data = zone_stats.get(zone)
            if not data:
                continue
            top1_vals = [t for t, _ in data]
            best5_vals = [b for _, b in data if b is not None]
            n = len(top1_vals)
            n2 = sum(1 for t in top1_vals if t < 2.0)
            n5 = sum(1 for b in best5_vals if b < 2.0)
            mt = sorted(top1_vals)[n // 2]
            mb = sorted(best5_vals)[len(best5_vals) // 2] if best5_vals else 0.0
            print(f"  {rid:<14}  {zone:<8}  {n:>4}  {n2:>4}({100*n2/n:>4.1f}%)  {n5:>4}({100*n5/len(best5_vals) if best5_vals else 0:>4.1f}%)  {mt:>9.2f}  {mb:>10.2f}")
            all_top1.extend(top1_vals)
            all_best5.extend(best5_vals)
        if all_top1:
            n = len(all_top1)
            n2 = sum(1 for t in all_top1 if t < 2.0)
            n5 = sum(1 for b in all_best5 if b < 2.0)
            mt = sorted(all_top1)[n // 2]
            mb = sorted(all_best5)[len(all_best5) // 2] if all_best5 else 0.0
            print(f"  {rid:<14}  {'TOTAL':<8}  {n:>4}  {n2:>4}({100*n2/n:>4.1f}%)  {n5:>4}({100*n5/len(all_best5) if all_best5 else 0:>4.1f}%)  {mt:>9.2f}  {mb:>10.2f}")
        print()


if __name__ == "__main__":
    main()
