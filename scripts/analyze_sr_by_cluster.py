#!/usr/bin/env python3
"""Cluster-aware SR (success-rate) analysis for novel2025.

Many of the 499 novel2025 targets are XChem fragment-screen series on
a single enzyme — one cluster alone holds 130 targets, so per-target
averages are heavily biased by whichever cluster happens to have the
most bound ligands. This script reports SR under three aggregations
that strip that bias:

  per_target            current behaviour — every row counted equally,
                        biased toward big clusters
  cluster_rep_only      one row per cluster (the cluster representative
                        chosen by mmseqs); macro-average over enzymes
  cluster_any           a cluster passes if ANY of its members passes
                        (lower-bound: how many *unique enzymes* did we
                        solve at least once?)
  cluster_mean          each cluster contributes one row equal to the
                        mean SR of its members; macro-average that

Multiple identity thresholds are tested (100% / 95% / 70% / 50% / 30%).
Output goes to stdout + ``experiments/novel2025_test/sr_by_cluster.csv``
so the same numbers can be re-plotted later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PPS = REPO / "experiments" / "novel2025_test" / "per_pose_scores.csv"
DEFAULT_CLUSTER_DIR = REPO / "experiments" / "novel2025_test"
DEFAULT_OUT = REPO / "experiments" / "novel2025_test" / "sr_by_cluster.csv"


# --------------------------------------------------------------------------- #
# Per-target metrics                                                          #
# --------------------------------------------------------------------------- #

def compute_per_target(df: pd.DataFrame, native_thr: float = 2.0) -> pd.DataFrame:
    """Per-target ranker outputs.

    Top-1: pose with max ``lscore``. Top-5 (no-diversity approximation):
    five highest ``lscore`` poses; best ``true_rmsd`` among them. Oracle:
    minimum ``true_rmsd`` over the entire pose pool. Native bool = below
    ``native_thr`` Å.

    The diversity-aware top-5 used in production needs pairwise pose RMSD
    which would mean re-loading every SDF. For SR-summary granularity the
    no-diversity top-5 is a tight upper bound (the production pipeline
    just enforces a 2 Å pairwise cut on the same ordering).
    """
    df = df.dropna(subset=["lscore", "true_rmsd"]).copy()

    rows: list[dict] = []
    # ``seq_zone`` is target-level; pull it from the first row per target.
    zone_lookup = df.drop_duplicates("target").set_index("target")["seq_zone"]

    for target, g in df.groupby("target"):
        if g.empty:
            continue
        sorted_by_lscore = g.sort_values("lscore", ascending=False)
        top1_rmsd = float(sorted_by_lscore.iloc[0]["true_rmsd"])
        top5_pool = sorted_by_lscore.head(5)
        top5_rmsd = float(top5_pool["true_rmsd"].min())
        oracle_rmsd = float(g["true_rmsd"].min())
        rows.append({
            "target": target,
            "seq_zone": zone_lookup.get(target, "?"),
            "n_poses": len(g),
            "top1_rmsd": top1_rmsd,
            "top5_rmsd": top5_rmsd,
            "oracle_rmsd": oracle_rmsd,
            "top1_native": top1_rmsd < native_thr,
            "top5_native": top5_rmsd < native_thr,
            "oracle_native": oracle_rmsd < native_thr,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregation policies                                                        #
# --------------------------------------------------------------------------- #

def _sr_table(metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-zone + total SR for top-1 / top-5 / oracle. Returns a long-format
    DataFrame keyed by (label, zone, metric)."""
    out: list[dict] = []
    metric_cols = ["top1_native", "top5_native", "oracle_native"]
    for zone in ("novel", "remote", "related", "ALL"):
        sub = metrics if zone == "ALL" else metrics[metrics.seq_zone == zone]
        n = len(sub)
        for col in metric_cols:
            v = sub[col].mean() if n else float("nan")
            out.append({
                "policy": label,
                "zone": zone,
                "metric": col.replace("_native", ""),
                "n": n,
                "sr": v,
            })
    return pd.DataFrame(out)


def aggregate_per_target(metrics: pd.DataFrame) -> pd.DataFrame:
    return _sr_table(metrics, "per_target")


def aggregate_cluster_rep_only(
    metrics: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Drop everything that isn't the cluster's chosen representative."""
    merged = metrics.merge(clusters, on="target", how="left")
    reps = merged[merged.target == merged.cluster_rep]
    return _sr_table(reps, "cluster_rep_only")


def aggregate_cluster_any(
    metrics: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Cluster passes if ANY of its members passes. Zone is the rep's zone
    (singletons trivially keep their own zone)."""
    merged = metrics.merge(clusters, on="target", how="left")
    rep_zone = merged.groupby("cluster_rep")["seq_zone"].first()
    cluster_pass = merged.groupby("cluster_rep").agg(
        top1_native=("top1_native", "any"),
        top5_native=("top5_native", "any"),
        oracle_native=("oracle_native", "any"),
    )
    cluster_pass["seq_zone"] = rep_zone
    cluster_pass = cluster_pass.reset_index().rename(columns={"cluster_rep": "target"})
    return _sr_table(cluster_pass, "cluster_any")


def aggregate_cluster_mean(
    metrics: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Each cluster contributes the mean of its members' booleans (range 0-1).
    Macro-average across clusters; un-biases by cluster size."""
    merged = metrics.merge(clusters, on="target", how="left")
    rep_zone = merged.groupby("cluster_rep")["seq_zone"].first()
    cluster_mean = merged.groupby("cluster_rep").agg(
        top1_native=("top1_native", "mean"),
        top5_native=("top5_native", "mean"),
        oracle_native=("oracle_native", "mean"),
    )
    cluster_mean["seq_zone"] = rep_zone
    cluster_mean = cluster_mean.reset_index().rename(columns={"cluster_rep": "target"})
    return _sr_table(cluster_mean, "cluster_mean")


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def print_table(df: pd.DataFrame, threshold_label: str | None = None) -> None:
    pivot = df.pivot_table(
        index=["policy", "zone"],
        columns="metric",
        values="sr",
    )
    n_pivot = df.pivot_table(
        index=["policy", "zone"],
        columns="metric",
        values="n",
    )
    pivot = pivot.reindex(columns=["top1", "top5", "oracle"])
    n_pivot = n_pivot.reindex(columns=["top1", "top5", "oracle"])
    if threshold_label:
        print(f"\n=== {threshold_label} ===")
    header = (
        f"  {'policy':<20}{'zone':<10}{'n':>7}"
        f"{'top1':>10}{'top5':>10}{'oracle':>10}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for (policy, zone), row in pivot.iterrows():
        n_val = n_pivot.loc[(policy, zone), "top1"]
        n_int = int(n_val) if pd.notna(n_val) else 0
        sr_strs = [
            f"{row[col]:>9.1%}" if pd.notna(row[col]) else "       —"
            for col in ("top1", "top5", "oracle")
        ]
        print(f"  {policy:<20}{zone:<10}{n_int:>7}{sr_strs[0]:>10}{sr_strs[1]:>10}{sr_strs[2]:>10}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-pose-csv", type=Path, default=DEFAULT_PPS)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR,
                        help="Directory containing cluster_targets_<NNN>.csv files")
    parser.add_argument("--thresholds", type=int, nargs="+",
                        default=[100, 95, 70, 50, 30])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--native-rmsd", type=float, default=2.0)
    args = parser.parse_args()

    print(f"[sr] loading per_pose_scores: {args.per_pose_csv}")
    df = pd.read_csv(args.per_pose_csv)
    print(f"[sr]   raw rows: {len(df):,}")
    df = df.dropna(subset=["lscore", "true_rmsd"])
    print(f"[sr]   rows w/ lscore + true_rmsd: {len(df):,}")
    print(f"[sr]   unique targets: {df['target'].nunique()}")

    print("[sr] computing per-target metrics …")
    metrics = compute_per_target(df, native_thr=args.native_rmsd)
    print(f"[sr]   per-target rows: {len(metrics)}")
    metrics_path = args.cluster_dir / "sr_per_target.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[sr]   saved per-target metrics → {metrics_path}")

    all_results: list[pd.DataFrame] = []
    per_target_table = aggregate_per_target(metrics)
    per_target_table["threshold"] = "n/a"
    all_results.append(per_target_table)
    print_table(per_target_table, "Per-target (current — biased by cluster size)")

    for thr in args.thresholds:
        cluster_path = args.cluster_dir / f"cluster_targets_{thr:03d}.csv"
        if not cluster_path.exists():
            print(f"[sr] missing {cluster_path}; skip threshold {thr}")
            continue
        clusters = pd.read_csv(cluster_path)
        # Strip the leading "_input" suffix used inside cluster_targets so
        # joins with per-pose data line up regardless of how either side
        # spelled the target id.
        if metrics["target"].str.endswith("_input").any() != clusters["target"].str.endswith("_input").any():
            metrics["target"] = metrics["target"].str.replace("_input$", "", regex=True)
            clusters["target"] = clusters["target"].str.replace("_input$", "", regex=True)
            clusters["cluster_rep"] = clusters["cluster_rep"].str.replace("_input$", "", regex=True)

        threshold_label = f"identity threshold = {thr/100:.2f}"
        for fn, label in (
            (aggregate_cluster_rep_only, "cluster_rep_only"),
            (aggregate_cluster_any, "cluster_any"),
            (aggregate_cluster_mean, "cluster_mean"),
        ):
            tbl = fn(metrics, clusters)
            tbl["threshold"] = f"{thr}%"
            all_results.append(tbl)

        # Pretty-print the three policies grouped by threshold
        thr_chunk = pd.concat(
            [r for r in all_results[-3:]], ignore_index=True
        )
        print_table(thr_chunk, threshold_label)

    out_df = pd.concat(all_results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\n[sr] wrote {args.output} ({len(out_df)} rows; long-format SR table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
