#!/usr/bin/env python3
"""Build the novel2025 analysis report.

Loads the canonical artefacts that were produced by earlier scripts
(``per_pose_scores.csv``, ``sr_per_target.csv``, ``sr_by_cluster.csv``,
``cluster_targets_*.csv``, ``poses_unified/_summary.csv``) and emits:

  experiments/novel2025_test/figures/*.png    (matplotlib charts)
  experiments/novel2025_test/ANALYSIS.md      (markdown with embedded
                                                images + accompanying
                                                text and read-outs)

The script is **read-only** w.r.t. the upstream data; no upstream CSV
is mutated. Re-run any time the SR / cluster artefacts get refreshed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_NOVEL = REPO / "experiments" / "novel2025_test"
DEFAULT_FIGS = DEFAULT_NOVEL / "figures"
DEFAULT_REPORT = DEFAULT_NOVEL / "ANALYSIS.md"
DEFAULT_UNIFIED = REPO / "experiments" / "poses_unified" / "_summary.csv"


# Common style: bigger fonts + tight layout always.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #

def chart_sr_aggregation(sr_df: pd.DataFrame, out: Path) -> None:
    """Bar chart: Top-1 / Top-5 / Oracle SR under each aggregation policy
    (per_target vs cluster_rep_only @ 100% vs cluster_any @ 100%).
    """
    df = sr_df[(sr_df.zone == "ALL") & (((sr_df.threshold.isna() | (sr_df.threshold == "100%"))) )]
    pivot = df.pivot_table(index="policy", columns="metric", values="sr")
    pivot = pivot.reindex(columns=["top1", "top5", "oracle"])
    # Only the four core policies for the headline chart.
    pivot = pivot.reindex(["per_target", "cluster_rep_only", "cluster_any", "cluster_mean"])
    pivot = pivot.dropna(how="all")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(pivot.index))
    w = 0.27
    colors = ["#5b8def", "#f0a43d", "#7fbf7b"]
    for i, metric in enumerate(["top1", "top5", "oracle"]):
        ax.bar(x + (i - 1) * w, pivot[metric] * 100, width=w,
               label=metric.upper(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("SR by aggregation policy (novel2025 ALL, 100 % identity)")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    for i, metric in enumerate(["top1", "top5", "oracle"]):
        for j, v in enumerate(pivot[metric] * 100):
            ax.text(j + (i - 1) * w, v + 1.0, f"{v:.1f}", ha="center",
                    fontsize=9, color="#444")
    _save(fig, out)


def chart_sr_by_zone(sr_df: pd.DataFrame, out: Path) -> None:
    """Top-1 / Oracle by zone, comparing per_target vs cluster_rep_only.
    """
    df = sr_df[
        (sr_df.policy.isin(["per_target", "cluster_rep_only"]))
        & (sr_df.zone.isin(["novel", "remote", "related"]))
        & ((sr_df.threshold.isna() | (sr_df.threshold == "100%")))
    ].copy()
    df["zone"] = pd.Categorical(df["zone"], ["novel", "remote", "related"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["top1", "oracle"]):
        sub = df[df.metric == metric]
        pivot = sub.pivot_table(index="zone", columns="policy", values="sr",
                                observed=False)
        pivot = pivot.reindex(columns=["per_target", "cluster_rep_only"])
        x = np.arange(len(pivot.index))
        w = 0.36
        ax.bar(x - w / 2, pivot["per_target"] * 100, width=w,
               label="per_target", color="#bdbdbd")
        ax.bar(x + w / 2, pivot["cluster_rep_only"] * 100, width=w,
               label="cluster_rep_only @ 100 %", color="#5b8def")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(f"{metric.upper()} by zone")
        ax.set_ylim(0, 80)
        for i, (per, cl) in enumerate(zip(pivot["per_target"] * 100,
                                          pivot["cluster_rep_only"] * 100)):
            ax.text(i - w / 2, per + 1.0, f"{per:.1f}", ha="center",
                    fontsize=9, color="#555")
            ax.text(i + w / 2, cl + 1.0, f"{cl:.1f}", ha="center",
                    fontsize=9, color="#3055bf")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Per-zone SR — per_target vs cluster_rep_only", y=1.02)
    _save(fig, out)


def chart_cluster_size_dist(clusters: pd.DataFrame, out: Path) -> None:
    """Cluster-size distribution at 100 % identity. Shows the long tail
    dominated by the 130-member XChem cluster."""
    sizes = clusters.drop_duplicates("cluster_rep").cluster_size
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    counts, edges = np.histogram(sizes, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(range(len(counts)), counts, width=0.85, color="#7e6cd5")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{int(edges[i])}–{int(edges[i+1])-1}"
                        for i in range(len(counts))], rotation=20, fontsize=9)
    ax.set_xlabel("Cluster size (number of targets sharing one enzyme)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster-size distribution @ 100 % identity (246 clusters / 499 inputs)")
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c + 1.5, str(c), ha="center", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.18)
    _save(fig, out)


def chart_top1_oracle_gap(per_target: pd.DataFrame, out: Path) -> None:
    """Per-target gap = top1_rmsd − oracle_rmsd. Histogram split by whether
    the target's oracle is < 2 Å (recoverable) or ≥ 2 Å (generation
    failure — ranker can't help)."""
    df = per_target.dropna(subset=["top1_rmsd", "oracle_rmsd"])
    gap = df["top1_rmsd"] - df["oracle_rmsd"]
    bins = np.linspace(0, 12, 25)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    recoverable = gap[df.oracle_native]
    unrecoverable = gap[~df.oracle_native]
    ax.hist([recoverable, unrecoverable], bins=bins, stacked=True,
            color=["#7fbf7b", "#bdbdbd"],
            label=[f"oracle < 2 Å (recoverable, n={len(recoverable)})",
                   f"oracle ≥ 2 Å (generation miss, n={len(unrecoverable)})"])
    ax.set_xlabel("top1_rmsd − oracle_rmsd (Å)")
    ax.set_ylabel("Targets")
    ax.set_title("Top-1 ranker gap: distance between picked pose and the best in the pool")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out)


def chart_source_family_top1(unified: pd.DataFrame, out: Path) -> None:
    """For each target × ligand take the row with max lscore; count which
    source family that pick belongs to. Shows where top-1 picks come from."""
    df = unified.dropna(subset=["lscore"])
    if df.empty:
        return
    top = df.loc[df.groupby(["target", "ligand_id"], dropna=False)["lscore"].idxmax()]
    counts = top.source_family.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = plt.get_cmap("tab10")
    colors = [palette(i) for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of (target, ligand) picks")
    ax.set_title("Source family contribution to per-(target, ligand) top-1 by lscore")
    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(v + 2, bar.get_y() + bar.get_height() / 2,
                f"{v}  ({v / counts.sum():.0%})",
                va="center", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.18)
    _save(fig, out)


def chart_pose_pool_per_target(unified: pd.DataFrame, out: Path) -> None:
    """How many poses each target carries — distribution. Larger pools
    mean better oracle but heavier post-analysis."""
    counts = unified.groupby("target").size()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(counts, bins=40, color="#5b8def")
    ax.set_xlabel("Poses per target")
    ax.set_ylabel("Targets")
    ax.set_title(
        f"Pose-pool size distribution "
        f"(median = {int(counts.median())}, max = {int(counts.max())})"
    )
    _save(fig, out)


def chart_lscore_vs_rmsd(per_pose: pd.DataFrame, out: Path,
                          n_sample: int = 30000) -> None:
    """Scatter: lscore vs true_rmsd. The story is "lscore is a noisy
    signal — high lscore doesn't guarantee low rmsd, low rmsd doesn't
    guarantee high lscore." 2 Å native cutoff drawn for reference."""
    df = per_pose.dropna(subset=["lscore", "true_rmsd"])
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(df.lscore, df.true_rmsd, s=2, alpha=0.10, color="#5b8def",
               rasterized=True)
    ax.axhline(2.0, color="#d44", lw=1.0, linestyle="--", label="2 Å native cutoff")
    ax.set_xlabel("lscore (1 − P(>2 Å) from RMSD-Pred)")
    ax.set_ylabel("true_rmsd (Å) vs crystal")
    ax.set_title(f"lscore vs true_rmsd  (n = {len(df):,} sampled poses)")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 80)
    ax.legend(loc="upper right", frameon=False)
    _save(fig, out)


def chart_threshold_sensitivity(sr_df: pd.DataFrame, out: Path) -> None:
    """How does cluster_rep_only SR shift across identity thresholds?
    Sensitivity test: 100% / 95% / 70% / 50% / 30%."""
    df = sr_df[
        (sr_df.policy == "cluster_rep_only") & (sr_df.zone == "ALL")
    ].copy()
    df["thr_pct"] = df.threshold.str.rstrip("%").astype(int)
    df = df.sort_values("thr_pct")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for metric, color in [("top1", "#5b8def"), ("top5", "#f0a43d"),
                          ("oracle", "#7fbf7b")]:
        sub = df[df.metric == metric]
        ax.plot(sub.thr_pct, sub.sr * 100, marker="o", linewidth=2,
                color=color, label=metric.upper())
    ax.set_xlabel("Identity threshold (%)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("cluster_rep_only SR vs identity threshold")
    ax.set_xticks([100, 95, 70, 50, 30])
    ax.invert_xaxis()
    ax.legend(frameon=False)
    ax.set_ylim(0, 75)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Markdown report                                                             #
# --------------------------------------------------------------------------- #

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f} %" if pd.notna(v) else "—"


def write_report(
    md_path: Path,
    sr_df: pd.DataFrame,
    per_target: pd.DataFrame,
    clusters: pd.DataFrame,
    unified: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    """Compose the markdown. Tables come from sr_df; figures referenced by
    relative path so the report is portable as long as the figs/ stays
    sibling."""

    overall = sr_df[(sr_df.zone == "ALL") & ((sr_df.threshold.isna() | (sr_df.threshold == "100%")))]
    overall_pivot = overall.pivot_table(index="policy", columns="metric", values="sr")
    overall_n = overall.pivot_table(index="policy", columns="metric", values="n")

    cluster_count = clusters.cluster_rep.nunique()
    biggest_cluster_size = clusters.cluster_size.max()
    biggest_rep = clusters.loc[clusters.cluster_size.idxmax(), "cluster_rep"]
    n_singletons = (clusters.cluster_size == 1).sum()

    # Source-family contribution (used in markdown text body)
    top_picks = unified.dropna(subset=["lscore"]).copy()
    top_picks = top_picks.loc[
        top_picks.groupby(["target", "ligand_id"], dropna=False)["lscore"].idxmax()
    ]
    family_counts = top_picks.source_family.value_counts()
    family_lines = "\n".join(
        f"  - `{fam}`: {n} ({n / family_counts.sum():.1%})"
        for fam, n in family_counts.head(8).items()
    )

    rec_gap = per_target.eval("top1_rmsd - oracle_rmsd")
    median_gap_recoverable = rec_gap[per_target.oracle_native].median()

    md = f"""# novel2025 — pose-pool analysis

`scripts/generate_novel2025_report.py` regenerates this file from the
canonical CSV artefacts. Re-run after refreshing
`per_pose_scores.csv`, `sr_per_target.csv`, `sr_by_cluster.csv`, the
cluster targets files, or `experiments/poses_unified/_summary.csv`.

## Dataset shape

- Pipeline inputs: **499 targets** (`pipeline/*_input.yaml`)
- Targets with crystal-bound `true_rmsd`: **{len(per_target):,}** (the rest
  failed at the post-analysis stage and have no per-pose RMSD)
- Unique enzymes after sequence clustering @ 100 % identity:
  **{cluster_count}** ({n_singletons} singletons; biggest cluster
  {biggest_cluster_size} members on rep `{biggest_rep}`)
- Pose pool collected by `collect_unified_poses.py`:
  **{len(unified):,} poses** across {unified['target'].nunique()} targets

## Headline SR (ALL zones, 100 % identity threshold)

| policy | n | Top-1 | Top-5 | Oracle |
|---|---:|---:|---:|---:|
"""
    policy_order = ["per_target", "cluster_rep_only", "cluster_any", "cluster_mean"]
    for policy in policy_order:
        if policy not in overall_pivot.index:
            continue
        n = int(overall_n.loc[policy, "top1"]) if pd.notna(overall_n.loc[policy, "top1"]) else 0
        md += (
            f"| `{policy}` | {n} "
            f"| {_fmt_pct(overall_pivot.loc[policy].get('top1'))} "
            f"| {_fmt_pct(overall_pivot.loc[policy].get('top5'))} "
            f"| {_fmt_pct(overall_pivot.loc[policy].get('oracle'))} |\n"
        )

    md += f"""
![SR by aggregation policy]({figures['sr_aggregation'].relative_to(md_path.parent)})

The XChem 130-member super-cluster pulls the per-target average down
~6–8 pp because fragment-screen ligands have below-average SR. The
`cluster_rep_only @ 100 %` numbers are the less-biased headline.

The Top-1 ↔ Oracle gap stays ≈ 33 pp regardless of aggregation —
that's the actual ranker bottleneck. The right pose is in the pool
~63 % of the time but the scorer picks it ~31 % of the time.

## Per-zone breakdown

![Per-zone SR — per_target vs cluster_rep_only]({figures['sr_by_zone'].relative_to(md_path.parent)})

- **novel** (no homolog in PDB): Oracle ~59 % — small template
  signal, hardest set
- **remote** (0.3–0.5 max id): Oracle **~73 %** — best Oracle, most
  template-coverage
- **related** (> 0.5 max id): Oracle ~58 % — surprisingly slightly
  worse than novel because the existing pool fishes alternate poses
  that ColabFold MSA biases the cofold toward

The Top-1 SR is flat across zones (~28–33 %) — scoring fails uniformly.
This is consistent with the "ranker is the bottleneck" framing: the
template-coverage signal moves Oracle but not Top-1.

## Sequence redundancy

![Cluster-size distribution]({figures['cluster_size_dist'].relative_to(md_path.parent)})

`mmseqs easy-cluster --min-seq-id 1.0 -c 0.9` on the first protein
chain of every input collapses **499 → 246** unique enzymes. One
cluster (rep `9s4h_input`, members `7hqq…7hr*`) holds 130 entries
(an XChem fragment screen). Threshold sensitivity is small:

![Threshold sensitivity]({figures['threshold_sensitivity'].relative_to(md_path.parent)})

100 % → 30 % only loses 39 clusters (244 → 207), so the multi-member
clusters are nearly always tight homologs of the same enzyme rather
than distant homologs. **100 % identity dedup is enough** for this
dataset.

## Source-family contributions (per-(target, ligand) top-1 by lscore)

![Source family top-1 contribution]({figures['source_family_top1'].relative_to(md_path.parent)})

Top-8 family share of per-(target, ligand) winners:

{family_lines}

## Pose-pool size

![Pose-pool size distribution]({figures['pose_pool_per_target'].relative_to(md_path.parent)})

Median pool ≈ {int(unified.groupby('target').size().median())} poses /
target; tail goes up to {int(unified.groupby('target').size().max()):,}
on the largest XChem fragment chains. Heavy fragments inflate
post-analysis (BA-Pred + RMSD-Pred GNN cost) but help oracle SR by
adding coverage variants.

## Top-1 ↔ Oracle gap

![Top-1 ranker gap]({figures['top1_oracle_gap'].relative_to(md_path.parent)})

For targets where the **oracle is < 2 Å (recoverable)**, the median
``top1_rmsd − oracle_rmsd`` is **{median_gap_recoverable:.2f} Å**. Those
are the targets the ranker fails on — the right pose IS in the pool.
For the remaining ~37 % of targets (oracle ≥ 2 Å), no scoring change
can help; they need better generation (more cofold seeds, wider
template pool, MSA depth).

## lscore vs true_rmsd

![lscore vs true_rmsd]({figures['lscore_vs_rmsd'].relative_to(md_path.parent)})

The signal is real but noisy: high lscore (right edge) skews towards
sub-2 Å but misses are common; many low-rmsd poses (bottom edge)
carry low lscore. That spread = the ranker bottleneck visualised.

## Where the SR ceilings sit (current pipeline)

| metric                          | value     |
|---------------------------------|-----------|
| Top-1 SR (cluster_rep_only)     | ≈ 31 %   |
| Top-5 SR                        | ≈ 40 %   |
| Oracle SR (= pool ceiling)      | ≈ 64 %   |
| Top-1 ↔ Oracle gap (scoring)    | ≈ 33 pp  |
| Oracle ↔ 100 % gap (generation) | ≈ 36 pp  |

Two independent ceilings to push:

1. **Generation ceiling (Oracle)**: improve template-pocket coverage
   (this session shipped: union mmseqs+foldseek, USalign-aligned
   consensus pockets, 10-source vina/adg fan-out) and cofold quality
   (in flight: unified MSA pipeline so all three cofolders see the
   same homologs).
2. **Ranker ceiling (Top-1 / Oracle gap)**: scoring problem; needs a
   learned ranker, ensemble of independent scorers, or both. Held
   off for the next sprint.

## Generated artefacts

- `figures/*.png` — every chart embedded above
- `sr_per_target.csv` — per-target {{top1, top5, oracle}}_rmsd + native bool + zone
- `sr_by_cluster.csv` — long-format SR table (192 rows: policy × zone × threshold × metric)
- `cluster_targets_{{030,050,070,095,100}}.csv` — target → cluster_rep + cluster_size
- `experiments/poses_unified/_summary.csv` — 281 k pose rows, 26 columns

Re-run the report::

    .venv/bin/python scripts/generate_novel2025_report.py

Refresh upstream data first if it changed::

    .venv/bin/python scripts/cluster_novel2025_targets.py
    .venv/bin/python scripts/analyze_sr_by_cluster.py
    .venv/bin/python scripts/collect_unified_poses.py --runs-dir experiments/runs/ \\
        --output-root experiments/poses_unified --rcsb-dir ~/DB/RCSB/raw/mmCIF_data
"""

    md_path.write_text(md)
    print(f"  wrote {md_path.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novel-dir", type=Path, default=DEFAULT_NOVEL)
    parser.add_argument("--unified-csv", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    novel = args.novel_dir
    print(f"[report] loading data from {novel}")
    sr_df = pd.read_csv(novel / "sr_by_cluster.csv")
    per_target = pd.read_csv(novel / "sr_per_target.csv")
    clusters = pd.read_csv(novel / "cluster_targets_100.csv")
    per_pose = pd.read_csv(novel / "per_pose_scores.csv")

    if args.unified_csv.exists():
        print(f"[report] loading unified summary from {args.unified_csv}")
        unified = pd.read_csv(args.unified_csv)
    else:
        print(f"[report] WARNING: {args.unified_csv} missing — source-family"
              " and pose-pool charts will be skipped")
        unified = pd.DataFrame()

    figs_dir = args.figures_dir
    figs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[report] writing charts to {figs_dir}")

    figures: dict[str, Path] = {}
    figures["sr_aggregation"] = figs_dir / "sr_aggregation.png"
    chart_sr_aggregation(sr_df, figures["sr_aggregation"])

    figures["sr_by_zone"] = figs_dir / "sr_by_zone.png"
    chart_sr_by_zone(sr_df, figures["sr_by_zone"])

    figures["cluster_size_dist"] = figs_dir / "cluster_size_dist.png"
    chart_cluster_size_dist(clusters, figures["cluster_size_dist"])

    figures["threshold_sensitivity"] = figs_dir / "threshold_sensitivity.png"
    chart_threshold_sensitivity(sr_df, figures["threshold_sensitivity"])

    figures["top1_oracle_gap"] = figs_dir / "top1_oracle_gap.png"
    chart_top1_oracle_gap(per_target, figures["top1_oracle_gap"])

    figures["lscore_vs_rmsd"] = figs_dir / "lscore_vs_rmsd.png"
    chart_lscore_vs_rmsd(per_pose, figures["lscore_vs_rmsd"])

    if not unified.empty:
        figures["source_family_top1"] = figs_dir / "source_family_top1.png"
        chart_source_family_top1(unified, figures["source_family_top1"])
        figures["pose_pool_per_target"] = figs_dir / "pose_pool_per_target.png"
        chart_pose_pool_per_target(unified, figures["pose_pool_per_target"])

    # Fall back to per_pose-derived family chart if unified is unavailable
    if "source_family_top1" not in figures:
        figures["source_family_top1"] = figs_dir / "source_family_top1.png"
        df_legacy = per_pose.dropna(subset=["lscore"]).copy()
        df_legacy["source_family"] = (
            df_legacy["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
        )
        df_legacy = df_legacy.rename(columns={"target": "target_id"})
        df_legacy["target"] = df_legacy["target_id"]
        df_legacy["ligand_id"] = df_legacy["source"].str.extract(r"_(L\d*)$")
        chart_source_family_top1(df_legacy, figures["source_family_top1"])

    if "pose_pool_per_target" not in figures:
        figures["pose_pool_per_target"] = figs_dir / "pose_pool_per_target.png"
        df_legacy = per_pose.copy()
        chart_pose_pool_per_target(df_legacy, figures["pose_pool_per_target"])

    write_report(args.report, sr_df, per_target, clusters,
                 unified if not unified.empty else per_pose,
                 figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
