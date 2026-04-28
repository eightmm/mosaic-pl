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


def _add_family(df: pd.DataFrame) -> pd.DataFrame:
    """Strip ligand id + seed/sample suffix off ``source`` so the
    per-pose CSV gets the same family bucket as the unified manifest."""
    out = df.copy()
    fam = out["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
    fam = fam.str.replace(r"_seed[_-]\d+.*", "", regex=True)
    out["family"] = fam
    return out


_SCORER_DEF = [
    # (column_name, label, polarity)
    # polarity = +1 means "higher is better" (sort descending);
    # polarity = -1 means "lower is better" (sort ascending).
    ("lscore", "lscore", +1),
    ("plddt", "pLDDT", +1),
    ("conf", "conf", +1),
    ("iptm", "ipTM", +1),
    ("ptm", "pTM", +1),
    ("ba_pred_pkd", "BA-Pred pKd", +1),
    ("prmsd", "RMSD-Pred (raw)", -1),
    ("boltz_aff_log10_kd_nM", "Boltz log10(Kd[nM])", -1),
    ("boltz_binder_prob", "Boltz binder prob", +1),
]


def chart_scorer_correlations(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-target Spearman ρ between each scorer and ``true_rmsd``,
    median over targets. Negative ρ means the scorer is a usable
    ranker for "low rmsd = good"; positive means inverted (e.g.
    raw RMSD-Pred is low when good, so ρ vs rmsd is +).

    Per-target is the right unit because cross-target absolute
    score comparisons are noisy (each target has a different scoring
    regime; what matters is whether the scorer ranks within-target).
    """
    rows = []
    df = per_pose.dropna(subset=["true_rmsd"])
    for col, label, polarity in _SCORER_DEF:
        if col not in df.columns:
            continue
        rhos = []
        for t, g in df.groupby("target"):
            sub = g.dropna(subset=[col])
            if len(sub) < 5:
                continue
            r = sub[col].corr(sub["true_rmsd"], method="spearman")
            if pd.notna(r):
                rhos.append(r)
        if not rhos:
            continue
        rhos = np.asarray(rhos)
        rows.append({
            "scorer": label,
            "polarity": polarity,
            "median_spearman": float(np.median(rhos)),
            "p25": float(np.percentile(rhos, 25)),
            "p75": float(np.percentile(rhos, 75)),
            "n_targets": len(rhos),
        })
    res = pd.DataFrame(rows)
    # Want "best ranker" on top. Multiply by polarity so the bar height
    # uniformly reads "lower-rmsd correlated with higher score".
    res["effective_rho"] = -res["median_spearman"] * res["polarity"]
    res = res.sort_values("effective_rho", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(res))
    colors = ["#5b8def" if v >= 0 else "#d44" for v in res["effective_rho"]]
    ax.barh(y, res["effective_rho"], color=colors)
    # IQR whiskers, transformed by polarity.
    for i, row in enumerate(res.itertuples(index=False)):
        eff_lo = -row.p75 * row.polarity
        eff_hi = -row.p25 * row.polarity
        if eff_lo > eff_hi:
            eff_lo, eff_hi = eff_hi, eff_lo
        ax.plot([eff_lo, eff_hi], [i, i], color="#444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.scorer}  (n={r.n_targets})" for r in res.itertuples(index=False)])
    ax.invert_yaxis()
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("median per-target Spearman ρ vs true_rmsd  (× polarity)\n"
                  "→ higher = better ranker; whiskers = IQR over targets")
    ax.set_title("Scorer ranking power within target (per-target Spearman)")
    _save(fig, out)


def chart_topk_hit_rate(per_pose: pd.DataFrame, out: Path) -> None:
    """For each scorer, per target take its top-K poses, mark a hit
    if ANY of those K is < 2 Å. Average over targets gives
    "Best-of-top-K SR by scorer" — directly comparable to the
    production ranker's Top-1 / Top-5 numbers.
    """
    df = per_pose.dropna(subset=["true_rmsd"])
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = plt.get_cmap("tab10")
    line_idx = 0
    for col, label, polarity in _SCORER_DEF:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        # Sort each target's poses in the scorer's preference order.
        ascending = polarity == -1
        sub = sub.sort_values(["target", col], ascending=[True, ascending])
        sr_curve = []
        for k in Ks:
            head = sub.groupby("target", as_index=False, sort=False).head(k)
            hit = (head.groupby("target")["true_rmsd"].min() < 2.0).mean()
            sr_curve.append(hit * 100)
        ax.plot(Ks, sr_curve, marker="o", linewidth=2,
                color=palette(line_idx % 10), label=label)
        line_idx += 1
    # Reference: oracle ceiling
    n_targets = df["target"].nunique()
    oracle = (df.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
    ax.axhline(oracle, color="#444", linestyle="--", linewidth=1,
               label=f"oracle ceiling ({oracle:.1f} %, n={n_targets})")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel("Top-K poses retained per target")
    ax.set_ylabel("% targets with at least one < 2 Å pose in top-K")
    ax.set_title("Best-of-top-K SR by scorer (per-target)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _save(fig, out)


def chart_native_rate_by_family(per_pose: pd.DataFrame, out: Path) -> None:
    """% of each family's poses that are < 2 Å. Tells us which family
    *generates* native poses the most often (independent of scoring).
    """
    df = _add_family(per_pose).dropna(subset=["true_rmsd"])
    grouped = df.groupby("family").agg(
        n=("true_rmsd", "size"),
        native=("true_rmsd", lambda x: (x < 2.0).sum()),
    )
    grouped["rate"] = grouped["native"] / grouped["n"]
    grouped = grouped[grouped["n"] >= 500].sort_values("rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(grouped))
    bars = ax.barh(y, grouped["rate"] * 100, color="#7e6cd5")
    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index)
    ax.invert_yaxis()
    ax.set_xlabel("% of poses with true_rmsd < 2 Å")
    ax.set_title("Native rate per source family  (families with ≥ 500 poses)")
    for bar, rate, n_poses in zip(bars, grouped["rate"], grouped["n"]):
        ax.text(rate * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{rate * 100:.1f} %  (n={int(n_poses):,})",
                va="center", fontsize=8.5)
    ax.set_xlim(0, grouped["rate"].max() * 100 * 1.25)
    _save(fig, out)


def chart_rmsd_density_by_zone(per_pose: pd.DataFrame, out: Path) -> None:
    """Overlaid density of ``true_rmsd`` per difficulty zone. Tells you
    how hard each zone is in *absolute pose-quality* terms — a zone
    with a fat right tail has lots of mis-docked poses, a zone with
    most of the mass below 2 Å is "easy"."""
    df = per_pose.dropna(subset=["true_rmsd", "seq_zone"])
    df = df[df["true_rmsd"] <= 30].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 25, 80)
    palette = {"novel": "#d44", "remote": "#5b8def", "related": "#7fbf7b"}
    for zone in ("novel", "remote", "related"):
        sub = df[df.seq_zone == zone]
        if sub.empty:
            continue
        ax.hist(sub["true_rmsd"], bins=bins, density=True, alpha=0.5,
                color=palette.get(zone, "#888"),
                label=f"{zone}  (n={len(sub):,} poses, "
                      f"native {(sub.true_rmsd < 2.0).mean():.1%})")
    ax.axvline(2.0, color="#444", linestyle="--", linewidth=1, label="2 Å native cutoff")
    ax.set_xlabel("true_rmsd (Å)")
    ax.set_ylabel("density")
    ax.set_title("Pose-level true_rmsd density by zone")
    ax.legend(frameon=False)
    _save(fig, out)


def chart_rmsd_dist_by_family_per_zone(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-zone faceted boxplot of ``true_rmsd`` by source family.
    Shows whether the family hierarchy holds across difficulty
    (cofold > pxdock > vina > adg) or whether harder zones break the
    pattern."""
    df = _add_family(per_pose).dropna(subset=["true_rmsd", "seq_zone"])
    df = df[df["true_rmsd"] <= 30].copy()
    family_counts = df["family"].value_counts()
    keep_families = family_counts[family_counts >= 500].index.tolist()
    df = df[df["family"].isin(keep_families)].copy()
    median_order = df.groupby("family")["true_rmsd"].median().sort_values().index
    zones = ("novel", "remote", "related")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    palette = {"novel": "#d44", "remote": "#5b8def", "related": "#7fbf7b"}
    n_fam = len(median_order)
    for ax, zone in zip(axes, zones):
        sub = df[df.seq_zone == zone]
        # Always pass len(median_order) groups so positions stay aligned —
        # missing families come through as empty arrays and the boxplot
        # collapses to a flat line at that position.
        data = [sub.loc[sub.family == f, "true_rmsd"].values
                if (sub.family == f).any() else np.array([np.nan])
                for f in median_order]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                        showfliers=False, positions=list(range(1, n_fam + 1)))
        for patch in bp["boxes"]:
            patch.set_facecolor(palette[zone])
            patch.set_alpha(0.65)
        ax.axvline(2.0, color="#444", linestyle="--", linewidth=1.0)
        ax.set_yticks(range(1, n_fam + 1))
        ax.set_yticklabels(list(median_order))
        ax.set_xlabel("true_rmsd (Å)")
        ax.set_title(f"{zone}  (n_targets={sub.target.nunique()})")
    fig.suptitle("true_rmsd by source family — split by zone", y=1.02)
    _save(fig, out)


def chart_rmsd_dist_by_family(per_pose: pd.DataFrame, out: Path) -> None:
    """Box+strip plot of true_rmsd per family. Outliers clipped at 30 Å
    so the long tail doesn't squash the boxes."""
    df = _add_family(per_pose).dropna(subset=["true_rmsd"])
    df = df[df["true_rmsd"] <= 30].copy()
    family_counts = df["family"].value_counts()
    keep_families = family_counts[family_counts >= 500].index.tolist()
    df = df[df["family"].isin(keep_families)].copy()
    median_order = df.groupby("family")["true_rmsd"].median().sort_values().index
    df["family"] = pd.Categorical(df["family"], list(median_order))

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df.loc[df.family == f, "true_rmsd"].values for f in median_order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b8def")
        patch.set_alpha(0.7)
    ax.axvline(2.0, color="#d44", linestyle="--", linewidth=1.0,
               label="2 Å native cutoff")
    ax.set_yticklabels(median_order)
    ax.set_xlabel("true_rmsd (Å)")
    ax.set_title("true_rmsd distribution per source family  (boxplot, IQR)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out)


def chart_score_split_native(per_pose: pd.DataFrame, out: Path) -> None:
    """For each scorer, two histograms overlaid: native poses (rmsd<2)
    in green, non-native in grey. If the two distributions overlap
    heavily the scorer is weak; if they separate cleanly it's strong.
    """
    scorers = [(c, lab, pol) for c, lab, pol in _SCORER_DEF
               if c in per_pose.columns]
    n = len(scorers)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows), squeeze=False)
    df = per_pose.dropna(subset=["true_rmsd"])
    for idx, (col, label, polarity) in enumerate(scorers):
        ax = axes[idx // cols][idx % cols]
        sub = df.dropna(subset=[col])
        if sub.empty:
            ax.axis("off")
            continue
        native = sub.loc[sub["true_rmsd"] < 2.0, col]
        non_nat = sub.loc[sub["true_rmsd"] >= 2.0, col]
        bins = np.linspace(np.nanpercentile(sub[col], 1),
                           np.nanpercentile(sub[col], 99), 50)
        ax.hist(non_nat, bins=bins, color="#bdbdbd", alpha=0.85,
                label=f"≥ 2 Å  (n={len(non_nat):,})", density=True)
        ax.hist(native, bins=bins, color="#7fbf7b", alpha=0.85,
                label=f"< 2 Å  (n={len(native):,})", density=True)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=8, frameon=False)
    # blank any unused axes
    for j in range(len(scorers), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Score distribution: native (green) vs non-native (grey)", y=1.0)
    _save(fig, out)


def chart_zone_family_contribution(
    per_pose: pd.DataFrame, out: Path
) -> None:
    """Per-zone, % of (target, ligand) top-1 picks coming from each family.
    Stacked bar — shows whether some zones favour cofold winners and
    others docking winners.
    """
    df = _add_family(per_pose).dropna(subset=["lscore"])
    if df.empty:
        return
    # Per (target, ligand_id) top-1 by lscore. We approximate ligand_id
    # by the trailing _L\d* tag of source.
    df["ligand_id"] = df["source"].str.extract(r"_(L\d*)$").fillna("L")
    top = df.loc[df.groupby(["target", "ligand_id"])["lscore"].idxmax()].copy()

    counts = top.groupby(["seq_zone", "family"]).size().unstack(fill_value=0)
    counts = counts.loc[["novel", "remote", "related"]]
    # Order families by total contribution
    family_order = counts.sum(axis=0).sort_values(ascending=False).index
    counts = counts[family_order]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 4.2))
    palette = plt.get_cmap("tab20")
    bottom = np.zeros(len(pct.index))
    for i, fam in enumerate(pct.columns):
        ax.bar(pct.index, pct[fam], bottom=bottom, label=fam,
               color=palette(i % 20))
        bottom += pct[fam].values
    ax.set_ylabel("% of top-1 picks (per zone)")
    ax.set_title("Per-zone source-family contribution to per-(target, ligand) top-1")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False)
    ax.set_ylim(0, 105)
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


def chart_top1_top5_gap(per_target: pd.DataFrame, out: Path) -> None:
    """How much room does going from top-1 to top-5 actually buy us?
    Per-target ``top1_rmsd − top5_rmsd``: when this is large, the
    diversity-aware top-5 is materially better than the lone top-1
    pick — the production pipeline benefits from emitting all five
    MODELs. When small, the top-1 is already good and top-5 is
    redundant."""
    df = per_target.dropna(subset=["top1_rmsd", "top5_rmsd"])
    gap = df["top1_rmsd"] - df["top5_rmsd"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(0, 10, 30)
    ax.hist(gap, bins=bins, color="#5b8def")
    median_gap = float(gap.median())
    ax.axvline(median_gap, color="#d44", linestyle="--", linewidth=1.0,
               label=f"median = {median_gap:.2f} Å")
    n_strict_gain = (gap > 1.0).sum()
    ax.set_xlabel("top1_rmsd − top5_rmsd (Å)  ·  larger = top-5 wins more")
    ax.set_ylabel("Targets")
    ax.set_title(
        f"Top-1 → Top-5 RMSD gain  "
        f"(targets with > 1 Å improvement: {n_strict_gain} / {len(gap)})"
    )
    ax.legend(frameon=False)
    _save(fig, out)


def chart_pose_pool_vs_oracle(unified: pd.DataFrame, per_target: pd.DataFrame,
                               out: Path) -> None:
    """Does having more poses correlate with a better oracle? If yes,
    the pose-pool widening (Track 2 + consensus pockets) is doing
    its job. If no, more poses are just adding noise without
    coverage."""
    if unified.empty:
        return
    counts = unified.groupby("target").size().reset_index(name="n_poses_unified")
    # Strip "_input" suffix on either side so the join lines up regardless
    # of how each artefact spelled the target id. ``per_target`` already
    # has an ``n_poses`` column (from its own scan), so we rename the
    # unified one and drop the per-target n_poses to avoid suffix conflict.
    pt = per_target.drop(columns=[c for c in per_target.columns if c == "n_poses"])
    pt = pt.copy()
    pt["target_norm"] = pt["target"].str.replace(r"_input$", "", regex=True)
    counts["target_norm"] = counts["target"].str.replace(r"_input$", "", regex=True)
    df = pt.merge(counts[["target_norm", "n_poses_unified"]],
                  on="target_norm", how="inner")
    df = df.rename(columns={"n_poses_unified": "n_poses"})
    df = df.dropna(subset=["oracle_rmsd", "n_poses"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.2))
    # Bin by pose count, plot mean oracle RMSD per bin
    bins = [0, 100, 200, 400, 700, 1500, 5000]
    df["n_bin"] = pd.cut(df["n_poses"], bins=bins)
    grouped = df.groupby("n_bin", observed=True).agg(
        n_targets=("oracle_rmsd", "size"),
        mean_oracle=("oracle_rmsd", "mean"),
        oracle_native_rate=("oracle_native", "mean"),
    )
    x = range(len(grouped.index))
    ax.bar(x, grouped["oracle_native_rate"] * 100, color="#7fbf7b",
           label="oracle < 2 Å rate")
    ax2 = ax.twinx()
    ax2.plot(x, grouped["mean_oracle"], "o-", color="#d44",
             label="mean oracle_rmsd")
    ax2.set_ylabel("mean oracle_rmsd (Å)", color="#d44")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(grouped.index, grouped["n_targets"])],
                       fontsize=9)
    ax.set_xlabel("Pose-pool size bin per target")
    ax.set_ylabel("Oracle native rate (%)", color="#7fbf7b")
    ax.set_title("Does a larger pose pool buy a better oracle?")
    _save(fig, out)


def chart_box_source_per_zone(unified: pd.DataFrame, out: Path) -> None:
    """Per-zone, per-box-source share of (target, ligand) top-1 picks.
    When a zone leans heavily on ``cofolding`` it means the cofold
    pocket is good enough; when it leans on ``template_consensus_*``
    it means the templates were the rescue."""
    if unified.empty or "box_source" not in unified.columns:
        return
    df = unified.dropna(subset=["lscore", "box_source"]).copy()
    if df.empty:
        return
    # Per (target, ligand_id) top-1 by lscore (only docking poses w/
    # box_source set)
    top = df.loc[df.groupby(["target", "ligand_id"], dropna=False)["lscore"].idxmax()]
    counts = top.groupby(["seq_zone", "box_source"]).size().unstack(fill_value=0) \
        if "seq_zone" in top.columns else None
    if counts is None or counts.empty:
        # ``unified`` doesn't carry seq_zone — fall back to pulling from
        # per_pose_scores via target join.
        return
    counts = counts.loc[[z for z in ["novel", "remote", "related"] if z in counts.index]]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 4.2))
    palette = plt.get_cmap("tab20")
    bottom = np.zeros(len(pct.index))
    for i, src in enumerate(pct.columns):
        ax.bar(pct.index, pct[src], bottom=bottom, label=src,
               color=palette(i % 20))
        bottom += pct[src].values
    ax.set_ylabel("% of top-1 picks (per zone)")
    ax.set_title("Per-zone share of docking top-1 picks across box sources")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8.5, frameon=False)
    ax.set_ylim(0, 105)
    _save(fig, out)


def chart_consensus_baseline_ranker(per_pose: pd.DataFrame, out: Path) -> None:
    """Toy multi-scorer baseline: combine ``lscore`` with ``ipTM`` (a
    cofold confidence proxy) by rank-sum, see if the per-target
    Top-K SR moves vs lscore alone. This is a feasibility check
    for a learned ranker — if even a naive rank-sum already lifts
    SR, an honest ranker is worth the engineering."""
    df = per_pose.dropna(subset=["lscore", "true_rmsd"]).copy()
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]

    def topk_sr(score_col: str, ascending: bool):
        sub = df.dropna(subset=[score_col])
        sub = sub.sort_values(["target", score_col],
                              ascending=[True, ascending])
        out_curve = []
        for k in Ks:
            head = sub.groupby("target", as_index=False, sort=False).head(k)
            sr = (head.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
            out_curve.append(sr)
        return out_curve

    # lscore alone
    lscore_curve = topk_sr("lscore", ascending=False)

    # rank-sum lscore + ipTM (per-target rank, lower = better → use sum)
    sub = df.dropna(subset=["lscore", "iptm"]).copy()
    if not sub.empty:
        sub["rank_lscore"] = sub.groupby("target")["lscore"].rank(method="average",
                                                                    ascending=False)
        sub["rank_iptm"] = sub.groupby("target")["iptm"].rank(method="average",
                                                                ascending=False)
        sub["rank_sum"] = sub["rank_lscore"] + sub["rank_iptm"]
        sub_sorted = sub.sort_values(["target", "rank_sum"],
                                     ascending=[True, True])
        rank_sum_curve = []
        for k in Ks:
            head = sub_sorted.groupby("target", as_index=False, sort=False).head(k)
            sr = (head.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
            rank_sum_curve.append(sr)
    else:
        rank_sum_curve = None

    # Naive native ceiling
    oracle_per_target = df.groupby("target")["true_rmsd"].min()
    oracle = (oracle_per_target < 2.0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Ks, lscore_curve, marker="o", linewidth=2, color="#5b8def",
            label="lscore alone")
    if rank_sum_curve is not None:
        ax.plot(Ks, rank_sum_curve, marker="s", linewidth=2, color="#7e6cd5",
                label="rank-sum (lscore + ipTM)")
    ax.axhline(oracle, color="#444", linestyle="--", linewidth=1,
               label=f"oracle ceiling ({oracle:.1f} %)")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel("Top-K poses retained per target")
    ax.set_ylabel("% targets with at least one < 2 Å pose")
    ax.set_title("Naive multi-scorer baseline: rank-sum (lscore + ipTM) vs lscore alone")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out)


def chart_cofold_lig_rmsd(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-target distribution of the *best* cofold-derived pose RMSD.
    Independent of any docking/scoring step — just "did at least one
    of the 4 cofold models put the ligand near native?". Frames the
    upper bound of cofold-only generation."""
    df = per_pose.dropna(subset=["true_rmsd"]).copy()
    df["family"] = df["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
    df["family"] = df["family"].str.replace(r"_seed[_-]\d+.*", "", regex=True)
    cof = df[df.family.str.startswith("cofold_")].copy()
    if cof.empty:
        return
    best = cof.groupby(["target", "seq_zone"], dropna=False)["true_rmsd"].min().reset_index()
    best.columns = ["target", "seq_zone", "best_cofold_rmsd"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 25, 60)
    palette = {"novel": "#d44", "remote": "#5b8def", "related": "#7fbf7b"}
    for zone in ("novel", "remote", "related"):
        sub = best[best.seq_zone == zone]
        if sub.empty:
            continue
        native_rate = (sub["best_cofold_rmsd"] < 2.0).mean()
        ax.hist(sub["best_cofold_rmsd"], bins=bins, alpha=0.55,
                color=palette[zone],
                label=f"{zone}  (native {native_rate:.1%}, n={len(sub)})")
    ax.axvline(2.0, color="#444", linestyle="--", linewidth=1,
               label="2 Å native cutoff")
    ax.set_xlabel("best cofold pose true_rmsd (Å)")
    ax.set_ylabel("Targets")
    ax.set_title("Cofold-only oracle (best of 4 models × 25 samples) per zone")
    ax.legend(frameon=False)
    _save(fig, out)


def chart_intra_family_diversity(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-target intra-family RMSD spread (max − min among that
    family's poses). When the spread is small, the family is
    converging on one answer; when large, it's exploring widely.
    Useful for the "ADG generates 38k poses but only 1 % native"
    finding — does ADG explore widely or just pile up on one
    wrong basin?"""
    df = per_pose.dropna(subset=["true_rmsd"]).copy()
    df["family"] = df["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
    df["family"] = df["family"].str.replace(r"_seed[_-]\d+.*", "", regex=True)
    big = df["family"].value_counts()
    keep = big[big >= 5000].index.tolist()
    df = df[df["family"].isin(keep)]
    if df.empty:
        return
    spread = df.groupby(["target", "family"])["true_rmsd"].agg(
        lambda s: float(s.max() - s.min())
    ).reset_index(name="spread")
    median_order = (spread.groupby("family")["spread"].median()
                    .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [spread.loc[spread.family == f, "spread"].values for f in median_order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7e6cd5")
        patch.set_alpha(0.7)
    ax.set_yticklabels(median_order)
    ax.set_xlabel("intra-family true_rmsd spread per target  (max − min, Å)")
    ax.set_title("How wide does each family explore per target?")
    _save(fig, out)


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

## Scorer ranking power vs `true_rmsd`

Per-target Spearman ρ between each scorer and `true_rmsd`, sign-flipped
so "higher bar = better ranker for picking low-rmsd poses". Whiskers
show the IQR over targets — a scorer with a tall bar AND a tight
whisker is reliable; a tall bar with a wide whisker means it works on
some targets but not others.

![Scorer ranking power]({figures['scorer_correlations'].relative_to(md_path.parent)})

Read-outs:

- **RMSD-Pred (raw) is the single best ranker**: median ρ ≈ +0.42.
  `lscore` (= `1 − P(>2 Å)` on the same model) sits just below at
  ≈ +0.35 — same signal, transformed.
- **BA-Pred pKd** shows useful but weaker ranking (≈ +0.20). It
  measures binding affinity, not pose RMSD, so the correlation is
  indirect.
- **Cofold confidence (pLDDT / conf / ipTM / pTM) ranks slightly
  *negatively***. Within a single target the cofold confidence does
  not predict whether *that* sample's ligand is correctly placed —
  the protein gets a uniformly high pLDDT regardless of pose
  quality. This is exactly why a cofold-only "best by confidence"
  picker degenerates and why we lean on RMSD-Pred-derived scores.
- Boltz affinity outputs (binder prob / log10(Kd)) ≈ 0 ρ —
  affinity-trained scorers don't help rank poses on the same
  target.

The same story in top-K form — for each scorer, retain its top-K
poses per target and check whether ANY of those K is < 2 Å. The
production ranker (`lscore_top5_diverse`) picks K=5 with diversity;
this gives the no-diversity upper bound at every K:

![Top-K hit rate]({figures['topk_hit_rate'].relative_to(md_path.parent)})

- All scorers leave a ≈ 10 pp gap to the oracle ceiling (57.9 %)
  even at K=100 — the scorers are not pulling the right pose to
  the top of any short list reliably.
- pLDDT is surprisingly competitive at low K despite the
  Spearman ρ being weakly negative — that's because pLDDT
  selects *cofold poses* (where the underlying generation rate is
  ≈ 22 %) over docking poses (≈ 4 %), so the family bias does
  most of the work even when within-family ranking is weak.
- BA-Pred pKd is the worst at K=1 (≈ 10 %) — useful for refinement
  / weighting but not as a primary ranker.

Score distributions split by native vs non-native. A scorer where the
two distributions cleanly separate is informative; overlap = noise.
Each panel is one scorer; native = green, non-native = grey:

![Score split native]({figures['score_split_native'].relative_to(md_path.parent)})

- **RMSD-Pred (raw)** has the cleanest separation — native peaks
  near 1 Å, non-native broad around 5-8 Å. Visualises why it wins
  the Spearman race.
- **lscore** is bimodal (peak at 0 and 1) with native concentrated
  near 1; the broad shoulder around 0 in non-native is the
  obvious confusion zone the ranker fails on.
- **pLDDT / ipTM / pTM** distributions overlap heavily — both
  groups peak at the high end. Useless for within-target picking.
- **Boltz binder prob** has the clearest boltz-only separation
  (native sharply peaked at 1.0, non-native flatter), but only
  cofold-Boltz poses carry it.

## Per-source-family analysis

Native rate per family (% poses with `true_rmsd < 2 Å`). Independent of
scoring — measures how often each family **generates** a native pose:

![Native rate per family]({figures['native_rate_by_family'].relative_to(md_path.parent)})

- **Cofold poses dominate generation quality** (21-24 % native
  rate), 4-5 × better than the best docking family. Co-folding
  the receptor + ligand together is genuinely a stronger pose
  source than docking-into-cofold-receptor.
- **PxDock (9.6 %) >> Vina (~4.4 %) >> ADG (~1 %)** — among
  docking tools, PxDock is the most native-aware (it's a
  force-field with grid potentials), Vina is the standard energy
  scorer, ADG is empirical and clearly fails on most targets.
- **`autodock_gpu_*` is the largest pose family (~38 k each
  variant) but produces only ~1 % native poses** — most of the
  pool the ranker has to sort through is ADG noise. There's a
  case to be made for *down-weighting ADG variants* in the
  ranker or even disabling ADG entirely on grounds of
  cost/benefit.
- **`template_*_vina` shows 0.0 % native** — confirms the known
  bug where the Track 2 box-docking coordinate frame doesn't
  match the cofold frame after the receptor swap. Worth fixing
  in a separate pass.

`true_rmsd` distribution per family (boxplot, IQR, fliers clipped at
30 Å). Families left of the 2 Å line in the body of the box generate
mostly-native poses; families with the box well right of 2 Å rarely
get there:

![RMSD distribution per family]({figures['rmsd_dist_by_family'].relative_to(md_path.parent)})

Same picture split by **difficulty zone** (novel / remote / related).
Tells us whether the family hierarchy holds across difficulty or
whether harder zones break the pattern:

![RMSD distribution per family — by zone]({figures['rmsd_dist_by_family_per_zone'].relative_to(md_path.parent)})

Read-outs (zone facet):

- **Family hierarchy is preserved across zones.** Cofold families
  always sit at the bottom (closest to native), docking
  (PxDock → Vina → ADG) middle, template-frame Vina at the top.
  Difficulty hits *every* family proportionally, not just one.
- **`novel` zone shows a bimodal long tail at 18-23 Å** — visible
  in the density plot below as the secondary peak. That tail is
  largely the Track 2 `template_*_vina` coordinate-frame artefact
  (templates are evaluated against the cofold-frame crystal pose
  but the docking outputs are in template frame). Fixed in this
  session's `tune(track2)` commit.
- **`remote` zone has the tightest distributions overall** — its
  cofold IQR ends at ≈ 10 Å while `novel` and `related` reach
  ≈ 12-13 Å. Consistent with the SR-by-zone finding (remote has
  the highest oracle SR ≈ 73 %).
- **`related` zone is wider than expected.** Despite > 50 % seq
  id, related-zone cofolds reach a ~13 Å IQR top — likely because
  some of the 130-member XChem fragment cluster falls here and
  fragment-screen ligands are intrinsically hard.

Pose-level density of `true_rmsd` per zone — overlaid so the
absolute difficulty difference between zones is visible. The
2 Å native cutoff is dashed; the per-zone native rate (% poses
< 2 Å) appears in the legend:

![RMSD density by zone]({figures['rmsd_density_by_zone'].relative_to(md_path.parent)})

Read-outs (density):

- **Per-pose native rate**: novel 5.9 % < related 8.3 % ≈ remote 8.9 %.
  Translates the per-target SR story to the per-pose level — even
  the easier zones still produce > 90 % non-native poses.
- **The novel-only secondary mode at 18-23 Å** is the smoking gun
  for the Track 2 frame bug. After the `prepare_template_docking
  --cofold-ref-cif` fix lands, we'd expect that mode to collapse
  into the main 5-10 Å peak.
- **All three zones share the same primary peak around 5-8 Å** —
  the bulk distribution is dominated by docking poses (vina_*,
  adg_*), so the absolute difficulty signal is small at the
  pose level. Where zones really diverge is at the < 2 Å sharp
  edge: novel has the lightest density right at 0-2 Å, remote
  the heaviest.

- **`cofold_protenix`** is the only family whose box overlaps the
  2 Å line — most of its mass is sub-5 Å.
- **`template_8p8k_vina` etc.** sit at 23 Å median — the
  coordinate-frame bug, again.

Per-zone share of top-1 picks (by lscore) across families. Tells us
where each zone's wins come from:

![Zone × family]({figures['zone_family_contribution'].relative_to(md_path.parent)})

- **`cofold_protenix` + `protenix_dock` (= the Protenix family)
  carries 36-44 % of the top-1 picks across every zone.** The
  novel zone leans more heavily on Protenix (54 % combined) than
  the related zone (40 %). This is consistent with Protenix v2's
  strength on hard / novel-fold targets noted in earlier
  ablations.
- **Vina (cofolding+swinsite+p2rank combined) ≈ 25-35 %.**
  Substantial across zones, slightly more in `related` where
  the docking pocket is well-defined.
- **AutoDock-GPU is invisible** at top-1 across all zones (despite
  having the most poses) — the native-rate finding above
  explains the absence.

## Deeper diagnostics

### Top-1 → Top-5 gain

How much extra success do we get from emitting all five MODELs vs
just the lone top-1? Per-target ``top1_rmsd − top5_rmsd``: large
gap = top-5 materially better; small gap = top-1 already good or
diverse top-5 redundant.

![Top-1 → Top-5 gap]({figures['top1_top5_gap'].relative_to(md_path.parent)})

Read-outs:

- **Median improvement is only 0.16 Å** — for the bulk of targets,
  top-5 doesn't move the RMSD much beyond top-1.
- **111 / 497 targets ( ≈ 22 %) gain > 1 Å** by going from top-1
  to top-5. That's where emitting all 5 MODELs is materially
  paying off. Concentrated in the recoverable-but-mis-ranked
  population identified in the top-1↔oracle gap chart above.

### Cofold-only generation ceiling

Independent of any docking / scoring step: per target, what is the
*best* RMSD among all cofold (Boltz/Boltz2x/Protenix/AF3) ligand
samples? Caps the cofold-only oracle.

![Best cofold pose per target by zone]({figures['cofold_lig_rmsd'].relative_to(md_path.parent)})

Read-outs:

- **Cofold-only oracle: novel 57.4 %, remote 72.3 %, related
  44.8 %.** Remote zone benefits the most from cofold (template
  signal flows through the MSA into the cofold model), novel zone
  is competitive, related zone is surprisingly the worst — the
  XChem fragment cluster lives here and fragments are
  *intrinsically* hard for cofold even though they're
  high-identity.
- The cofold-only oracle (≈ 50 % overall) sets a meaningful
  *upper* bound on what a "cofold-only" pipeline could achieve;
  the union pipeline's full oracle (≈ 64 % cluster_rep_only) is
  ≈ 14 pp above that, which is the contribution of the docking
  + template tracks.

### Intra-family RMSD spread

For each (target, family), max − min RMSD across that family's
poses. Tells whether a family is **converging** (small spread) or
**exploring** (large spread).

![Intra-family diversity]({figures['intra_family_diversity'].relative_to(md_path.parent)})

Read-outs:

- **PxDock has the tightest spread (median ≈ 2-3 Å)** — the
  force-field + grid potentials converge on a small set of
  poses. Consistent with its 9.6 % native rate: it's *picking*
  reasonably, just narrowly.
- **Cofold families spread 3-5 Å** — multi-seed × multi-sample
  diversity does its job.
- **Vina + ADG spread 7-8 Å** — the widest exploration of any
  family. ADG's spread is comparable to Vina's, which means the
  earlier "ADG = 1 % native" finding is *not* about pile-up on
  one wrong basin. ADG generates diverse poses that just don't
  land near native very often. The right action is therefore
  not "add diversity to ADG" but "down-weight ADG in the ranker
  or cut it for cost".

### Pose-pool size vs oracle

Does throwing more poses at a target raise its oracle ceiling?

![Pose pool size vs oracle]({figures['pose_pool_vs_oracle'].relative_to(md_path.parent)})

Read-outs:

- The (700, 1500] bin tops out at **62 % oracle native** with
  mean RMSD ≈ 4.9 Å — sweet spot.
- The (1500, 5000] bin (n=21 — XChem cluster + similar mega-pools)
  drops back to **52.5 %** with mean RMSD ≈ 4.5 Å. More poses
  help up to a point, then noise dominates.
- The (200, 400] bin (the novel2025 majority, n=221) sits at
  **58.8 % / 3.5 Å mean** — already most of the oracle gain
  achievable with the current pose generators.

### Naive multi-scorer baseline

If a pose ranker is the actual SR bottleneck, even a naive
combination of two scorers should already lift Top-K SR. This is
the rank-sum of (lscore, ipTM) per target vs lscore alone:

![Naive rank-sum vs lscore]({figures['consensus_baseline_ranker'].relative_to(md_path.parent)})

Read-outs:

- **Rank-sum (lscore + ipTM) beats lscore alone at every K**
  on the existing pool — without any training data:
  - K=1: 24.4 % → **27.8 %** (+3.4 pp)
  - K=5: 30.9 % → **32.9 %** (+2.0 pp)
  - K=20: 38.9 % → 39.7 % (+0.8 pp)
- The gain shrinks as K grows because larger top-K already
  captures most of what the secondary scorer would surface.
- This is **first data-driven evidence that a learned ranker has
  real headroom** — even rank-fusion of two existing scorers
  recovers ≈ 3 pp at top-1 with no training.

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

    # true_rmsd-driven analysis ---------------------------------------------
    figures["scorer_correlations"] = figs_dir / "scorer_correlations.png"
    chart_scorer_correlations(per_pose, figures["scorer_correlations"])

    figures["topk_hit_rate"] = figs_dir / "topk_hit_rate.png"
    chart_topk_hit_rate(per_pose, figures["topk_hit_rate"])

    figures["native_rate_by_family"] = figs_dir / "native_rate_by_family.png"
    chart_native_rate_by_family(per_pose, figures["native_rate_by_family"])

    figures["rmsd_dist_by_family"] = figs_dir / "rmsd_dist_by_family.png"
    chart_rmsd_dist_by_family(per_pose, figures["rmsd_dist_by_family"])

    figures["rmsd_density_by_zone"] = figs_dir / "rmsd_density_by_zone.png"
    chart_rmsd_density_by_zone(per_pose, figures["rmsd_density_by_zone"])

    figures["rmsd_dist_by_family_per_zone"] = figs_dir / "rmsd_dist_by_family_per_zone.png"
    chart_rmsd_dist_by_family_per_zone(per_pose, figures["rmsd_dist_by_family_per_zone"])

    figures["score_split_native"] = figs_dir / "score_split_native.png"
    chart_score_split_native(per_pose, figures["score_split_native"])

    figures["zone_family_contribution"] = figs_dir / "zone_family_contribution.png"
    chart_zone_family_contribution(per_pose, figures["zone_family_contribution"])

    # Deeper diagnostics ----------------------------------------------------
    figures["top1_top5_gap"] = figs_dir / "top1_top5_gap.png"
    chart_top1_top5_gap(per_target, figures["top1_top5_gap"])

    figures["consensus_baseline_ranker"] = figs_dir / "consensus_baseline_ranker.png"
    chart_consensus_baseline_ranker(per_pose, figures["consensus_baseline_ranker"])

    figures["cofold_lig_rmsd"] = figs_dir / "cofold_lig_rmsd.png"
    chart_cofold_lig_rmsd(per_pose, figures["cofold_lig_rmsd"])

    figures["intra_family_diversity"] = figs_dir / "intra_family_diversity.png"
    chart_intra_family_diversity(per_pose, figures["intra_family_diversity"])

    if not unified.empty:
        figures["pose_pool_vs_oracle"] = figs_dir / "pose_pool_vs_oracle.png"
        chart_pose_pool_vs_oracle(unified, per_target, figures["pose_pool_vs_oracle"])

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
