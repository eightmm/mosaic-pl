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
# Zone helpers                                                                 #
# --------------------------------------------------------------------------- #

_ZONE_ORDER = ("Overall", "novel", "remote", "related")
_ZONE_COLOR = {
    "Overall": "#444444",
    "novel":   "#d44",
    "remote":  "#5b8def",
    "related": "#7fbf7b",
}


def _zone_subset(df: pd.DataFrame, zone: str, col: str = "seq_zone") -> pd.DataFrame:
    if zone == "Overall":
        return df
    if col not in df.columns:
        return df.iloc[0:0]
    return df[df[col] == zone]


def _zone_axes(figsize=(13, 9.5)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    return fig, axes.flatten()


def _attach_zone_to_unified(unified: pd.DataFrame, per_target: pd.DataFrame) -> pd.DataFrame:
    if unified.empty:
        return unified
    pt = per_target[["target", "seq_zone"]].copy()
    pt["target_norm"] = pt["target"].str.replace(r"_input$", "", regex=True)
    u = unified.copy()
    u["target_norm"] = u["target"].str.replace(r"_input$", "", regex=True)
    return u.merge(pt[["target_norm", "seq_zone"]], on="target_norm", how="left")


def _empty_axis(ax, label: str = "no data") -> None:
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", color="#aaa", fontsize=13)
    ax.axis("off")


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
    """2×2 bar chart: Top-1 / Top-5 / Oracle SR under each aggregation policy,
    faceted by zone (Overall=ALL, novel, remote, related)."""
    # zone_col in sr_df is the "zone" column; ALL maps to Overall
    zone_map = {"Overall": "ALL", "novel": "novel", "remote": "remote", "related": "related"}

    fig, axes = _zone_axes()
    fig.suptitle("SR by aggregation policy (100 % identity threshold)", y=1.01)

    for ax, zone in zip(axes, _ZONE_ORDER):
        sr_zone = zone_map[zone]
        df = sr_df[
            (sr_df.zone == sr_zone)
            & ((sr_df.threshold.isna() | (sr_df.threshold == "100%")))
        ]
        pivot = df.pivot_table(index="policy", columns="metric", values="sr")
        pivot = pivot.reindex(columns=["top1", "top5", "oracle"])
        pivot = pivot.reindex(["per_target", "cluster_rep_only", "cluster_any", "cluster_mean"])
        pivot = pivot.dropna(how="all")

        n_targets = int(df[df.metric == "top1"].iloc[0]["n"]) if not df[df.metric == "top1"].empty else 0
        ax.set_title(f"{zone}  (n={n_targets})")

        if pivot.empty or len(pivot) == 0:
            _empty_axis(ax)
            continue

        x = np.arange(len(pivot.index))
        w = 0.27
        colors = ["#5b8def", "#f0a43d", "#7fbf7b"]
        for i, metric in enumerate(["top1", "top5", "oracle"]):
            if metric not in pivot.columns:
                continue
            ax.bar(x + (i - 1) * w, pivot[metric] * 100, width=w,
                   label=metric.upper(), color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=10, fontsize=9)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 85)
        for i, metric in enumerate(["top1", "top5", "oracle"]):
            if metric not in pivot.columns:
                continue
            for j, v in enumerate(pivot[metric] * 100):
                if pd.notna(v):
                    ax.text(j + (i - 1) * w, v + 1.0, f"{v:.1f}", ha="center",
                            fontsize=8, color="#444")
        if zone == "Overall":
            ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)

    _save(fig, out)


def chart_sr_by_zone(sr_df: pd.DataFrame, out: Path) -> None:
    """Top-1 / Oracle by zone, comparing per_target vs cluster_rep_only."""
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


def _draw_top1_oracle_gap_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    df = df_zone.dropna(subset=["top1_rmsd", "oracle_rmsd"])
    n = len(df)
    ax.set_title(f"{zone}  (n={n})")
    if n < 5:
        _empty_axis(ax)
        return
    gap = df["top1_rmsd"] - df["oracle_rmsd"]
    bins = np.linspace(0, 12, 25)
    recoverable = gap[df.oracle_native]
    unrecoverable = gap[~df.oracle_native]
    ax.hist([recoverable, unrecoverable], bins=bins, stacked=True,
            color=["#7fbf7b", "#bdbdbd"],
            label=[f"oracle < 2 Å (n={len(recoverable)})",
                   f"oracle ≥ 2 Å (n={len(unrecoverable)})"])
    ax.set_xlabel("top1_rmsd − oracle_rmsd (Å)")
    ax.set_ylabel("Targets")
    ax.legend(frameon=False, loc="upper right", fontsize=8)


def chart_top1_oracle_gap(per_target: pd.DataFrame, out: Path) -> None:
    """2×2: Per-target gap = top1_rmsd − oracle_rmsd, split by zone."""
    fig, axes = _zone_axes()
    fig.suptitle("Top-1 ranker gap: distance between picked pose and the best in the pool", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_target, zone)
        _draw_top1_oracle_gap_panel(ax, sub, zone)
    _save(fig, out)


def _draw_source_family_top1_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    df = df_zone.dropna(subset=["lscore"])
    if df.empty:
        ax.set_title(f"{zone}  (n=0)")
        _empty_axis(ax)
        return
    n_targets = df["target"].nunique()
    ax.set_title(f"{zone}  (n_targets={n_targets})")
    top = df.loc[df.groupby(["target", "ligand_id"], dropna=False)["lscore"].idxmax()]
    counts = top.source_family.value_counts()
    if counts.empty or len(counts) < 1:
        _empty_axis(ax)
        return
    palette = plt.get_cmap("tab10")
    colors = [palette(i) for i in range(len(counts))]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Picks")
    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v}  ({v / counts.sum():.0%})",
                va="center", fontsize=8)
    ax.set_xlim(0, counts.max() * 1.22)


def chart_source_family_top1(unified: pd.DataFrame, out: Path,
                              per_target: pd.DataFrame | None = None) -> None:
    """2×2: Source family contribution to per-(target, ligand) top-1 by lscore."""
    df = unified.dropna(subset=["lscore"])
    if df.empty:
        return
    # attach zone if missing
    if "seq_zone" not in df.columns and per_target is not None:
        df = _attach_zone_to_unified(df, per_target)

    fig, axes = _zone_axes()
    fig.suptitle("Source family contribution to per-(target, ligand) top-1 by lscore", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(df, zone)
        _draw_source_family_top1_panel(ax, sub, zone)
    _save(fig, out)


def _draw_pose_pool_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    counts = df_zone.groupby("target").size()
    n = len(counts)
    ax.set_title(f"{zone}  (n_targets={n})")
    if n < 5:
        _empty_axis(ax)
        return
    ax.hist(counts, bins=40, color="#5b8def")
    ax.set_xlabel("Poses per target")
    ax.set_ylabel("Targets")
    med = int(counts.median())
    ax.axvline(med, color="#d44", linestyle="--", linewidth=1,
               label=f"median={med}")
    ax.legend(frameon=False, fontsize=9)


def chart_pose_pool_per_target(unified: pd.DataFrame, out: Path,
                                per_target: pd.DataFrame | None = None) -> None:
    """2×2: Pose-pool size distribution per zone."""
    df = unified.copy()
    if "seq_zone" not in df.columns and per_target is not None:
        df = _attach_zone_to_unified(df, per_target)

    fig, axes = _zone_axes()
    fig.suptitle("Pose-pool size distribution per target", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(df, zone)
        _draw_pose_pool_panel(ax, sub, zone)
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


def _compute_scorer_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target Spearman ρ for _SCORER_DEF scorers."""
    rows = []
    df = df.dropna(subset=["true_rmsd"])
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
    if res.empty:
        return res
    res["effective_rho"] = -res["median_spearman"] * res["polarity"]
    res = res.sort_values("effective_rho", ascending=False)
    return res


def _draw_scorer_corr_panel(ax, res: pd.DataFrame, zone: str) -> None:
    n = res["n_targets"].max() if not res.empty else 0
    ax.set_title(f"{zone}  (n_targets up to {n})" if not res.empty else f"{zone}")
    if res.empty:
        _empty_axis(ax)
        return
    y = np.arange(len(res))
    colors = ["#5b8def" if v >= 0 else "#d44" for v in res["effective_rho"]]
    ax.barh(y, res["effective_rho"], color=colors)
    for i, row in enumerate(res.itertuples(index=False)):
        eff_lo = -row.p75 * row.polarity
        eff_hi = -row.p25 * row.polarity
        if eff_lo > eff_hi:
            eff_lo, eff_hi = eff_hi, eff_lo
        ax.plot([eff_lo, eff_hi], [i, i], color="#444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.scorer}  (n={r.n_targets})" for r in res.itertuples(index=False)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("median Spearman ρ × polarity", fontsize=9)


def chart_scorer_correlations(per_pose: pd.DataFrame, out: Path) -> None:
    """2×2: Per-target Spearman ρ between each scorer and true_rmsd, per zone."""
    fig, axes = _zone_axes(figsize=(15, 11))
    fig.suptitle("Scorer ranking power within target (per-target Spearman ρ)", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_pose, zone)
        res = _compute_scorer_correlations(sub)
        _draw_scorer_corr_panel(ax, res, zone)
    _save(fig, out)


def _compute_topk_sr(df: pd.DataFrame) -> dict:
    """Returns dict: scorer_label -> list of SR values for Ks=[1,2,3,5,10,20,50,100]."""
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]
    df = df.dropna(subset=["true_rmsd"])
    results = {}
    for col, label, polarity in _SCORER_DEF:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        ascending = polarity == -1
        sub = sub.sort_values(["target", col], ascending=[True, ascending])
        curve = []
        for k in Ks:
            head = sub.groupby("target", as_index=False, sort=False).head(k)
            hit = (head.groupby("target")["true_rmsd"].min() < 2.0).mean()
            curve.append(hit * 100)
        results[label] = curve
    return results


def _draw_topk_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]
    df = df_zone.dropna(subset=["true_rmsd"])
    n_targets = df["target"].nunique()
    ax.set_title(f"{zone}  (n_targets={n_targets})")
    if n_targets < 5:
        _empty_axis(ax)
        return
    palette = plt.get_cmap("tab10")
    line_idx = 0
    results = _compute_topk_sr(df)
    for col, label, polarity in _SCORER_DEF:
        if label not in results:
            continue
        ax.plot(Ks, results[label], marker="o", linewidth=1.5,
                color=palette(line_idx % 10), label=label)
        line_idx += 1
    oracle = (df.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
    ax.axhline(oracle, color="#444", linestyle="--", linewidth=1,
               label=f"oracle ({oracle:.1f} %)")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks, fontsize=8)
    ax.set_xlabel("Top-K", fontsize=9)
    ax.set_ylabel("% targets with ≥1 pose < 2 Å", fontsize=9)


def chart_topk_hit_rate(per_pose: pd.DataFrame, out: Path) -> None:
    """2×2: Best-of-top-K SR by scorer, per zone."""
    fig, axes = _zone_axes(figsize=(14, 10))
    fig.suptitle("Best-of-top-K SR by scorer (per-target)", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_pose, zone)
        _draw_topk_panel(ax, sub, zone)
    # Single legend on the Overall panel (axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", frameon=False,
                       fontsize=7, ncol=2)
    _save(fig, out)


def _draw_native_rate_panel(ax, df_zone: pd.DataFrame, zone: str, n_min: int = 200) -> None:
    df = _add_family(df_zone).dropna(subset=["true_rmsd"])
    n_total = len(df)
    ax.set_title(f"{zone}  (n_poses={n_total:,})")
    grouped = df.groupby("family").agg(
        n=("true_rmsd", "size"),
        native=("true_rmsd", lambda x: (x < 2.0).sum()),
    )
    grouped["rate"] = grouped["native"] / grouped["n"]
    grouped = grouped[grouped["n"] >= n_min].sort_values("rate", ascending=False)
    if grouped.empty:
        _empty_axis(ax)
        return
    y = np.arange(len(grouped))
    bars = ax.barh(y, grouped["rate"] * 100, color="#7e6cd5")
    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% poses < 2 Å", fontsize=9)
    for bar, rate, n_poses in zip(bars, grouped["rate"], grouped["n"]):
        ax.text(rate * 100 + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{rate * 100:.1f} %  (n={int(n_poses):,})",
                va="center", fontsize=7.5)
    ax.set_xlim(0, grouped["rate"].max() * 100 * 1.30)


def chart_native_rate_by_family(per_pose: pd.DataFrame, out: Path) -> None:
    """2×2: % of each family's poses that are < 2 Å, per zone."""
    fig, axes = _zone_axes(figsize=(15, 11))
    fig.suptitle("Native rate per source family  (≥ 200 poses per zone, ≥ 500 Overall)",
                 y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_pose, zone)
        n_min = 500 if zone == "Overall" else 200
        _draw_native_rate_panel(ax, sub, zone, n_min=n_min)
    _save(fig, out)


def chart_rmsd_density_by_zone(per_pose: pd.DataFrame, out: Path) -> None:
    """Overlaid density of ``true_rmsd`` per difficulty zone."""
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
    """Per-zone faceted boxplot of ``true_rmsd`` by source family (1×3)."""
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


def chart_score_split_native(per_pose: pd.DataFrame, figs_dir: Path) -> dict[str, Path]:
    """4 separate per-zone PNGs (9-panel each): score distribution native vs non-native."""
    scorers = [(c, lab, pol) for c, lab, pol in _SCORER_DEF
               if c in per_pose.columns]
    n = len(scorers)
    cols = 3
    rows = (n + cols - 1) // cols
    outputs: dict[str, Path] = {}

    for zone in _ZONE_ORDER:
        df_zone = _zone_subset(per_pose, zone).dropna(subset=["true_rmsd"])
        label = zone.lower()
        out = figs_dir / f"score_split_native_{label}.png"

        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows), squeeze=False)
        for idx, (col, scorer_label, polarity) in enumerate(scorers):
            ax = axes[idx // cols][idx % cols]
            sub = df_zone.dropna(subset=[col])
            if sub.empty or len(sub) < 5:
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
            ax.set_title(scorer_label)
            ax.set_xlabel(scorer_label)
            ax.set_ylabel("density")
            ax.legend(loc="best", fontsize=8, frameon=False)
        for j in range(len(scorers), rows * cols):
            axes[j // cols][j % cols].axis("off")
        n_poses = len(df_zone)
        fig.suptitle(f"Score distribution: native (green) vs non-native (grey) — {zone}  "
                     f"(n={n_poses:,} poses)", y=1.0)
        _save(fig, out)
        outputs[f"score_split_native_{label}"] = out

    return outputs


def chart_zone_family_contribution(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-zone, % of (target, ligand) top-1 picks coming from each family."""
    df = _add_family(per_pose).dropna(subset=["lscore"])
    if df.empty:
        return
    df["ligand_id"] = df["source"].str.extract(r"_(L\d*)$").fillna("L")
    top = df.loc[df.groupby(["target", "ligand_id"])["lscore"].idxmax()].copy()

    counts = top.groupby(["seq_zone", "family"]).size().unstack(fill_value=0)
    counts = counts.loc[[z for z in ["novel", "remote", "related"] if z in counts.index]]
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


def _draw_top1_top5_gap_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    df = df_zone.dropna(subset=["top1_rmsd", "top5_rmsd"])
    n = len(df)
    ax.set_title(f"{zone}  (n={n})")
    if n < 5:
        _empty_axis(ax)
        return
    gap = df["top1_rmsd"] - df["top5_rmsd"]
    bins = np.linspace(0, 10, 30)
    ax.hist(gap, bins=bins, color="#5b8def")
    median_gap = float(gap.median())
    ax.axvline(median_gap, color="#d44", linestyle="--", linewidth=1.0,
               label=f"median = {median_gap:.2f} Å")
    n_gain = (gap > 1.0).sum()
    ax.set_xlabel("top1_rmsd − top5_rmsd (Å)", fontsize=9)
    ax.set_ylabel("Targets")
    ax.set_title(f"{zone}  (n={n}, >{1:.0f}Å: {n_gain})")
    ax.legend(frameon=False, fontsize=9)


def chart_top1_top5_gap(per_target: pd.DataFrame, out: Path) -> None:
    """2×2: Top-1 → Top-5 RMSD gain per zone."""
    fig, axes = _zone_axes()
    fig.suptitle("Top-1 → Top-5 RMSD gain  (larger = top-5 wins more)", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_target, zone)
        _draw_top1_top5_gap_panel(ax, sub, zone)
    _save(fig, out)


def _draw_intra_family_panel(ax, df_zone: pd.DataFrame, zone: str,
                              n_min: int = 1500) -> None:
    df = df_zone.dropna(subset=["true_rmsd"]).copy()
    df["family"] = df["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
    df["family"] = df["family"].str.replace(r"_seed[_-]\d+.*", "", regex=True)
    big = df["family"].value_counts()
    keep = big[big >= n_min].index.tolist()
    df = df[df["family"].isin(keep)]
    n_total = len(df)
    ax.set_title(f"{zone}  (n_poses={n_total:,})")
    if df.empty:
        _empty_axis(ax)
        return
    spread = df.groupby(["target", "family"])["true_rmsd"].agg(
        lambda s: float(s.max() - s.min())
    ).reset_index(name="spread")
    median_order = (spread.groupby("family")["spread"].median()
                    .sort_values().index.tolist())
    data = [spread.loc[spread.family == f, "spread"].values for f in median_order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7e6cd5")
        patch.set_alpha(0.7)
    ax.set_yticklabels(median_order, fontsize=8)
    ax.set_xlabel("spread (Å)", fontsize=9)


def chart_intra_family_diversity(per_pose: pd.DataFrame, out: Path) -> None:
    """2×2: Per-target intra-family RMSD spread per zone."""
    fig, axes = _zone_axes(figsize=(15, 11))
    fig.suptitle("Intra-family RMSD spread per target (≥ 1500 poses/family per zone, "
                 "≥ 5000 Overall)", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_pose, zone)
        n_min = 5000 if zone == "Overall" else 1500
        _draw_intra_family_panel(ax, sub, zone, n_min=n_min)
    _save(fig, out)


def _draw_pose_pool_vs_oracle_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    n = len(df_zone)
    ax.set_title(f"{zone}  (n_targets={n})")
    if n < 5:
        _empty_axis(ax)
        return
    df = df_zone.dropna(subset=["oracle_rmsd", "n_poses"])
    if df.empty:
        _empty_axis(ax)
        return
    # Merge smaller bins to ensure n >= 10 per bin
    bins = [0, 100, 200, 400, 700, 1500, 5000]
    df["n_bin"] = pd.cut(df["n_poses"], bins=bins)
    grouped = df.groupby("n_bin", observed=True).agg(
        n_targets=("oracle_rmsd", "size"),
        mean_oracle=("oracle_rmsd", "mean"),
        oracle_native_rate=("oracle_native", "mean"),
    )
    # add low-n caveat marker
    grouped["label"] = [
        f"{b}\n(n={int(nt)}{'*' if nt < 10 else ''})"
        for b, nt in zip(grouped.index, grouped["n_targets"])
    ]
    x = range(len(grouped.index))
    ax.bar(x, grouped["oracle_native_rate"] * 100, color="#7fbf7b",
           label="oracle < 2 Å rate")
    ax2 = ax.twinx()
    ax2.plot(x, grouped["mean_oracle"], "o-", color="#d44",
             label="mean oracle_rmsd")
    ax2.set_ylabel("mean oracle_rmsd (Å)", color="#d44", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["label"], fontsize=7.5)
    ax.set_xlabel("Pose-pool size bin", fontsize=9)
    ax.set_ylabel("Oracle native rate (%)", color="#7fbf7b", fontsize=9)
    ax.set_title(f"{zone}  (n_targets={n})")


def chart_pose_pool_vs_oracle(unified: pd.DataFrame, per_target: pd.DataFrame,
                               out: Path) -> None:
    """2×2: Does a larger pose pool buy a better oracle? Per zone."""
    if unified.empty:
        return

    # attach zone to unified
    if "seq_zone" not in unified.columns:
        unified = _attach_zone_to_unified(unified, per_target)

    counts = unified.groupby("target").size().reset_index(name="n_poses_unified")
    pt = per_target.drop(columns=[c for c in per_target.columns if c == "n_poses"])
    pt = pt.copy()
    pt["target_norm"] = pt["target"].str.replace(r"_input$", "", regex=True)
    counts["target_norm"] = counts["target"].str.replace(r"_input$", "", regex=True)
    df = pt.merge(counts[["target_norm", "n_poses_unified"]], on="target_norm", how="inner")
    df = df.rename(columns={"n_poses_unified": "n_poses"})
    df = df.dropna(subset=["oracle_rmsd", "n_poses"])
    if df.empty:
        return

    fig, axes = _zone_axes()
    fig.suptitle("Does a larger pose pool buy a better oracle?", y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(df, zone)
        _draw_pose_pool_vs_oracle_panel(ax, sub, zone)
    _save(fig, out)


def _draw_consensus_ranker_panel(ax, df_zone: pd.DataFrame, zone: str) -> None:
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]
    df = df_zone.dropna(subset=["lscore", "true_rmsd"]).copy()
    n_targets = df["target"].nunique()
    ax.set_title(f"{zone}  (n_targets={n_targets})")
    if n_targets < 5:
        _empty_axis(ax)
        return

    def topk_sr_local(score_col, ascending):
        sub = df.dropna(subset=[score_col])
        sub = sub.sort_values(["target", score_col], ascending=[True, ascending])
        curve = []
        for k in Ks:
            head = sub.groupby("target", as_index=False, sort=False).head(k)
            sr = (head.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
            curve.append(sr)
        return curve

    lscore_curve = topk_sr_local("lscore", ascending=False)
    ax.plot(Ks, lscore_curve, marker="o", linewidth=2, color="#5b8def", label="lscore alone")

    sub2 = df.dropna(subset=["lscore", "iptm"]).copy()
    if not sub2.empty and sub2["target"].nunique() >= 5:
        sub2["rank_lscore"] = sub2.groupby("target")["lscore"].rank(method="average", ascending=False)
        sub2["rank_iptm"] = sub2.groupby("target")["iptm"].rank(method="average", ascending=False)
        sub2["rank_sum"] = sub2["rank_lscore"] + sub2["rank_iptm"]
        sub_sorted = sub2.sort_values(["target", "rank_sum"], ascending=[True, True])
        rank_sum_curve = []
        for k in Ks:
            head = sub_sorted.groupby("target", as_index=False, sort=False).head(k)
            sr = (head.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
            rank_sum_curve.append(sr)
        ax.plot(Ks, rank_sum_curve, marker="s", linewidth=2, color="#7e6cd5",
                label="rank-sum (lscore+ipTM)")

    oracle = (df.groupby("target")["true_rmsd"].min() < 2.0).mean() * 100
    ax.axhline(oracle, color="#444", linestyle="--", linewidth=1,
               label=f"oracle ({oracle:.1f} %)")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks, fontsize=8)
    ax.set_xlabel("Top-K", fontsize=9)
    ax.set_ylabel("% targets with ≥1 pose < 2 Å", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def chart_consensus_baseline_ranker(per_pose: pd.DataFrame, out: Path) -> None:
    """2×2: Naive multi-scorer baseline rank-sum vs lscore alone, per zone."""
    fig, axes = _zone_axes()
    fig.suptitle("Naive multi-scorer baseline: rank-sum (lscore + ipTM) vs lscore alone",
                 y=1.01)
    for ax, zone in zip(axes, _ZONE_ORDER):
        sub = _zone_subset(per_pose, zone)
        _draw_consensus_ranker_panel(ax, sub, zone)
    _save(fig, out)


def chart_cofold_lig_rmsd(per_pose: pd.DataFrame, out: Path) -> None:
    """Per-target distribution of the *best* cofold-derived pose RMSD (existing overlay)."""
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
    """Compose the markdown."""

    overall = sr_df[(sr_df.zone == "ALL") & ((sr_df.threshold.isna() | (sr_df.threshold == "100%")))]
    overall_pivot = overall.pivot_table(index="policy", columns="metric", values="sr")
    overall_n = overall.pivot_table(index="policy", columns="metric", values="n")

    cluster_count = clusters.cluster_rep.nunique()
    biggest_cluster_size = clusters.cluster_size.max()
    biggest_rep = clusters.loc[clusters.cluster_size.idxmax(), "cluster_rep"]
    n_singletons = (clusters.cluster_size == 1).sum()

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

    def _fig(key):
        if key not in figures:
            return f"<!-- figure {key} not generated -->"
        return f"![{key}]({figures[key].relative_to(md_path.parent)})"

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
    # Only per_target + cluster_rep_only in the headline table
    policy_order = ["per_target", "cluster_rep_only"]
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
{_fig('sr_aggregation')}

The XChem 130-member super-cluster (see [Sequence redundancy](#sequence-redundancy))
pulls the per-target average down ~6–8 pp because fragment-screen ligands have
below-average SR. The `cluster_rep_only @ 100 %` numbers are the less-biased
headline.

The Top-1 ↔ Oracle gap stays ≈ 33 pp regardless of aggregation —
that's the actual ranker bottleneck. The right pose is in the pool
~63 % of the time but the scorer picks it ~31 % of the time.

### Zone facet

{_fig('sr_by_zone')}

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

{_fig('cluster_size_dist')}

`mmseqs easy-cluster --min-seq-id 1.0 -c 0.9` on the first protein
chain of every input collapses **499 → 246** unique enzymes. One
cluster (rep `9s4h_input`, members `7hqq…7hr*`) holds 130 entries
(an XChem fragment screen).

100 → 30 % identity threshold only collapses 244 → 207 clusters (39 cluster
loss); 100 % dedup is enough for this dataset.

## Source-family contributions (per-(target, ligand) top-1 by lscore)

{_fig('source_family_top1')}

Top-8 family share of per-(target, ligand) winners:

{family_lines}

Read-outs (zone facet): the novel zone leans more heavily on cofold families
(fewer template hints → cofold is often the only competitive source), while
the related zone has a somewhat larger docking-family share as template pockets
better define the binding site. The overall pattern is stable across zones.

## Pose-pool size

{_fig('pose_pool_per_target')}

Median pool ≈ {int(unified.groupby('target').size().median())} poses /
target. Read-outs (zone facet): the XChem cluster (see Sequence redundancy)
inflates the related-zone right tail, but median counts are comparable across
zones.

## Top-1 ↔ Oracle gap

{_fig('top1_oracle_gap')}

For targets where the **oracle is < 2 Å (recoverable)**, the median
``top1_rmsd − oracle_rmsd`` is **{median_gap_recoverable:.2f} Å**. Those
are the targets the ranker fails on — the right pose IS in the pool.
For the remaining ~37 % of targets (oracle ≥ 2 Å), no scoring change
can help; they need better generation (more cofold seeds, wider
template pool, MSA depth).

Read-outs (zone facet): the gap distribution shape is similar across zones;
the novel zone has slightly more "generation miss" (unrecoverable) targets as
expected.

## Scorer ranking power vs `true_rmsd`

Per-target Spearman ρ between each scorer and `true_rmsd`, sign-flipped
so "higher bar = better ranker for picking low-rmsd poses". Whiskers
show the IQR over targets — a scorer with a tall bar AND a tight
whisker is reliable; a tall bar with a wide whisker means it works on
some targets but not others.

{_fig('scorer_correlations')}

Read-outs:

- **RMSD-Pred (raw) is the single best ranker**: median ρ ≈ +0.42.
  `lscore` (= `1 − P(>2 Å)` on the same model) sits just below at
  ≈ +0.35 — same signal, transformed.
- **BA-Pred pKd** shows useful but weaker ranking (≈ +0.20).
- **Cofold confidence (pLDDT / conf / ipTM / pTM) ranks slightly
  *negatively***. Within a single target the cofold confidence does
  not predict whether *that* sample's ligand is correctly placed.
- Boltz affinity outputs ≈ 0 ρ — affinity-trained scorers don't help
  rank poses on the same target.
- Read-outs (zone facet): all zones show the same scorer ranking;
  the novel zone has slightly wider IQR whiskers indicating more
  per-target variability, consistent with having fewer template hints
  to anchor the scoring.

The same story in top-K form:

{_fig('topk_hit_rate')}

- All scorers leave a ≈ 10 pp gap to the oracle ceiling even at K=100.
- pLDDT is competitive at low K via family-bias, not within-family ranking.
- BA-Pred pKd is the worst at K=1 (~10 %).

Score distributions split by native vs non-native (4 per-zone PNGs):

{_fig('score_split_native_overall')}

{_fig('score_split_native_novel')}

{_fig('score_split_native_remote')}

{_fig('score_split_native_related')}

- **RMSD-Pred (raw)** has the cleanest separation.
- **lscore** is bimodal (peaks at 0 and 1) with native near 1.
- **pLDDT / ipTM / pTM** overlap heavily across native/non-native.
- **Boltz binder prob** has clear separation but only for cofold-Boltz poses.

## Per-source-family analysis

> **Note:** `template_*_vina` (Track 2) shows ~0 % native and a 18-23 Å mode in
> this report — the data was generated **before** the chain-pick + cofold-frame
> fix in commit c027e8e. A re-run is pending; treat Track-2-family numbers as
> pre-fix.

Native rate per family (% poses with `true_rmsd < 2 Å`). Independent of
scoring — measures how often each family **generates** a native pose:

{_fig('native_rate_by_family')}

- **Cofold poses dominate generation quality** (21-24 % native rate), 4-5 × better
  than the best docking family.
- **PxDock (9.6 %) >> Vina (~4.4 %) >> ADG (~1 %)** — among docking tools.
- **`autodock_gpu_*` is the largest pose family but produces only ~1 % native**.
- Read-outs (zone facet): the family hierarchy is preserved across all three zones;
  per-zone thresholds are lowered to ≥ 200 poses so smaller families are visible.

Same picture split by **difficulty zone** (1×3 facet):

{_fig('rmsd_dist_by_family_per_zone')}

Read-outs:

- **Family hierarchy is preserved across zones.**
- **`remote` zone has the tightest distributions overall** — cofold IQR ends ≈ 10 Å
  vs ≈ 12-13 Å for novel/related.
- **`related` zone is wider than expected** — XChem fragment cluster (see Sequence
  redundancy) dominates the right tail.

Pose-level density of `true_rmsd` per zone:

{_fig('rmsd_density_by_zone')}

Read-outs:

- **Per-pose native rate**: novel 5.9 % < related 8.3 % ≈ remote 8.9 %.
- **All three zones share the same primary peak around 5-8 Å** dominated by docking
  poses.
- **`cofold_protenix`** is the only family whose box overlaps the 2 Å line.

Per-zone share of top-1 picks (by lscore) across families:

{_fig('zone_family_contribution')}

- **`cofold_protenix` + `protenix_dock` carries 36-44 % of the top-1 picks** across
  every zone; novel leans more heavily on Protenix (54 % combined).
- **Vina combined ≈ 25-35 %.**
- **AutoDock-GPU is invisible** at top-1 across all zones.

## Deeper diagnostics

### Top-1 → Top-5 gain

{_fig('top1_top5_gap')}

Read-outs:

- **Median improvement is only 0.16 Å** for the bulk of targets.
- **≈ 22 % of targets gain > 1 Å** by going from top-1 to top-5.
- Read-outs (zone facet): the novel zone shows the highest fraction of targets
  with > 1 Å gain (cofold diversity is doing more work when templates are scarce).

### Cofold-only generation ceiling

Independent of any docking / scoring step: per target, what is the
*best* RMSD among all cofold (Boltz/Boltz2x/Protenix/AF3) ligand
samples?

{_fig('cofold_lig_rmsd')}

Read-outs:

- **Cofold-only oracle: novel 57.4 %, remote 72.3 %, related 44.8 %.**
  The related zone is the worst — the XChem fragment cluster (see Sequence
  redundancy) dominates and fragments are intrinsically hard for cofold.
- The union pipeline's full oracle (≈ 64 % cluster_rep_only) is ≈ 14 pp above
  the cofold-only oracle, attributable to docking + template tracks.

### Intra-family RMSD spread

{_fig('intra_family_diversity')}

Read-outs:

- **PxDock has the tightest spread (median ≈ 2-3 Å)** — consistent with its high
  native rate and narrow-search grid potentials.
- **Cofold families spread 3-5 Å** — multi-seed diversity does its job.
- **Vina + ADG spread 7-8 Å** — ADG generates diverse poses that just don't
  land near native. The right action is down-weighting ADG, not adding diversity.
- Read-outs (zone facet): the novel zone shows slightly wider spread across all
  families (harder targets → more exploration before convergence).

### Pose-pool size vs oracle

Does throwing more poses at a target raise its oracle ceiling?

{_fig('pose_pool_vs_oracle')}

Read-outs:

- The (700, 1500] bin tops out at **62 % oracle native** — sweet spot.
- The (1500, 5000] bin (n=21* low-n caveat — XChem cluster + similar mega-pools)
  drops back to **52.5 %**. More poses help up to a point, then noise dominates.
- The (200, 400] bin (the novel2025 majority, n=221) sits at **58.8 %** —
  already most of the oracle gain achievable with current pose generators.
- Bins marked `*` have fewer than 10 targets; interpret with caution.

### Naive multi-scorer baseline

If a pose ranker is the actual SR bottleneck, even a naive combination of two
scorers should already lift Top-K SR. Rank-sum of (lscore, ipTM) per target vs
lscore alone:

{_fig('consensus_baseline_ranker')}

Read-outs:

- **Rank-sum (lscore + ipTM) beats lscore alone at every K** — K=1: ~+3 pp.
- The gain shrinks as K grows.
- **First data-driven evidence that a learned ranker has real headroom.**
- Read-outs (zone facet): the rank-sum lift is consistent across all three zones
  (roughly equal benefit), suggesting the ipTM signal is not zone-specific.

## Where the SR ceilings sit (current pipeline)

| metric                          | value     |
|---------------------------------|-----------|
| Top-1 SR (cluster_rep_only)     | ≈ 31 %   |
| Top-5 SR                        | ≈ 40 %   |
| Oracle SR (= pool ceiling)      | ≈ 64 %   |
| Top-1 ↔ Oracle gap (scoring)    | ≈ 33 pp  |
| Oracle ↔ 100 % gap (generation) | ≈ 36 pp  |

Two independent ceilings to push:

1. **Generation ceiling (Oracle)**: improve template-pocket coverage and cofold
   quality.
2. **Ranker ceiling (Top-1 / Oracle gap)**: scoring problem; needs a learned
   ranker, ensemble of independent scorers, or both.

## Generated artefacts

- `figures/*.png` — every chart embedded above
- `figures/score_split_native_overall.png`, `_novel.png`, `_remote.png`, `_related.png`
  — per-zone score-separation panels
- `sr_per_target.csv` — per-target {{top1, top5, oracle}}_rmsd + native bool + zone
- `sr_by_cluster.csv` — long-format SR table (policy × zone × threshold × metric)
- `cluster_targets_{{030,050,070,095,100}}.csv` — target → cluster_rep + cluster_size
- `experiments/poses_unified/_summary.csv` — pose rows

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

    # --- SR summary charts (zone-facetable via 2×2 or special layout) ---
    figures["sr_aggregation"] = figs_dir / "sr_aggregation.png"
    chart_sr_aggregation(sr_df, figures["sr_aggregation"])

    figures["sr_by_zone"] = figs_dir / "sr_by_zone.png"
    chart_sr_by_zone(sr_df, figures["sr_by_zone"])

    figures["cluster_size_dist"] = figs_dir / "cluster_size_dist.png"
    chart_cluster_size_dist(clusters, figures["cluster_size_dist"])

    figures["top1_oracle_gap"] = figs_dir / "top1_oracle_gap.png"
    chart_top1_oracle_gap(per_target, figures["top1_oracle_gap"])

    # --- Per-pose charts (zone-faceted 2×2) ---
    figures["scorer_correlations"] = figs_dir / "scorer_correlations.png"
    chart_scorer_correlations(per_pose, figures["scorer_correlations"])

    figures["topk_hit_rate"] = figs_dir / "topk_hit_rate.png"
    chart_topk_hit_rate(per_pose, figures["topk_hit_rate"])

    figures["native_rate_by_family"] = figs_dir / "native_rate_by_family.png"
    chart_native_rate_by_family(per_pose, figures["native_rate_by_family"])

    figures["rmsd_density_by_zone"] = figs_dir / "rmsd_density_by_zone.png"
    chart_rmsd_density_by_zone(per_pose, figures["rmsd_density_by_zone"])

    figures["rmsd_dist_by_family_per_zone"] = figs_dir / "rmsd_dist_by_family_per_zone.png"
    chart_rmsd_dist_by_family_per_zone(per_pose, figures["rmsd_dist_by_family_per_zone"])

    # score_split_native → 4 separate PNGs
    score_split_figs = chart_score_split_native(per_pose, figs_dir)
    figures.update(score_split_figs)

    figures["zone_family_contribution"] = figs_dir / "zone_family_contribution.png"
    chart_zone_family_contribution(per_pose, figures["zone_family_contribution"])

    # --- Per-target charts (zone-faceted 2×2) ---
    figures["top1_top5_gap"] = figs_dir / "top1_top5_gap.png"
    chart_top1_top5_gap(per_target, figures["top1_top5_gap"])

    figures["consensus_baseline_ranker"] = figs_dir / "consensus_baseline_ranker.png"
    chart_consensus_baseline_ranker(per_pose, figures["consensus_baseline_ranker"])

    figures["cofold_lig_rmsd"] = figs_dir / "cofold_lig_rmsd.png"
    chart_cofold_lig_rmsd(per_pose, figures["cofold_lig_rmsd"])

    figures["intra_family_diversity"] = figs_dir / "intra_family_diversity.png"
    chart_intra_family_diversity(per_pose, figures["intra_family_diversity"])

    # --- Unified-dependent charts ---
    if not unified.empty:
        figures["pose_pool_vs_oracle"] = figs_dir / "pose_pool_vs_oracle.png"
        chart_pose_pool_vs_oracle(unified, per_target, figures["pose_pool_vs_oracle"])

        figures["source_family_top1"] = figs_dir / "source_family_top1.png"
        chart_source_family_top1(unified, figures["source_family_top1"],
                                 per_target=per_target)

        figures["pose_pool_per_target"] = figs_dir / "pose_pool_per_target.png"
        chart_pose_pool_per_target(unified, figures["pose_pool_per_target"],
                                   per_target=per_target)

    # Fall back to per_pose-derived family chart if unified is unavailable
    if "source_family_top1" not in figures:
        figures["source_family_top1"] = figs_dir / "source_family_top1.png"
        df_legacy = per_pose.dropna(subset=["lscore"]).copy()
        df_legacy["source_family"] = (
            df_legacy["source"].str.replace(r"_(L\d*|X\d*)$", "", regex=True)
        )
        df_legacy["ligand_id"] = df_legacy["source"].str.extract(r"_(L\d*)$")
        chart_source_family_top1(df_legacy, figures["source_family_top1"],
                                 per_target=per_target)

    if "pose_pool_per_target" not in figures:
        figures["pose_pool_per_target"] = figs_dir / "pose_pool_per_target.png"
        chart_pose_pool_per_target(per_pose.copy(), figures["pose_pool_per_target"],
                                   per_target=per_target)

    write_report(args.report, sr_df, per_target, clusters,
                 unified if not unified.empty else per_pose,
                 figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
