#!/usr/bin/env python3
"""Figures for the CASP17 ligand-series stage-1 postmortem.

Renders what `scripts/analyze_stage1_signals.py` prints as tables:

  fig1_auc_ranking      per-signal AUC, L01 vs L02 (small multiples)
  fig2_roc              ROC curves faceted by signal family
  fig3_auc_vs_bedroc    the AUC/BEDROC disagreement, the postmortem's finding 1
  fig4_enrichment       cumulative recall vs fraction screened (early enrichment)
  fig5_l02_provenance   what the submitted L02 score is, and where its hits sit

Every signal is already normalised to higher-is-better by ``collect``. The table
view for all of these lives in ``docs/ligand_series_stage1_postmortem.md``; each
figure also writes its own CSV next to the PNG.

    uv run python scripts/plot_stage1_signals.py --out-dir docs/figures

Colours are the validated categorical palette (blue, orange, aqua, yellow,
magenta …) in fixed slot order — see the dataviz reference. Scatter forms cap at
three slots because the all-pairs CVD floor does not clear past that.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# palette — validated categorical slots, light and dark steps
# --------------------------------------------------------------------------- #
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
               "#008300", "#9085e9", "#e66767"]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#8a8a85", grid="#e6e5e1", series=SERIES_LIGHT),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                 muted="#84837b", grid="#33332f", series=SERIES_DARK),
}

# --------------------------------------------------------------------------- #
# metrics (same definitions as analyze_stage1_signals.py)
# --------------------------------------------------------------------------- #
def avg_ranks_desc(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def bedroc_from_ranks(ranks: np.ndarray, y: np.ndarray, alpha: float = 20.0) -> float:
    n_tot, n_act = len(ranks), int(y.sum())
    ra = n_act / n_tot
    ri = ranks[y] / n_tot
    rie_num = np.exp(-alpha * ri).sum() / n_act
    rie_den = (1 / n_tot) * (1 - math.exp(-alpha)) / (math.exp(alpha / n_tot) - 1)
    rie = rie_num / rie_den
    return (rie * ra * math.sinh(alpha / 2)
            / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1 / (1 - math.exp(alpha * (1 - ra))))


def roc_curve(scores: np.ndarray, y: np.ndarray):
    """FPR/TPR with ties handled as a single step, plus the AUC."""
    ok = ~np.isnan(scores)
    s, yy = scores[ok], y[ok]
    order = np.argsort(-s, kind="mergesort")
    s, yy = s[order], yy[order]
    n_pos, n_neg = int(yy.sum()), int((~yy).sum())
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    # collapse ties so the curve steps once per distinct score
    keep = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0.0, tp[keep] / n_pos]
    fpr = np.r_[0.0, fp[keep] / n_neg]
    ranks = avg_ranks_desc(s)
    auc = ((len(s) + 1 - ranks)[yy].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return fpr, tpr, auc


def enrichment_curve(scores: np.ndarray, y: np.ndarray):
    ok = ~np.isnan(scores)
    s, yy = scores[ok], y[ok]
    order = np.argsort(-s, kind="mergesort")
    yy = yy[order]
    frac_screened = np.r_[0.0, np.arange(1, len(yy) + 1) / len(yy)]
    recall = np.r_[0.0, np.cumsum(yy) / yy.sum()]
    return frac_screened, recall


def signal_metrics(scores: np.ndarray, y: np.ndarray, alpha: float = 20.0):
    ok = ~np.isnan(scores)
    s, yy = scores[ok], y[ok]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return None
    ranks = avg_ranks_desc(s)
    n, na = len(s), int(yy.sum())
    auc = ((n + 1 - ranks)[yy].sum() - na * (na + 1) / 2) / (na * (n - na))
    order = np.argsort(-s, kind="mergesort")
    yo = yy[order]
    ra = na / n
    ef = {f: (yo[:max(1, int(round(n * f)))].sum() / max(1, int(round(n * f)))) / ra
          for f in (0.01, 0.05, 0.10)}
    return dict(n=n, n_act=na, auc=float(auc),
                bedroc=float(bedroc_from_ranks(ranks, yy, alpha)),
                ef1=ef[0.01], ef5=ef[0.05], ef10=ef[0.10])


def random_bedroc(n_tot: int, n_act: int, alpha: float = 20.0, draws: int = 2000):
    rng = np.random.default_rng(0)
    y = np.zeros(n_tot, dtype=bool)
    y[:n_act] = True
    ranks = np.arange(1, n_tot + 1, dtype=float)
    vals = [bedroc_from_ranks(ranks, rng.permutation(y), alpha) for _ in range(draws)]
    return float(np.mean(vals)), float(np.percentile(vals, 95))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
#: signal -> (family, display label). Order inside a family is the plot order.
FAMILIES = {
    "boltz": [("boltz_prob_mean", "binder prob (mean)"),
              ("boltz_aff_min", "affinity (min)"),
              ("boltz_aff_mean", "affinity (mean)"),
              ("boltz_LE", "ligand efficiency")],
    "akscore2": [("ak_dock_best", "ak_dock (best)"),
                 ("ak_score_best", "ak_score (best)"),
                 ("ak_ens_best", "ak_ens (best)"),
                 ("ak_score_mean", "ak_score (mean)")],
    "genscore": [("gen_score_mean", "GenScore (mean)"),
                 ("gen_score_p90", "GenScore (p90)"),
                 ("gen_score_best", "GenScore (best)")],
    "descriptors": [("mol.QED", "QED"),
                    ("mol.heavy_atoms", "heavy atoms"),
                    ("mol.cLogP", "cLogP"),
                    ("mol.arom_rings", "aromatic rings"),
                    ("mol.HBA", "HBA")],
}

PRETTY = {k: lbl for fam in FAMILIES.values() for k, lbl in fam}
PRETTY["submitted"] = "submitted score"

#: Fixed categorical slot per signal, so a signal keeps its hue across panels
#: and targets. Colour follows the entity, never its rank within a panel — if
#: L02 is missing a signal the survivors must not be repainted.
SLOT = {}
for _fam in FAMILIES.values():
    for _i, (_k, _) in enumerate(_fam):
        SLOT[_k] = _i
#: Signals that share a panel in fig4 need their own fixed slots.
SLOT_CROSS_FAMILY = {"gen_score_mean": 0, "boltz_LE": 1, "mol.QED": 2,
                     "ak_dock_best": 3}


def load(target: str, data_dir: Path):
    path = data_dir / f"{target}_signals.csv"
    rows = list(csv.DictReader(path.open()))
    y = np.array([r["label"] in ("True", "1", "true") for r in rows], dtype=bool)

    def col(key):
        out = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            v = r.get(key, "")
            if v not in ("", "None", "nan", "NaN"):
                try:
                    out[i] = float(v)
                except ValueError:
                    pass
        return out

    return rows, y, col


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style(theme: dict):
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink2"],
        "axes.edgecolor": theme["grid"],
        "xtick.color": theme["ink2"],
        "ytick.color": theme["ink2"],
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "600",
        "axes.grid": True,
        "grid.color": theme["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.dpi": 160,
    })


def fig_auc_ranking(data, theme, out_png, out_csv):
    """Per-signal AUC for both targets.

    Dots on a stem from 0.5, not bars: the useful range is 0.45-0.65, and a bar
    chart on a truncated axis makes length stop meaning value. The stem length
    *is* the distance from random, which is the quantity that matters.
    """
    import matplotlib.pyplot as plt

    keys = ["submitted"] + [k for fam in FAMILIES.values() for k, _ in fam]
    table = []
    for T, (rows, y, col) in data.items():
        for k in keys:
            m = signal_metrics(col(k), y)
            if m:
                table.append(dict(target=T, signal=k, label=PRETTY.get(k, k), **m))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.6), sharex=True)
    for ax, T in zip(axes, data):
        sub = sorted([r for r in table if r["target"] == T], key=lambda r: r["auc"])
        ypos = np.arange(len(sub))
        vals = np.array([r["auc"] for r in sub])
        for yi, v, r in zip(ypos, vals, sub):
            is_sub = r["signal"] == "submitted"
            c = theme["ink2"] if is_sub else theme["series"][0]
            ax.plot([0.5, v], [yi, yi], color=c, lw=2.0, solid_capstyle="round",
                    zorder=3)
            ax.scatter([v], [yi], s=46, color=c, edgecolor=theme["surface"],
                       linewidth=2.0, zorder=4)
        ax.axvline(0.5, color=theme["muted"], lw=1.0, zorder=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in sub], fontsize=8.5)
        for tick, r in zip(ax.get_yticklabels(), sub):
            if r["signal"] == "submitted":
                tick.set_color(theme["ink"])
                tick.set_fontweight("600")
        ax.set_xlim(0.44, 0.665)
        ax.set_ylim(-0.8, len(sub) - 0.2)
        ax.set_xlabel("AUC")
        n_act = max(r["n_act"] for r in sub)
        n_tot = max(r["n"] for r in sub)
        ax.set_title(f"{T}  ·  N={n_tot}, {n_act} binders", loc="left",
                     color=theme["ink"])
        ax.grid(axis="y", visible=False)
        # every dot gets its value: the contrast relief rule, and the axis is
        # too compressed to read a 0.01 difference off it
        for yi, v in zip(ypos, vals):
            off = 0.005 if v >= 0.5 else -0.005
            ax.text(v + off, yi, f"{v:.3f}", va="center",
                    ha="left" if v >= 0.5 else "right",
                    fontsize=7.5, color=theme["ink2"])
        ax.text(0.5, -0.65, " random", color=theme["muted"], fontsize=8, va="center")
    fig.suptitle("Stage-1 signals ranked by AUC against the released binder labels",
                 x=0.008, y=0.975, ha="left", fontsize=12.5, color=theme["ink"])
    fig.text(0.008, 0.933,
             "Stem length = distance from random. Nothing clears 0.64; a usable "
             "screen needs ~0.8. The submitted score is the grey row.",
             ha="left", fontsize=9, color=theme["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_png)
    plt.close(fig)
    _write_csv(out_csv, table)


def fig_roc(data, theme, out_png, out_csv):
    """ROC per signal family — small multiples, one panel per family per target."""
    import matplotlib.pyplot as plt

    fams = list(FAMILIES)
    fig, axes = plt.subplots(len(data), len(fams),
                             figsize=(4.0 * len(fams), 4.1 * len(data)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    rows_out = []
    for ri, (T, (rows, y, col)) in enumerate(data.items()):
        sub_fpr, sub_tpr, sub_auc = roc_curve(col("submitted"), y)
        for ci, fam in enumerate(fams):
            ax = axes[ri, ci]
            ax.plot([0, 1], [0, 1], color=theme["muted"], lw=1.0, zorder=2)
            # the submitted file is the benchmark, drawn in ink rather than a
            # categorical hue — it is a reference, not one series among peers
            ax.plot(sub_fpr, sub_tpr, color=theme["ink2"], lw=1.6, zorder=3)
            n_drawn = 0
            for key, label in FAMILIES[fam]:
                v = col(key)
                if np.isnan(v).all():
                    continue
                fpr, tpr, auc = roc_curve(v, y)
                ax.plot(fpr, tpr, color=theme["series"][SLOT[key]], lw=2.0,
                        zorder=4, label=f"{label}  {auc:.3f}")
                n_drawn += 1
                rows_out.append(dict(target=T, family=fam, signal=key,
                                     label=label, auc=round(float(auc), 4)))
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if n_drawn:
                ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.35)
            else:
                # L02 has no AKScore2/GenScore columns — rescoring never ran for
                # that target. Say so rather than shipping an empty panel.
                ax.text(0.5, 0.14, "not run for this target",
                        transform=ax.transAxes, ha="center", fontsize=9,
                        color=theme["muted"])
            if ri == 0:
                ax.set_title(fam, loc="left", color=theme["ink"])
            if ci == 0:
                ax.set_ylabel(f"{T}\ntrue-positive rate")
            if ri == len(data) - 1:
                ax.set_xlabel("false-positive rate")
            ax.text(0.03, 0.93, f"submitted {sub_auc:.3f}", transform=ax.transAxes,
                    fontsize=8, color=theme["ink2"])
    fig.suptitle("ROC by signal family — the submitted score in grey, the diagonal is random",
                 x=0.006, y=0.995, ha="left", fontsize=12.5, color=theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    fig.savefig(out_png)
    plt.close(fig)
    _write_csv(out_csv, rows_out)


def fig_auc_vs_bedroc(data, theme, out_png, out_csv):
    """Global rank quality vs early enrichment — they disagree, which is the point."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data), figsize=(11, 5.2))
    axes = np.atleast_1d(axes)
    rows_out = []
    # named exemplars from the postmortem: the two axes rank these oppositely
    HIGHLIGHT = {"mol.cLogP", "ak_dock_best", "mol.QED", "submitted"}
    for ax, (T, (rows, y, col)) in zip(axes, data.items()):
        keys = ["submitted"] + [k for fam in FAMILIES.values() for k, _ in fam]
        pts = []
        for k in keys:
            m = signal_metrics(col(k), y)
            if m:
                pts.append((k, m))
        n_tot = max(m["n"] for _, m in pts)
        n_act = max(m["n_act"] for _, m in pts)
        rb_mean, rb_p95 = random_bedroc(n_tot, n_act)
        ax.axhline(rb_mean, color=theme["muted"], lw=1.0)
        ax.axhline(rb_p95, color=theme["muted"], lw=1.0, alpha=0.55)
        ax.axvline(0.5, color=theme["muted"], lw=1.0)
        aucs = [m["auc"] for _, m in pts]
        beds = [m["bedroc"] for _, m in pts]
        pad_x = 0.1 * (max(aucs) - min(aucs))
        pad_y = 0.1 * (max(beds) - min(beds))
        # room on the right for the annotation of the right-most highlighted point
        ax.set_xlim(min(aucs) - pad_x, max(aucs) + 3.2 * pad_x)
        ax.set_ylim(min(min(beds), rb_mean) - pad_y, max(max(beds), rb_p95) + pad_y)
        for k, m in pts:
            hl = k in HIGHLIGHT
            ax.scatter(m["auc"], m["bedroc"], s=64 if hl else 34,
                       color=theme["series"][0] if hl else theme["muted"],
                       alpha=1.0 if hl else 0.55,
                       edgecolor=theme["surface"], linewidth=1.6,
                       zorder=5 if hl else 3)
            if hl:
                # flip the label inward when the point sits near the right edge
                right = m["auc"] > min(aucs) + 0.72 * (max(aucs) - min(aucs))
                ax.annotate(PRETTY.get(k, k), (m["auc"], m["bedroc"]),
                            textcoords="offset points",
                            xytext=(-8, 8) if right else (8, 4),
                            ha="right" if right else "left",
                            fontsize=8.5, color=theme["ink"])
            rows_out.append(dict(target=T, signal=k, label=PRETTY.get(k, k),
                                 auc=round(m["auc"], 4),
                                 bedroc=round(m["bedroc"], 4),
                                 random_bedroc_mean=round(rb_mean, 4),
                                 random_bedroc_p95=round(rb_p95, 4)))
        x1 = ax.get_xlim()[1]
        ax.text(x1, rb_p95, "random 95th pct ", color=theme["muted"],
                fontsize=8, va="bottom", ha="right")
        ax.text(x1, rb_mean, "random mean ", color=theme["muted"],
                fontsize=8, va="bottom", ha="right")
        ax.set_xlabel("AUC  (global rank quality)")
        ax.set_ylabel("BEDROC α=20  (early enrichment)")
        ax.set_title(f"{T}  ·  N={n_tot}, {n_act} binders", loc="left",
                     color=theme["ink"])
    fig.suptitle("The two metrics rank the signals differently",
                 x=0.006, ha="left", fontsize=12, color=theme["ink"])
    fig.text(0.006, 0.925,
             "cLogP is third on AUC yet below random on BEDROC; ak_dock is mid-AUC "
             "with the best top-1 % hit rate.",
             ha="left", fontsize=9, color=theme["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_png)
    plt.close(fig)
    _write_csv(out_csv, rows_out)


def fig_enrichment(data, theme, out_png, out_csv):
    """Cumulative recall vs fraction screened — what a screen would actually return."""
    import matplotlib.pyplot as plt

    PICK = ["submitted", "gen_score_mean", "boltz_LE", "mol.QED", "ak_dock_best"]
    # Top row is the whole library; bottom row zooms to the first 10 %, which is
    # the only part a real screen would buy — and the part the full curve hides.
    fig, axes = plt.subplots(2, len(data), figsize=(11, 8.6), sharey="row")
    axes = np.atleast_2d(axes)
    rows_out = []
    for ci, (T, (rows, y, col)) in enumerate(data.items()):
        for ri, xmax in enumerate((1.0, 0.10)):
            ax = axes[ri, ci]
            ax.plot([0, 1], [0, 1], color=theme["muted"], lw=1.0, zorder=2)
            for k in PICK:
                v = col(k)
                if np.isnan(v).all():
                    continue
                x, r = enrichment_curve(v, y)
                if k == "submitted":
                    ax.plot(x, r, color=theme["ink2"], lw=1.8, zorder=3,
                            label="submitted")
                else:
                    # fixed slot per signal: a signal missing from L02 must not
                    # repaint the ones that survive
                    ax.plot(x, r, color=theme["series"][SLOT_CROSS_FAMILY[k]],
                            lw=2.0, zorder=4, label=PRETTY.get(k, k))
                if ri == 0:
                    j = min(np.searchsorted(x, 0.05), len(r) - 1)
                    rows_out.append(dict(
                        target=T, signal=k, label=PRETTY.get(k, k),
                        recall_at_5pct=round(float(r[j]), 4)))
            ax.set_xlim(0, xmax)
            ax.axvline(0.05, color=theme["muted"], lw=1.0, zorder=2)
            if ri == 0:
                ax.set_ylim(0, 1)
                ax.set_title(T, loc="left", color=theme["ink"])
                ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.35)
                ax.text(0.058, 0.93, "top 5 %", color=theme["muted"], fontsize=8)
            else:
                ax.set_ylim(0, 0.22)
                ax.set_title(f"{T} — first 10 %", loc="left", color=theme["ink2"],
                             fontsize=9)
                ax.text(0.0515, 0.205, "top 5 %", color=theme["muted"], fontsize=8)
            ax.set_xlabel("fraction of the library screened")
    axes[0, 0].set_ylabel("fraction of real binders found")
    axes[1, 0].set_ylabel("fraction of real binders found")
    fig.suptitle("Enrichment — buying the top 5 % returns barely more than chance",
                 x=0.006, y=0.99, ha="left", fontsize=12.5, color=theme["ink"])
    fig.text(0.006, 0.955,
             "The diagonal is what random picking returns. A curve hugging it in "
             "the zoom row means the signal bought nothing.",
             ha="left", fontsize=9, color=theme["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2.6)
    fig.savefig(out_png)
    plt.close(fig)
    _write_csv(out_csv, rows_out)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


FIGURES = [
    ("fig1_auc_ranking", fig_auc_ranking),
    ("fig2_roc", fig_roc),
    ("fig3_auc_vs_bedroc", fig_auc_vs_bedroc),
    ("fig4_enrichment", fig_enrichment),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/tmp/claude-1001/figdata"),
                    help="directory holding {target}_signals.csv from collect()")
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs" / "figures")
    ap.add_argument("--targets", nargs="+", default=["L01", "L02"])
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = {T: load(T, args.data_dir) for T in args.targets}
    for T, (rows, y, _) in data.items():
        print(f"{T}: {len(rows)} fragments, {int(y.sum())} binders")

    for theme_name in args.themes:
        theme = THEMES[theme_name]
        _style(theme)
        suffix = "" if theme_name == "light" else "_dark"
        for stem, fn in FIGURES:
            png = args.out_dir / f"{stem}{suffix}.png"
            csv_path = args.out_dir / f"{stem}.csv"
            fn(data, theme, png, csv_path)
            print(f"  wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
