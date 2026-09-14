#!/usr/bin/env python3
"""Matched-signal comparison of the ligand-series stage-1 targets.

`fig1_auc_ranking` shows every signal that exists per target, so L01 carries the
rescoring families (GenScore / AKScore2) that never ran on L02 and the two panels
are not comparable. This figure keeps only the signals both targets share — the
submitted score, the folded-model signals, and the molecular descriptors — and
plots them as one paired dot plot so L01 and L02 read off the same axis.

    uv run python scripts/plot_stage1_matched_signals.py --out-dir docs/figures

Reads the metrics already computed by `plot_stage1_signals.py`
(`fig1_auc_ranking.csv`) so the numbers match the postmortem exactly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: signals present for BOTH targets, grouped; order inside a group is plot order
MATCHED = [
    ("submitted", "submitted score", "submitted"),
    ("boltz_prob_mean", "binder probability (mean)", "folded"),
    ("boltz_aff_min", "affinity (min)", "folded"),
    ("boltz_aff_mean", "affinity (mean)", "folded"),
    ("boltz_LE", "ligand efficiency", "folded"),
    ("mol.QED", "QED", "descriptor"),
    ("mol.heavy_atoms", "heavy atoms", "descriptor"),
    ("mol.cLogP", "cLogP", "descriptor"),
    ("mol.arom_rings", "aromatic rings", "descriptor"),
    ("mol.HBA", "HBA", "descriptor"),
]

SERIES_LIGHT = ["#2a78d6", "#eb6834"]
SERIES_DARK = ["#3987e5", "#d95926"]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#8a8a85", grid="#e6e5e1", series=SERIES_LIGHT),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                 muted="#84837b", grid="#33332f", series=SERIES_DARK),
}


def load_metrics(csv_path: Path) -> dict:
    out: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(csv_path.open()):
        out[(r["target"], r["signal"])] = r
    return out


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
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })


def render(metrics: dict, theme_name: str, out_png: Path, out_csv: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[theme_name]
    _style(theme)
    c01, c02 = theme["series"]

    rows = []
    for key, label, group in MATCHED:
        a = metrics.get(("L01", key))
        b = metrics.get(("L02", key))
        if not a or not b:
            continue
        rows.append(dict(key=key, label=label, group=group,
                         auc01=float(a["auc"]), auc02=float(b["auc"]),
                         n01=int(a["n"]), n02=int(b["n"])))
    rows.sort(key=lambda r: r["auc01"], reverse=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.1), dpi=220)
    ys = list(range(len(rows)))[::-1]

    for y, r in zip(ys, rows):
        lo, hi = sorted((r["auc01"], r["auc02"]))
        ax.plot([lo, hi], [y, y], color=theme["grid"], lw=2.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([r["auc01"]], [y], s=64, color=c01, zorder=4,
                   edgecolors=theme["surface"], linewidths=1.6)
        ax.scatter([r["auc02"]], [y], s=64, color=c02, zorder=4,
                   edgecolors=theme["surface"], linewidths=1.6)
        for val, colr, dx in ((r["auc01"], c01, -1), (r["auc02"], c02, 1)):
            ha = "right" if (val == lo) else "left"
            off = -0.006 if ha == "right" else 0.006
            ax.text(val + off, y, f"{val:.3f}", ha=ha, va="center",
                    fontsize=9, color=theme["ink2"], zorder=5)

    ax.axvline(0.5, color=theme["muted"], lw=1.0, zorder=1)
    ax.text(0.5, len(rows) - 0.35, "random", ha="center", va="bottom",
            fontsize=9, color=theme["muted"])

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(theme["ink"] if r["group"] == "descriptor" else theme["ink2"])
        if r["group"] == "submitted":
            tick.set_fontweight("bold")
    ax.set_xlabel("AUC against the released binder labels")
    ax.set_xlim(0.44, 0.68)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.grid(axis="x", color=theme["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)

    ax.scatter([], [], s=64, color=c01, label="L01 · 1210 fragments, 80 binders")
    ax.scatter([], [], s=64, color=c02, label="L02 · 647 fragments, 29 binders")
    ax.legend(loc="lower left", frameon=False, fontsize=10, handletextpad=0.4,
              borderaxespad=0.2)

    fig.text(0.012, 0.965, "The same signals, scored on both targets",
             ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.905,
             "Rescoring families are dropped — they ran on L01 only. What is left ranks the library by "
             "molecular character,\nand the two targets disagree on almost every one of them.",
             ha="left", va="top", fontsize=10, color=theme["ink2"])
    fig.text(0.012, 0.015,
             "Folded-model rows cover 269 of 647 L02 fragments; descriptor rows cover all of both libraries.",
             ha="left", va="bottom", fontsize=8.5, color=theme["muted"])
    fig.tight_layout(rect=(0, 0.045, 1, 0.80))
    fig.savefig(out_png)
    plt.close(fig)

    if out_csv:
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["signal", "label", "group", "auc_L01", "auc_L02", "n_L01", "n_L02"])
            for r in rows:
                w.writerow([r["key"], r["label"], r["group"],
                            f"{r['auc01']:.4f}", f"{r['auc02']:.4f}", r["n01"], r["n02"]])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-csv", type=Path,
                    default=REPO / "docs/figures/fig1_auc_ranking.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "docs/figures")
    args = ap.parse_args()

    metrics = load_metrics(args.metrics_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = render(metrics, "light", args.out_dir / "fig1b_matched_signals.png",
                  args.out_dir / "fig1b_matched_signals.csv")
    render(metrics, "dark", args.out_dir / "fig1b_matched_signals_dark.png", None)
    print(f"wrote fig1b_matched_signals.png ({len(rows)} matched signals)")
    for r in rows:
        print(f"  {r['label']:26s} L01 {r['auc01']:.3f}   L02 {r['auc02']:.3f}")


if __name__ == "__main__":
    main()
