#!/usr/bin/env python3
"""Render the published CASP17 L01/L02 stage-1 submission summary.

The values below are the submitted-file metrics documented in
``docs/status.md`` and ``docs/ligand_series_stage1_postmortem.md``.  This
small self-contained chart intentionally does not recompute scores: it makes
the recorded submission results easy to compare and keeps their provenance
explicit.
"""
from __future__ import annotations

import argparse
from pathlib import Path


METRICS = {
    "L01": {"AUC": 0.557, "BEDROC20": 0.116, "EF1%": 0.00, "EF5%": 1.76},
    "L02": {"AUC": 0.626, "BEDROC20": 0.180, "EF1%": 7.44, "EF5%": 2.09},
}
BASELINES = {"AUC": 0.500, "BEDROC20": None, "EF1%": 1.00, "EF5%": 1.00}
BEDROC_RANDOM = {"L01": 0.090, "L02": 0.075}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ligand_series_stage1_submission_metrics.png"),
        help="output PNG path",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#d8d8d2",
            "axes.labelcolor": "#42423f",
            "xtick.color": "#42423f",
            "ytick.color": "#42423f",
            "grid.color": "#e7e7e2",
            "grid.linewidth": 0.8,
        }
    )

    labels = list(BASELINES)
    fig, axes = plt.subplots(1, len(labels), figsize=(12, 4.3))
    colors = {"L01": "#2a78d6", "L02": "#eb6834"}
    x = np.array([0, 1])
    width = 0.34

    for ax, metric in zip(axes, labels):
        values = [METRICS[target][metric] for target in ("L01", "L02")]
        bars = ax.bar(x, values, width=0.62, color=[colors["L01"], colors["L02"]], zorder=3)
        baseline = BASELINES[metric]
        if metric == "BEDROC20":
            for target, xpos in zip(("L01", "L02"), x):
                ax.hlines(BEDROC_RANDOM[target], xpos - width, xpos + width,
                          color="#7f7f78", linewidth=1.4, linestyle="--", zorder=4)
            ax.text(0.5, 0.012, "dashed: target-specific random mean",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#666661")
        else:
            ax.axhline(baseline, color="#7f7f78", linewidth=1.4, linestyle="--", zorder=4)
            ax.text(0.5, 0.012, f"dashed: random = {baseline:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#666661")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035,
                    f"{value:.3f}" if metric in {"AUC", "BEDROC20"} else f"{value:.2f}",
                    ha="center", va="bottom", fontsize=10, color="#20201e", weight="semibold")
        ax.set_title(metric, loc="left", weight="semibold", fontsize=11)
        ax.set_xticks(x, ["L01", "L02"])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

    fig.suptitle("CASP17 ligand-series stage 1: submitted-file performance",
                 x=0.055, y=0.99, ha="left", fontsize=15, weight="bold")
    fig.text(0.055, 0.92,
             "L01: 1,210 truth labels / 1,209 submitted scores / 80 binders     "
             "L02: 647 fragments / 29 binders",
             ha="left", fontsize=9.5, color="#52514e")
    fig.text(0.055, 0.015,
             "Source: docs/status.md and docs/ligand_series_stage1_postmortem.md. "
             "Higher is better for every displayed metric.",
             ha="left", fontsize=8, color="#666661")
    fig.tight_layout(rect=(0.03, 0.07, 1, 0.88))
    fig.savefig(args.output, dpi=220)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
