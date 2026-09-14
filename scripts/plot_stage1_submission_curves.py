#!/usr/bin/env python3
"""Plot ROC and early-enrichment curves for the submitted L01/L02 scores.

The plotted scores are read directly from ``experiments/ligand_series/*LG129``
and joined with the revealed ``ligands_truth.csv`` labels.  Consequently, this
uses the L01 submission exactly as sent (including its one missing score), not
the later ID-shift-corrected signal analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_stage1_signals import metrics, read_submitted, read_truth  # noqa: E402


COLORS = {"L01": "#2a78d6", "L02": "#eb6834"}


def _ranked_labels(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return labels ordered by descending score with stable tie handling."""
    order = np.argsort(-scores, kind="mergesort")
    return labels[order]


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores, ranked_labels = scores[order], labels[order]
    true_positive = np.cumsum(ranked_labels)
    false_positive = np.cumsum(~ranked_labels)
    keep = np.r_[np.diff(ranked_scores) != 0, True]
    return (
        np.r_[0.0, false_positive[keep] / false_positive[-1]],
        np.r_[0.0, true_positive[keep] / true_positive[-1]],
    )


def _enrichment_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranked_labels = _ranked_labels(scores, labels)
    screened = np.arange(len(ranked_labels) + 1) / len(ranked_labels)
    recall = np.r_[0.0, np.cumsum(ranked_labels) / ranked_labels.sum()]
    return screened, recall


def _load(target: str) -> tuple[np.ndarray, np.ndarray]:
    truth = read_truth(REPO / "inputs" / "ligand_series" / target / "ligands_truth.csv")
    submitted = read_submitted(REPO / "experiments" / "ligand_series" / f"{target}LG129.bind.txt")
    labels, scores = [], []
    for fragment_id, (label, _) in truth.items():
        score = submitted.get(fragment_id)
        if score is not None:
            labels.append(label)
            scores.append(score)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)


def _metric_label(metric: dict[str, float]) -> str:
    return (
        f"AUC {metric['AUC']:.3f}\n"
        f"BEDROC20 {metric['BEDROC']:.3f}\n"
        f"EF1% {metric['EF1']:.2f}  |  EF5% {metric['EF5']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ligand_series_stage1_submission_curves.png"),
        help="output PNG path",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex="row", sharey="row")
    for column, target in enumerate(("L01", "L02")):
        scores, labels = _load(target)
        result = metrics(scores, labels)
        color = COLORS[target]

        roc_ax = axes[0, column]
        fpr, tpr = _roc_curve(scores, labels)
        roc_ax.plot([0, 1], [0, 1], color="#85857d", linewidth=1.25, linestyle="--", label="random")
        roc_ax.plot(fpr, tpr, color=color, linewidth=2.6, label="submitted score")
        roc_ax.fill_between(fpr, tpr, fpr, color=color, alpha=0.12)
        roc_ax.text(0.96, 0.05, _metric_label(result), transform=roc_ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color="#222220",
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fcfcfb", "edgecolor": "#d8d8d2"})
        roc_ax.set_title(f"{target} ROC  |  N={result['N']}, binders={result['n']}",
                         loc="left", fontsize=12, weight="semibold")
        roc_ax.set_xlim(0, 1)
        roc_ax.set_ylim(0, 1)
        roc_ax.set_aspect("equal")
        roc_ax.grid(zorder=0)
        roc_ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        if column == 0:
            roc_ax.set_ylabel("true-positive rate")

        enrichment_ax = axes[1, column]
        screened, recall = _enrichment_curve(scores, labels)
        enrichment_ax.plot([0, 0.20], [0, 0.20], color="#85857d", linewidth=1.25, linestyle="--", label="random")
        enrichment_ax.plot(screened, recall, color=color, linewidth=2.6, label="submitted score")
        enrichment_ax.axvline(0.01, color="#a6a69e", linewidth=1.0, linestyle=":")
        enrichment_ax.axvline(0.05, color="#85857d", linewidth=1.0, linestyle="--")
        enrichment_ax.text(0.01, 0.198, "1%", ha="center", va="top", fontsize=8, color="#666661")
        enrichment_ax.text(0.05, 0.198, "5%", ha="center", va="top", fontsize=8, color="#666661")
        enrichment_ax.set_xlim(0, 0.20)
        enrichment_ax.set_ylim(0, 0.55)
        enrichment_ax.grid(zorder=0)
        enrichment_ax.set_xlabel("fraction of library screened")
        if column == 0:
            enrichment_ax.set_ylabel("fraction of binders recovered")

    fig.suptitle("CASP17 ligand-series stage 1: submitted ranking behavior",
                 x=0.07, y=0.98, ha="left", fontsize=15, weight="bold")
    fig.text(0.07, 0.935,
             "Top: global discrimination (ROC-AUC). Bottom: practical early enrichment; dashed lines are random ranking.",
             ha="left", fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.015,
             "Source: submitted *.bind.txt joined to revealed ligands_truth.csv. Higher is better for ROC-AUC, BEDROC, and EF.",
             ha="left", fontsize=8, color="#666661")
    fig.tight_layout(rect=(0.04, 0.06, 1, 0.91))
    fig.savefig(args.output, dpi=220)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
