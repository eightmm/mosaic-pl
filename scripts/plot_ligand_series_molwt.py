#!/usr/bin/env python3
"""Plot molecular-weight distributions for the revealed L01/L02 libraries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TARGETS = ("L01", "L02")
COLORS = {"all": "#7d7d75", "nonbinder": "#8fa8c7", "binder": "#e55934"}


def _load(target: str) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    path = REPO / "inputs" / "ligand_series" / target / "ligands_truth.csv"
    weights: list[float] = []
    binders: list[bool] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:
                raise ValueError(f"invalid SMILES for {row['CASP ID']}: {row['canonical_smiles']}")
            weights.append(Descriptors.MolWt(molecule))
            binders.append(row["binding"].upper() == "TRUE")
    return np.asarray(weights), np.asarray(binders, dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ligand_series_fragment_molwt_distribution.png"),
        help="output PNG path",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="histogram bin width in Da (default: 10)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {target: _load(target) for target in TARGETS}
    minimum = min(weights.min() for weights, _ in data.values())
    maximum = max(weights.max() for weights, _ in data.values())
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    start = np.floor(minimum / args.bin_width) * args.bin_width
    stop = np.ceil(maximum / args.bin_width) * args.bin_width + args.bin_width
    bins = np.arange(start, stop + args.bin_width * 0.5, args.bin_width)

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
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    for axis, target in zip(axes, TARGETS):
        weights, binder = data[target]
        nonbinder = weights[~binder]
        actual_binders = weights[binder]
        nonbinder_median = np.median(nonbinder)
        binder_median = np.median(actual_binders)
        axis.hist(
            nonbinder,
            bins=bins,
            density=True,
            color=COLORS["nonbinder"],
            alpha=0.80,
            label=f"non-binder (n={len(nonbinder)})\nmedian {nonbinder_median:.1f} Da",
            zorder=2,
        )
        axis.hist(
            actual_binders,
            bins=bins,
            density=True,
            color=COLORS["binder"],
            alpha=0.85,
            label=f"binder (n={len(actual_binders)})\nmedian {binder_median:.1f} Da",
            zorder=3,
        )
        axis.axvline(nonbinder_median, color="#555550", linewidth=1.3, linestyle="--", zorder=4)
        axis.axvline(binder_median, color=COLORS["binder"], linewidth=1.5, linestyle=":", zorder=4)
        axis.set_title(
            f"{target}  |  {len(actual_binders)} binder / {len(nonbinder)} non-binder",
            loc="left",
            fontsize=12,
            weight="semibold",
        )
        axis.grid(axis="y", zorder=0)
        axis.set_axisbelow(True)
        axis.set_xlabel("molecular weight (Da)")
        axis.legend(frameon=False, loc="upper left", fontsize=8.5, labelspacing=0.8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
