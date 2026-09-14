#!/usr/bin/env python3
"""Render molecular-weight representatives split by revealed binder status."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ligand_series_representatives import REPO, SLOTS, TARGETS, _load_library, _molecule_image, _select


GROUPS = (("binder", True, "#b9453a"), ("non-binder", False, "#386c9e"))


def _write_selection(path: Path, selected) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target", "binding_group", "slot", "fragment_id", "molecular_weight_da", "canonical_smiles"],
        )
        writer.writeheader()
        for target, group_name, _, ligands in selected:
            for slot, ligand in ligands:
                writer.writerow(
                    {
                        "target": target,
                        "binding_group": group_name,
                        "slot": slot,
                        "fragment_id": ligand.fragment_id,
                        "molecular_weight_da": f"{ligand.molecular_weight:.2f}",
                        "canonical_smiles": ligand.smiles,
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "figures" / "ligand_series_binder_nonbinder_representatives.png",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=REPO / "docs" / "figures" / "ligand_series_binder_nonbinder_representatives.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)

    selected = []
    for target in TARGETS:
        library = _load_library(target)
        for group_name, binder_value, color in GROUPS:
            group = [ligand for ligand in library if ligand.is_binder is binder_value]
            if len(group) < len(SLOTS):
                raise ValueError(f"{target} {group_name} has only {len(group)} ligands")
            selected.append((target, group_name, color, _select(group)))
    _write_selection(args.selection_output, selected)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
        }
    )
    figure, axes = plt.subplots(4, 5, figsize=(15, 12.6), constrained_layout=True)
    for row, (target, group_name, color, ligands) in enumerate(selected):
        for column, (slot, ligand) in enumerate(ligands):
            axis = axes[row, column]
            axis.imshow(_molecule_image(ligand.smiles))
            axis.set_axis_off()
            axis.set_title(
                f"{slot.upper()}\n{ligand.fragment_id}  |  {ligand.molecular_weight:.1f} Da",
                fontsize=9.2,
                fontweight="bold" if slot in {"smallest", "largest"} else "normal",
                pad=7,
            )
        axes[row, 0].text(
            -0.17,
            0.5,
            f"{target}\n{group_name}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13.5,
            fontweight="bold",
            color=color,
        )
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"wrote {args.selection_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
