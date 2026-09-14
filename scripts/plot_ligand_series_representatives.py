#!/usr/bin/env python3
"""Render five molecular-weight representatives for each CASP17 LG library.

For L01 and L02 the selection is deterministic: the molecular-weight minimum,
the nearest molecule to each 25th/50th/75th percentile, and the maximum.
The source is the revealed ``ligands_truth.csv`` files, so the binder marker is
an annotation only and has no role in selecting molecules.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


REPO = Path(__file__).resolve().parents[1]
TARGETS = ("L01", "L02")
SLOTS = (("smallest", 0.0), ("P25", 0.25), ("median", 0.50), ("P75", 0.75), ("largest", 1.0))


@dataclass(frozen=True)
class Ligand:
    target: str
    fragment_id: str
    smiles: str
    is_binder: bool
    molecular_weight: float


def _load_library(target: str) -> list[Ligand]:
    path = REPO / "inputs" / "ligand_series" / target / "ligands_truth.csv"
    ligands: list[Ligand] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:
                raise ValueError(f"invalid SMILES for {row['CASP ID']}: {row['canonical_smiles']}")
            ligands.append(
                Ligand(
                    target=target,
                    fragment_id=row["CASP ID"],
                    smiles=row["canonical_smiles"],
                    is_binder=row["binding"].upper() == "TRUE",
                    molecular_weight=Descriptors.MolWt(molecule),
                )
            )
    return sorted(ligands, key=lambda ligand: (ligand.molecular_weight, ligand.fragment_id))


def _select(ligands: list[Ligand]) -> list[tuple[str, Ligand]]:
    weights = np.array([ligand.molecular_weight for ligand in ligands])
    selected: list[tuple[str, Ligand]] = []
    used: set[str] = set()
    for label, quantile in SLOTS:
        target_weight = np.quantile(weights, quantile)
        candidates = sorted(
            ligands,
            key=lambda ligand: (ligand.fragment_id in used, abs(ligand.molecular_weight - target_weight)),
        )
        ligand = candidates[0]
        selected.append((label, ligand))
        used.add(ligand.fragment_id)
    return selected


def _molecule_image(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    Draw.rdMolDraw2D.PrepareMolForDrawing(molecule)
    return Draw.MolToImage(molecule, size=(360, 230), kekulize=True)


def _write_selection(path: Path, selected: dict[str, list[tuple[str, Ligand]]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target", "slot", "fragment_id", "molecular_weight_da", "binding", "canonical_smiles"],
        )
        writer.writeheader()
        for target in TARGETS:
            for slot, ligand in selected[target]:
                writer.writerow(
                    {
                        "target": target,
                        "slot": slot,
                        "fragment_id": ligand.fragment_id,
                        "molecular_weight_da": f"{ligand.molecular_weight:.2f}",
                        "binding": str(ligand.is_binder).lower(),
                        "canonical_smiles": ligand.smiles,
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "figures" / "ligand_series_representative_ligands.png",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=REPO / "docs" / "figures" / "ligand_series_representative_ligands.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)

    selected = {target: _select(_load_library(target)) for target in TARGETS}
    _write_selection(args.selection_output, selected)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
        }
    )
    figure, axes = plt.subplots(2, 5, figsize=(15, 6.9), constrained_layout=True)
    for row, target in enumerate(TARGETS):
        for column, (slot, ligand) in enumerate(selected[target]):
            axis = axes[row, column]
            axis.imshow(_molecule_image(ligand.smiles))
            axis.set_axis_off()
            binder_text = "binder" if ligand.is_binder else "non-binder"
            axis.set_title(
                f"{slot.upper()}\n{ligand.fragment_id}  |  {ligand.molecular_weight:.1f} Da\n{binder_text}",
                fontsize=9.3,
                fontweight="bold" if slot in {"smallest", "largest"} else "normal",
                pad=7,
            )
        axes[row, 0].text(
            -0.16,
            0.5,
            target,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color="#203647",
        )
    figure.suptitle(
        "CASP17 ligand-series: molecular-weight representatives",
        x=0.05,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.05,
        0.94,
        "Per library: minimum, nearest 25th/50th/75th percentile, and maximum molecular weight. "
        "Structure selection does not use binder status.",
        ha="left",
        fontsize=9.5,
        color="#555550",
    )
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"wrote {args.selection_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
