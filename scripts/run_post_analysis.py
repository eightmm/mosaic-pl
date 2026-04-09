#!/usr/bin/env python3
"""Post-analysis: run BA-Pred and RMSD-Pred on all cofolding + docking results.

Collects protein structures from cofolding outputs and docking poses,
then runs BA-Pred (binding affinity) and RMSD-Pred (pose RMSD) on each pair.

Usage:
    python run_post_analysis.py --run-dir experiments/runs/full_pipeline_test

Output:
    <run-dir>/outputs/analysis/
        ba_pred_results.tsv       (per-model × per-docking affinity)
        rmsd_pred_results.tsv     (per-model × per-docking RMSD)
        summary.json              (aggregated best picks)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_receptor_pdbs(run_dir: Path) -> dict[str, Path]:
    """Find receptor PDB files from cofolding outputs."""
    receptors: dict[str, Path] = {}

    # Docking prep receptor (protonated, best model auto-selected)
    prep_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if prep_pdb.exists():
        receptors["docking_prep"] = prep_pdb

    # Per-model CIF → we use the docking prep PDB since it's already converted
    # But also check for raw cofolding CIFs for model-specific analysis
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue
        for cif in sorted(model_dir.rglob("*.cif")):
            if "model" in cif.name or "sample" in cif.name:
                receptors[model] = cif
                break

    return receptors


def find_ligand_files(run_dir: Path) -> dict[str, Path]:
    """Find ligand files from docking outputs."""
    ligands: dict[str, Path] = {}

    # Docking prep SDF (3D from SMILES)
    for sdf in (run_dir / "inputs" / "docking").glob("ligand_*.sdf"):
        ligands["input_sdf"] = sdf
        break

    # Vina docked poses
    vina_pdbqt = run_dir / "outputs" / "vina" / "docked.pdbqt"
    if vina_pdbqt.exists():
        ligands["vina"] = vina_pdbqt

    # AutoDock-GPU DLG
    adg_dlg = run_dir / "outputs" / "autodock_gpu" / "docking.dlg"
    if adg_dlg.exists():
        ligands["autodock_gpu"] = adg_dlg

    # Protenix-Dock SDF results
    for sdf in (run_dir / "outputs" / "protenix_dock").rglob("*.sdf"):
        ligands["protenix_dock"] = sdf
        break

    return ligands


def run_prediction(
    tool: str, pred_bin: Path, receptor: Path, ligand: Path, output: Path, device: str = "cuda"
) -> bool:
    """Run bapred or rmsdpred."""
    cmd = [str(pred_bin), "-r", str(receptor), "-l", str(ligand), "-o", str(output), "--device", device]
    print(f"  {tool}: {receptor.name} × {ligand.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"    → {output}")
            return True
        else:
            print(f"    FAILED: {result.stderr[-200:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BA-Pred and RMSD-Pred on pipeline results.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Pipeline run directory.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    analysis_dir = args.run_dir / "outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pred_venv = Path(__file__).resolve().parent.parent / ".venvs" / "pred" / "bin"
    bapred_bin = pred_venv / "bapred"
    rmsdpred_bin = pred_venv / "rmsdpred"

    if not bapred_bin.exists():
        print(f"ERROR: bapred not found at {bapred_bin}")
        return 1

    # Find receptor PDB (use docking prep receptor as primary)
    receptors = find_receptor_pdbs(args.run_dir)
    ligands = find_ligand_files(args.run_dir)

    if not receptors:
        print("ERROR: No receptor PDB files found.")
        return 1
    if not ligands:
        print("ERROR: No ligand files found.")
        return 1

    print(f"Receptors: {list(receptors.keys())}")
    print(f"Ligands: {list(ligands.keys())}")

    # Use docking_prep receptor as primary (protonated, best model)
    primary_receptor = receptors.get("docking_prep") or next(iter(receptors.values()))
    print(f"Primary receptor: {primary_receptor}")

    # Run BA-Pred on each docking result
    print("\n=== BA-Pred (Binding Affinity) ===")
    ba_results = {}
    for lig_name, lig_path in ligands.items():
        out = analysis_dir / f"ba_pred_{lig_name}.tsv"
        ok = run_prediction("BA-Pred", bapred_bin, primary_receptor, lig_path, out, args.device)
        if ok and out.exists():
            ba_results[lig_name] = out.read_text().strip()

    # Run RMSD-Pred on each docking result
    print("\n=== RMSD-Pred (Pose RMSD) ===")
    rmsd_results = {}
    for lig_name, lig_path in ligands.items():
        out = analysis_dir / f"rmsd_pred_{lig_name}.tsv"
        ok = run_prediction("RMSD-Pred", rmsdpred_bin, primary_receptor, lig_path, out, args.device)
        if ok and out.exists():
            rmsd_results[lig_name] = out.read_text().strip()

    # Write summary
    summary = {
        "receptor": str(primary_receptor),
        "ligands": {k: str(v) for k, v in ligands.items()},
        "ba_pred": {k: str(analysis_dir / f"ba_pred_{k}.tsv") for k in ba_results},
        "rmsd_pred": {k: str(analysis_dir / f"rmsd_pred_{k}.tsv") for k in rmsd_results},
    }
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
