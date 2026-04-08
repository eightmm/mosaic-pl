#!/usr/bin/env python3
"""Prepare docking inputs from unified YAML and cofolding outputs.

Converts:
  - Ligand SMILES → 3D SDF → PDBQT (for Vina)
  - Cofolding output CIF → receptor PDB → receptor PDBQT (for Vina)

Protenix-Dock needs receptor PDB + ligand SDF.
Vina needs receptor PDBQT + ligand PDBQT.

Usage:
    python prepare_docking_inputs.py \
        --input-yaml runs/target/inputs/boltz_input.yaml \
        --cofolding-dir runs/target/outputs/boltz \
        --output-dir runs/target/inputs/docking \
        --model boltz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def smiles_to_sdf(smiles: str, output_path: Path, name: str = "ligand") -> Path:
    """Convert SMILES to 3D SDF using RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", name)

    # Generate 3D conformer
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        # Fallback: use random coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    print(f"  Ligand SDF: {output_path}")
    return output_path


def sdf_to_pdbqt(sdf_path: Path, output_path: Path) -> Path:
    """Convert SDF to PDBQT using meeko."""
    from meeko import MoleculePreparation
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Failed to read SDF: {sdf_path}")

    preparator = MoleculePreparation()
    preparator.prepare(mol)
    if not preparator.is_ok:
        raise ValueError(f"meeko failed to prepare {sdf_path}: {preparator.log}")

    pdbqt_string = preparator.write_pdbqt_string()
    output_path.write_text(pdbqt_string)
    print(f"  Ligand PDBQT: {output_path}")
    return output_path


def cif_to_pdb(cif_path: Path, output_path: Path) -> Path:
    """Convert mmCIF to PDB using gemmi."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    structure.remove_ligands_and_waters()
    structure.write_pdb(str(output_path))
    print(f"  Receptor PDB: {output_path}")
    return output_path


def pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> Path:
    """Convert PDB to PDBQT for receptor: pdb2pqr (protonation + charges) → PDBQT format."""
    import subprocess
    import tempfile

    # Step 1: pdb2pqr adds hydrogens and assigns charges
    pqr_path = output_path.with_suffix(".pqr")
    try:
        subprocess.run(
            ["pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try python module
        subprocess.run(
            [sys.executable, "-m", "pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )

    # Step 2: PQR → PDBQT (PQR has charges in the occupancy/bfactor columns)
    AD_TYPE_MAP = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
        "F": "F", "P": "P", "CL": "Cl", "BR": "Br", "I": "I",
    }
    pqr_lines = pqr_path.read_text().splitlines()
    pdbqt_lines = []
    for line in pqr_lines:
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith(("TER", "END")):
                pdbqt_lines.append(line)
            continue
        # PQR format: cols 55-62 = charge, 63-70 = radius
        atom_name = line[12:16].strip()
        element = atom_name.lstrip("0123456789")[0:2].upper().strip()
        if len(element) > 1 and element not in ("CL", "BR"):
            element = element[0]
        ad_type = AD_TYPE_MAP.get(element, element)

        # Extract charge from PQR (columns vary, parse from end)
        parts = line.split()
        try:
            charge = float(parts[-2])  # second to last = charge
        except (ValueError, IndexError):
            charge = 0.0

        # Build PDBQT line: first 54 chars from PQR + reformatted tail
        pdb_prefix = f"{line[:54]:<54s}"
        pdbqt_lines.append(f"{pdb_prefix}  0.00  0.00    {charge:+.3f} {ad_type:<2s}")

    output_path.write_text("\n".join(pdbqt_lines) + "\n")
    pqr_path.unlink(missing_ok=True)
    print(f"  Receptor PDBQT: {output_path} (protonated, Gasteiger charges)")
    return output_path


def find_best_cofolding_structure(cofolding_dir: Path, model: str) -> Path | None:
    """Find the best-ranked structure from cofolding output."""
    if model == "boltz":
        # Boltz: boltz_results_*/predictions/*/*.cif
        for cif in sorted(cofolding_dir.rglob("predictions/**/*.cif")):
            return cif
    elif model == "protenix":
        # Protenix: **/*.cif ranked by name
        for cif in sorted(cofolding_dir.rglob("*.cif")):
            return cif
    elif model == "alphafold3":
        # AF3: **/fold_*_model.cif
        for cif in sorted(cofolding_dir.rglob("*model*.cif")):
            return cif

    # Generic fallback
    for cif in sorted(cofolding_dir.rglob("*.cif")):
        return cif
    for pdb in sorted(cofolding_dir.rglob("*.pdb")):
        return pdb
    return None


def extract_smiles_from_yaml(input_yaml: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, smiles) pairs from unified input YAML."""
    # Simple YAML parsing to avoid heavy dependency
    text = input_yaml.read_text()
    results = []
    current_id = None
    in_ligand = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ligand:") or stripped == "ligand:":
            in_ligand = True
            current_id = None
            continue
        if in_ligand:
            if stripped.startswith("id:"):
                current_id = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("smiles:"):
                smiles = stripped.split(":", 1)[1].strip()
                results.append((current_id or "L", smiles))
                in_ligand = False
            elif stripped.startswith("- ") or (stripped and not stripped.startswith((" ", "#"))):
                in_ligand = False

    return results


def extract_smiles_from_json(input_json: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, smiles) from Protenix/AF3 JSON input."""
    data = json.loads(input_json.read_text())
    if isinstance(data, list):
        data = data[0]

    results = []
    for seq in data.get("sequences", []):
        if "ligand" in seq:
            lig = seq["ligand"]
            smiles = lig.get("smiles") or lig.get("ligand", "")
            lig_id = lig.get("id", "L")
            if isinstance(lig_id, list):
                lig_id = lig_id[0]
            if smiles:
                results.append((str(lig_id), smiles))
    return results


def compute_box_from_ligand(sdf_path: Path, padding: float = 10.0) -> tuple[list[float], list[float]]:
    """Compute docking box center and size from ligand 3D coordinates."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(supplier)
    if mol is None:
        return [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    min_xyz = positions.min(axis=0)
    max_xyz = positions.max(axis=0)
    center = ((min_xyz + max_xyz) / 2).tolist()
    size = (max_xyz - min_xyz + padding).tolist()

    return center, size


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare docking inputs from unified input + cofolding output.")
    parser.add_argument("--input-yaml", type=Path, help="Unified input YAML (Boltz format).")
    parser.add_argument("--input-json", type=Path, help="Protenix/AF3 input JSON (alternative to YAML).")
    parser.add_argument("--cofolding-dir", type=Path, required=True, help="Cofolding output directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for docking inputs.")
    parser.add_argument("--model", type=str, default="boltz", choices=["boltz", "protenix", "alphafold3"],
                        help="Which cofolding model produced the structure.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract SMILES
    if args.input_yaml and args.input_yaml.exists():
        ligands = extract_smiles_from_yaml(args.input_yaml)
    elif args.input_json and args.input_json.exists():
        ligands = extract_smiles_from_json(args.input_json)
    else:
        print("  ERROR: No input file found.")
        return 1

    if not ligands:
        print("  No ligands with SMILES found, skipping ligand preparation.")
        return 0

    # 2. Convert SMILES → SDF → PDBQT for each ligand
    for lig_id, smiles in ligands:
        print(f"  Preparing ligand {lig_id}: {smiles}")
        sdf_path = smiles_to_sdf(smiles, args.output_dir / f"ligand_{lig_id}.sdf", name=lig_id)
        sdf_to_pdbqt(sdf_path, args.output_dir / f"ligand_{lig_id}.pdbqt")

    # 3. Find best cofolding structure → receptor PDB → PDBQT
    structure = find_best_cofolding_structure(args.cofolding_dir, args.model)
    if structure is None:
        print(f"  WARNING: No cofolding structure found in {args.cofolding_dir}")
        return 0

    print(f"  Using cofolding structure: {structure}")
    if structure.suffix in (".cif", ".mmcif"):
        pdb_path = cif_to_pdb(structure, args.output_dir / "receptor.pdb")
    else:
        pdb_path = structure  # Already PDB

    pdb_to_pdbqt(pdb_path, args.output_dir / "receptor.pdbqt")

    # 4. Compute docking box from ligand
    first_sdf = args.output_dir / f"ligand_{ligands[0][0]}.sdf"
    center, size = compute_box_from_ligand(first_sdf)

    # 5. Write summary JSON
    summary = {
        "receptor_pdb": str(args.output_dir / "receptor.pdb"),
        "receptor_pdbqt": str(args.output_dir / "receptor.pdbqt"),
        "ligands": [
            {
                "id": lig_id,
                "smiles": smiles,
                "sdf": str(args.output_dir / f"ligand_{lig_id}.sdf"),
                "pdbqt": str(args.output_dir / f"ligand_{lig_id}.pdbqt"),
            }
            for lig_id, smiles in ligands
        ],
        "box_center": center,
        "box_size": size,
        "cofolding_structure": str(structure),
        "cofolding_model": args.model,
    }
    summary_path = args.output_dir / "docking_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  Summary: {summary_path}")
    print(f"  Box center: {center}")
    print(f"  Box size: {size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
