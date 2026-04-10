#!/usr/bin/env python3
"""Prepare docking inputs from template search hits.

Given filtered template hits (with ligand info), extracts the best
template structure, uses the template ligand position for docking box,
and prepares receptor/ligand files for docking.

Usage:
    python prepare_template_docking.py \
        --hits-tsv outputs/template_search_sequence/filtered_hits.tsv \
        --rcsb-dir ~/DB/RCSB/raw/mmCIF_data \
        --input-yaml inputs/boltz_input.yaml \
        --output-dir inputs/template_docking \
        --rcsb-db ~/DB/RCSB/processed/rcsb_index.db
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_template_cif(pdb_id: str, rcsb_dir: Path) -> Path | None:
    """Find CIF file for a PDB ID in the RCSB raw data directory."""
    pdb_id = pdb_id.lower()
    hash_dir = pdb_id[1:3]  # middle two characters
    cif_gz = rcsb_dir / hash_dir / f"{pdb_id}.cif.gz"
    if cif_gz.exists():
        return cif_gz
    cif = rcsb_dir / hash_dir / f"{pdb_id}.cif"
    if cif.exists():
        return cif
    return None


def extract_cif(cif_path: Path, output_dir: Path) -> Path:
    """Extract gzipped CIF to output directory."""
    output = output_dir / cif_path.name.replace(".gz", "")
    if cif_path.suffix == ".gz":
        import gzip
        with gzip.open(cif_path, "rb") as f_in, open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(cif_path, output)
    return output


def extract_ligand_center(cif_path: Path, ligand_ccd: str) -> list[float] | None:
    """Extract ligand center coordinates from CIF using gemmi."""
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == ligand_ccd:
                    for atom in residue:
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        return None
    import numpy as np
    arr = np.array(coords)
    return arr.mean(axis=0).tolist()


def extract_template_ligand_sdf(cif_path: Path, ligand_ccd: str, output_sdf: Path) -> Path | None:
    """Extract template ligand from CIF as SDF with bound-state 3D coordinates.

    Writes the first matching residue as a HETATM PDB block, then converts
    to SDF via RDKit (or openbabel as fallback).  The resulting SDF preserves
    the crystallographic pose, which is needed for lig-align reference.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    target_residue = None
    target_chain = None
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == ligand_ccd:
                    target_residue = residue
                    target_chain = chain
                    break
            if target_residue:
                break
        if target_residue:
            break

    if target_residue is None:
        return None

    # Write ligand atoms as minimal PDB
    ligand_pdb = output_sdf.with_suffix(".pdb")
    with open(ligand_pdb, "w") as f:
        for i, atom in enumerate(target_residue):
            name = f" {atom.name:<3s}" if len(atom.name) < 4 else atom.name
            f.write(
                f"HETATM{i + 1:5d} {name:4s} {target_residue.name:>3s}"
                f" {target_chain.name:>1s}{target_residue.seqid.num:4d}    "
                f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"  1.00  0.00          {atom.element.name:>2s}\n"
            )
        f.write("END\n")

    # Convert PDB → SDF
    try:
        from rdkit import Chem
        mol = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # keep unsanitized — coords are what matter
            writer = Chem.SDWriter(str(output_sdf))
            writer.write(mol)
            writer.close()
            return output_sdf
    except Exception:
        pass

    # Fallback: try openbabel
    try:
        result = subprocess.run(
            ["obabel", str(ligand_pdb), "-O", str(output_sdf)],
            check=True, capture_output=True, text=True,
        )
        if output_sdf.exists():
            return output_sdf
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def cif_to_receptor_pdb(cif_path: Path, output_pdb: Path) -> Path:
    """Extract protein chains from CIF to PDB."""
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    structure.remove_ligands_and_waters()
    structure.write_pdb(str(output_pdb))
    return output_pdb


def prepare_receptor_pdbqt(pdb_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Run pdb2pqr and convert to PDBQT."""
    pqr_path = output_dir / "receptor.pqr"
    pdbqt_path = output_dir / "receptor.pdbqt"
    protonated_pdb = output_dir / "receptor_protonated.pdb"

    # pdb2pqr
    try:
        subprocess.run(
            [sys.executable, "-m", "pdb2pqr", "--ff=AMBER", "--ffout=AMBER",
             "--keep-chain", str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If pdb2pqr not in this venv, just copy PDB as-is
        shutil.copy2(pdb_path, protonated_pdb)
        shutil.copy2(pdb_path, pdbqt_path)
        return protonated_pdb, pdbqt_path

    # PQR → protonated PDB
    pdb_lines = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:1].upper()
            pdb_lines.append(f"{line[:54]:<54s}  1.00  0.00          {element:>2s}  ")
        elif line.startswith(("TER", "END")):
            pdb_lines.append(line)
    protonated_pdb.write_text("\n".join(pdb_lines) + "\n")

    # PQR → PDBQT
    AD_TYPE_MAP = {"C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD"}
    pdbqt_lines = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:2].upper().strip()
            if len(element) > 1 and element not in ("CL", "BR"):
                element = element[0]
            ad_type = AD_TYPE_MAP.get(element, element)
            parts = line.split()
            try:
                charge = float(parts[-2])
            except (ValueError, IndexError):
                charge = 0.0
            pdb_prefix = f"{line[:54]:<54s}"
            pdbqt_lines.append(f"{pdb_prefix}  0.00  0.00    {charge:+.3f} {ad_type:<2s}")
        elif line.startswith(("TER", "END")):
            pdbqt_lines.append(line)
    pdbqt_path.write_text("\n".join(pdbqt_lines) + "\n")
    pqr_path.unlink(missing_ok=True)

    return protonated_pdb, pdbqt_path


def prepare_ligand_files(smiles: str, lig_id: str, output_dir: Path) -> tuple[Path, Path]:
    """SMILES → 3D SDF → PDBQT."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation

    sdf_path = output_dir / f"ligand_{lig_id}.sdf"
    pdbqt_path = output_dir / f"ligand_{lig_id}.pdbqt"

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", lig_id)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    preparator = MoleculePreparation()
    preparator.prepare(mol)
    if preparator.is_ok:
        pdbqt_path.write_text(preparator.write_pdbqt_string())

    return sdf_path, pdbqt_path


def extract_smiles_from_yaml(yaml_path: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, smiles) from unified YAML."""
    results = []
    text = yaml_path.read_text()
    current_id, in_ligand = None, False
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
                smiles = stripped.split(":", 1)[1].strip().strip("'\"")
                results.append((current_id or "L", smiles))
                in_ligand = False
            elif stripped.startswith("- ") or (stripped and not stripped.startswith((" ", "#"))):
                in_ligand = False
    return results


def parse_filtered_hits(tsv_path: Path) -> list[dict]:
    """Parse filtered hits TSV from template_filter."""
    hits = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("num_ligands", 0)) > 0:
                hits.append(row)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare template-based docking inputs.")
    parser.add_argument("--hits-tsv", type=Path, required=True, help="Filtered template hits TSV.")
    parser.add_argument("--rcsb-dir", type=Path, default=Path.home() / "DB/RCSB/raw/mmCIF_data")
    parser.add_argument("--input-yaml", type=Path, help="Unified input YAML (for ligand SMILES).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-templates", type=int, default=3, help="Max templates to prepare.")
    parser.add_argument("--box-size", type=float, default=22.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse hits
    hits = parse_filtered_hits(args.hits_tsv)
    if not hits:
        print("No template hits with ligands found.")
        return 0

    # 2. Extract ligand SMILES from input
    ligands = []
    if args.input_yaml and args.input_yaml.exists():
        ligands = extract_smiles_from_yaml(args.input_yaml)
    if not ligands:
        print("No ligand SMILES found in input.")
        return 1

    print(f"Found {len(hits)} template hits with ligands, using top {args.max_templates}")

    all_templates = []
    for i, hit in enumerate(hits[:args.max_templates]):
        pdb_id = hit["pdb_id"]
        ligand_codes = hit.get("ligand_codes", "").split(";")
        ligand_smiles_list = hit.get("ligand_smiles", "").split(";")

        print(f"\n{'='*60}")
        print(f"  Template {i+1}: {pdb_id} (pident={hit.get('pident', '?')}%)")
        print(f"  Ligands: {', '.join(ligand_codes)}")
        print(f"{'='*60}")

        # Find and extract CIF
        cif_path = find_template_cif(pdb_id, args.rcsb_dir)
        if not cif_path:
            print(f"  CIF not found for {pdb_id}, skipping.")
            continue

        template_dir = args.output_dir / f"template_{pdb_id}"
        template_dir.mkdir(parents=True, exist_ok=True)

        cif = extract_cif(cif_path, template_dir)
        print(f"  Extracted: {cif.name}")

        # Extract ligand center for docking box
        center = None
        template_ligand_ccd = None
        for ccd_code in ligand_codes:
            ccd_code = ccd_code.strip()
            if not ccd_code:
                continue
            center = extract_ligand_center(cif, ccd_code)
            if center:
                template_ligand_ccd = ccd_code
                break

        if not center:
            print(f"  Could not extract ligand center, skipping.")
            continue

        box_size = [args.box_size] * 3
        print(f"  Ligand {template_ligand_ccd} center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")

        # Extract template ligand SDF (bound pose for lig-align)
        template_ligand_sdf = None
        template_lig_sdf_path = template_dir / f"template_ligand_{template_ligand_ccd}.sdf"
        extracted = extract_template_ligand_sdf(cif, template_ligand_ccd, template_lig_sdf_path)
        if extracted:
            template_ligand_sdf = str(extracted)
            print(f"  Template ligand SDF: {extracted.name}")
        else:
            print(f"  WARNING: Could not extract template ligand SDF for {template_ligand_ccd}")

        # Prepare receptor
        receptor_pdb = cif_to_receptor_pdb(cif, template_dir / "receptor.pdb")
        protonated_pdb, receptor_pdbqt = prepare_receptor_pdbqt(receptor_pdb, template_dir)
        print(f"  Receptor PDB: {receptor_pdb.name}")
        print(f"  Receptor PDBQT: {receptor_pdbqt.name}")

        # Prepare target ligand
        for lig_id, smiles in ligands:
            sdf_path, pdbqt_path = prepare_ligand_files(smiles, lig_id, template_dir)
            print(f"  Ligand {lig_id}: {sdf_path.name}, {pdbqt_path.name}")

        # Write docking prep summary (same format as cofolding-based prep)
        summary = {
            "receptor_pdb": str(protonated_pdb),
            "receptor_pdb_raw": str(receptor_pdb),
            "receptor_pdbqt": str(receptor_pdbqt),
            "ligands": [
                {
                    "id": lig_id,
                    "smiles": smiles,
                    "sdf": str(template_dir / f"ligand_{lig_id}.sdf"),
                    "pdbqt": str(template_dir / f"ligand_{lig_id}.pdbqt"),
                }
                for lig_id, smiles in ligands
            ],
            "box_center": center,
            "box_size": box_size,
            "box_method": "template_ligand",
            "template_pdb_id": pdb_id,
            "template_ligand_ccd": template_ligand_ccd,
            "template_ligand_sdf": template_ligand_sdf,
            "template_pident": float(hit.get("pident", 0)),
            "best_mcs_coverage": float(hit.get("best_mcs_coverage", 0)),
            "best_tanimoto": float(hit.get("best_tanimoto", 0)),
        }
        summary_path = template_dir / "docking_prep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  Summary: {summary_path}")

        all_templates.append(summary)

    # Write master summary
    master_summary = args.output_dir / "template_docking_summary.json"
    master_summary.write_text(json.dumps(all_templates, indent=2) + "\n")
    print(f"\n{len(all_templates)} templates prepared → {master_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
