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


def run_pdb2pqr(pdb_path: Path, pqr_path: Path) -> None:
    """Run pdb2pqr to add hydrogens and assign AMBER charges."""
    import subprocess
    try:
        subprocess.run(
            ["pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(
            [sys.executable, "-m", "pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )


def pqr_to_protonated_pdb(pqr_path: Path, output_path: Path) -> Path:
    """Convert PQR back to PDB format (keeps protonation, HIS→HID/HIE/HIP)."""
    lines = pqr_path.read_text().splitlines()
    pdb_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:1].upper()
            # PDB format: cols 1-54 (coords), 55-60 (occ), 61-66 (bfactor), 77-78 (element)
            pdb_lines.append(f"{line[:54]:<54s}  1.00  0.00          {element:>2s}  ")
        elif line.startswith(("TER", "END")):
            pdb_lines.append(line)
    output_path.write_text("\n".join(pdb_lines) + "\n")
    print(f"  Receptor PDB (protonated): {output_path}")
    return output_path


def pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> Path:
    """Convert PDB to PDBQT for receptor: pdb2pqr (protonation + charges) → PDBQT format."""

    # Step 1: pdb2pqr adds hydrogens and assigns charges
    pqr_path = output_path.with_suffix(".pqr")
    run_pdb2pqr(pdb_path, pqr_path)

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
    print(f"  Receptor PDBQT: {output_path} (protonated, Gasteiger charges)")
    return output_path


def find_best_cofolding_structure(cofolding_dir: Path, model: str) -> Path | None:
    """Find the best-ranked structure from cofolding output.

    Prefers ``_aligned.cif`` (produced by ``align_cofolding_outputs.py``) over
    the raw CIF so that downstream stages work in the common reference frame.
    """
    def _prefer_aligned(cifs: list[Path]) -> Path | None:
        # First pass: return the first cif that has a sibling ``_aligned``.
        # Second pass: fall back to the first raw cif.
        for cif in cifs:
            aligned = cif.with_name(cif.stem + "_aligned.cif")
            if aligned.exists():
                return aligned
        return cifs[0] if cifs else None

    if model.startswith("boltz"):
        return _prefer_aligned(sorted(cofolding_dir.rglob("predictions/**/*.cif")))
    elif model == "protenix":
        raw = [c for c in sorted(cofolding_dir.rglob("*.cif")) if "_aligned" not in c.name]
        return _prefer_aligned(raw)
    elif model == "alphafold3":
        raw = [c for c in sorted(cofolding_dir.rglob("*model*.cif")) if "_aligned" not in c.name]
        return _prefer_aligned(raw)

    raw = [c for c in sorted(cofolding_dir.rglob("*.cif")) if "_aligned" not in c.name]
    result = _prefer_aligned(raw)
    if result:
        return result
    for pdb in sorted(cofolding_dir.rglob("*.pdb")):
        return pdb
    return None


def read_confidence_score(cofolding_dir: Path, model: str) -> float:
    """Read mean per-residue/atom pLDDT, normalised to the ``[0, 100]`` scale.

    Each cofolding backend stores pLDDT differently:
      * **Boltz-2 / Boltz-2x** write a per-residue array in ``plddt_*.npz`` on
        the ``[0, 1]`` scale. We multiply by 100 so it is comparable with AF3.
      * **Protenix v2** writes a scalar mean pLDDT already on ``[0, 100]``
        inside the per-sample ``confidences.json`` (``"plddt": <float>``).
        The earlier version of this function called ``sum(float)`` on that
        scalar and fell through to the exception handler, returning ``-1`` —
        the bug that made Protenix invisible to the auto-selector.
      * **AlphaFold3** writes a ``"atom_plddts"`` list already on ``[0, 100]``.
    """
    try:
        if model.startswith("boltz"):
            import numpy as np
            for npz in cofolding_dir.rglob("plddt_*model_0.npz"):
                data = np.load(str(npz))
                return float(data[data.files[0]].mean()) * 100.0
        elif model == "protenix":
            for json_f in cofolding_dir.rglob("*confidence*.json"):
                data = json.loads(json_f.read_text())
                if "plddt" in data:
                    val = data["plddt"]
                    if isinstance(val, (list, tuple)) and val:
                        return float(sum(val) / len(val))
                    if isinstance(val, (int, float)):
                        return float(val)
        elif model == "alphafold3":
            for json_f in cofolding_dir.rglob("*confidence*.json"):
                if "summary" in json_f.name:
                    continue
                data = json.loads(json_f.read_text())
                if "atom_plddts" in data and data["atom_plddts"]:
                    return float(sum(data["atom_plddts"]) / len(data["atom_plddts"]))
    except Exception:
        pass
    return -1.0


def select_best_model(output_root: Path) -> tuple[str, Path]:
    """Auto-select best cofolding model by confidence score."""
    candidates = []
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = output_root / "outputs" / model
        if not model_dir.exists():
            continue
        structure = find_best_cofolding_structure(model_dir, model)
        if structure is None:
            continue
        score = read_confidence_score(model_dir, model)
        candidates.append((model, model_dir, structure, score))
        print(f"  Model {model}: pLDDT={score:.1f}, structure={structure.name}")

    if not candidates:
        return ("", Path())

    # Sort by confidence score descending, pick best
    candidates.sort(key=lambda x: x[3], reverse=True)
    best = candidates[0]
    print(f"  Selected: {best[0]} (pLDDT={best[3]:.1f})")
    return (best[0], best[1])


def run_p2rank(pdb_path: Path, output_dir: Path) -> tuple[list[float], list[float]] | None:
    """Run P2Rank binding site prediction and return (center, size) of top pocket."""
    import subprocess
    import csv

    prank_bin = Path(__file__).resolve().parent.parent / ".local" / "bin" / "prank"
    if not prank_bin.exists():
        print("  P2Rank not found, skipping binding site prediction.")
        return None

    p2rank_out = output_dir / "p2rank"
    try:
        subprocess.run(
            [str(prank_bin), "predict", "-f", str(pdb_path), "-o", str(p2rank_out)],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  P2Rank failed: {e}")
        return None

    # Parse predictions CSV
    pred_file = p2rank_out / f"{pdb_path.name}_predictions.csv"
    if not pred_file.exists():
        print("  P2Rank produced no predictions file.")
        return None

    with open(pred_file) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # Normalize keys (P2Rank pads headers with spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}
            # First row = top-ranked pocket
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            cz = float(row["center_z"])
            box_side = 22.5
            print(f"  P2Rank pocket 1: center=[{cx:.1f}, {cy:.1f}, {cz:.1f}], score={row['score'].strip()}")
            return ([cx, cy, cz], [box_side, box_side, box_side])

    print("  P2Rank found no pockets.")
    return None


def run_swinsite(pdb_path: Path, output_dir: Path) -> tuple[list[float], list[float]] | None:
    """Run SwinSite binding site prediction and return (center, size) of top pocket."""
    repo_root = Path(__file__).resolve().parent.parent
    swinsite_dir = repo_root / "external" / "swinsite"
    pred_python = repo_root / ".venvs" / "pred" / "bin" / "python"

    if not swinsite_dir.exists() or not pred_python.exists():
        print("  SwinSite not found, skipping.")
        return None

    swinsite_out = output_dir / "swinsite"
    # SwinSite expects input_dir/<sample_name>/protein.pdb
    input_dir = swinsite_out / "input"
    sample_dir = input_dir / "receptor"
    sample_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(pdb_path), str(sample_dir / "protein.pdb"))

    try:
        import subprocess as _sp
        _sp.run(
            [str(pred_python), str(swinsite_dir / "predict.py"),
             "-i", str(input_dir), "-f", "pdb", "-of", "pdb",
             "-o", str(swinsite_out / "results"),
             "-l", str(swinsite_out / "log.txt"),
             "-m",
             str(swinsite_dir / "model/fold_1/best_epoch.h5"),
             str(swinsite_dir / "model/fold_2/best_epoch.h5"),
             str(swinsite_dir / "model/fold_3/best_epoch.h5"),
             str(swinsite_dir / "model/fold_4/best_epoch.h5"),
            ],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        print(f"  SwinSite failed: {e}")
        return None

    # Parse pocket PDB files to find center
    results_dir = swinsite_out / "results" / "input" / "receptor"
    if not results_dir.exists():
        print("  SwinSite produced no output.")
        return None

    for pocket_pdb in sorted(results_dir.glob("pocket_*.pdb")):
        # Read coordinates and compute centroid
        coords = []
        for line in pocket_pdb.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
        if not coords:
            continue
        import numpy as _np
        arr = _np.array(coords)
        center = arr.mean(axis=0).tolist()
        box_side = 22.5
        print(f"  SwinSite pocket 1: center=[{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}], atoms={len(coords)}")
        return (center, [box_side, box_side, box_side])

    print("  SwinSite found no pockets.")
    return None


def extract_smiles_from_yaml(input_yaml: Path) -> list[tuple[str, str]]:
    """Extract ``(ligand_id, smiles)`` pairs from the unified input YAML.

    Uses ``yaml.safe_load`` so block-style nested mappings parse correctly
    and multi-ligand inputs (candidate + cofactors + metals) preserve their
    declared order. Ligand entries that carry only a ``ccd`` field
    (metals/ions addressed by CCD code) are skipped — docking can only
    consume real SMILES.
    """
    try:
        import yaml
    except Exception:
        return []
    try:
        data = yaml.safe_load(input_yaml.read_text()) or {}
    except Exception:
        return []
    results: list[tuple[str, str]] = []
    for entry in data.get("sequences", []) or []:
        if not isinstance(entry, dict) or "ligand" not in entry:
            continue
        lig = entry["ligand"] or {}
        smi = lig.get("smiles")
        if not smi:
            continue  # CCD-only ligand (metal/ion) — not dockable
        lid = str(lig.get("id") or "L").strip()
        results.append((lid, str(smi).strip().strip("'\"")))
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


def _extract_cofolding_ligand_centroid(cif_path: Path) -> list[float] | None:
    """Extract the centroid of ligand (non-polymer) heavy atoms from a cofolding CIF.

    Boltz/Protenix/AF3 cofolding outputs contain both protein and ligand atoms.
    Non-polymer entities (ligand) are identified by ``entity_type == NonPolymer``
    or residue names starting with ``LIG``. Returns ``[x, y, z]`` or ``None``
    if no ligand atoms are found.
    """
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        coords: list[tuple[float, float, float]] = []
        for model in st:
            for chain in model:
                for res in chain:
                    is_ligand = (
                        res.entity_type == gemmi.EntityType.NonPolymer
                        or res.name.startswith("LIG")
                    )
                    if not is_ligand:
                        continue
                    for atom in res:
                        if atom.element.is_hydrogen:
                            continue
                        coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
        if not coords:
            return None
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        return [round(cx, 4), round(cy, 4), round(cz, 4)]
    except Exception as e:
        print(f"  WARNING: could not extract cofolding ligand centroid: {e}")
        return None


def compute_box_from_ligand(sdf_path: Path, box_side: float = 22.5) -> tuple[list[float], list[float]]:
    """Compute docking box center from ligand 3D coordinates. Box size fixed."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(supplier)
    if mol is None:
        return [0.0, 0.0, 0.0], [box_side, box_side, box_side]

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    min_xyz = positions.min(axis=0)
    max_xyz = positions.max(axis=0)
    center = ((min_xyz + max_xyz) / 2).tolist()

    return center, [box_side, box_side, box_side]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare docking inputs from unified input + cofolding output.")
    parser.add_argument("--input-yaml", type=Path, help="Unified input YAML (Boltz format).")
    parser.add_argument("--input-json", type=Path, help="Protenix/AF3 input JSON (alternative to YAML).")
    parser.add_argument("--cofolding-dir", type=Path, help="Cofolding output directory (single model).")
    parser.add_argument("--run-dir", type=Path, help="Run directory (auto-selects best model from outputs/).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for docking inputs.")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "boltz", "protenix", "alphafold3"],
                        help="Which cofolding model to use. 'auto' selects by confidence score.")
    parser.add_argument("--use-p2rank", action="store_true", default=True,
                        help="Use P2Rank for binding site prediction (default: true).")
    parser.add_argument("--no-p2rank", action="store_true", help="Disable P2Rank binding site prediction.")
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

    # 2. Convert SMILES → SDF → PDBQT for ALL ligands
    for lig_id, smiles in ligands:
        print(f"  Preparing ligand {lig_id}: {smiles}")
        sdf_path = smiles_to_sdf(smiles, args.output_dir / f"ligand_{lig_id}.sdf", name=lig_id)
        sdf_to_pdbqt(sdf_path, args.output_dir / f"ligand_{lig_id}.pdbqt")

    # 3. Select best cofolding structure
    cofolding_dir = args.cofolding_dir
    model = args.model

    if model == "auto" and args.run_dir:
        # Auto-select by comparing confidence scores across all models
        print("  Auto-selecting best cofolding model...")
        model, cofolding_dir = select_best_model(args.run_dir)
        if not model:
            print("  ERROR: No cofolding outputs found for auto-selection.")
            return 1
    elif model == "auto" and cofolding_dir:
        # Single dir provided, guess model from path
        dirname = cofolding_dir.name
        model = dirname if dirname in ("boltz", "protenix", "alphafold3") else "boltz"

    structure = find_best_cofolding_structure(cofolding_dir, model)
    if structure is None:
        print(f"  WARNING: No cofolding structure found in {cofolding_dir}")
        return 0

    print(f"  Using cofolding structure: {structure}")
    if structure.suffix in (".cif", ".mmcif"):
        pdb_path = cif_to_pdb(structure, args.output_dir / "receptor.pdb")
    else:
        pdb_path = structure

    pdb_to_pdbqt(pdb_path, args.output_dir / "receptor.pdbqt")

    # Also generate protonated PDB for Protenix-Dock (HIS→HID/HIE/HIP)
    pqr_path = args.output_dir / "receptor.pqr"
    if not pqr_path.exists():
        run_pdb2pqr(pdb_path, pqr_path)
    pqr_to_protonated_pdb(pqr_path, args.output_dir / "receptor_protonated.pdb")
    pqr_path.unlink(missing_ok=True)

    # 4. Determine docking box
    # Priority: cofolding ligand > SwinSite > P2Rank > input ligand coords
    center, size = None, None
    box_method = "fallback"
    binding_site_results = {}

    # 4a. Cofolding predicted ligand centroid (most direct signal — the
    #     cofolding model already placed the ligand in what it thinks is the
    #     binding pocket). Extract from the selected CIF.
    cofold_center = _extract_cofolding_ligand_centroid(structure)
    if cofold_center is not None:
        default_size = [22.5, 22.5, 22.5]
        binding_site_results["cofolding"] = (cofold_center, default_size)
        print(f"  Cofolding ligand centroid: {cofold_center}")

    # 4b. SwinSite (ML-based surface pocket predictor, needs GPU)
    print("  Running SwinSite binding site prediction...")
    swinsite_result = run_swinsite(pdb_path, args.output_dir)
    if swinsite_result:
        binding_site_results["swinsite"] = swinsite_result

    # 4c. P2Rank (surface geometry-based)
    if not args.no_p2rank:
        print("  Running P2Rank binding site prediction...")
        p2rank_result = run_p2rank(pdb_path, args.output_dir)
        if p2rank_result:
            binding_site_results["p2rank"] = p2rank_result

    # Pick best by priority
    for method in ("cofolding", "swinsite", "p2rank"):
        if method in binding_site_results:
            center, size = binding_site_results[method]
            box_method = method
            break

    if center is None:
        first_sdf = args.output_dir / f"ligand_{ligands[0][0]}.sdf"
        center, size = compute_box_from_ligand(first_sdf)
        box_method = "ligand_coordinates"

    # 5. Write summary JSON
    summary = {
        "receptor_pdb": str(args.output_dir / "receptor_protonated.pdb"),
        "receptor_pdb_raw": str(args.output_dir / "receptor.pdb"),
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
        "box_method": box_method,
        "binding_site_predictions": {
            k: {"center": v[0], "size": v[1]} for k, v in binding_site_results.items()
        },
        "cofolding_structure": str(structure),
        "cofolding_model": model,
    }
    summary_path = args.output_dir / "docking_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  Summary: {summary_path}")
    print(f"  Box center: {center} (method: {box_method})")
    print(f"  Box size: {size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
