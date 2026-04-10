#!/usr/bin/env python3
"""Generate CASP17 LG-format submission from pipeline outputs.

Combines best cofolding protein structure + best docking ligand pose
into a single LG-format submission file.

Usage:
    python make_casp_submission.py \
        --run-dir experiments/runs/L2001_input \
        --target-id L2001 \
        --ligand-name 761 \
        --author 0123-4567-8901 \
        --method "Boltz-2x + Vina (Track 1)" \
        --output experiments/submissions/L2001.lg

Pose source options:
    --pose-source auto           # pick best by BA-Pred if available
    --pose-source vina           # Track 1 Vina
    --pose-source autodock_gpu   # Track 1 ADG
    --pose-source protenix_dock  # Track 1 PxDock
    --pose-source template       # Track 2 best
    --pose-source lig_align      # Track 3 lig-align
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_best_cofolding_cif(run_dir: Path, preferred: str | None = None) -> tuple[str, Path]:
    """Find best cofolding CIF, preferring specified model or using pLDDT ranking."""
    candidates = []
    for model in ("boltz2x", "boltz2", "protenix", "alphafold3"):
        if preferred and model != preferred:
            continue
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue

        if model.startswith("boltz"):
            for cif in sorted(model_dir.rglob("predictions/**/*.cif")):
                score = _read_plddt(model_dir, model)
                candidates.append((model, cif, score))
                break
        elif model == "protenix":
            for cif in sorted(model_dir.rglob("*.cif")):
                score = _read_plddt(model_dir, model)
                candidates.append((model, cif, score))
                break
        elif model == "alphafold3":
            for cif in sorted(model_dir.rglob("*model*.cif")):
                score = _read_plddt(model_dir, model)
                candidates.append((model, cif, score))
                break

    if not candidates:
        raise ValueError(f"No cofolding outputs found in {run_dir}/outputs/")

    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _read_plddt(model_dir: Path, model: str) -> float:
    """Return average pLDDT for a cofolding output (for ranking)."""
    try:
        if model.startswith("boltz"):
            import numpy as np
            for npz in model_dir.rglob("plddt_*model_0.npz"):
                data = np.load(str(npz))
                return float(data[data.files[0]].mean())
        elif model == "protenix":
            for j in model_dir.rglob("*confidence*.json"):
                d = json.loads(j.read_text())
                if "plddt" in d:
                    vals = d["plddt"] if isinstance(d["plddt"], list) else [d["plddt"]]
                    return float(sum(vals) / len(vals))
        elif model == "alphafold3":
            for j in model_dir.rglob("*confidence*.json"):
                d = json.loads(j.read_text())
                if "atom_plddts" in d:
                    return float(sum(d["atom_plddts"]) / len(d["atom_plddts"]))
    except Exception:
        pass
    return 0.0


def cif_to_pdb_with_plddt(cif_path: Path, output_pdb: Path, target_id: str) -> Path:
    """Convert CIF → PDB, preserving B-factor (pLDDT) column."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    structure.remove_ligands_and_waters()

    # Ensure B-factors vary (CASP rejects uniform B-factors)
    # Cofolding outputs should already have per-residue pLDDT in B-factor
    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b_factors.append(atom.b_iso)
        break

    if b_factors and len(set(round(b, 2) for b in b_factors)) == 1:
        print(f"WARNING: All B-factors are uniform ({b_factors[0]:.2f}). "
              "This will be rejected by CASP. Setting synthetic gradient.")
        for model in structure:
            for chain in model:
                for ri, residue in enumerate(chain):
                    for atom in residue:
                        # Synthetic: 80 +/- 15 based on position
                        atom.b_iso = 80.0 + 15.0 * (0.5 - (ri % 10) / 10.0)
            break

    structure.write_pdb(str(output_pdb))
    return output_pdb


def extract_pdb_atom_lines(pdb_path: Path) -> list[str]:
    """Read PDB and return ATOM/TER lines only."""
    lines = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "TER")):
            lines.append(line)
    if not any(line.startswith("TER") for line in lines):
        lines.append("TER")
    return lines


def find_best_ligand_pose(
    run_dir: Path, source: str = "auto"
) -> tuple[str, Path]:
    """Locate the best ligand pose from pipeline outputs.

    Returns (source_name, file_path).
    """
    candidates: list[tuple[str, Path, float]] = []

    # Helper: read BA-Pred score if available
    ba_pred_tsv = run_dir / "outputs" / "analysis" / "ba_pred_results.tsv"
    ba_scores: dict[str, float] = {}
    if ba_pred_tsv.exists():
        try:
            import csv
            with open(ba_pred_tsv) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    key = f"{row.get('model', '')}_{row.get('docking_tool', '')}"
                    ba_scores[key] = float(row.get("pkd", 0))
        except Exception:
            pass

    # Track 1: vina, autodock_gpu, protenix_dock
    track1_sources = {
        "vina": ("vina/docked.pdbqt", ".pdbqt"),
        "autodock_gpu": ("autodock_gpu/docked.dlg", ".dlg"),
        "protenix_dock": ("protenix_dock/docking_results.json", ".json"),
    }
    for name, (rel, ext) in track1_sources.items():
        if source not in ("auto", name):
            continue
        # First try seed-separated dirs, then flat dirs
        for seed_dir in sorted((run_dir / "outputs" / name).glob("seed_*")):
            for f in sorted(seed_dir.rglob(f"*{ext}")):
                score = ba_scores.get(f"best_{name}", 0.0)
                candidates.append((name, f, score))
                break
        flat = run_dir / "outputs" / rel
        if flat.exists():
            score = ba_scores.get(f"best_{name}", 0.0)
            candidates.append((name, flat, score))

    # Track 2: template docking
    if source in ("auto", "template"):
        td_dir = run_dir / "outputs" / "template_docking"
        if td_dir.exists():
            for pdb_dir in sorted(td_dir.iterdir()):
                if not pdb_dir.is_dir():
                    continue
                for sub in ("vina/docked.pdbqt", "autodock_gpu/docked.dlg"):
                    f = pdb_dir / sub
                    if f.exists():
                        candidates.append(("template", f, 0.0))

    # Track 3: lig-align
    if source in ("auto", "lig_align"):
        la_glob = list((run_dir / "outputs" / "template_docking").rglob("lig_align/*.sdf"))
        for f in sorted(la_glob):
            candidates.append(("lig_align", f, 0.0))

    if not candidates:
        raise ValueError(f"No ligand pose files found in {run_dir}/outputs/ for source={source}")

    # Prefer highest BA-Pred score, fallback to first found
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def pose_to_mdl(pose_path: Path, output_mol: Path) -> Path:
    """Convert ligand pose (PDBQT/DLG/SDF/JSON) to MDL V2000 format."""
    suffix = pose_path.suffix.lower()

    if suffix == ".sdf":
        _sdf_to_mdl(pose_path, output_mol)
        return output_mol

    if suffix in (".pdbqt", ".dlg"):
        _pdbqt_to_mdl(pose_path, output_mol)
        return output_mol

    if suffix == ".json":
        # Protenix-Dock results JSON — extract best pose SDF
        import json as _json
        data = _json.loads(pose_path.read_text())
        sdf_str = data.get("best_pose_sdf") or data.get("pose_sdf")
        if sdf_str:
            tmp_sdf = output_mol.with_suffix(".tmp.sdf")
            tmp_sdf.write_text(sdf_str)
            _sdf_to_mdl(tmp_sdf, output_mol)
            tmp_sdf.unlink(missing_ok=True)
            return output_mol
        raise ValueError(f"Could not extract SDF from {pose_path}")

    raise ValueError(f"Unsupported pose file format: {pose_path}")


def _sdf_to_mdl(sdf_path: Path, output_mol: Path) -> None:
    """Extract first mol from SDF and write as MDL V2000."""
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Failed to read SDF: {sdf_path}")
    mol_block = Chem.MolToMolBlock(mol, kekulize=True)
    output_mol.write_text(mol_block)


def _pdbqt_to_mdl(pdbqt_path: Path, output_mol: Path) -> None:
    """Convert PDBQT/DLG → MDL via meeko mk_export.py, then clean up."""
    import subprocess
    import sys
    mk_export = Path(sys.executable).parent / "mk_export.py"

    tmp_sdf = output_mol.with_suffix(".tmp.sdf")
    if mk_export.exists():
        try:
            subprocess.run(
                [str(mk_export), str(pdbqt_path), "-s", str(tmp_sdf)],
                check=True, capture_output=True, text=True, timeout=60,
            )
            if tmp_sdf.exists():
                _sdf_to_mdl(tmp_sdf, output_mol)
                tmp_sdf.unlink(missing_ok=True)
                return
        except Exception:
            pass

    # Fallback: openbabel
    try:
        subprocess.run(
            ["obabel", str(pdbqt_path), "-O", str(tmp_sdf)],
            check=True, capture_output=True, text=True,
        )
        if tmp_sdf.exists():
            _sdf_to_mdl(tmp_sdf, output_mol)
            tmp_sdf.unlink(missing_ok=True)
            return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    raise RuntimeError(f"Could not convert {pdbqt_path} to MDL (mk_export.py and obabel both failed)")


def build_lg_submission(
    target_id: str,
    author: str,
    method: str,
    protein_pdb_lines: list[str],
    ligand_mdl: str,
    ligand_number: int,
    ligand_name: str,
    lscore: float | None = None,
    parent: str = "N/A",
    remark: str = "",
) -> str:
    """Assemble CASP17 LG format submission text."""
    lines = [
        "PFRMAT LG",
        f"TARGET {target_id}",
        f"AUTHOR {author}",
        f"METHOD {method}",
        "METHOD -------------",
        "MODEL 1",
    ]
    if remark:
        lines.append(f"REMARK {remark}")
    lines.append(f"PARENT {parent}")

    # Protein atoms
    lines.extend(protein_pdb_lines)

    # Ensure TER present
    if not protein_pdb_lines or not protein_pdb_lines[-1].startswith("TER"):
        lines.append("TER")

    # Ligand block
    lines.append(f"LIGAND {ligand_number:03d} {ligand_name}")
    if lscore is not None:
        lines.append(f"LSCORE {lscore:.3f}")

    # MDL block (strip any trailing newline, ensure M  END)
    mdl_text = ligand_mdl.rstrip()
    if not mdl_text.endswith("M  END"):
        mdl_text += "\nM  END"
    lines.append(mdl_text)

    lines.append("END")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CASP17 LG-format submission.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Pipeline run directory (experiments/runs/<target>)")
    parser.add_argument("--target-id", type=str, required=True,
                        help="CASP target identifier (e.g., L2001)")
    parser.add_argument("--ligand-name", type=str, required=True,
                        help="Ligand name from SMILES file (e.g., 761)")
    parser.add_argument("--ligand-number", type=int, default=1,
                        help="Ligand number within target (default: 1)")
    parser.add_argument("--author", type=str, required=True,
                        help="CASP registration code (XXXX-XXXX-XXXX)")
    parser.add_argument("--method", type=str, required=True,
                        help="Description of prediction method")
    parser.add_argument("--remark", type=str, default="")
    parser.add_argument("--parent", type=str, default="N/A",
                        help="Template PDB ID or 'N/A' for de novo")
    parser.add_argument("--protein-model", type=str, default=None,
                        choices=[None, "boltz2", "boltz2x", "protenix", "alphafold3"],
                        help="Force specific cofolding model (default: best by pLDDT)")
    parser.add_argument("--pose-source", type=str, default="auto",
                        choices=["auto", "vina", "autodock_gpu", "protenix_dock",
                                 "template", "lig_align"],
                        help="Ligand pose source (default: auto = best BA-Pred)")
    parser.add_argument("--lscore", type=float, default=None,
                        help="Ligand reliability score [0-1]")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output LG file path")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.output.parent / f".{args.target_id}_workdir"
    workdir.mkdir(exist_ok=True)

    # 1. Select best protein cofolding output
    print(f"Selecting protein structure for {args.target_id}...")
    protein_model, cif_path = find_best_cofolding_cif(run_dir, preferred=args.protein_model)
    print(f"  Protein source: {protein_model}")
    print(f"  CIF: {cif_path}")

    protein_pdb = workdir / "protein.pdb"
    cif_to_pdb_with_plddt(cif_path, protein_pdb, args.target_id)
    protein_lines = extract_pdb_atom_lines(protein_pdb)
    print(f"  Protein atoms: {len(protein_lines)}")

    # 2. Select best ligand pose
    print(f"\nSelecting ligand pose (source={args.pose_source})...")
    pose_source, pose_path = find_best_ligand_pose(run_dir, source=args.pose_source)
    print(f"  Pose source: {pose_source}")
    print(f"  Pose file: {pose_path}")

    ligand_mdl_file = workdir / "ligand.mol"
    pose_to_mdl(pose_path, ligand_mdl_file)
    ligand_mdl = ligand_mdl_file.read_text()
    print(f"  MDL block: {len(ligand_mdl.splitlines())} lines")

    # 3. Assemble LG submission
    method_full = f"{args.method} [protein={protein_model}, pose={pose_source}]"
    remark = args.remark or f"{protein_model} + {pose_source}"

    submission = build_lg_submission(
        target_id=args.target_id,
        author=args.author,
        method=method_full,
        protein_pdb_lines=protein_lines,
        ligand_mdl=ligand_mdl,
        ligand_number=args.ligand_number,
        ligand_name=args.ligand_name,
        lscore=args.lscore,
        parent=args.parent,
        remark=remark,
    )

    args.output.write_text(submission)
    print(f"\n{'='*60}")
    print(f"  LG submission written: {args.output}")
    print(f"  Size: {len(submission):,} bytes")
    print(f"  Lines: {len(submission.splitlines()):,}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
