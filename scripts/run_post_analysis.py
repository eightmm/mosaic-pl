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
import shutil
import subprocess
import sys
from pathlib import Path


def _stage_pose_file(src: Path, staged_dir: Path, stem: str) -> Path | None:
    """Copy a docked pose file to ``staged_dir`` with a unique stem and, when
    possible, convert it to SDF alongside the original.

    BA-Pred/RMSD-Pred name each pose ``{base_stem}_{index}``, so if we pass a
    ``.txt`` list of files that all have the same basename (``docked.pdbqt``),
    the emitted TSVs have name collisions. Staging each file under a unique
    stem fixes this, and producing a parallel ``{stem}.sdf`` lets downstream
    pose-extraction code (``make_casp_submission.py``) load any selected pose
    by its canonical name without caring about the original format.

    Returns the preferred representative for downstream tools: the SDF if the
    conversion succeeded, otherwise the original staged file.
    """
    if not src.exists():
        return None
    dst_orig = staged_dir / f"{stem}{src.suffix}"
    if dst_orig.resolve() != src.resolve():
        shutil.copyfile(src, dst_orig)
    dst_sdf = staged_dir / f"{stem}.sdf"
    if not dst_sdf.exists():
        _pdbqt_to_sdf(dst_orig, dst_sdf)
    return dst_sdf if dst_sdf.exists() else dst_orig


def _pdbqt_to_sdf(pdbqt_path: Path, sdf_path: Path) -> Path | None:
    """Convert PDBQT/DLG to SDF using meeko mk_export.py CLI."""
    mk_export = Path(sys.executable).parent / "mk_export.py"
    if not mk_export.exists():
        return None
    try:
        subprocess.run(
            [str(mk_export), str(pdbqt_path), "-s", str(sdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if sdf_path.exists() and sdf_path.stat().st_size > 0:
            return sdf_path
        return None
    except Exception:
        return None


def _write_list_file(paths: list[Path], list_path: Path) -> Path | None:
    if not paths:
        return None
    list_path.write_text("\n".join(str(p) for p in paths) + "\n")
    return list_path


def _pxdock_json_to_sdf(json_path: Path, sdf_path: Path) -> Path | None:
    """Convert a Protenix-Dock ``*_out.json`` pose file to a multi-pose SDF.

    The JSON stores an atom-mapped SMILES plus per-pose Cartesian coordinates.
    We build an RDKit Mol from the mapped SMILES, re-order atoms to match the
    mapping indices, then stamp each pose's xyz onto a conformer and write
    them out as a multi-record SDF.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # noqa: F401  (needed for conformer ops)
    except Exception as e:  # pragma: no cover - optional dep path
        print(f"    protenix_dock: rdkit unavailable ({e}), skipping SDF conversion")
        return None

    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        print(f"    protenix_dock: failed to parse {json_path.name}: {e}")
        return None

    smiles = data.get("mapped_smiles")
    poses = data.get("poses") or []
    if not smiles or not poses:
        print(f"    protenix_dock: {json_path.name} missing mapped_smiles or poses")
        return None

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        print("    protenix_dock: rdkit failed to parse mapped_smiles")
        return None
    template = Chem.AddHs(template)

    # Atom index from the mapped SMILES (:N) → RDKit atom index.
    map_to_idx: dict[int, int] = {}
    for atom in template.GetAtoms():
        m = atom.GetAtomMapNum()
        if m > 0:
            map_to_idx[m] = atom.GetIdx()
    if not map_to_idx:
        print("    protenix_dock: mapped_smiles has no atom maps")
        return None

    # Build a stable ordering for xyz assignment. The JSON's ligand.xyz array
    # is stored in atom-map order (1..N).
    ordered_indices = [map_to_idx[k] for k in sorted(map_to_idx.keys())]
    n_atoms = template.GetNumAtoms()

    writer = Chem.SDWriter(str(sdf_path))
    written = 0
    for i, pose in enumerate(poses):
        xyz = pose.get("ligand", {}).get("xyz")
        if not xyz or len(xyz) < len(ordered_indices):
            continue
        conf = Chem.Conformer(n_atoms)
        # Initialize all atoms to origin, then overlay mapped atoms.
        for j in range(n_atoms):
            conf.SetAtomPosition(j, (0.0, 0.0, 0.0))
        for pos_i, atom_idx in enumerate(ordered_indices):
            x, y, z = xyz[pos_i]
            conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
        mol = Chem.Mol(template)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        mol.SetProp("_Name", f"pxdock_pose_{i}")
        pscore = pose.get("pscore")
        if pscore is not None:
            mol.SetProp("pxdock_pscore", str(pscore))
        writer.write(mol)
        written += 1
    writer.close()

    if written == 0:
        print(f"    protenix_dock: no poses converted from {json_path.name}")
        try:
            sdf_path.unlink()
        except OSError:
            pass
        return None
    print(f"    protenix_dock: converted {written} poses → {sdf_path.name}")
    return sdf_path


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


def _extract_cofolding_ligand(cif_path: Path, template_mol):
    """Extract the ligand from an aligned cofolding CIF with correct bond orders.

    Reads HETATM-like atoms (chain ``L`` or residues named LIG*/UNK/UNL) via
    gemmi, builds a minimal PDB block, parses with RDKit, then assigns bond
    orders from ``template_mol`` (built from the input ligand SDF/SMILES so
    the atom graph is authoritative). Returns an RDKit Mol or None.
    """
    try:
        import gemmi  # type: ignore
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate  # type: ignore
    except Exception:
        return None
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return None
    lines = []
    serial = 1
    for model in st:
        for chain in model:
            for res in chain:
                is_lig = (
                    res.name in ("LIG", "LIG1", "UNL", "UNK")
                    or getattr(res, "het_flag", "") == "H"
                    or chain.name == "L"
                )
                if not is_lig:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                    elem = atom.element.name
                    lines.append(
                        f"HETATM{serial:>5d} {atom.name:>4s} LIG L"
                        f"{1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2s}\n"
                    )
                    serial += 1
        break
    if not lines:
        return None
    raw = Chem.MolFromPDBBlock("".join(lines) + "END\n", removeHs=True, sanitize=False)
    if raw is None:
        return None
    try:
        return AssignBondOrdersFromTemplate(template_mol, raw)
    except Exception:
        return None


def _stage_cofolding_poses(run_dir: Path, staged_dir: Path) -> dict[str, Path]:
    """Extract aligned ligand poses from cofolding CIFs into multi-pose SDFs.

    Writes ``staged_dir / cofold_{model}.sdf`` for each cofolding tool that
    produced aligned outputs. Returns ``{tool_key: sdf_path}`` for successfully
    written SDFs, where ``tool_key`` is ``cofold_boltz2``/``cofold_boltz2x``/
    ``cofold_protenix``/``cofold_af3`` so downstream BA-Pred/RMSD-Pred tsvs
    carry the cofolding provenance in their filenames.

    The bond-order template is loaded from ``inputs/docking/ligand_*.sdf``
    (meeko-prepped from input SMILES) which has the authoritative molecular
    graph; cofolding CIFs alone don't encode bond orders.
    """
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return {}

    # Load template from the docking prep ligand (authoritative bond graph).
    template = None
    for cand in sorted((run_dir / "inputs" / "docking").glob("ligand_*.sdf")):
        template = next(iter(Chem.SDMolSupplier(str(cand), removeHs=False, sanitize=True)), None)
        if template is not None:
            break
    if template is None:
        print("  cofold: no ligand template SDF, skipping cofolding pose extraction")
        return {}
    template_heavy = Chem.RemoveHs(template)

    patterns = {
        "cofold_boltz2": ("boltz2", "*_aligned.cif"),
        "cofold_boltz2x": ("boltz2x", "*_aligned.cif"),
        "cofold_protenix": ("protenix", "*_aligned.cif"),
        "cofold_af3": ("alphafold3", "*_aligned.cif"),
    }
    written: dict[str, Path] = {}
    for key, (subdir, pat) in patterns.items():
        cifs = sorted((run_dir / "outputs" / subdir).rglob(pat))
        if not cifs:
            continue
        sdf_path = staged_dir / f"{key}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        n = 0
        for cif in cifs:
            mol = _extract_cofolding_ligand(cif, template_heavy)
            if mol is None:
                continue
            # Set a meaningful _Name so downstream MDL blocks (in the CASP
            # LG submission) carry the pose source instead of RDKit's
            # generic "RDKit          3D" default title. BA-Pred reads
            # _Name directly while RMSD-Pred appends a record index on top
            # of it, producing a double-indexed name (``cofold_af3_0_0``);
            # compute_submission_scores._canonicalize collapses that to
            # ``cofold_af3_0`` so the two tools still join 1-1.
            mol.SetProp("_Name", f"{key}_{n}")
            writer.write(mol)
            n += 1
        writer.close()
        if n > 0:
            written[key] = sdf_path
            print(f"  {key}: {n} cofolding poses → {sdf_path.name}")
        else:
            sdf_path.unlink(missing_ok=True)
    return written


def find_ligand_files(run_dir: Path) -> dict[str, Path]:
    """Collect pose files per source (docking + cofolding), for BA-Pred/RMSD-Pred.

    Returns a mapping ``tool_key -> pose_input``. ``pose_input`` is either a
    concrete pose file (SDF/DLG/PDBQT) or a ``.txt`` list of per-seed staged
    poses. BA-Pred and RMSD-Pred both accept all of these formats.

    Cofolding aligned poses are extracted from ``*_aligned.cif`` and staged as
    ``analysis/poses/cofold_{model}.sdf`` so the same scorer that picks the
    final top-5 sees cofolding candidates alongside docking outputs.

    The raw input SDF (``inputs/docking/ligand_*.sdf``) is intentionally
    excluded: its conformer comes from RDKit's SMILES embedding and is not
    aligned with the receptor, which makes BA-Pred's 8Å protein-context
    extraction return an empty mol and crash. Docking outputs only.
    """
    ligands: dict[str, Path] = {}
    analysis_dir = run_dir / "outputs" / "analysis"
    staged_dir = analysis_dir / "poses"
    staged_dir.mkdir(parents=True, exist_ok=True)

    # --- Vina: auto-discover per binding-site variant (vina / vina_cofolding /
    #     vina_swinsite / vina_p2rank). Each variant's seed_*/docked.pdbqt is
    #     staged under a unique key so BA-Pred/RMSD-Pred and top-5 selection
    #     can rank poses across binding-site hypotheses independently.
    outputs_dir = run_dir / "outputs"
    vina_variant_dirs = sorted(
        [outputs_dir / "vina"]
        + [d for d in outputs_dir.glob("vina_*") if d.is_dir()]
    )
    for var_dir in vina_variant_dirs:
        if not var_dir.exists():
            continue
        tool_key = var_dir.name  # "vina" or "vina_<source>"
        seed_poses: list[Path] = []
        for seed_dir in sorted(var_dir.glob("seed_*")):
            pdbqt = seed_dir / "docked.pdbqt"
            staged = _stage_pose_file(pdbqt, staged_dir, f"{tool_key}_{seed_dir.name}")
            if staged is not None:
                seed_poses.append(staged)
        if not seed_poses:
            flat = var_dir / "docked.pdbqt"
            staged = _stage_pose_file(flat, staged_dir, f"{tool_key}_flat")
            if staged is not None:
                seed_poses.append(staged)
        if seed_poses:
            lst = _write_list_file(seed_poses, analysis_dir / f"{tool_key}_poses.txt")
            if lst is not None:
                ligands[tool_key] = lst

    # --- AutoDock-GPU: same auto-discovery over autodock_gpu{,_<source>} dirs ---
    adg_variant_dirs = sorted(
        [outputs_dir / "autodock_gpu"]
        + [d for d in outputs_dir.glob("autodock_gpu_*") if d.is_dir()]
    )
    for var_dir in adg_variant_dirs:
        if not var_dir.exists():
            continue
        tool_key = var_dir.name
        seed_poses = []
        for seed_dir in sorted(var_dir.glob("seed_*")):
            dlg = seed_dir / "docking.dlg"
            staged = _stage_pose_file(dlg, staged_dir, f"{tool_key}_{seed_dir.name}")
            if staged is not None:
                seed_poses.append(staged)
        if not seed_poses:
            flat = var_dir / "docking.dlg"
            staged = _stage_pose_file(flat, staged_dir, f"{tool_key}_flat")
            if staged is not None:
                seed_poses.append(staged)
        if seed_poses:
            lst = _write_list_file(seed_poses, analysis_dir / f"{tool_key}_poses.txt")
            if lst is not None:
                ligands[tool_key] = lst

    # --- Protenix-Dock: JSON -> multi-record SDF ---
    pxdock_dir = run_dir / "outputs" / "protenix_dock"
    if pxdock_dir.exists():
        # Prefer any pre-existing SDF if PxDock ever emits one.
        existing_sdf = next(pxdock_dir.rglob("*.sdf"), None)
        if existing_sdf is not None:
            ligands["protenix_dock"] = existing_sdf
        else:
            out_json = next((p for p in pxdock_dir.glob("*_out.json")), None)
            if out_json is not None:
                sdf_path = pxdock_dir / "poses.sdf"
                converted = _pxdock_json_to_sdf(out_json, sdf_path)
                if converted is not None:
                    ligands["protenix_dock"] = converted

    # --- Cofolding aligned ligand poses (boltz2/2x/protenix/af3) ---
    cofold_sdfs = _stage_cofolding_poses(run_dir, staged_dir)
    ligands.update(cofold_sdfs)

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
        print("    TIMEOUT")
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
