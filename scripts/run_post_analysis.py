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

    Idempotency: when the upstream ``src`` is newer than a previously staged
    SDF, the staged SDF is deleted before re-conversion so selective reruns
    pick up the refreshed pose. Without this, a re-docked seed silently
    reuses the prior staging and BA/RMSD-Pred score the stale conformer.

    Returns the preferred representative for downstream tools: the SDF if the
    conversion succeeded, otherwise the original staged file.
    """
    if not src.exists():
        return None
    dst_orig = staged_dir / f"{stem}{src.suffix}"
    if dst_orig.resolve() != src.resolve():
        shutil.copyfile(src, dst_orig)
    dst_sdf = staged_dir / f"{stem}.sdf"
    src_mtime = src.stat().st_mtime
    if dst_sdf.exists() and dst_sdf.stat().st_mtime < src_mtime:
        dst_sdf.unlink()
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


_USALIGN_BIN = Path(__file__).resolve().parent.parent / ".local" / "bin" / "USalign"


def _usalign_transform(src_pdb: Path, ref_pdb: Path) -> tuple[list[list[float]], list[float]] | None:
    """Return (R, t) such that ``X_ref = t + R @ X_src`` for each atom.

    Runs USalign on the two receptor PDBs and parses the rotation matrix.
    Returns ``None`` if USalign fails or the matrix file is not written
    (low sequence overlap / disjoint chains). Callers should treat a
    failure as "skip the transform and leave the pose untouched".
    """
    if not _USALIGN_BIN.exists():
        return None
    if not (src_pdb.exists() and ref_pdb.exists()):
        return None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=True) as mat_f:
            mat_path = mat_f.name
        proc = subprocess.run(
            [str(_USALIGN_BIN), str(src_pdb), str(ref_pdb), "-m", mat_path],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not Path(mat_path).exists():
            return None
        R = [[0.0] * 3 for _ in range(3)]
        t = [0.0, 0.0, 0.0]
        for line in Path(mat_path).read_text().splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] not in ("0", "1", "2"):
                continue
            i = int(parts[0])
            t[i] = float(parts[1])
            R[i][0] = float(parts[2])
            R[i][1] = float(parts[3])
            R[i][2] = float(parts[4])
        Path(mat_path).unlink(missing_ok=True)
        # Sanity: if every row is zero the parse failed.
        if all(abs(v) < 1e-12 for row in R for v in row):
            return None
        return R, t
    except Exception:
        return None


def _apply_transform_to_sdf(sdf_path: Path, R: list[list[float]], t: list[float]) -> bool:
    """Apply rigid transform ``X' = t + R @ X`` to every conformer in ``sdf_path``
    and rewrite the file in place. Returns True on success.

    Used to map template-docked poses (in the template receptor's crystal
    frame) into the cofold receptor's frame, so downstream BA-Pred runs
    and the final CASP LG MODEL block see the pose and the receptor in
    one consistent coordinate system.
    """
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        mols = [m for m in supp if m is not None]
        if not mols:
            return False
        for mol in mols:
            for conf in mol.GetConformers():
                for i in range(mol.GetNumAtoms()):
                    p = conf.GetAtomPosition(i)
                    x = R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + t[0]
                    y = R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + t[1]
                    z = R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + t[2]
                    conf.SetAtomPosition(i, Point3D(x, y, z))
        writer = Chem.SDWriter(str(sdf_path))
        for mol in mols:
            writer.write(mol)
        writer.close()
        return True
    except Exception:
        return False


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
    dropped_outliers = 0
    for i, pose in enumerate(poses):
        xyz = pose.get("ligand", {}).get("xyz")
        if not xyz or len(xyz) < len(ordered_indices):
            continue
        # PxDock's minimization can numerically blow up on a small fraction
        # of poses, producing coords up to 1e7 Å. These poison RMSD scoring
        # and any submission MDL we build from them. 200 Å is well outside
        # any plausible binding-site frame; anything past that is garbage.
        if any(abs(c) > 200.0 for triple in xyz[: len(ordered_indices)] for c in triple):
            dropped_outliers += 1
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
    if dropped_outliers:
        print(f"    protenix_dock: dropped {dropped_outliers} pose(s) with |coord| > 200 Å")
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


def _extract_cofolding_ligand(cif_path: Path, template_mol, ligand_id: str = "L"):
    """Extract ONE specific ligand (by id) from an aligned cofolding CIF.

    ``ligand_id`` is the CommonInput ligand id (``"L"``, ``"L2"``, ...) which
    is also the chain label used by Boltz/Protenix/AF3 in their aligned
    output CIFs. We first try an exact chain-name match, then fall back to
    heavy-atom count matching against the template graph (for models that
    rename the chain — e.g. AF3 sometimes uses ``A``/``B``/``C``).

    Returns an RDKit Mol with bond orders assigned from ``template_mol``,
    or ``None`` if no matching ligand chain is found.
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

    target_heavy = template_mol.GetNumHeavyAtoms()

    # Pass 1: collect candidate ligand chains. A chain is "ligand-like" if
    # any residue has non-protein heteroatom nature.
    candidates: list[tuple[str, list]] = []  # (chain_name, atoms)
    for model in st:
        for chain in model:
            heavy_atoms = []
            for res in chain:
                is_lig_res = (
                    res.name in ("LIG", "LIG1", "UNL", "UNK")
                    or getattr(res, "het_flag", "") == "H"
                    or chain.name == ligand_id
                )
                if not is_lig_res:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    heavy_atoms.append(atom)
            if heavy_atoms:
                candidates.append((chain.name, heavy_atoms))
        break

    if not candidates:
        return None

    # Pick chain by id match first, else by heavy-atom-count match, else first.
    selected = None
    for name, atoms in candidates:
        if name == ligand_id:
            selected = atoms
            break
    if selected is None:
        for _, atoms in candidates:
            if len(atoms) == target_heavy:
                selected = atoms
                break
    if selected is None:
        # Single-ligand legacy fallback
        if len(candidates) == 1:
            selected = candidates[0][1]
        else:
            return None

    lines = []
    for i, atom in enumerate(selected, start=1):
        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
        elem = atom.element.name
        lines.append(
            f"HETATM{i:>5d} {atom.name:>4s} LIG L"
            f"{1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2s}\n"
        )
    raw = Chem.MolFromPDBBlock("".join(lines) + "END\n", removeHs=True, sanitize=False)
    if raw is None:
        return None
    try:
        return AssignBondOrdersFromTemplate(template_mol, raw)
    except Exception:
        return None


def _load_ligand_templates(run_dir: Path) -> dict[str, "object"]:
    """Load per-ligand template mols from ``inputs/docking/ligand_*.sdf``.

    Returns ``{lig_id: template_heavy_mol}`` where ``lig_id`` is derived
    from the filename stem (``ligand_L.sdf`` → ``L``, ``ligand_L2.sdf`` →
    ``L2``). Template is heavy-atom-only to match extraction output.
    """
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return {}
    templates: dict[str, object] = {}
    for cand in sorted((run_dir / "inputs" / "docking").glob("ligand_*.sdf")):
        stem = cand.stem  # e.g. "ligand_L" or "ligand_L2"
        lig_id = stem[len("ligand_"):] if stem.startswith("ligand_") else stem
        mol = next(iter(Chem.SDMolSupplier(str(cand), removeHs=False, sanitize=True)), None)
        if mol is None:
            continue
        templates[lig_id] = Chem.RemoveHs(mol)
    return templates


def _stage_cofolding_poses(run_dir: Path, staged_dir: Path) -> dict[str, Path]:
    """Extract aligned ligand poses from cofolding CIFs into multi-pose SDFs.

    Multi-ligand support: iterates every ligand found in
    ``inputs/docking/ligand_*.sdf`` and extracts that specific ligand from
    each cofolding CIF. Output key is ``cofold_{model}_{lig_id}`` — e.g.
    ``cofold_boltz2_L``, ``cofold_protenix_L2`` — so BA-Pred / RMSD-Pred
    TSVs distinguish per-ligand poses downstream.

    Returns ``{tool_key: sdf_path}`` for all (model, ligand) pairs that
    yielded at least one pose. Bond orders come from the per-ligand SDF
    template (authoritative graph from meeko/RDKit prep).
    """
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return {}

    templates = _load_ligand_templates(run_dir)
    if not templates:
        print("  cofold: no ligand template SDF, skipping cofolding pose extraction")
        return {}

    model_patterns = {
        "boltz2": "*_aligned.cif",
        "boltz2x": "*_aligned.cif",
        "protenix": "*_aligned.cif",
        "alphafold3": "*_aligned.cif",
    }
    model_short = {
        "boltz2": "boltz2", "boltz2x": "boltz2x",
        "protenix": "protenix", "alphafold3": "af3",
    }
    written: dict[str, Path] = {}

    for subdir, pat in model_patterns.items():
        cifs = sorted((run_dir / "outputs" / subdir).rglob(pat))
        if not cifs:
            continue
        short = model_short[subdir]
        for lig_id, template_heavy in templates.items():
            key = f"cofold_{short}_{lig_id}"
            sdf_path = staged_dir / f"{key}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            n = 0
            for cif in cifs:
                mol = _extract_cofolding_ligand(cif, template_heavy, ligand_id=lig_id)
                if mol is None:
                    continue
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


def _discover_ligand_ids(run_dir: Path) -> list[str]:
    """Return the list of ligand ids known to this run.

    Primary source is ``inputs/docking/docking_prep_summary.json``. Falls
    back to scanning ``inputs/docking/ligand_*.sdf`` names, and finally to
    ``["L"]`` if nothing else is available.
    """
    summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            ids = [str(lig.get("id")) for lig in (data.get("ligands") or []) if lig.get("id")]
            if ids:
                return ids
        except Exception:
            pass
    sdfs = sorted((run_dir / "inputs" / "docking").glob("ligand_*.sdf"))
    if sdfs:
        return [p.stem[len("ligand_"):] for p in sdfs if p.stem.startswith("ligand_")] or ["L"]
    return ["L"]


def _discover_pose_under_ligand_dir(
    seed_dir: Path, lig_id: str, filename: str,
) -> Path | None:
    """Locate a pose file honoring both new per-ligand layout and legacy flat layout.

    New layout (post multi-ligand refactor): ``seed_dir/ligand_{id}/{filename}``.
    Legacy: ``seed_dir/{filename}`` (docking ran on only the first ligand). The
    legacy file is mapped to whichever ligand id is "primary" (first in list).
    """
    new = seed_dir / f"ligand_{lig_id}" / filename
    if new.exists():
        return new
    return None


def find_ligand_files(run_dir: Path) -> dict[str, Path]:
    """Collect pose files per (tool, ligand), for BA-Pred/RMSD-Pred.

    Keys are ``{tool_variant}_{lig_id}`` — e.g. ``vina_cofolding_L``,
    ``autodock_gpu_swinsite_L2``, ``protenix_dock_L``. This lets the
    scoring aggregator partition poses per ligand so the final top-5
    submission can pick ONE pose per ligand per MODEL (CASP LG spec: a
    MODEL is the whole complex, all ligands together).

    Legacy layout (single-ligand docking runs that predate the multi-ligand
    refactor) is handled by treating the whole seed directory's output as
    the primary ligand (first entry in the summary).
    """
    ligands: dict[str, Path] = {}
    analysis_dir = run_dir / "outputs" / "analysis"
    staged_dir = analysis_dir / "poses"
    staged_dir.mkdir(parents=True, exist_ok=True)

    lig_ids = _discover_ligand_ids(run_dir)
    primary_lig = lig_ids[0] if lig_ids else "L"
    outputs_dir = run_dir / "outputs"

    # --- Vina: per binding-site variant × per ligand × per seed ---
    vina_variant_dirs = sorted(
        [outputs_dir / "vina"]
        + [d for d in outputs_dir.glob("vina_*") if d.is_dir()]
    )
    for var_dir in vina_variant_dirs:
        if not var_dir.exists():
            continue
        variant = var_dir.name
        for lig_id in lig_ids:
            tool_key = f"{variant}_{lig_id}"
            seed_poses: list[Path] = []
            for seed_dir in sorted(var_dir.glob("seed_*")):
                pose = _discover_pose_under_ligand_dir(seed_dir, lig_id, "docked.pdbqt")
                if pose is None and lig_id == primary_lig:
                    legacy = seed_dir / "docked.pdbqt"
                    if legacy.exists():
                        pose = legacy
                if pose is not None:
                    staged = _stage_pose_file(pose, staged_dir, f"{tool_key}_{seed_dir.name}")
                    if staged is not None:
                        seed_poses.append(staged)
            if not seed_poses and lig_id == primary_lig:
                flat = var_dir / "docked.pdbqt"
                staged = _stage_pose_file(flat, staged_dir, f"{tool_key}_flat")
                if staged is not None:
                    seed_poses.append(staged)
            if seed_poses:
                lst = _write_list_file(seed_poses, analysis_dir / f"{tool_key}_poses.txt")
                if lst is not None:
                    ligands[tool_key] = lst

    # --- AutoDock-GPU: same per-ligand, per-seed iteration ---
    adg_variant_dirs = sorted(
        [outputs_dir / "autodock_gpu"]
        + [d for d in outputs_dir.glob("autodock_gpu_*") if d.is_dir()]
    )
    for var_dir in adg_variant_dirs:
        if not var_dir.exists():
            continue
        variant = var_dir.name
        for lig_id in lig_ids:
            tool_key = f"{variant}_{lig_id}"
            seed_poses = []
            for seed_dir in sorted(var_dir.glob("seed_*")):
                pose = _discover_pose_under_ligand_dir(seed_dir, lig_id, "docking.dlg")
                if pose is None and lig_id == primary_lig:
                    legacy = seed_dir / "docking.dlg"
                    if legacy.exists():
                        pose = legacy
                if pose is not None:
                    staged = _stage_pose_file(pose, staged_dir, f"{tool_key}_{seed_dir.name}")
                    if staged is not None:
                        seed_poses.append(staged)
            if not seed_poses and lig_id == primary_lig:
                flat = var_dir / "docking.dlg"
                staged = _stage_pose_file(flat, staged_dir, f"{tool_key}_flat")
                if staged is not None:
                    seed_poses.append(staged)
            if seed_poses:
                lst = _write_list_file(seed_poses, analysis_dir / f"{tool_key}_poses.txt")
                if lst is not None:
                    ligands[tool_key] = lst

    # --- Protenix-Dock: JSON -> multi-record SDF, per-ligand when available ---
    pxdock_dir = outputs_dir / "protenix_dock"
    if pxdock_dir.exists():
        for lig_id in lig_ids:
            lig_dir = pxdock_dir / f"ligand_{lig_id}"
            source_dir = lig_dir if lig_dir.is_dir() else (pxdock_dir if lig_id == primary_lig else None)
            if source_dir is None:
                continue
            existing_sdf = next(source_dir.rglob("*.sdf"), None)
            if existing_sdf is not None:
                ligands[f"protenix_dock_{lig_id}"] = existing_sdf
                continue
            out_json = next((p for p in source_dir.glob("*_out.json")), None)
            if out_json is not None:
                # Write to ``protenix_dock/poses_<lig_id>.sdf`` so it matches
                # the layout compute_submission_scores expects (see
                # _resolve_pose_file there). The legacy ``poses.sdf`` path
                # left poses unfindable for the multi-ligand path.
                sdf_path = pxdock_dir / f"poses_{lig_id}.sdf"
                converted = _pxdock_json_to_sdf(out_json, sdf_path)
                if converted is not None:
                    ligands[f"protenix_dock_{lig_id}"] = converted

    # --- Cofolding aligned ligand poses (per-ligand: cofold_{model}_{lig_id}) ---
    cofold_sdfs = _stage_cofolding_poses(run_dir, staged_dir)
    ligands.update(cofold_sdfs)

    # --- Template-based docking poses (Track 2 + Track 3, multi-ligand) ---
    # Multi-track docking now runs per-dockable-ligand. Each template_dir is
    # ``outputs/template_docking/<template_pdb_id>/{vina,autodock_gpu,protenix_dock,lig_align}/ligand_<lig_id>/``.
    # Legacy single-ligand layout (no ``ligand_<id>/`` sub-dir) is also
    # accepted as a fallback so old runs on disk keep working.
    # Filename conventions (fixed upstream in prepare_template_docking.py /
    # run_multi_track_docking.py):
    #   vina           → docked.pdbqt
    #   autodock_gpu   → docked.dlg      (NOT ``docking.dlg`` — that was a typo)
    #   lig_align      → *.sdf           (produced by lig_align.run_pipeline)
    #
    # Template docking runs in the TEMPLATE receptor's crystal frame, which
    # generally does not overlap the cofold receptor. Before staging we
    # compute one USalign(template_receptor → cofold_receptor) rigid transform
    # per template and rewrite every staged pose SDF in the cofold frame.
    # This keeps BA-Pred's pocket-extraction and the final CASP LG MODEL
    # block (which embeds the cofold receptor) in one consistent frame.
    template_root = outputs_dir / "template_docking"
    cofold_receptor_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if template_root.exists():
        for template_dir in sorted(template_root.iterdir()):
            if not template_dir.is_dir():
                continue
            pdb_id = template_dir.name
            tpl_receptor_pdb = (
                run_dir / "inputs" / "template_docking" / f"template_{pdb_id}" / "receptor.pdb"
            )
            transform = None
            if cofold_receptor_pdb.exists() and tpl_receptor_pdb.exists():
                transform = _usalign_transform(tpl_receptor_pdb, cofold_receptor_pdb)
                if transform is None:
                    print(f"  WARN: USalign failed for template {pdb_id}; "
                          f"poses kept in template frame (BA-Pred likely to fail).")

            def _stage_and_align(src: Path, tool_key: str) -> Path | None:
                staged = _stage_pose_file(src, staged_dir, tool_key)
                if staged is None:
                    return None
                if transform is not None and staged.suffix.lower() == ".sdf":
                    R, t = transform
                    if not _apply_transform_to_sdf(staged, R, t):
                        print(f"  WARN: transform application failed for {staged.name}")
                return staged

            # Legacy single-ligand template-docking layouts predate the
            # ligand_<id>/ sub-dir refactor. They map to ``primary_lig`` only,
            # which silently dropped second ligands on multi-ligand targets.
            # Gate the fallback to single-ligand targets so multi-ligand
            # legacy outputs surface as a warning instead of misattributed.
            single_ligand_target = len(lig_ids) <= 1

            # Vina: per-ligand sub-dir (new) or single-ligand legacy layout.
            for vina_dir in (template_dir / "vina").glob("ligand_*"):
                lig_id = vina_dir.name[len("ligand_"):]
                vina_pdbqt = vina_dir / "docked.pdbqt"
                if vina_pdbqt.exists():
                    tool_key = f"template_{pdb_id}_vina_{lig_id}"
                    staged = _stage_and_align(vina_pdbqt, tool_key)
                    if staged is not None:
                        ligands[tool_key] = staged
            legacy_vina = template_dir / "vina" / "docked.pdbqt"
            if legacy_vina.exists() and single_ligand_target:
                tool_key = f"template_{pdb_id}_vina_{primary_lig}"
                if tool_key not in ligands:
                    staged = _stage_and_align(legacy_vina, tool_key)
                    if staged is not None:
                        ligands[tool_key] = staged
            elif legacy_vina.exists():
                print(f"  WARN: legacy template_docking/{pdb_id}/vina/ "
                      f"output but target has {len(lig_ids)} ligands; skipping "
                      "(re-run multi-track docking to materialize per-ligand outputs).")

            # AutoDock-GPU: same per-ligand pattern.
            for adg_dir in (template_dir / "autodock_gpu").glob("ligand_*"):
                lig_id = adg_dir.name[len("ligand_"):]
                adg_dlg = adg_dir / "docked.dlg"
                if adg_dlg.exists():
                    tool_key = f"template_{pdb_id}_adg_{lig_id}"
                    staged = _stage_and_align(adg_dlg, tool_key)
                    if staged is not None:
                        ligands[tool_key] = staged
            legacy_adg = template_dir / "autodock_gpu" / "docked.dlg"
            if legacy_adg.exists() and single_ligand_target:
                tool_key = f"template_{pdb_id}_adg_{primary_lig}"
                if tool_key not in ligands:
                    staged = _stage_and_align(legacy_adg, tool_key)
                    if staged is not None:
                        ligands[tool_key] = staged
            elif legacy_adg.exists():
                print(f"  WARN: legacy template_docking/{pdb_id}/autodock_gpu/ "
                      f"output but target has {len(lig_ids)} ligands; skipping.")

            # lig-align: per-ligand sub-dir or legacy.
            lig_align_root = template_dir / "lig_align"
            if lig_align_root.exists():
                for la_dir in lig_align_root.glob("ligand_*"):
                    lig_id = la_dir.name[len("ligand_"):]
                    mcs_sdf = next(la_dir.glob("*.sdf"), None)
                    if mcs_sdf is not None:
                        tool_key = f"template_{pdb_id}_lig_align_{lig_id}"
                        staged = _stage_and_align(mcs_sdf, tool_key)
                        if staged is not None:
                            ligands[tool_key] = staged
                legacy_mcs = next(lig_align_root.glob("*.sdf"), None)
                if legacy_mcs is not None and single_ligand_target:
                    tool_key = f"template_{pdb_id}_lig_align_{primary_lig}"
                    if tool_key not in ligands:
                        staged = _stage_and_align(legacy_mcs, tool_key)
                        if staged is not None:
                            ligands[tool_key] = staged
                elif legacy_mcs is not None:
                    print(f"  WARN: legacy template_docking/{pdb_id}/lig_align/*.sdf "
                          f"output but target has {len(lig_ids)} ligands; skipping.")

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


_BAPRED_MIN_CC_MAJOR = 6  # Pascal+
_BAPRED_MAX_CC_MAJOR = 8  # Ada/Ampere; sm_90 (Hopper) and sm_100 (Blackwell) are unsupported


def _verify_cuda_capability(device: str) -> tuple[bool, str]:
    """Refuse sm_90+ devices for BA-Pred / RMSD-Pred.

    The bundled CUDA kernels in ``.venvs/pred`` were compiled for sm_60–sm_89.
    On Hopper (H100) and Blackwell the prediction binaries return exit 0 with
    empty TSVs — undetectable in downstream pipelines because the missing
    rows just look like "no qualifying poses". Read the runtime capability
    explicitly and fail loud instead.
    """
    if device == "cpu":
        return True, "cpu"
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        # Without torch we can't probe; fall through and let bapred fail naturally.
        return True, "cuda (torch unavailable for capability probe)"
    if not torch.cuda.is_available():
        return False, "no CUDA device visible — submit on a GPU partition"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if major > _BAPRED_MAX_CC_MAJOR:
        return False, (
            f"GPU {name} reports sm_{major}{minor}; BA-Pred / RMSD-Pred "
            f"kernels in .venvs/pred only support sm_{_BAPRED_MIN_CC_MAJOR}0 "
            f"through sm_{_BAPRED_MAX_CC_MAJOR}9 (Pascal/Volta/Turing/Ampere/Ada). "
            "Heavy partition (H100/Blackwell) silently produces empty results. "
            "Submit post-analysis on the 6000ada partition or rebuild kernels."
        )
    if major < _BAPRED_MIN_CC_MAJOR:
        return False, (
            f"GPU {name} reports sm_{major}{minor}; BA-Pred kernels need "
            f"sm_{_BAPRED_MIN_CC_MAJOR}0 or newer."
        )
    return True, f"cuda sm_{major}{minor} ({name})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BA-Pred and RMSD-Pred on pipeline results.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Pipeline run directory.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cap_ok, cap_msg = _verify_cuda_capability(args.device)
    print(f"Compute capability check: {cap_msg}")
    if not cap_ok:
        print(f"ERROR: {cap_msg}")
        return 2

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
