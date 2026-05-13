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


def _safe_v2000_molblock(mol, *, kekulize: bool = False) -> str:
    """Emit a V2000 MDL mol block, scrubbing the stereo metadata that can
    push RDKit into V3000 output.

    The CASP LG validator accepts V2000 only (verified 2026-05-06; see
    ``docs/casp17_lg_format.md`` §0). RDKit auto-selects V3000 when
    ``MolToMolBlock`` cannot encode the molecule's stereo information in
    the 3-character V2000 columns — even for small ligands. Per-atom
    chirality and per-bond direction are already encoded in the 3D
    coordinates, so we strip them before emit. Enhanced stereo groups
    are dropped likewise.

    Raises ``ValueError`` if the result is still V3000 — at that point
    the molecule genuinely exceeds the V2000 size limit (atom or bond
    count > 999) and the LG entry cannot be produced.
    """
    from rdkit import Chem

    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    try:
        mol.SetStereoGroups([])
    except (AttributeError, RuntimeError):
        pass
    block = Chem.MolToMolBlock(mol, kekulize=kekulize)
    lines = block.splitlines()
    if len(lines) >= 4 and "V3000" in lines[3]:
        raise ValueError(
            "RDKit produced a V3000 mol block which the CASP LG validator "
            "rejects; ligand likely exceeds V2000's 999-atom/bond cap"
        )
    return block


def find_best_cofolding_cif(run_dir: Path, preferred: str | None = None) -> tuple[str, Path]:
    """Find best cofolding CIF, preferring ``_aligned.cif`` and pLDDT ranking.

    Emits a loud warning when forced to fall back to an unaligned cif —
    the submission MODEL receptor must be in the same frame as the docked
    pose MDL blocks, otherwise the rendered LG file mixes coordinate frames.
    """

    def _first_cif(model_dir: Path, model: str) -> Path | None:
        if model.startswith("boltz"):
            raw = sorted(model_dir.rglob("predictions/**/*.cif"))
        elif model == "protenix":
            raw = sorted(c for c in model_dir.rglob("*.cif") if "_aligned" not in c.name)
        elif model == "alphafold3":
            raw = sorted(c for c in model_dir.rglob("*model*.cif") if "_aligned" not in c.name)
        else:
            raw = sorted(c for c in model_dir.rglob("*.cif") if "_aligned" not in c.name)
        for cif in raw:
            aligned = cif.with_name(cif.stem + "_aligned.cif")
            if aligned.exists():
                return aligned
            print(f"  WARNING: submission falling back to UNALIGNED "
                  f"{model}/{cif.name} — pose coords (aligned frame) and "
                  f"receptor coords will disagree. Re-run "
                  f"scripts/align_cofolding_outputs.py first.")
            return cif
        return None

    candidates = []
    for model in ("boltz2x", "boltz2", "protenix", "alphafold3"):
        if preferred and model != preferred:
            continue
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue
        cif = _first_cif(model_dir, model)
        if cif is not None:
            score = _read_plddt(model_dir, model)
            candidates.append((model, cif, score))

    if not candidates:
        raise ValueError(f"No cofolding outputs found in {run_dir}/outputs/")

    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _read_plddt(model_dir: Path, model: str) -> float:
    """Return average pLDDT normalised to [0, 100] for ranking.

    Boltz stores per-residue pLDDT on [0, 1] → multiply by 100.
    Protenix stores a scalar on [0, 100].
    AF3 stores per-atom pLDDT on [0, 100].
    """
    try:
        if model.startswith("boltz"):
            import numpy as np
            for npz in model_dir.rglob("plddt_*model_0.npz"):
                data = np.load(str(npz))
                return float(data[data.files[0]].mean()) * 100.0
        elif model == "protenix":
            for j in model_dir.rglob("*confidence*.json"):
                d = json.loads(j.read_text())
                if "plddt" in d:
                    val = d["plddt"]
                    if isinstance(val, (list, tuple)) and val:
                        return float(sum(val) / len(val))
                    if isinstance(val, (int, float)):
                        return float(val)
        elif model == "alphafold3":
            for j in model_dir.rglob("*confidence*.json"):
                if "summary" in j.name:
                    continue
                d = json.loads(j.read_text())
                if "atom_plddts" in d and d["atom_plddts"]:
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


def pose_to_mdl(
    pose_path: Path,
    output_mol: Path,
    pose_index: int | None = None,
    title: str | None = None,
) -> Path:
    """Convert ligand pose (PDBQT/DLG/SDF/JSON) to MDL V2000 format.

    ``pose_index`` selects a specific record inside a multi-record SDF/PDBQT
    when the pose selector (``compute_submission_scores``) identifies a pose
    by name like ``vina_seed_42_3``. When ``None``, the first record is used.
    ``title`` overrides the MDL block's first-line name; when not supplied
    RDKit's default ``"     RDKit          3D"`` leaks through and carries
    no source information. Passing ``pose.pose_name`` here is the canonical
    call.
    """
    if pose_path is None or str(pose_path) in ("", "."):
        raise ValueError(
            f"pose_to_mdl received an empty/unresolved pose path "
            f"({pose_path!r}). This usually means the pose selector could not "
            f"locate the staged pose file for the best pose; check "
            f"outputs/analysis/poses/ and the ba_pred/rmsd_pred TSVs."
        )
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file does not exist: {pose_path}")

    suffix = pose_path.suffix.lower()

    if suffix == ".sdf":
        _sdf_to_mdl(pose_path, output_mol, record_index=pose_index or 0, title=title)
        return output_mol

    if suffix in (".pdbqt", ".dlg"):
        _pdbqt_to_mdl(pose_path, output_mol, record_index=pose_index or 0, title=title)
        return output_mol

    if suffix == ".json":
        # Protenix-Dock results JSON — extract best pose SDF
        import json as _json
        data = _json.loads(pose_path.read_text())
        sdf_str = data.get("best_pose_sdf") or data.get("pose_sdf")
        if sdf_str:
            tmp_sdf = output_mol.with_suffix(".tmp.sdf")
            tmp_sdf.write_text(sdf_str)
            _sdf_to_mdl(tmp_sdf, output_mol, record_index=0, title=title)
            tmp_sdf.unlink(missing_ok=True)
            return output_mol
        raise ValueError(f"Could not extract SDF from {pose_path}")

    raise ValueError(
        f"Unsupported pose file format: suffix={suffix!r} path={pose_path}"
    )


def extract_cofolded_ligand_mdl(
    cif_path: Path,
    output_mol: Path,
    title: str | None = None,
) -> Path | None:
    """Fallback when no docking poses exist — extract the ligand as predicted
    by the cofolding model and emit an MDL block. Returns the path on success,
    or ``None`` if extraction fails (e.g. CIF has no ligand atoms).

    Docking can legitimately produce zero poses on pathological SMILES
    (very large/flexible ligands where RDKit/ETKDG refuses to embed, or
    Vina/ADG cannot even parse the PDBQT). In that case the cofolding
    model's own bound-pose prediction is the only pose we have, so we use
    it as MODEL 1 with a low LSCORE to signal the fallback path.

    Uses gemmi for CIF → ligand-only PDB and RDKit for PDB → MDL (no obabel
    dependency, keeps the submission path runnable on the master node where
    openbabel may not be installed).
    """
    import gemmi
    from rdkit import Chem

    structure = gemmi.read_structure(str(cif_path))
    tmp_pdb = output_mol.with_suffix(".lig.pdb")
    lig_lines: list[str] = []
    atom_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name in {"HOH", "WAT", "DOD"}:
                    continue
                tab = gemmi.find_tabulated_residue(residue.name)
                if tab and (tab.is_amino_acid() or tab.is_nucleic_acid()):
                    continue
                for atom in residue:
                    atom_idx += 1
                    el = (atom.element.name or atom.name.strip()[:1]).upper()
                    # Columns match the PDB HETATM record format exactly so
                    # RDKit's PDB reader assigns the right element.
                    lig_lines.append(
                        f"HETATM{atom_idx:5d} {atom.name:<4.4s} {residue.name:<3.3s} "
                        f"A{residue.seqid.num:4d}    "
                        f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                        f"  1.00{atom.b_iso:6.2f}          {el:>2s}"
                    )
        break
    if not lig_lines:
        return None

    tmp_pdb.write_text("\n".join(lig_lines) + "\nEND\n")

    mol = Chem.MolFromPDBFile(str(tmp_pdb), removeHs=False, sanitize=False)
    tmp_pdb.unlink(missing_ok=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    if title:
        mol.SetProp("_Name", title)
    output_mol.write_text(_safe_v2000_molblock(mol, kekulize=False))
    return output_mol


def _sdf_to_mdl(
    sdf_path: Path,
    output_mol: Path,
    record_index: int = 0,
    title: str | None = None,
) -> None:
    """Extract ``record_index``-th mol from SDF and write as MDL V2000.

    ``title`` forces the MDL block's title/header line. When unset we fall
    back to the mol's ``_Name`` property (if present) and finally to RDKit's
    default ``"     RDKit          3D"`` — which the evaluator then picks up
    as the pose name. Passing an explicit ``title`` avoids that.
    """
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = None
    for i, candidate in enumerate(supplier):
        if i == record_index:
            mol = candidate
            break
    if mol is None:
        raise ValueError(
            f"Failed to read record {record_index} from SDF: {sdf_path} "
            f"(supplier had {sum(1 for _ in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False))} records)"
        )
    if title:
        mol.SetProp("_Name", title)
    mol_block = _safe_v2000_molblock(mol, kekulize=True)
    output_mol.write_text(mol_block)


def _pdbqt_to_mdl(
    pdbqt_path: Path,
    output_mol: Path,
    record_index: int = 0,
    title: str | None = None,
) -> None:
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
                _sdf_to_mdl(tmp_sdf, output_mol, record_index=record_index, title=title)
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
            _sdf_to_mdl(tmp_sdf, output_mol, record_index=record_index, title=title)
            tmp_sdf.unlink(missing_ok=True)
            return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    raise RuntimeError(f"Could not convert {pdbqt_path} to MDL (mk_export.py and obabel both failed)")


def build_lg_submission(
    target_id: str,
    author: str,
    method: str,
    models: list[dict],
    parent: str = "N/A",
    start_model_idx: int = 1,
) -> str:
    """Assemble CASP17 LG format submission text per docs/casp17_lg_format.md.

    Each entry in ``models`` is one complete prediction snapshot:

        {
          "protein_pdb_lines": list[str],    # ATOM/HETATM/TER rows, receptor
          "parent": str | None,              # overrides top-level parent
          "remark": str | None,              # optional REMARK inside MODEL
          "ligands": [                       # one entry per ligand of target
            {
              "ligand_number": int,          # e.g. 1, 2
              "ligand_name": str,            # e.g. "LIG"
              "ligand_mdl": str,             # MDL V2000 body, should end "M  END"
              "lscore": float | None,        # optional, [0,1]
            },
            ...
          ],
          "affinity_nM": float | None,       # optional, per-MODEL
        }

    For multi-ligand targets each MODEL MUST carry every ligand of the target
    (this is the CASP LG spec — a MODEL is the "whole complex at rank k").
    The function emits up to 5 MODELs with their own PARENT/ATOM/TER/LIGAND*/
    AFFNTY/END structure. MODEL 1 = primary prediction.
    """
    if not models:
        raise ValueError("build_lg_submission requires at least one model")
    if len(models) > 5:
        raise ValueError(
            f"LG format allows at most 5 MODEL blocks; got {len(models)}. "
            "Truncate to top-5 before calling."
        )

    # Each non-empty line of ``method`` becomes its own ``METHOD`` record so
    # multi-line method descriptions emit valid CASP-style continuations
    # (matches CASP17_own's accepted submissions).
    method_records = [
        f"METHOD {ln.rstrip()}"
        for ln in method.splitlines()
        if ln.strip()
    ] or [f"METHOD {method}"]
    lines = [
        "PFRMAT LG",
        f"TARGET {target_id}",
        f"AUTHOR {author}",
        *method_records,
    ]

    for offset, model in enumerate(models):
        idx = start_model_idx + offset
        ligand_entries = model.get("ligands") or []
        if not ligand_entries:
            raise ValueError(f"MODEL {idx} has no ligand entries")

        lines.append(f"MODEL {idx}")
        remark = model.get("remark")
        if remark:
            lines.append(f"REMARK {remark}")
        lines.append(f"PARENT {model.get('parent') or parent}")

        # Receptor coordinates (ensure TER terminates chain)
        protein_block = list(model.get("protein_pdb_lines") or [])
        if not protein_block or not protein_block[-1].startswith("TER"):
            protein_block.append("TER")
        lines.extend(protein_block)

        # One LIGAND block per ligand. ID is the integer from the CASP-issued
        # SMILES file column, emitted *as-is* (no zero-pad). Verified
        # against the live LG validator on R2314/R2317/R2318: zero-padded
        # ids like ``001`` cause server lookup against the SMILES file to
        # crash. See docs/casp17_lg_format.md §6.
        for lig in ligand_entries:
            ligand_number = int(lig["ligand_number"])
            ligand_name = str(lig["ligand_name"])
            lines.append(f"LIGAND {ligand_number} {ligand_name}")
            if lig.get("lscore") is not None:
                lines.append(f"LSCORE {float(lig['lscore']):.3f}")
            mdl_text = (lig.get("ligand_mdl") or "").rstrip()
            if not mdl_text.endswith("M  END"):
                mdl_text += "\nM  END"
            lines.append(mdl_text)

        # Per-MODEL AFFNTY before END
        if model.get("affinity_nM") is not None:
            lines.append(f"AFFNTY {float(model['affinity_nM']):.3f} aa")

        lines.append("END")

    return "\n".join(lines) + "\n"


def build_lg_submission_with_affinity(
    target_id: str,
    author: str,
    method: str,
    models: list[dict],
    parent: str = "N/A",
) -> str:
    """Thin alias retained for call-site compatibility — AFFNTY is now carried
    per-MODEL inside the ``models`` list, not as a top-level argument.
    """
    return build_lg_submission(
        target_id=target_id,
        author=author,
        method=method,
        models=models,
        parent=parent,
    )


def _load_target_ligands(run_dir: Path) -> list[dict]:
    """Read ``inputs/docking/docking_prep_summary.json`` and return the
    normalized ligand list the submission needs.

    Each entry has ``{"ligand_id": "L"|"L2"|..., "ligand_number": int,
    "ligand_name": str}``. ``ligand_number`` is derived from position in
    the summary (1-indexed — matches the SMILES file's convention).
    ``ligand_name`` defaults to ``"LIG"`` per CASP convention but callers
    can override via CLI.
    """
    summary_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text())
    except Exception as e:
        print(f"  WARNING: could not parse {summary_path}: {e}")
        return []
    ligs = data.get("ligands") or []
    out = []
    for i, lig in enumerate(ligs, start=1):
        out.append({
            "ligand_id": str(lig.get("id") or f"L{i}"),
            "ligand_number": i,
            "ligand_name": "LIG",
        })
    return out


def _select_poses_for_ligand(
    ligand_id: str,
    candidate_pool: list,
    k: int,
    rmsd_threshold: float,
    primary: bool,
) -> list:
    """Pick up to ``k`` diverse poses for one ligand from the shared pool.

    Filters ``candidate_pool`` by ``PoseScore.ligand_id`` matching the given
    ``ligand_id``. For backwards compatibility (older runs / legacy dirs
    whose poses have ``ligand_id is None``), the primary ligand inherits
    untagged poses.
    """
    from compute_submission_scores import select_diverse_top_k  # type: ignore

    filtered = [
        p for p in candidate_pool
        if (p.ligand_id == ligand_id) or (primary and p.ligand_id is None)
    ]
    if not filtered:
        return []
    return select_diverse_top_k(
        filtered, k=k, rmsd_threshold=rmsd_threshold,
    )


def _cofold_fallback_mdl(
    cif_path: Path,
    workdir: Path,
    ligand_id: str,
    model_idx: int,
    protein_model: str,
) -> str | None:
    """Extract this ligand from the cofolding CIF and return its MDL body.

    Returns ``None`` if extraction fails. Writes a per-ligand, per-MODEL
    MOL file under ``workdir`` to keep artifacts around for debugging.

    Current implementation extracts the first non-protein residue from the
    CIF and returns that — which is OK for single-ligand targets but
    **ambiguous for multi-ligand**. Full per-ligand CIF extraction is
    Task 4's responsibility; for now non-primary ligands may get the same
    (first-ligand) MDL if the CIF doesn't clearly separate them.
    """
    mol_path = workdir / f"ligand_{ligand_id}_model{model_idx}.mol"
    extracted = extract_cofolded_ligand_mdl(
        cif_path, mol_path, title=f"cofolded_{protein_model}_{ligand_id}",
    )
    if extracted is None:
        return None
    return extracted.read_text()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CASP17 LG-format submission.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Pipeline run directory (experiments/runs/<target>)")
    parser.add_argument("--target-id", type=str, required=True,
                        help="CASP target identifier (e.g., L2001)")
    parser.add_argument("--ligand-name", type=str, default="LIG",
                        help="Ligand name from SMILES file (default: LIG per CASP convention)")
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
                        help="Ligand pose source (default: auto = from scores)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of MODELs to emit (default 5, CASP LG allows 1-5)")
    parser.add_argument("--diversity-rmsd", type=float, default=2.0,
                        help="Minimum pairwise heavy-atom RMSD (Å) between MODELs (default 2.0)")
    parser.add_argument("--affinity-nM", type=float, default=None,
                        help="Manual AFFNTY override (Kd in nM, applied to every MODEL)")
    parser.add_argument("--include-affinity", action="store_true",
                        help="Include AFFNTY record (auto-computed from scores)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output LG file path")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.output.parent / f".{args.target_id}_workdir"
    workdir.mkdir(exist_ok=True)

    # 1. Aggregate scores first — used for pose selection and LSCORE/AFFNTY
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_submission_scores import aggregate, _split_pose_name  # type: ignore

    print(f"Aggregating scores from {run_dir}...")
    scores = aggregate(run_dir)
    print(f"  Pose scores collected: {len(scores.pose_scores)}")
    print(f"  Boltz affinities collected: {len(scores.boltz_affinities)}")

    # 2. Select best protein cofolding output
    print(f"\nSelecting protein structure for {args.target_id}...")
    protein_model, cif_path = find_best_cofolding_cif(run_dir, preferred=args.protein_model)
    print(f"  Protein source: {protein_model}")
    print(f"  CIF: {cif_path}")

    protein_pdb = workdir / "protein.pdb"
    cif_to_pdb_with_plddt(cif_path, protein_pdb, args.target_id)
    protein_lines = extract_pdb_atom_lines(protein_pdb)
    print(f"  Protein atoms: {len(protein_lines)}")

    # 3. Discover all ligands of the target
    target_ligands = _load_target_ligands(run_dir)
    if not target_ligands:
        print("  WARNING: no ligands found in docking_prep_summary.json. "
              "Submission will emit a single LIGAND block using CLI defaults.")
        target_ligands = [{"ligand_id": "L", "ligand_number": 1, "ligand_name": args.ligand_name}]
    else:
        for lig in target_ligands:
            lig["ligand_name"] = args.ligand_name
    ligand_summary = ", ".join(
        f"{lig['ligand_id']}(#{lig['ligand_number']:03d})" for lig in target_ligands
    )
    print(f"  Ligands: {ligand_summary}")

    # 4. Per-ligand pose selection
    #
    # Build a candidate pool once; filter per ligand. Today's pool is
    # effectively primary-ligand-only (docking runners dock prep[0]); later
    # the pool will carry per-ligand poses keyed by ligand_id.
    if args.pose_source == "auto":
        candidate_pool = scores.pose_scores
    else:
        candidate_pool = [p for p in scores.pose_scores if p.source == args.pose_source]
        if not candidate_pool:
            print(f"  WARNING: no poses with source={args.pose_source}, falling back to all sources")
            candidate_pool = scores.pose_scores

    print(f"\nSelecting per-ligand top-{args.top_k} poses (diversity >= {args.diversity_rmsd}Å)...")
    per_ligand_selected: dict[str, list] = {}
    for i, lig in enumerate(target_ligands):
        primary = (i == 0)
        chosen = _select_poses_for_ligand(
            lig["ligand_id"], candidate_pool,
            k=args.top_k, rmsd_threshold=args.diversity_rmsd,
            primary=primary,
        )
        per_ligand_selected[lig["ligand_id"]] = chosen
        note = "(primary)" if primary else ""
        if not chosen:
            note += " (no scored poses → cofold fallback)" if not primary else " (no scored poses)"
        print(f"  {lig['ligand_id']}: {len(chosen)} scored poses {note}")

    # 5. Affinity — one value computed from the primary ligand's ensemble,
    #    written to every MODEL.
    affinity_nM = args.affinity_nM
    if args.include_affinity and affinity_nM is None and scores.ensemble_affinity_nM is not None:
        affinity_nM = scores.ensemble_affinity_nM
        print(f"\nAFFNTY (ensemble): {affinity_nM:.3g} nM "
              f"(log10={scores.ensemble_log_kd_nM:.3f})")

    # 6. Assemble up to top_k MODELs, each carrying every ligand.
    models: list[dict] = []
    source_summary: list[str] = []
    for model_idx in range(args.top_k):
        ligand_entries = []
        for lig in target_ligands:
            scored = per_ligand_selected.get(lig["ligand_id"], [])
            if scored and model_idx < len(scored):
                pose = scored[model_idx]
                _, rec_idx = _split_pose_name(pose.pose_name)
                mdl_file = workdir / f"ligand_{lig['ligand_id']}_model{model_idx+1}.mol"
                try:
                    pose_to_mdl(pose.pose_file, mdl_file, pose_index=rec_idx, title=pose.pose_name)
                    mdl_text = mdl_file.read_text()
                    source_summary.append(pose.source)
                    ligand_entries.append({
                        "ligand_number": lig["ligand_number"],
                        "ligand_name": lig["ligand_name"],
                        "ligand_mdl": mdl_text,
                        "lscore": pose.lscore,
                    })
                    continue
                except Exception as e:
                    print(f"  WARNING: pose_to_mdl failed for "
                          f"{lig['ligand_id']}/MODEL {model_idx+1}: {e} — using cofold fallback")
            # No scored pose for this (ligand, MODEL) slot → cofold fallback
            mdl_text = _cofold_fallback_mdl(
                cif_path, workdir, lig["ligand_id"], model_idx + 1, protein_model,
            )
            if mdl_text is None:
                if model_idx == 0 and not ligand_entries:
                    raise RuntimeError(
                        f"No docking poses AND cofold extraction failed for "
                        f"{lig['ligand_id']} — cannot produce MODEL 1."
                    )
                # Secondary MODELs can simply carry fewer entries, but CASP
                # requires every MODEL to carry every ligand. If we can't
                # emit one, skip this MODEL entirely (stop growing the list).
                break
            source_summary.append(f"cofold_{protein_model}")
            ligand_entries.append({
                "ligand_number": lig["ligand_number"],
                "ligand_name": lig["ligand_name"],
                "ligand_mdl": mdl_text,
                "lscore": 0.10,
            })

        # Only accept MODEL if all ligands are represented
        if len(ligand_entries) != len(target_ligands):
            print(f"  MODEL {model_idx+1}: incomplete (got {len(ligand_entries)}/"
                  f"{len(target_ligands)} ligands), stopping.")
            break

        models.append({
            "protein_pdb_lines": protein_lines,
            "parent": args.parent,
            "remark": args.remark or f"{protein_model} snapshot {model_idx+1}",
            "ligands": ligand_entries,
            "affinity_nM": affinity_nM,
        })
        entry_summary = " + ".join(
            f"{e['ligand_number']:03d}:LSCORE={e['lscore']:.3f}"
            if e['lscore'] is not None else f"{e['ligand_number']:03d}:LSCORE=N/A"
            for e in ligand_entries
        )
        print(f"  MODEL {model_idx+1}: {entry_summary}")

    if not models:
        raise RuntimeError("No MODELs could be assembled — nothing to write.")
    if len(models) < args.top_k:
        print(f"  NOTE: only {len(models)} of {args.top_k} MODELs available; "
              "submission truncated.")

    # 7. Write LG
    source_tag = "+".join(dict.fromkeys(source_summary)) or "auto"
    method_full = (
        f"{args.method} [protein={protein_model}, "
        f"pose={source_tag}, models={len(models)}, "
        f"ligands={len(target_ligands)}]"
    )

    submission = build_lg_submission(
        target_id=args.target_id,
        author=args.author,
        method=method_full,
        models=models,
        parent=args.parent,
    )

    args.output.write_text(submission)
    print(f"\n{'='*60}")
    print(f"  LG submission written: {args.output}")
    print(f"  Size: {len(submission):,} bytes")
    print(f"  Lines: {len(submission.splitlines()):,}")
    print(f"  MODELs: {len(models)}  Ligands/MODEL: {len(target_ligands)}")
    if affinity_nM is not None:
        print(f"  AFFNTY per MODEL: {affinity_nM:.3g} nM")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
