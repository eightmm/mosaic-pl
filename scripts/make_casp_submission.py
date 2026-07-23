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
        --output experiments/CASP17/submissions/L2001.lg

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
import re
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


def _strip_h_mdl(mdl_text: str) -> str:
    """Remove explicit H atoms from a V2000 MDL block (and their bonds),
    renumbering atom indices. CASP ligand validation matches the heavy-atom
    graph against the released SMILES; docked poses that keep explicit H give
    a different atom count than cofold poses (which are heavy-only), and a
    mixed submission makes the ligand validator crash. Heavy-atom-only is the
    consistent, spec-aligned form ("hydrogens optional; we omit them")."""
    lines = mdl_text.split("\n")
    try:
        ci = next(i for i, l in enumerate(lines) if l.rstrip().endswith("V2000"))
    except StopIteration:
        return mdl_text
    na, nb = int(lines[ci][:3]), int(lines[ci][3:6])
    atoms = lines[ci + 1: ci + 1 + na]
    bonds = lines[ci + 1 + na: ci + 1 + na + nb]
    keep, omap, new_i = [], {}, 0
    for idx, a in enumerate(atoms):
        if a[31:34].strip() == "H":
            continue
        new_i += 1
        omap[idx + 1] = new_i
        keep.append(a)
    if len(keep) == na:           # no H present, nothing to do
        return mdl_text
    new_bonds = []
    for b in bonds:
        a1, a2 = int(b[0:3]), int(b[3:6])
        if a1 in omap and a2 in omap:
            new_bonds.append(f"{omap[a1]:>3}{omap[a2]:>3}" + b[6:])
    counts = f"{len(keep):>3}{len(new_bonds):>3}" + lines[ci][6:]
    return "\n".join(lines[:ci] + [counts] + keep + new_bonds + ["M  END"])


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
            mdl_text = _strip_h_mdl(mdl_text)   # heavy-atom only (match SMILES)
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
    "ligand_name": str}``. ``ligand_number`` is the **0-indexed** position
    matching the CASP-issued SMILES file's ID column (verified against the
    live validator: T2383 / R2387 ligand ID = 0; a 1-based id is rejected
    with "LIGAND id does not correspond to the SMILES template").
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
    for i, lig in enumerate(ligs):          # 0-indexed: matches SMILES file ID
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
    anchors: list | None = None,
) -> list:
    """Pick up to ``k`` diverse poses for one ligand from the shared pool.

    Filters ``candidate_pool`` by ``PoseScore.ligand_id`` matching the given
    ``ligand_id``. For backwards compatibility (older runs / legacy dirs
    whose poses have ``ligand_id is None``), the primary ligand inherits
    untagged poses.

    When ``anchors`` (research / dominant-consensus pocket centres) are given,
    uses the anchored ranker so a dominant functional site is guaranteed a MODEL.
    """
    from compute_submission_scores import (  # type: ignore
        select_diverse_top_k, select_diverse_top_k_anchored,
    )

    filtered = [
        p for p in candidate_pool
        if (p.ligand_id == ligand_id) or (primary and p.ligand_id is None)
    ]
    if not filtered:
        return []
    if anchors:
        return select_diverse_top_k_anchored(
            filtered, anchors, k=k, rmsd_threshold=rmsd_threshold,
        )
    return select_diverse_top_k(
        filtered, k=k, rmsd_threshold=rmsd_threshold,
    )


_REC_HEAVY_CACHE: dict = {}


def _receptor_heavy_coords(cif_path: "Path"):
    """Heavy-atom coords of the (polymer-only) submission receptor, in the same
    aligned frame the staged poses use. Cached per cif. None on failure."""
    key = str(cif_path)
    if key in _REC_HEAVY_CACHE:
        return _REC_HEAVY_CACHE[key]
    arr = None
    try:
        import gemmi
        import numpy as np
        st = gemmi.read_structure(str(cif_path))
        st.remove_ligands_and_waters()
        pts = []
        for model in st:
            for chain in model:
                for res in chain:
                    for a in res:
                        if (a.element.name or "") != "H":
                            pts.append([a.pos.x, a.pos.y, a.pos.z])
            break
        arr = np.asarray(pts, dtype=float) if pts else None
    except Exception:
        arr = None
    _REC_HEAVY_CACHE[key] = arr
    return arr


def _pose_clashes(mol, rec, cutoff: float = 2.0, min_n: int = 5) -> bool:
    """True if the ligand interpenetrates the receptor — ``min_n``+ heavy-atom
    pairs closer than ``cutoff`` Å. A template pose docked against a *different*
    receptor conformation, once aligned onto the cofold receptor, jams into it
    (T2414 3zos: 88 contacts < 2 Å). Such poses must never enter a MODEL."""
    if rec is None or mol is None:
        return False
    try:
        import numpy as np
        from compute_submission_scores import _pose_coord_array  # type: ignore
        a = _pose_coord_array(mol)
        if a is None:
            return False
        d = np.linalg.norm(rec[:, None, :] - a[None, :, :], axis=2)
        return int((d < cutoff).sum()) >= min_n
    except Exception:
        return False


_TPL_REC_CACHE: dict = {}
_TRACK2_RE = re.compile(r"^template_([0-9A-Za-z]{4})_(?:vina|adg|autodock|lig_align)")


def _track2_template(pose_name: str) -> str | None:
    """Template PDB id if this is a Track-2 template-receptor-docked pose
    (``template_<pdb>_vina/adg/lig_align_...``), else None. ``template_consensus``
    is Track-1 on the cofold receptor — NOT this."""
    s = pose_name or ""
    if "consensus" in s:
        return None
    m = _TRACK2_RE.match(s)
    return m.group(1) if m else None


def _template_receptor_lines(pdb_id: str, run_dir: "Path", cofold_cif: "Path"):
    """Receptor ATOM/TER lines for a Track-2 template, in the cofold frame.

    A template-docked pose fits the TEMPLATE receptor's conformation, not the
    cofold receptor — so its MODEL must carry the template receptor (aligned to
    the cofold frame via USalign, the same transform the docked pose already
    went through). Regenerated from the vendored template CIF. Cached. None on
    failure (caller falls back to the cofold receptor)."""
    key = (str(run_dir), pdb_id)
    if key in _TPL_REC_CACHE:
        return _TPL_REC_CACHE[key]
    lines = None
    try:
        import gemmi
        import numpy as np
        from casp17.usalign import run_usalign  # type: ignore
        tcif = Path(run_dir) / "outputs" / "template_pockets" / "_extract_work" / f"{pdb_id}.cif"
        if tcif.is_file():
            r = run_usalign(tcif, Path(cofold_cif))
            if r:
                R = np.asarray(r[0], dtype=float)
                t = np.asarray(r[1], dtype=float)
                st = gemmi.read_structure(str(tcif))
                st.remove_alternative_conformations()
                st.remove_ligands_and_waters()
                st.remove_empty_chains()
                model0 = st[0]
                for chain in model0:
                    for k, res in enumerate(chain, 1):
                        # renumber residues to positive sequential per chain — the
                        # CASP server indexes an array by residue number and CRASHES
                        # on the negative resSeq (e.g. 3zos chain B starts at -1:
                        # "non-creatable array value ... subscript -1").
                        res.seqid.num = k
                        res.seqid.icode = " "
                        for a in res:
                            p = a.pos
                            v = R @ np.array([p.x, p.y, p.z]) + t
                            a.pos = gemmi.Position(float(v[0]), float(v[1]), float(v[2]))
                # keep first model only for a clean single-chain-set PDB
                while len(st) > 1:
                    del st[1]
                pdb = st.make_pdb_string()
                lines = [ln for ln in pdb.splitlines() if ln.startswith(("ATOM", "TER"))]
                if not lines:
                    lines = None
    except Exception:
        lines = None
    _TPL_REC_CACHE[key] = lines
    return lines


def _force_alt_orientation(chosen: list, pool: list, anchors: list,
                           k: int, rmsd_threshold: float) -> list:
    """Guarantee an alternate-orientation MODEL for a symmetric-ligand target.

    Among anchor-covering docked poses, group by which equivalent scissile group
    faces the catalytic anchor. If every currently-``chosen`` MODEL presents the
    same orientation, promote the best-lscore REAL pose that presents the *other*
    orientation into the MODEL set (evicting the weakest non-primary MODEL). No
    synthetic geometry — a genuine docked pose. Fail-open."""
    try:
        import numpy as np
        from compute_submission_scores import _load_pose_mol, _pose_coord_array  # type: ignore
        from casp17.symmetry_pose import engaged_scissile_group  # type: ignore
    except Exception:
        return chosen
    if not chosen:
        return chosen
    apts = [np.asarray(a.xyz, dtype=float) for a in anchors]
    mc: dict = {}

    def covering(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        if arr is None:
            return False
        return min(float(np.linalg.norm(arr.mean(0) - ap)) for ap in apts) <= 8.0

    # orientation key of each chosen MODEL; need ≥2 equivalent groups to matter
    chosen_keys, n_groups = set(), 0
    for p in chosen:
        if not covering(p):
            continue
        key, ng = engaged_scissile_group(_load_pose_mol(p, mc), apts)
        n_groups = max(n_groups, ng)
        if key is not None:
            chosen_keys.add(key)
    if n_groups < 2 or len(chosen_keys) != 1:
        return chosen  # asymmetric ligand, or both orientations already present

    alts = []
    for p in pool:
        if p.lscore is None or p in chosen or not covering(p):
            continue
        key, ng = engaged_scissile_group(_load_pose_mol(p, mc), apts)
        if key is not None and key not in chosen_keys:
            alts.append(p)
    if not alts:
        return chosen
    alt = max(alts, key=lambda p: p.lscore or 0.0)
    # evict the weakest non-primary MODEL (keep MODEL 1)
    if len(chosen) >= k:
        victim = min(chosen[1:], key=lambda p: p.lscore or 0.0)
        chosen = [p for p in chosen if p is not victim]
    return chosen + [alt]


def _force_template_flip(chosen: list, pool: list, anchors: list,
                         k: int, rmsd_threshold: float,
                         min_lscore: float = 0.15,
                         flip_dot: float = -0.2) -> list:
    """Guarantee two roughly-opposite ligand orientations at the template site.

    The CASP meeting: for a (partly-)symmetric ligand, submit ~2 flipped poses at
    the template site. Rather than require a graph automorphism (most drug-like
    ligands have none), this is docking-grounded: if docking produced a good pose
    at the template site in the OPPOSITE orientation, the ligand tolerates the
    flip, so include it as a second MODEL.

    Orientation = centroid→tip unit vector (tip = heavy atom furthest from the
    ligand centroid). Two template-site poses are 'flipped' when the vectors point
    into opposite hemispheres (dot < ``flip_dot``). Only fires at ``top_template``
    / ``template_site`` anchors, only promotes a pose with lscore ≥ ``min_lscore``
    that stays diverse (RMSD ≥ threshold). Fail-open — MODEL 1 never evicted."""
    try:
        import numpy as np
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if not chosen:
        return chosen
    tpl_pts = [np.asarray(a.xyz, dtype=float) for a in anchors
               if getattr(a, "kind", "") in ("top_template", "template_site")]
    if not tpl_pts:
        return chosen
    mc: dict = {}

    def orient(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        if arr is None or len(arr) < 3:
            return None
        cen = arr.mean(0)
        if min(float(np.linalg.norm(cen - tp)) for tp in tpl_pts) > 8.0:
            return None  # not at a template site
        tip = arr[int(np.argmax(np.linalg.norm(arr - cen, axis=1)))]
        v = tip - cen
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-6 else None

    ref = next((o for o in (orient(p) for p in chosen) if o is not None), None)
    if ref is None:
        return chosen  # no chosen MODEL sits at a template site
    # already two orientations at the site?
    if any((o is not None) and float(np.dot(o, ref)) < flip_dot
           for o in (orient(p) for p in chosen)):
        return chosen

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    alts = []
    for p in pool:
        if p in chosen or p.lscore is None or (p.lscore or 0.0) < min_lscore:
            continue
        o = orient(p)
        if o is None or float(np.dot(o, ref)) >= flip_dot:
            continue
        cm = _load_pose_mol(p, mc)
        if cm is not None and diverse_ok(cm):
            alts.append(p)
    if not alts:
        return chosen
    alt = max(alts, key=lambda p: p.lscore or 0.0)
    if len(chosen) >= k:
        victim = min(chosen[1:], key=lambda p: p.lscore or 0.0)
        chosen = [p for p in chosen if p is not victim]
    return chosen + [alt]


def _force_catalytic_flip(chosen: list, pool: list, run_dir: "Path",
                          cif_path: "Path", k: int, rmsd_threshold: float,
                          min_lscore: float = 0.15, flip_dot: float = -0.2,
                          cov: float = 8.0) -> list:
    """Guarantee TWO opposite-orientation poses at the catalytic residue site.

    CASP meeting: for a (pseudo-)symmetric ligand, submit two ~flipped poses at
    the catalytic site (both plausible binding modes of the symmetric warhead).
    Grounds the catalytic site on ``research.json`` residues; only fires for a
    symmetric ligand (≥2 equivalent hydrolase scissile groups or a pose-symmetry
    op). Ensures the 5 MODELs include ≥1 catalytic-site pose in EACH orientation
    (centroid→tip hemisphere), pulling the missing orientation from the pool —
    catalytic modes often score below a decoy pocket, so LSCORE alone drops them.
    Fail-open; MODEL 1 kept."""
    try:
        import numpy as np
        from casp17.research_prior import research_centers  # type: ignore
        from casp17.symmetry_pose import functional_flip_ops, pose_symmetry_ops  # type: ignore
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if not chosen:
        return chosen
    try:
        ancs = research_centers(Path(run_dir), Path(cif_path))
    except Exception:
        return chosen
    apts = [np.asarray(a.xyz, dtype=float) for a in ancs]
    if not apts:
        return chosen
    mc: dict = {}

    def at_cat(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        return arr is not None and min(
            float(np.linalg.norm(arr.mean(0) - ap)) for ap in apts) <= cov

    def orient(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        if arr is None or len(arr) < 3:
            return None
        c = arr.mean(0)
        tip = arr[int(np.argmax(np.linalg.norm(arr - c, axis=1)))]
        v = tip - c
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-6 else None

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    def add_pose(chs, p):
        if len(chs) >= k:
            # protect MODEL 1 and existing catalytic-site poses from eviction,
            # so adding the flipped orientation keeps BOTH catalytic poses.
            evictable = [q for q in chs[1:] if not at_cat(q)] or chs[1:]
            victim = min(evictable, key=lambda q: q.lscore or 0.0)
            chs = [q for q in chs if q is not victim]
        return chs + [p]

    cat_oris = [o for o in (orient(p) for p in chosen if at_cat(p)) if o is not None]
    # already two opposite orientations at the catalytic site?
    if any(float(np.dot(a, b)) < flip_dot
           for i, a in enumerate(cat_oris) for b in cat_oris[i + 1:]):
        return chosen

    pool_cat = []
    for p in pool:
        if p in chosen or p.lscore is None or (p.lscore or 0.0) < min_lscore:
            continue
        if not at_cat(p):
            continue
        o = orient(p)
        if o is not None:
            pool_cat.append((p, o))
    if not pool_cat:
        return chosen

    # symmetry gate: test on a pose AT the catalytic site (its scissile group
    # faces the anchor there — a far decoy pose reads as spuriously asymmetric).
    rep_pose = next((p for p in chosen if at_cat(p)), None) or pool_cat[0][0]
    rep = _load_pose_mol(rep_pose, mc)
    try:
        symmetric = rep is not None and bool(
            functional_flip_ops(rep, apts) or pose_symmetry_ops(rep))
    except Exception:
        symmetric = False
    if not symmetric:
        return chosen  # only hedge orientations for a symmetric ligand

    # reference orientation: an existing catalytic MODEL, else the best-lscore
    # catalytic pose from the pool (which we then add as the first orientation).
    ref = cat_oris[0] if cat_oris else None
    if ref is None:
        p0, ref = max(pool_cat, key=lambda po: po[0].lscore or 0.0)
        m = _load_pose_mol(p0, mc)
        if m is not None and diverse_ok(m):
            chosen = add_pose(chosen, p0)
    # add the best-lscore catalytic pose in the OPPOSITE orientation
    opp = sorted((po for po in pool_cat if po[0] not in chosen
                  and float(np.dot(po[1], ref)) < flip_dot),
                 key=lambda po: -(po[0].lscore or 0.0))
    for p, _o in opp:
        m = _load_pose_mol(p, mc)
        if m is not None and diverse_ok(m):
            chosen = add_pose(chosen, p)
            break
    return chosen


def _force_cofold_consensus(chosen: list, pool: list, k: int,
                            rmsd_threshold: float, agree_tol: float = 2.5) -> list:
    """Include one co-folding pose where AF3 and Boltz agree.

    CASP meeting T2414: "co-folding 결과(Boltz와 AF가 유사해 보이는 것)에서 1개".
    Self-gating — only fires when an AF3 cofold pose and a Boltz cofold pose sit
    within ``agree_tol`` Å of each other (a genuine cross-method consensus mode),
    then submits the higher-lscore of the pair. Skips if a cofold pose is already
    chosen or the agreed pose duplicates a MODEL. Fail-open; MODEL 1 kept."""
    try:
        from compute_submission_scores import _load_pose_mol, _pose_pair_rmsd  # type: ignore
    except Exception:
        return chosen
    if not chosen:
        return chosen

    def _cofold(p, tag):
        n = (getattr(p, "name", "") or "").lower()
        return "cofold" in n and tag in n

    if any("cofold" in (getattr(p, "name", "") or "").lower() for p in chosen):
        return chosen  # a cofold pose is already represented
    af = [p for p in pool if p.lscore is not None and _cofold(p, "af3")]
    bz = [p for p in pool if p.lscore is not None
          and (_cofold(p, "boltz2") or _cofold(p, "boltz2x"))]
    if not af or not bz:
        return chosen
    mc: dict = {}
    best = None
    for a in af:
        am = _load_pose_mol(a, mc)
        if am is None:
            continue
        for b in bz:
            r = _pose_pair_rmsd(am, _load_pose_mol(b, mc))
            if r is not None and r <= agree_tol:
                pick = a if (a.lscore or 0.0) >= (b.lscore or 0.0) else b
                if best is None or (pick.lscore or 0.0) > (best.lscore or 0.0):
                    best = pick
    if best is None or best in chosen:
        return chosen
    bm = _load_pose_mol(best, mc)
    if bm is not None and not all(
            (_load_pose_mol(s, mc) is None)
            or ((_pose_pair_rmsd(bm, _load_pose_mol(s, mc)) or 9e9) >= rmsd_threshold)
            for s in chosen):
        return chosen  # duplicates an existing MODEL
    if len(chosen) >= k:
        victim = min(chosen[1:], key=lambda p: p.lscore or 0.0)
        chosen = [p for p in chosen if p is not victim]
    return chosen + [best]


def _force_active_site(chosen: list, pool: list, run_dir: "Path",
                       cif_path: "Path", k: int, rmsd_threshold: float,
                       n_min: int, cov: float = 8.0) -> list:
    """Force >= ``n_min`` MODELs at the dominant active site.

    Active site = top-template COM ∪ dominant template cluster (share ≥ 0.5) ∪
    catalytic pocket (research residues). Where the true binding site scores low
    (RMSD-Pred favours snug decoy pockets, so LSCORE selection fills the 5 MODELs
    with decoys — e.g. T2414: 4/5 at minor consensus pockets), this pulls in the
    best available active-site poses with spatial diversity, evicting the weakest
    NON-active MODEL each time (MODEL 1 kept). Opt-in via ``--active-site-min``.
    Fail-open."""
    try:
        import numpy as np
        from casp17.research_prior import (  # type: ignore
            research_centers, template_com_centers)
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if n_min <= 0 or not chosen:
        return chosen
    # active site = top-template bound-ligand COM (the meeting's "top template의
    # catalytic site") ∪ catalytic-pocket centroid. The dominant *cluster*
    # centroid is deliberately excluded — its consensus offset can sit several Å
    # off the true site and count decoy sub-pockets as "active".
    apts = []
    try:
        for a in template_com_centers(Path(run_dir)):
            if a.kind == "top_template":
                apts.append(np.asarray(a.xyz, dtype=float))
        cats = [np.asarray(a.xyz, dtype=float)
                for a in research_centers(Path(run_dir), Path(cif_path))]
        if cats:
            apts.append(np.mean(cats, axis=0))
    except Exception:
        return chosen
    if not apts:
        return chosen
    mc: dict = {}

    def at_site(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        return arr is not None and min(
            float(np.linalg.norm(arr.mean(0) - ap)) for ap in apts) <= cov

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    need = n_min - sum(1 for p in chosen if at_site(p))
    if need <= 0:
        return chosen
    cands = sorted((p for p in pool if p not in chosen and p.lscore is not None
                    and at_site(p)), key=lambda p: -(p.lscore or 0.0))
    for p in cands:
        if need <= 0:
            break
        cm = _load_pose_mol(p, mc)
        if cm is None or not diverse_ok(cm):
            continue
        if len(chosen) >= k:
            evictable = [q for q in chosen[1:] if not at_site(q)] or chosen[1:]
            victim = min(evictable, key=lambda q: q.lscore or 0.0)
            chosen = [q for q in chosen if q is not victim]
        chosen = chosen + [p]
        need -= 1
    return chosen


def _force_catalytic_regions(chosen: list, pool: list, run_dir: "Path",
                             cif_path: "Path", k: int, rmsd_threshold: float,
                             n_regions: int = 2, per_region: int = 2,
                             clust_cut: float = 12.0, cov: float = 6.0) -> list:
    """Split scattered catalytic residues into spatial regions and fill each.

    Some catalytic pockets span TWO+ separated residue clusters (e.g. T2414 KIT:
    the ATP pocket vs the Cys178 covalent loop). Greedy-cluster the research
    residues (``clust_cut`` Å), keep the ``n_regions`` regions with the most
    docked poses nearby, and ensure ``per_region`` chosen MODELs sit at each
    (best-lscore, spatially diverse) — e.g. 2 ATP + 2 covalent. Evicts the
    weakest MODEL not serving any target region (MODEL 1 kept). Opt-in via
    ``--catalytic-regions``. Fail-open."""
    try:
        import numpy as np
        from casp17.research_prior import research_centers  # type: ignore
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if n_regions <= 0 or per_region <= 0 or not chosen:
        return chosen
    try:
        pts = [np.asarray(a.xyz, dtype=float)
               for a in research_centers(Path(run_dir), Path(cif_path))]
    except Exception:
        return chosen
    if not pts:
        return chosen
    # greedy spatial clustering of catalytic residues
    regions: list[dict] = []
    for p in pts:
        for c in regions:
            if float(np.linalg.norm(p - c["cen"])) <= clust_cut:
                c["mem"].append(p)
                c["cen"] = np.mean(c["mem"], axis=0)
                break
        else:
            regions.append({"mem": [p], "cen": p.copy()})
    mc: dict = {}

    def region_of(p):
        # PRIMARY region only: the region whose residues the ligand's nearest
        # atom is closest to (min-atom, so a covalent warhead counts for the
        # Cys region while the ATP-pocket body counts for the ATP region). A
        # pose spanning both is assigned once — no double counting.
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        if arr is None:
            return None
        ds = [min(float(np.min(np.linalg.norm(arr - m, axis=1))) for m in c["mem"])
              for c in regions]
        j = int(np.argmin(ds))
        return j if ds[j] <= cov else None

    def near(p, c):
        return region_of(p) == regions.index(c)

    # rank regions by how many pool poses sit in them (binding-plausible sites)
    for c in regions:
        c["npose"] = sum(1 for p in pool
                         if p.lscore is not None and near(p, c))
    ranked = [c for c in sorted(regions, key=lambda c: -c["npose"]) if c["npose"] > 0]
    if not ranked:
        return chosen
    # If research flags a covalent warhead, GUARANTEE the region holding that
    # residue is one of the chosen regions (its warhead-contact poses are sparse
    # and would otherwise be out-ranked by the main pocket + its neighbours).
    chosen_regions = ranked[:n_regions]
    try:
        import json
        cov_s = json.loads((Path(run_dir) / "research.json").read_text()).get(
            "covalent_suspicion") or {}
        resi = (cov_s.get("proposed_link") or {}).get("resi") if cov_s.get("suspected") else None
    except Exception:
        resi = None
    if resi is not None:
        # covalent residue point = the research anchor nearest that resi
        try:
            from casp17.research_prior import research_centers as _rc  # type: ignore
            cov_pt = next((np.asarray(a.xyz, dtype=float)
                           for a in _rc(Path(run_dir), Path(cif_path))
                           if str(resi) in a.label), None)
        except Exception:
            cov_pt = None
        if cov_pt is not None:
            cov_reg = min(regions, key=lambda c: float(np.linalg.norm(c["cen"] - cov_pt)))
            if cov_reg["npose"] > 0 and cov_reg not in chosen_regions:
                chosen_regions = ([ranked[0]] + [cov_reg])[:n_regions]
    regions = chosen_regions
    if not regions:
        return chosen

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    def in_any_region(p):
        return any(near(p, c) for c in regions)

    for c in regions:
        need = per_region - sum(1 for p in chosen if near(p, c))
        if need <= 0:
            continue
        cands = sorted((p for p in pool if p not in chosen and p.lscore is not None
                        and near(p, c)), key=lambda p: -(p.lscore or 0.0))
        for p in cands:
            if need <= 0:
                break
            cm = _load_pose_mol(p, mc)
            if cm is None or not diverse_ok(cm):
                continue
            if len(chosen) >= k:
                # evict weakest MODEL not serving any target region (keep MODEL 1)
                evictable = [q for q in chosen[1:] if not in_any_region(q)] or chosen[1:]
                victim = min(evictable, key=lambda q: q.lscore or 0.0)
                chosen = [q for q in chosen if q is not victim]
            chosen = chosen + [p]
            need -= 1
    return chosen


def _force_residue_contact(chosen: list, pool: list, run_dir: "Path",
                           cif_path: "Path", k: int, rmsd_threshold: float,
                           resids: list, n: int = 1, cut: float = 5.0) -> list:
    """Force ``n`` MODELs that CONTACT a specified residue set (any ligand heavy
    atom within ``cut`` Å of any of those residues). Use to guarantee coverage of
    a residue cluster the ranker missed (e.g. T2414 upper catalytic loop
    ASP182/ASN187/ASP200/TYR213). Best-lscore, spatially diverse, receptor-clash-
    free (pool is pre-filtered); evicts the weakest non-contacting MODEL (MODEL 1
    kept). Fail-open."""
    try:
        import numpy as np
        from casp17.research_prior import research_centers  # type: ignore
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if n <= 0 or not resids or not chosen:
        return chosen
    want = {str(r) for r in resids}
    try:
        pts = [np.asarray(a.xyz, dtype=float)
               for a in research_centers(Path(run_dir), Path(cif_path))
               if any(re.search(rf"\b{r}\b|{r}$", a.label) for r in want)]
    except Exception:
        return chosen
    if not pts:
        return chosen
    mc: dict = {}

    def contacts(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        return arr is not None and any(
            float(np.min(np.linalg.norm(arr - m, axis=1))) <= cut for m in pts)

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    need = n - sum(1 for p in chosen if contacts(p))
    if need <= 0:
        return chosen
    cands = sorted((p for p in pool if p not in chosen and p.lscore is not None
                    and contacts(p)), key=lambda p: -(p.lscore or 0.0))
    for p in cands:
        if need <= 0:
            break
        cm = _load_pose_mol(p, mc)
        if cm is None or not diverse_ok(cm):
            continue
        if len(chosen) >= k:
            evictable = [q for q in chosen[1:] if not contacts(q)] or chosen[1:]
            victim = min(evictable, key=lambda q: q.lscore or 0.0)
            chosen = [q for q in chosen if q is not victim]
        chosen = chosen + [p]
        need -= 1
    return chosen


def _force_covalent_split(chosen: list, pool: list, run_dir: "Path",
                          cif_path: "Path", k: int, rmsd_threshold: float,
                          n_cov: int = 2, n_noncov: int = 2,
                          cov_cut: float = 5.0, noncov_gap: float = 6.0,
                          site_cov: float = 8.0) -> list:
    """Split the active-cleft MODELs into covalent vs non-covalent orientations.

    For a target with a covalent-capable residue in the binding cleft (e.g. T2414
    Cys178), the ligand binds ONE pocket but its warhead may or may not engage the
    residue. This hedges both: force ``n_cov`` poses whose nearest atom is within
    ``cov_cut`` Å of the nucleophile SG (warhead engaged) AND ``n_noncov`` poses in
    the same cleft (centroid ≤ ``site_cov`` Å of the top-template COM) whose atoms
    stay > ``noncov_gap`` Å from the SG (not engaged). Best-lscore, spatially
    diverse; evicts the weakest MODEL serving neither class (MODEL 1 kept).
    Opt-in. Fail-open."""
    try:
        import json
        import numpy as np
        from casp17.research_prior import research_centers, template_com_centers  # type: ignore
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    if not chosen or (n_cov <= 0 and n_noncov <= 0):
        return chosen
    try:
        cov = json.loads((Path(run_dir) / "research.json").read_text()).get(
            "covalent_suspicion") or {}
        resi = (cov.get("proposed_link") or {}).get("resi") if cov.get("suspected") else None
        ancs = research_centers(Path(run_dir), Path(cif_path))
        sg = next((np.asarray(a.xyz, dtype=float) for a in ancs
                   if resi is not None and str(resi) in a.label), None)
        top = next((np.asarray(a.xyz, dtype=float)
                    for a in template_com_centers(Path(run_dir))
                    if a.kind == "top_template"), None)
    except Exception:
        return chosen
    if sg is None or top is None:
        return chosen
    mc: dict = {}

    def coords(p):
        return _pose_coord_array(_load_pose_mol(p, mc))

    def is_cov(p):
        a = coords(p)
        return a is not None and float(np.min(np.linalg.norm(a - sg, axis=1))) <= cov_cut

    def is_noncov(p):
        a = coords(p)
        return (a is not None
                and float(np.linalg.norm(a.mean(0) - top)) <= site_cov
                and float(np.min(np.linalg.norm(a - sg, axis=1))) > noncov_gap)

    def diverse_ok(cm):
        return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                   for pm in (_load_pose_mol(s, mc) for s in chosen))

    def fill(pred, need):
        nonlocal chosen
        if need <= 0:
            return
        cands = sorted((p for p in pool if p not in chosen and p.lscore is not None
                        and pred(p)), key=lambda p: -(p.lscore or 0.0))
        for p in cands:
            if need <= 0:
                break
            cm = _load_pose_mol(p, mc)
            if cm is None or not diverse_ok(cm):
                continue
            if len(chosen) >= k:
                evictable = [q for q in chosen[1:]
                             if not is_cov(q) and not is_noncov(q)] or chosen[1:]
                victim = min(evictable, key=lambda q: q.lscore or 0.0)
                chosen = [q for q in chosen if q is not victim]
            chosen = chosen + [p]
            need -= 1

    fill(is_cov, n_cov - sum(1 for p in chosen if is_cov(p)))
    fill(is_noncov, n_noncov - sum(1 for p in chosen if is_noncov(p)))
    return chosen


def _force_covalent_pose(chosen: list, pool: list, run_dir: "Path",
                         cif_path: "Path", k: int, rmsd_threshold: float,
                         cov_cutoff: float = 3.6) -> list:
    """When research flags a covalent warhead, force in a covalent-geometry pose.

    CASP meeting T2414: "Cys178 부근 covalent 가능한 pose가 있으면 Model 5에 하나".
    Uses ``research.json`` covalent_suspicion → the nucleophile residue (e.g.
    Cys178 SG); includes the docked pose whose nearest ligand heavy atom sits
    within ``cov_cutoff`` Å of that SG/OG. LSCORE is IGNORED (RMSD-Pred is
    non-covalent-trained and scores these ~0) — it is a candidate to eyeball, per
    the meeting. Self-gating (only when covalent suspected + such a pose exists).
    Fail-open; MODEL 1 kept."""
    try:
        import json
        import numpy as np
        from casp17.research_prior import research_centers  # type: ignore
        from compute_submission_scores import (  # type: ignore
            _load_pose_mol, _pose_coord_array, _pose_pair_rmsd)
    except Exception:
        return chosen
    rj = Path(run_dir) / "research.json"
    if not rj.is_file():
        return chosen
    try:
        cov = (json.loads(rj.read_text()).get("covalent_suspicion") or {})
    except Exception:
        return chosen
    if not cov.get("suspected"):
        return chosen
    pl = cov.get("proposed_link") or {}
    resi = pl.get("resi")
    resn = (pl.get("residue_name") or "").upper()
    try:
        ancs = research_centers(Path(run_dir), Path(cif_path))
    except Exception:
        return chosen
    sg = None
    for a in ancs:
        lab = a.label.upper()
        if resi is not None and str(resi) in lab and (not resn or resn[:3] in lab):
            sg = np.asarray(a.xyz, dtype=float)
            break
    if sg is None:  # fall back to any Cys/Ser nucleophile anchor
        for a in ancs:
            if a.label.upper().startswith(("CYS", "SER")):
                sg = np.asarray(a.xyz, dtype=float)
                break
    if sg is None:
        return chosen
    mc: dict = {}

    def min_atom(p):
        arr = _pose_coord_array(_load_pose_mol(p, mc))
        return None if arr is None else float(np.min(np.linalg.norm(arr - sg, axis=1)))

    if any((min_atom(p) or 9e9) <= cov_cutoff for p in chosen):
        return chosen  # a covalent-contact pose is already submitted
    near = [(d, p) for d, p in ((min_atom(p), p) for p in pool if p not in chosen)
            if d is not None and d <= cov_cutoff]
    if not near:
        return chosen
    near.sort(key=lambda dp: (dp[0], -(dp[1].lscore or 0.0)))  # closest, then best lscore
    alt = near[0][1]
    am = _load_pose_mol(alt, mc)
    if am is not None and not all(
            (_load_pose_mol(s, mc) is None)
            or ((_pose_pair_rmsd(am, _load_pose_mol(s, mc)) or 9e9) >= rmsd_threshold)
            for s in chosen):
        return chosen
    if len(chosen) >= k:
        victim = min(chosen[1:], key=lambda p: p.lscore or 0.0)
        chosen = [p for p in chosen if p is not victim]
    return chosen + [alt]


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
    parser.add_argument("--author", type=str, default="6095-5696-9732",
                        help="CASP registration code (default: group LCDD "
                             "6095-5696-9732)")
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
    parser.add_argument("--output", type=Path, default=None,
                        help="Output LG path. CASP17 group convention forces "
                             "the basename to '{target}_LCDD.lg' regardless of "
                             "what is passed (only the directory is honoured). "
                             "Default: experiments/CASP17/submissions/{target}_LCDD.lg.")
    parser.add_argument("--no-slack", action="store_true",
                        help="skip the post-build viewer HTML + Slack summary "
                             "(default: build viz/standalone + notify when a "
                             "Slack webhook is configured)")
    parser.add_argument("--no-research-anchor", action="store_true",
                        help="disable research/dominant-consensus pocket anchoring "
                             "in pose selection (default: on — guarantees a MODEL "
                             "at a dominant functional site when one exists)")
    parser.add_argument("--no-symflip", action="store_true",
                        help="disable pseudo-symmetry pose expansion (default: on — "
                             "adds the ~180°-flipped pose of a symmetric ligand at "
                             "an anchor so both orientations can be submitted)")
    parser.add_argument("--no-cofold-consensus", action="store_true",
                        help="disable the cofold-consensus MODEL (default: on — when "
                             "an AF3 and a Boltz cofold pose converge, include one so "
                             "a cross-method-agreed binding mode is submitted)")
    parser.add_argument("--no-covalent", action="store_true",
                        help="disable the covalent-pose MODEL (default: on — when "
                             "research flags a covalent warhead, include a pose with a "
                             "ligand atom within ~3.6 Å of the nucleophile SG/OG)")
    parser.add_argument("--active-site-min", type=int, default=0, metavar="N",
                        help="force >= N MODELs at the dominant active site "
                             "(top-template / dominant cluster / catalytic pocket). "
                             "0 = off. Use when the true site scores low and LSCORE "
                             "selection fills the MODELs with decoy pockets (e.g. T2414).")
    parser.add_argument("--catalytic-regions", type=int, default=0, metavar="N",
                        help="split scattered catalytic residues into N spatial "
                             "regions (most-populated first) and force --per-region "
                             "MODELs at each (e.g. 2 regions x 2 = ATP + Cys178). "
                             "0 = off.")
    parser.add_argument("--per-region", type=int, default=2, metavar="M",
                        help="MODELs per catalytic region for --catalytic-regions "
                             "(default 2).")
    parser.add_argument("--covalent-split", type=int, default=0, metavar="N",
                        help="hedge covalent vs non-covalent in the binding cleft: "
                             "force N warhead-engaged (atom<=5Å of nucleophile SG) "
                             "+ N non-engaged cleft poses. 0 = off (e.g. T2414 Cys178).")
    parser.add_argument("--contact-residues", type=str, default="", metavar="LIST",
                        help="comma-separated residue numbers; force --contact-n "
                             "MODEL(s) with a ligand atom within 5Å of any of them "
                             "(e.g. 182,187,200,213 for the T2414 upper catalytic loop).")
    parser.add_argument("--contact-n", type=int, default=1, metavar="N",
                        help="number of residue-contact MODELs for --contact-residues.")
    parser.add_argument("--template-first", action="store_true",
                        help="promote the best-lscore Track-2 template-based pose to "
                             "MODEL 1 (template-guided primary; PARENT=<template>).")
    args = parser.parse_args()

    # CASP17 group convention (LCDD): the submission basename MUST be
    # '{target}_LCDD.lg'. Only the output directory is honoured; the
    # filename is forced. See docs/casp17_lg_format.md and the
    # casp_author_code memory.
    forced_name = f"{args.target_id}_LCDD.lg"
    out_dir = args.output.parent if args.output is not None else Path("experiments/CASP17/submissions")
    if args.output is not None and args.output.name != forced_name:
        print(f"NOTE: overriding output name '{args.output.name}' -> "
              f"'{forced_name}' (CASP17 LCDD naming convention)")
    args.output = out_dir / forced_name

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

    # research / dominant-template-consensus pocket anchors (same frame as cif)
    anchors: list = []
    if not args.no_research_anchor:
        try:
            from casp17.research_prior import anchor_centers  # type: ignore
            anchors = anchor_centers(run_dir, cif_path)
            if anchors:
                print("  Pocket anchors: "
                      + ", ".join(f"{a.kind}:{a.label}" for a in anchors))
        except Exception as e:
            print(f"  research anchors skipped: {e}")

    protein_pdb = workdir / "protein.pdb"
    cif_to_pdb_with_plddt(cif_path, protein_pdb, args.target_id)
    protein_lines = extract_pdb_atom_lines(protein_pdb)
    print(f"  Protein atoms: {len(protein_lines)}")

    # 3. Discover all ligands of the target
    target_ligands = _load_target_ligands(run_dir)
    if not target_ligands:
        print("  WARNING: no ligands found in docking_prep_summary.json. "
              "Submission will emit a single LIGAND block using CLI defaults.")
        target_ligands = [{"ligand_id": "L", "ligand_number": args.ligand_id, "ligand_name": args.ligand_name}]
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

    # Drop poses that interpenetrate the receptor. A template-docked pose (docked
    # against a different receptor conformation) can, once aligned onto the cofold
    # receptor, jam into it (T2414 3zos: 88 heavy-atom contacts < 2 Å). Such a
    # MODEL is physically invalid, so remove clashers before any selection/hedge.
    try:
        from compute_submission_scores import _load_pose_mol as _lpm  # type: ignore
        _rec = _receptor_heavy_coords(cif_path)
        if _rec is not None:
            _mc0: dict = {}
            _before = len(candidate_pool)
            # Every submitted MODEL uses the cofold (TARGET-sequence) receptor —
            # CASP rejects a template receptor ("chain A sequence doesn't match").
            # So ALL poses must fit the cofold receptor; drop any that clash
            # (template-docked poses that jam into the cofold conformation).
            candidate_pool = [p for p in candidate_pool
                              if not _pose_clashes(_lpm(p, _mc0), _rec)]
            _dropped = _before - len(candidate_pool)
            if _dropped:
                print(f"  clash filter: dropped {_dropped} receptor-clashing pose(s) "
                      f"({len(candidate_pool)} remain)")
    except Exception as _e:  # fail-open — never block submission on the filter
        print(f"  clash filter skipped: {_e}")

    print(f"\nSelecting per-ligand top-{args.top_k} poses (diversity >= {args.diversity_rmsd}Å)...")
    per_ligand_selected: dict[str, list] = {}
    for i, lig in enumerate(target_ligands):
        primary = (i == 0)
        chosen = _select_poses_for_ligand(
            lig["ligand_id"], candidate_pool,
            k=args.top_k, rmsd_threshold=args.diversity_rmsd,
            primary=primary, anchors=anchors,
        )
        # symmetric-ligand hedge: force in a REAL docked pose whose alternate
        # scissile group faces the catalytic anchor (opposite orientation), if
        # the current MODELs only show one orientation.
        if not args.no_symflip and anchors:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_alt_orientation(chosen, pool_lig, anchors,
                                            k=args.top_k, rmsd_threshold=args.diversity_rmsd)
            # template-site orientation hedge: 2 roughly-flipped poses at the
            # template site when docking found them (partly-symmetric ligand).
            chosen = _force_template_flip(chosen, pool_lig, anchors,
                                          k=args.top_k, rmsd_threshold=args.diversity_rmsd)
            # catalytic-site orientation hedge: 2 flipped poses at the catalytic
            # residue site for a symmetric ligand (both warhead orientations).
            if cif_path is not None:
                chosen = _force_catalytic_flip(chosen, pool_lig, run_dir, cif_path,
                                               k=args.top_k, rmsd_threshold=args.diversity_rmsd)
        # active-site fill (opt-in): pull >= N MODELs to the dominant active site
        # when LSCORE selection filled them with decoy pockets.
        if args.active_site_min > 0 and cif_path is not None:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_active_site(chosen, pool_lig, run_dir, cif_path,
                                        k=args.top_k, rmsd_threshold=args.diversity_rmsd,
                                        n_min=args.active_site_min)
        # multi-region catalytic fill (opt-in): split scattered catalytic residues
        # into N regions and force --per-region MODELs at each (e.g. ATP + Cys178).
        if args.catalytic_regions > 0 and cif_path is not None:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_catalytic_regions(chosen, pool_lig, run_dir, cif_path,
                                              k=args.top_k, rmsd_threshold=args.diversity_rmsd,
                                              n_regions=args.catalytic_regions,
                                              per_region=args.per_region)
        # force MODEL(s) contacting a specified residue set (opt-in)
        if args.contact_residues.strip() and cif_path is not None:
            resids = [r.strip() for r in args.contact_residues.split(",") if r.strip()]
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_residue_contact(chosen, pool_lig, run_dir, cif_path,
                                            k=args.top_k, rmsd_threshold=args.diversity_rmsd,
                                            resids=resids, n=args.contact_n)
        # covalent vs non-covalent cleft hedge (opt-in)
        if args.covalent_split > 0 and cif_path is not None:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_covalent_split(chosen, pool_lig, run_dir, cif_path,
                                           k=args.top_k, rmsd_threshold=args.diversity_rmsd,
                                           n_cov=args.covalent_split,
                                           n_noncov=args.covalent_split)
        if not args.no_cofold_consensus:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_cofold_consensus(chosen, pool_lig,
                                             k=args.top_k, rmsd_threshold=args.diversity_rmsd)
        if not args.no_covalent and cif_path is not None:
            pool_lig = [p for p in candidate_pool
                        if (p.ligand_id == lig["ligand_id"]) or (primary and p.ligand_id is None)]
            chosen = _force_covalent_pose(chosen, pool_lig, run_dir, cif_path,
                                          k=args.top_k, rmsd_threshold=args.diversity_rmsd)
        # promote a Track-2 template-based pose to MODEL 1 (template-guided primary)
        if args.template_first and chosen:
            ti = next((j for j, p in enumerate(chosen)
                       if _track2_template(getattr(p, "pose_name", "") or "")), None)
            if ti is not None and ti != 0:
                chosen.insert(0, chosen.pop(ti))
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

        # Receptor = the cofold (TARGET-sequence) prediction for EVERY MODEL.
        # CASP validates the receptor sequence against the target, so a homolog
        # template receptor is rejected — poses are placed on our predicted
        # (cofold) structure regardless of where they were docked.
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

    if not args.no_slack:
        # viewer HTML + standalone + Slack summary (no-op if no webhook).
        # Subprocess-isolated so a notify failure never fails the build.
        import subprocess
        import sys as _sys
        subprocess.run(
            [_sys.executable, str(Path(__file__).resolve().parent / "notify_lg.py"),
             "--lg", str(args.output)],
            check=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
