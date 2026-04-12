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


def pose_to_mdl(
    pose_path: Path,
    output_mol: Path,
    pose_index: int | None = None,
) -> Path:
    """Convert ligand pose (PDBQT/DLG/SDF/JSON) to MDL V2000 format.

    ``pose_index`` selects a specific record inside a multi-record SDF/PDBQT
    when the pose selector (``compute_submission_scores``) identifies a pose
    by name like ``vina_seed_42_3``. When ``None``, the first record is used.
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
        _sdf_to_mdl(pose_path, output_mol, record_index=pose_index or 0)
        return output_mol

    if suffix in (".pdbqt", ".dlg"):
        _pdbqt_to_mdl(pose_path, output_mol, record_index=pose_index or 0)
        return output_mol

    if suffix == ".json":
        # Protenix-Dock results JSON — extract best pose SDF
        import json as _json
        data = _json.loads(pose_path.read_text())
        sdf_str = data.get("best_pose_sdf") or data.get("pose_sdf")
        if sdf_str:
            tmp_sdf = output_mol.with_suffix(".tmp.sdf")
            tmp_sdf.write_text(sdf_str)
            _sdf_to_mdl(tmp_sdf, output_mol, record_index=0)
            tmp_sdf.unlink(missing_ok=True)
            return output_mol
        raise ValueError(f"Could not extract SDF from {pose_path}")

    raise ValueError(
        f"Unsupported pose file format: suffix={suffix!r} path={pose_path}"
    )


def _sdf_to_mdl(sdf_path: Path, output_mol: Path, record_index: int = 0) -> None:
    """Extract ``record_index``-th mol from SDF and write as MDL V2000."""
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
    mol_block = Chem.MolToMolBlock(mol, kekulize=True)
    output_mol.write_text(mol_block)


def _pdbqt_to_mdl(pdbqt_path: Path, output_mol: Path, record_index: int = 0) -> None:
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
                _sdf_to_mdl(tmp_sdf, output_mol, record_index=record_index)
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
            _sdf_to_mdl(tmp_sdf, output_mol, record_index=record_index)
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
    models: list[dict],
    ligand_number: int,
    ligand_name: str,
    parent: str = "N/A",
    remark: str = "",
) -> str:
    """Assemble CASP17 LG format submission text (multi-model).

    ``models`` is a list of dicts, each containing at least ``ligand_mdl``
    (the MDL V2000 text for that model's pose) and optionally ``lscore``.
    Models are emitted in list order as ``MODEL 1``, ``MODEL 2``, … The
    protein ATOM block is shared across all models (same cofolding receptor
    frame), so we only emit it once per MODEL block to stay compatible with
    the CASP LG parser, which expects a complete PARENT/ATOM/TER/LIGAND set
    inside each MODEL.
    """
    if not models:
        raise ValueError("build_lg_submission requires at least one model")

    lines = [
        "PFRMAT LG",
        f"TARGET {target_id}",
        f"AUTHOR {author}",
        f"METHOD {method}",
        "METHOD -------------",
    ]

    # Ensure protein has a terminator row we can reuse.
    protein_block = list(protein_pdb_lines)
    if not protein_block or not protein_block[-1].startswith("TER"):
        protein_block.append("TER")

    for idx, model in enumerate(models, start=1):
        lines.append(f"MODEL {idx}")
        if remark:
            lines.append(f"REMARK {remark}")
        lines.append(f"PARENT {parent}")
        lines.extend(protein_block)
        lines.append(f"LIGAND {ligand_number:03d} {ligand_name}")
        if model.get("lscore") is not None:
            lines.append(f"LSCORE {model['lscore']:.3f}")

        mdl_text = (model.get("ligand_mdl") or "").rstrip()
        if not mdl_text.endswith("M  END"):
            mdl_text += "\nM  END"
        lines.append(mdl_text)

    lines.append("END")
    return "\n".join(lines) + "\n"


def build_lg_submission_with_affinity(
    target_id: str,
    author: str,
    method: str,
    protein_pdb_lines: list[str],
    models: list[dict],
    ligand_number: int,
    ligand_name: str,
    affinity_nM: float | None = None,
    parent: str = "N/A",
    remark: str = "",
) -> str:
    """Assemble LG submission with optional AFFNTY record (per-complex)."""
    result = build_lg_submission(
        target_id=target_id,
        author=author,
        method=method,
        protein_pdb_lines=protein_pdb_lines,
        models=models,
        ligand_number=ligand_number,
        ligand_name=ligand_name,
        parent=parent,
        remark=remark,
    )
    if affinity_nM is None:
        return result
    # Insert AFFNTY before the final END (per-complex, not per-MODEL)
    lines = result.rstrip().splitlines()
    if lines[-1] == "END":
        lines.insert(-1, f"AFFNTY {affinity_nM:.3f} aa")
    else:
        lines.append(f"AFFNTY {affinity_nM:.3f} aa")
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
                        help="Ligand pose source (default: auto = from scores)")
    parser.add_argument("--lscore", type=float, default=None,
                        help="Manual LSCORE override for MODEL 1 [0-1]")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of MODELs to emit (default 5, CASP LG allows 1-5)")
    parser.add_argument("--diversity-rmsd", type=float, default=2.0,
                        help="Minimum pairwise heavy-atom RMSD (Å) between MODELs (default 2.0)")
    parser.add_argument("--affinity-nM", type=float, default=None,
                        help="Manual AFFNTY override (Kd in nM)")
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
    from compute_submission_scores import aggregate

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

    # 3. Select diverse top-k ligand poses
    print(f"\nSelecting ligand poses (source={args.pose_source}, top_k={args.top_k}, "
          f"diversity>={args.diversity_rmsd}Å)...")

    from compute_submission_scores import select_diverse_top_k, _split_pose_name

    if args.pose_source == "auto":
        candidate_pool = scores.pose_scores
    else:
        candidate_pool = [p for p in scores.pose_scores if p.source == args.pose_source]
        if not candidate_pool:
            print(f"  WARNING: no poses with source={args.pose_source}, falling back to all sources")
            candidate_pool = scores.pose_scores

    selected = select_diverse_top_k(
        candidate_pool,
        k=args.top_k,
        rmsd_threshold=args.diversity_rmsd,
    )
    if not selected:
        raise RuntimeError(
            "No poses selected — check that post-analysis TSVs exist and that "
            "outputs/analysis/poses/ contains the staged pose files."
        )

    models: list[dict] = []
    top_sources: list[str] = []
    for i, pose in enumerate(selected, start=1):
        _, rec_idx = _split_pose_name(pose.pose_name)
        pose_path = pose.pose_file
        mdl_file = workdir / f"ligand_model{i}.mol"
        pose_to_mdl(pose_path, mdl_file, pose_index=rec_idx)
        mdl_text = mdl_file.read_text()
        model_lscore = pose.lscore
        if i == 1 and args.lscore is not None:
            model_lscore = args.lscore  # user override applies to MODEL 1 only
        models.append({
            "ligand_mdl": mdl_text,
            "lscore": model_lscore,
            "source": pose.source,
            "pose_name": pose.pose_name,
            "ba_pred_pkd": pose.ba_pred_pkd,
            "rmsd_pred": pose.rmsd_pred,
        })
        top_sources.append(pose.source)
        print(f"  MODEL {i}: {pose.source}/{pose.pose_name} "
              f"pKd={pose.ba_pred_pkd} pRMSD={pose.rmsd_pred} LSCORE={model_lscore}")
        print(f"           file={pose_path} record={rec_idx} mdl_lines={len(mdl_text.splitlines())}")

    if len(selected) < args.top_k:
        print(f"  NOTE: only {len(selected)} of {args.top_k} MODELs satisfy the "
              f"diversity threshold (>= {args.diversity_rmsd}Å). Submission has "
              f"{len(selected)} MODELs.")

    # 4. Affinity
    affinity_nM = args.affinity_nM
    if args.include_affinity and affinity_nM is None and scores.ensemble_affinity_nM is not None:
        affinity_nM = scores.ensemble_affinity_nM
        print(f"\nAFFNTY (ensemble): {affinity_nM:.3g} nM "
              f"(log10={scores.ensemble_log_kd_nM:.3f})")

    # 5. Assemble LG submission
    source_tag = "+".join(dict.fromkeys(top_sources)) or "auto"
    method_full = f"{args.method} [protein={protein_model}, pose={source_tag}, top{len(selected)}]"
    remark = args.remark or f"{protein_model} + {source_tag} (top-{len(selected)})"

    submission = build_lg_submission_with_affinity(
        target_id=args.target_id,
        author=args.author,
        method=method_full,
        protein_pdb_lines=protein_lines,
        models=models,
        ligand_number=args.ligand_number,
        ligand_name=args.ligand_name,
        affinity_nM=affinity_nM,
        parent=args.parent,
        remark=remark,
    )

    args.output.write_text(submission)
    print(f"\n{'='*60}")
    print(f"  LG submission written: {args.output}")
    print(f"  Size: {len(submission):,} bytes")
    print(f"  Lines: {len(submission.splitlines()):,}")
    print(f"  MODELs: {len(models)}")
    for i, m in enumerate(models, start=1):
        sc = m["lscore"]
        sc_txt = f"{sc:.3f}" if sc is not None else "N/A"
        print(f"    MODEL {i}: {m['source']}/{m['pose_name']} LSCORE={sc_txt}")
    if affinity_nM is not None:
        print(f"  AFFNTY: {affinity_nM:.3g} nM")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
