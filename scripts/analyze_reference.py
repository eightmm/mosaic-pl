#!/usr/bin/env python3
"""Analyse predicted ligand poses against an experimental reference.

For a pipeline run directory that has been through post-analysis, compute
heavy-atom RMSD between every predicted pose and an experimental ligand
(crystal structure). Reports:

- `reference_analysis.json` — full per-pose breakdown
- `reference_analysis.tsv` — one row per pose with actual + predicted metrics
- stdout summary — correlation between RMSD-Pred pRMSD and actual RMSD,
  top-5 diverse selection evaluation, protein CA RMSD sanity check.

Usage:
    python analyze_reference.py \
        --run-dir experiments/runs/L2001_input \
        --reference-ligand experiments/casp16_test/L2000/references/L2000_prepared/L2001/ligand_761_C_1.pdb \
        --reference-protein experiments/casp16_test/L2000/references/L2000_prepared/L2001/protein_aligned.pdb \
        --diversity-rmsd 2.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _setup_sys_path():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def load_reference_ligand(pdb_path: Path, template_sdf: Path):
    """Load a crystal ligand from PDB and assign bond orders from a template SDF.

    PDB files have no bond information. We rely on a known-good template
    (one of the staged docking-output SDFs for the same ligand) and either
    AssignBondOrdersFromTemplate or a manual xyz overlay as a fallback.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    template_mol = None
    supplier = Chem.SDMolSupplier(str(template_sdf), removeHs=True, sanitize=True)
    for m in supplier:
        if m is not None:
            template_mol = m
            break
    if template_mol is None:
        raise RuntimeError(f"Could not load ligand template from {template_sdf}")

    # Strategy 1: Let RDKit guess bonds (proximityBonding=True) then fix with
    # AssignBondOrdersFromTemplate. Works for typical ligands but can trip on
    # proximity overestimation with dense crystal coordinates.
    raw = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False, proximityBonding=True)
    if raw is not None:
        try:
            ref = AllChem.AssignBondOrdersFromTemplate(template_mol, raw)
            return ref, template_mol
        except Exception:
            pass

    # Strategy 2: manual xyz overlay — parse HETATM/ATOM coordinates from the
    # PDB, build a conformer directly on the template, assuming atom order
    # inside the HETATM block matches the template heavy-atom order. This is
    # the way most PDB ligand files are written (same ordering as the PDB
    # chemical component dictionary).
    ref = _overlay_pdb_xyz_on_template(pdb_path, template_mol)
    return ref, template_mol


def _overlay_pdb_xyz_on_template(pdb_path: Path, template_mol):
    """Last-resort fallback: take heavy-atom xyz from the PDB in file order and
    stamp it onto a copy of the template Mol. Works only when the PDB atom
    order matches the template heavy-atom order (common for reference PDBs
    that were built from the same chemical component template)."""
    from rdkit import Chem

    heavy_xyz: list[tuple[float, float, float]] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # Element is cols 77-78 but may be blank — fall back to first letter
        # of the atom name, skipping a leading digit.
        elt = line[76:78].strip()
        if not elt:
            name = line[12:16].strip()
            name_no_digits = name.lstrip("0123456789")
            elt = name_no_digits[:1] if name_no_digits else ""
        if elt.upper() == "H":
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        heavy_xyz.append((x, y, z))

    n_tmpl = template_mol.GetNumAtoms()
    if len(heavy_xyz) != n_tmpl:
        raise RuntimeError(
            f"PDB has {len(heavy_xyz)} heavy atoms but template has {n_tmpl} — "
            "cannot overlay without a bond-order fit."
        )

    ref = Chem.Mol(template_mol)
    conf = Chem.Conformer(n_tmpl)
    for i, (x, y, z) in enumerate(heavy_xyz):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    ref.RemoveAllConformers()
    ref.AddConformer(conf, assignId=True)
    return ref


def compute_pose_rmsd(pose_mol, ref_mol) -> float | None:
    """Heavy-atom RMSD between a pose and the reference ligand, no alignment.

    RDKit's `rdMolAlign.CalcRMS` finds the best symmetry-aware substructure
    mapping and computes positional RMSD without rotating/translating, which
    is what we want because the reference is already in (approximately) the
    same receptor frame as the predicted poses.
    """
    from rdkit.Chem import rdMolAlign

    if pose_mol is None or ref_mol is None:
        return None
    try:
        return float(rdMolAlign.CalcRMS(pose_mol, ref_mol))
    except Exception:
        return None


def _collect_ca(path: Path) -> dict:
    """Return ``{(chain, resi): (x, y, z)}`` for every CA in ``path``."""
    rows: dict[tuple[str, int], tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16:17].strip()
        if alt not in ("", "A"):
            continue
        chain = line[21:22]
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        rows.setdefault((chain, resi), (x, y, z))
    return rows


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "MSE": "M",
}


def _collect_ca_per_chain(path: Path):
    """Return ``{chain: list[(resi, one_letter, xyz)]}`` in file order."""
    per: dict[str, list[tuple[int, str, tuple[float, float, float]]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16:17].strip()
        if alt not in ("", "A"):
            continue
        chain = line[21:22]
        resname = line[17:20].strip()
        one = _AA3TO1.get(resname)
        if one is None:
            continue
        try:
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        per.setdefault(chain, []).append((resi, one, (x, y, z)))
    return per


def _find_best_chain_offset(ref_seq: str, pred_seq: str) -> tuple[int, int]:
    """Sliding-offset alignment: find the integer offset that maximises the
    number of matching residues between ``ref_seq`` and ``pred_seq``. Returns
    ``(offset, matches)`` where offset is how many positions to shift the
    reference sequence to the right so that ``ref[i]`` aligns with
    ``pred[i + offset]``."""
    best = (0, 0)
    n_ref = len(ref_seq)
    n_pred = len(pred_seq)
    if n_ref == 0 or n_pred == 0:
        return best
    for offset in range(-n_ref + 1, n_pred):
        matches = 0
        for i in range(n_ref):
            j = i + offset
            if 0 <= j < n_pred and ref_seq[i] == pred_seq[j]:
                matches += 1
        if matches > best[1]:
            best = (offset, matches)
    return best


def _match_ca_by_sequence(ref_pdb: Path, pred_pdb: Path):
    """Pair up CA coordinates between ref and pred by residue sequence (not by
    residue number) to tolerate crystallographic numbering offsets and
    multi-chain asymmetric units.

    Returns ``(ref_xyz, pred_xyz, info)`` where the first two are
    parallel-index lists of matched coordinates and ``info`` is a dict with
    per-chain metadata.
    """
    ref_per = _collect_ca_per_chain(ref_pdb)
    pred_per = _collect_ca_per_chain(pred_pdb)

    ref_xyz: list[tuple[float, float, float]] = []
    pred_xyz: list[tuple[float, float, float]] = []
    chain_info: list[dict] = []

    # Prefer the predicted chain with the most CAs (usually only one for
    # single-protein inputs), and match it against the best-scoring ref chain.
    pred_chains_sorted = sorted(pred_per.items(), key=lambda kv: -len(kv[1]))
    for pred_chain, pred_list in pred_chains_sorted:
        pred_seq = "".join(tri for _, tri, _ in pred_list)
        best_chain = None
        best_offset = 0
        best_matches = 0
        for ref_chain, ref_list in ref_per.items():
            ref_seq = "".join(tri for _, tri, _ in ref_list)
            offset, matches = _find_best_chain_offset(ref_seq, pred_seq)
            if matches > best_matches:
                best_matches = matches
                best_offset = offset
                best_chain = ref_chain
        if best_chain is None or best_matches == 0:
            continue
        ref_list = ref_per[best_chain]
        for i, (_, r_one, r_xyz) in enumerate(ref_list):
            j = i + best_offset
            if 0 <= j < len(pred_list) and pred_list[j][1] == r_one:
                ref_xyz.append(r_xyz)
                pred_xyz.append(pred_list[j][2])
        chain_info.append({
            "pred_chain": pred_chain,
            "ref_chain": best_chain,
            "offset": best_offset,
            "matches": best_matches,
        })
        # Use only the first (largest) predicted chain — CASP targets are
        # typically single protein; additional chains tend to be copies.
        break

    return ref_xyz, pred_xyz, {"chain_pairs": chain_info}


def protein_ca_rmsd(pred_pdb: Path, ref_pdb: Path) -> float | None:
    """Positional CA-only RMSD between two PDBs without alignment.

    Used as a sanity flag: if this is large, the reference protein is not in
    the predicted frame and raw ligand RMSD is misleading — alignment via
    `compute_protein_alignment` should be applied.
    """
    a = _collect_ca(pred_pdb)
    b = _collect_ca(ref_pdb)
    common = set(a) & set(b)
    if not common:
        return None
    acc = 0.0
    for key in common:
        ax, ay, az = a[key]
        bx, by, bz = b[key]
        acc += (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
    return math.sqrt(acc / len(common))


def compute_protein_alignment(ref_pdb: Path, pred_pdb: Path):
    """Kabsch alignment: rigid transform that maps REF CA coords onto PRED CA.

    The reference PDB often uses crystallographic residue numbering (e.g.
    starts at 21) or contains multiple chain copies, while the predicted
    protein numbers residues from 1. We therefore match residues by their
    1-letter amino-acid sequence using a sliding-offset alignment instead of
    exact ``(chain, resi)`` equality. Returns ``(R, t, rmsd_after, n_common,
    info)`` where ``info`` captures the chain pairs + offsets used.
    """
    import numpy as np

    ref_list, pred_list, info = _match_ca_by_sequence(ref_pdb, pred_pdb)
    if not ref_list:
        return None, None, None, 0, {"chain_pairs": []}

    ref_xyz = np.array(ref_list, dtype=float)
    pred_xyz = np.array(pred_list, dtype=float)

    ref_c = ref_xyz.mean(axis=0)
    pred_c = pred_xyz.mean(axis=0)
    X = ref_xyz - ref_c
    Y = pred_xyz - pred_c

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = pred_c - R @ ref_c

    aligned = (R @ ref_xyz.T).T + t
    diff = aligned - pred_xyz
    n_common = len(ref_list)
    rmsd_after = float(np.sqrt((diff * diff).sum() / n_common))
    return R, t, rmsd_after, n_common, info


def apply_transform_to_mol(mol, R, t):
    """Return a copy of ``mol`` whose conformer has been rotated and translated
    by ``(R, t)``. Works in-place on the copy's conformer."""
    import numpy as np
    from rdkit import Chem

    out = Chem.Mol(mol)
    if out.GetNumConformers() == 0:
        return out
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z], dtype=float)
        v2 = R @ v + t
        conf.SetAtomPosition(i, (float(v2[0]), float(v2[1]), float(v2[2])))
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 <= 0 or sy2 <= 0:
        return None
    return sxy / math.sqrt(sx2 * sy2)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        paired = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(paired):
            j = i
            while j + 1 < len(paired) and values[paired[j + 1]] == values[paired[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based ranks averaged across ties
            for k in range(i, j + 1):
                out[paired[k]] = avg
            i = j + 1
        return out

    return pearson(ranks(xs), ranks(ys))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse predicted poses vs experimental reference.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-ligand", type=Path, required=True,
                        help="Reference ligand PDB (experimental crystal).")
    parser.add_argument("--reference-protein", type=Path, default=None,
                        help="Reference protein PDB for CA RMSD sanity check.")
    parser.add_argument("--diversity-rmsd", type=float, default=2.0,
                        help="Diversity threshold used in top-k selection (default 2.0 Å).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--success-rmsd", type=float, default=2.0,
                        help="RMSD (Å) under which a pose is counted as a hit (default 2.0).")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    _setup_sys_path()
    from compute_submission_scores import (  # noqa: E402
        aggregate,
        select_diverse_top_k,
        _split_pose_name,
        _load_pose_mol,
    )

    print(f"Aggregating scores from {run_dir}...")
    scores = aggregate(run_dir)
    print(f"  Pose scores collected: {len(scores.pose_scores)}")
    if not scores.pose_scores:
        print("ERROR: no pose scores available — run post-analysis first.")
        return 1

    # Template SDF: first staged pose from Vina (or any available); required for
    # AssignBondOrdersFromTemplate.
    template_candidates = [
        p for p in scores.pose_scores
        if p.pose_file and p.pose_file.exists() and p.pose_file.suffix.lower() == ".sdf"
    ]
    if not template_candidates:
        print("ERROR: no staged SDF poses found to use as a bond-order template.")
        return 1
    template_sdf = template_candidates[0].pose_file
    print(f"  Using {template_sdf.name} as bond-order template")

    ref_mol, template_mol = load_reference_ligand(args.reference_ligand, template_sdf)
    print(f"  Reference ligand heavy atoms: {ref_mol.GetNumAtoms()}")
    print(f"  Template heavy atoms:        {template_mol.GetNumAtoms()}")

    # Protein alignment: the reference crystal and the predicted cofolding model
    # typically live in different coordinate frames. To make ligand RMSD
    # meaningful, we Kabsch-align the ref protein onto the predicted protein
    # (using common CA atoms) and transport the ref ligand by the same
    # rigid transform before comparing it to the predicted poses.
    protein_ca_before = None
    protein_ca_after = None
    alignment_n = 0
    alignment_info: dict = {}
    if args.reference_protein is not None:
        pred_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
        if pred_pdb.exists():
            protein_ca_before = protein_ca_rmsd(pred_pdb, args.reference_protein)
            R, t, protein_ca_after, alignment_n, alignment_info = compute_protein_alignment(
                ref_pdb=args.reference_protein,
                pred_pdb=pred_pdb,
            )
            if R is not None:
                ref_mol = apply_transform_to_mol(ref_mol, R, t)
                pairs = alignment_info.get("chain_pairs", [])
                pair_txt = ", ".join(
                    f"pred[{p['pred_chain']}]↔ref[{p['ref_chain']}] (offset {p['offset']}, {p['matches']} match)"
                    for p in pairs
                )
                before_txt = f"{protein_ca_before:.2f}" if protein_ca_before is not None else "N/A"
                print(f"  Protein Kabsch alignment: {alignment_n} matched CA, "
                      f"before={before_txt} Å, after={protein_ca_after:.2f} Å [{pair_txt}]")
            else:
                print("  Protein alignment: no common residues — skipping transform")
        else:
            print(f"  Predicted receptor.pdb not found at {pred_pdb} — skipping CA RMSD")

    # Per-pose RMSD
    mol_cache: dict = {}
    rows = []
    for pose in scores.pose_scores:
        mol = _load_pose_mol(pose, mol_cache)
        actual_rmsd = compute_pose_rmsd(mol, ref_mol)
        rows.append({
            "source": pose.source,
            "pose_name": pose.pose_name,
            "pose_file": str(pose.pose_file),
            "ba_pred_pkd": pose.ba_pred_pkd,
            "rmsd_pred": pose.rmsd_pred,
            "lscore": pose.lscore,
            "actual_rmsd": actual_rmsd,
        })

    scored_rows = [r for r in rows if r["actual_rmsd"] is not None]
    if not scored_rows:
        print("ERROR: could not compute RMSD for any pose (template/ref mismatch?).")
        return 1

    # Best-by-actual and best-by-LSCORE
    best_actual = min(scored_rows, key=lambda r: r["actual_rmsd"])
    best_lscore = max(
        (r for r in scored_rows if r["lscore"] is not None),
        key=lambda r: r["lscore"],
        default=None,
    )

    # Top-k diverse from compute_submission_scores — how do they fare?
    top_selected = select_diverse_top_k(
        scores.pose_scores,
        k=args.top_k,
        rmsd_threshold=args.diversity_rmsd,
    )
    top_rows = []
    by_key = {(r["source"], r["pose_name"]): r for r in rows}
    for pose in top_selected:
        key = (pose.source, pose.pose_name)
        r = by_key.get(key)
        if r is not None:
            top_rows.append(r)

    hit_threshold = args.success_rmsd
    n_hits = sum(1 for r in scored_rows if r["actual_rmsd"] is not None and r["actual_rmsd"] <= hit_threshold)
    top_hit_rank = None
    for i, r in enumerate(top_rows, start=1):
        if r["actual_rmsd"] is not None and r["actual_rmsd"] <= hit_threshold:
            top_hit_rank = i
            break

    # pRMSD ↔ actual RMSD correlation
    xs = [r["rmsd_pred"] for r in scored_rows if r["rmsd_pred"] is not None]
    ys = [r["actual_rmsd"] for r in scored_rows if r["rmsd_pred"] is not None]
    pearson_r = pearson(xs, ys)
    spearman_rho = spearman(xs, ys)

    lscore_xs = [r["lscore"] for r in scored_rows if r["lscore"] is not None]
    lscore_ys = [r["actual_rmsd"] for r in scored_rows if r["lscore"] is not None]
    pearson_lscore = pearson(lscore_xs, lscore_ys)
    spearman_lscore = spearman(lscore_xs, lscore_ys)

    summary = {
        "run_dir": str(run_dir),
        "reference_ligand": str(args.reference_ligand),
        "reference_protein": str(args.reference_protein) if args.reference_protein else None,
        "protein_ca_rmsd_before_alignment": protein_ca_before,
        "protein_ca_rmsd_after_alignment": protein_ca_after,
        "protein_alignment_n_residues": alignment_n,
        "protein_alignment_info": alignment_info,
        "n_poses_total": len(rows),
        "n_poses_scored": len(scored_rows),
        "n_poses_within_hit_threshold": n_hits,
        "hit_threshold_A": hit_threshold,
        "best_actual": {
            "source": best_actual["source"],
            "pose_name": best_actual["pose_name"],
            "actual_rmsd_A": best_actual["actual_rmsd"],
            "rmsd_pred": best_actual["rmsd_pred"],
            "lscore": best_actual["lscore"],
            "ba_pred_pkd": best_actual["ba_pred_pkd"],
        },
        "best_by_lscore": {
            "source": best_lscore["source"],
            "pose_name": best_lscore["pose_name"],
            "actual_rmsd_A": best_lscore["actual_rmsd"],
            "rmsd_pred": best_lscore["rmsd_pred"],
            "lscore": best_lscore["lscore"],
        } if best_lscore else None,
        "top_k_diverse": {
            "k": args.top_k,
            "diversity_rmsd": args.diversity_rmsd,
            "selected": [
                {
                    "rank": i,
                    "source": r["source"],
                    "pose_name": r["pose_name"],
                    "lscore": r["lscore"],
                    "rmsd_pred": r["rmsd_pred"],
                    "actual_rmsd_A": r["actual_rmsd"],
                }
                for i, r in enumerate(top_rows, start=1)
            ],
            "first_hit_rank_within_topk": top_hit_rank,
        },
        "correlations": {
            "pRMSD_vs_actual_pearson": pearson_r,
            "pRMSD_vs_actual_spearman": spearman_rho,
            "lscore_vs_actual_pearson": pearson_lscore,
            "lscore_vs_actual_spearman": spearman_lscore,
        },
    }

    output_json = args.output_json or (run_dir / "outputs" / "analysis" / "reference_analysis.json")
    output_tsv = args.output_tsv or (run_dir / "outputs" / "analysis" / "reference_analysis.tsv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    with open(output_tsv, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["source", "pose_name", "lscore", "ba_pred_pkd", "rmsd_pred", "actual_rmsd_A", "pose_file"])
        for r in rows:
            w.writerow([
                r["source"],
                r["pose_name"],
                f"{r['lscore']:.4f}" if r["lscore"] is not None else "",
                f"{r['ba_pred_pkd']:.4f}" if r["ba_pred_pkd"] is not None else "",
                f"{r['rmsd_pred']:.4f}" if r["rmsd_pred"] is not None else "",
                f"{r['actual_rmsd']:.4f}" if r["actual_rmsd"] is not None else "",
                r["pose_file"],
            ])

    # stdout summary
    print()
    print("=" * 60)
    print(f"  Poses scored: {len(scored_rows)} / {len(rows)}")
    print(f"  Poses within {hit_threshold:.1f} Å of truth: {n_hits}")
    print(f"  Best actual: {best_actual['source']}/{best_actual['pose_name']} "
          f"→ {best_actual['actual_rmsd']:.2f} Å (pRMSD={best_actual['rmsd_pred']}, LSCORE={best_actual['lscore']})")
    if best_lscore:
        print(f"  Best by LSCORE: {best_lscore['source']}/{best_lscore['pose_name']} "
              f"→ {best_lscore['actual_rmsd']:.2f} Å")
    print(f"  Top-{args.top_k} diverse (selected for submission):")
    for i, r in enumerate(top_rows, start=1):
        print(f"    {i}. {r['source']}/{r['pose_name']}: "
              f"LSCORE={r['lscore']}, pRMSD={r['rmsd_pred']}, actual={r['actual_rmsd']:.2f} Å")
    if top_hit_rank is not None:
        print(f"  First hit inside top-{args.top_k}: rank {top_hit_rank}")
    else:
        print(f"  No top-{args.top_k} pose under {hit_threshold:.1f} Å")
    print(f"  Correlations (pRMSD vs actual): pearson={pearson_r}, spearman={spearman_rho}")
    print(f"  Correlations (LSCORE vs actual): pearson={pearson_lscore}, spearman={spearman_lscore}")
    if protein_ca_before is not None:
        after_txt = f"{protein_ca_after:.2f}" if protein_ca_after is not None else "N/A"
        print(f"  Protein CA RMSD: before={protein_ca_before:.2f} Å, after_alignment={after_txt} Å")
    print(f"  JSON: {output_json}")
    print(f"  TSV:  {output_tsv}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
