"""For each L1xxx target, compare two pose-selection strategies against the
experimental crystal structure:

    A. pRMSD ascending (RMSD-Pred primary output)
    B. LSCORE descending (= prob_gt_2A ascending)

For each strategy, pick the top-5 poses (no diversity filter — raw ranking),
compute the actual heavy-atom RMSD of each pose to the experimental ligand,
and report top-1 + best-of-top-5 for both strategies side by side.

Pool: vina / autodock_gpu / protenix_dock poses and any cofolding pool
SDFs staged by run_post_analysis.py (cofold_boltz2, cofold_boltz2x,
cofold_protenix, cofold_af3). TSVs are read from analysis/ and keyed by
pose name. Pose files come from analysis/poses/{stem}.sdf +
protenix_dock/poses.sdf.

Usage:
    /home/jaemin/project/CASP17/.venvs/protenix-dock/bin/python compare_selection.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import (  # noqa: E402
    kabsch,
    parse_ca,
    pose_rmsd,
    reassign_bonds,
    transform_mol,
)

ROOT = Path(__file__).parent
EXPER = ROOT / "L1000_prepared"
RUNS = ROOT.parents[1] / "runs"
AFF_CSV = ROOT / "L1000_exper_affinity.csv"


def load_smiles_and_exper_affinity() -> dict[str, dict]:
    out = {}
    with AFF_CSV.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tid = row["Target ID"].strip()
            try:
                ba = float(row["binding_affinity"])
            except ValueError:
                ba = math.nan
            out[tid] = {"smiles": row["ligand_smiles"].strip(), "ba_kcal_mol": ba}
    return out


# ---------- ligand loader (experimental crystal ligand) ----------

def _strip_and_reassign(mol: Chem.Mol | None, template: Chem.Mol) -> Chem.Mol | None:
    """Remove explicit hydrogens (if any) then reassign bond orders from
    the SMILES template. Returns None if atom counts can't be reconciled."""
    if mol is None:
        return None
    if mol.GetNumAtoms() != template.GetNumAtoms():
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return None
    return reassign_bonds(mol, template)


def load_exper(tdir: Path, template) -> Chem.Mol | None:
    candidates = sorted(tdir.glob("ligand_*.pdb"))
    if not candidates:
        return None
    raw = Chem.MolFromPDBFile(str(candidates[0]), removeHs=False, sanitize=False)
    if raw is None:
        return None
    try:
        raw = Chem.RemoveHs(raw, sanitize=False)
    except Exception:
        return None
    return reassign_bonds(raw, template)


# ---------- TSV parsing ----------

def parse_ba_tsv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["Name"].strip()] = float(row["pKd"])
            except (ValueError, KeyError):
                pass
    return out


def parse_rmsd_tsv(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["Name"].strip()] = (float(row["pRMSD"]), float(row["Is_Above_2A"]))
            except (ValueError, KeyError):
                pass
    return out


# ---------- pose pool ----------

def build_pose_pool(run_dir: Path, template: Chem.Mol) -> list[dict]:
    """Return list of {name, tool, mol} for all pool poses.

    Pose name convention matches BA-Pred/RMSD-Pred output: {stem}_{record_idx}
    for list-based inputs (vina/adg/cofold_*) and a custom name from
    mol.SetProp("_Name") for single-file inputs (pxdock).
    """
    poses: list[dict] = []
    analysis_poses = run_dir / "outputs" / "analysis" / "poses"

    # vina / adg / cofold multi-record SDFs
    patterns = [
        ("vina", "vina_seed_*.sdf"),
        ("autodock_gpu", "autodock_gpu_seed_*.sdf"),
        ("cofold_boltz2", "cofold_boltz2.sdf"),
        ("cofold_boltz2x", "cofold_boltz2x.sdf"),
        ("cofold_protenix", "cofold_protenix.sdf"),
        ("cofold_af3", "cofold_af3.sdf"),
    ]
    for tool, glob in patterns:
        for sdf in sorted(analysis_poses.glob(glob)):
            stem = sdf.stem
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
            for i, mol in enumerate(suppl):
                if mol is None:
                    continue
                name = f"{stem}_{i}"
                fixed = _strip_and_reassign(mol, template)
                if fixed is None:
                    continue
                poses.append({"name": name, "tool": tool, "mol": fixed})

    # protenix_dock: single multi-record SDF (poses.sdf)
    pxdock_sdf = run_dir / "outputs" / "protenix_dock" / "poses.sdf"
    if pxdock_sdf.exists():
        suppl = Chem.SDMolSupplier(str(pxdock_sdf), removeHs=True, sanitize=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"poses_{i}"
            # pxdock records embed the mapped template already; still force bond
            # reassignment to match the canonical template used everywhere else.
            fixed = _strip_and_reassign(mol, template)
            if fixed is None:
                continue
            poses.append({"name": name, "tool": "protenix_dock", "mol": fixed})

    return poses


# ---------- per-target comparison ----------

def evaluate_target(tid: str, smiles: str) -> dict:
    run_dir = RUNS / f"{tid}_input"
    tdir = EXPER / tid
    if not run_dir.exists() or not tdir.exists():
        return {"target": tid, "status": "missing dirs"}

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return {"target": tid, "status": "bad template"}

    ref_lig = load_exper(tdir, template)
    if ref_lig is None:
        return {"target": tid, "status": "exper ligand parse failed"}

    # Protein alignment: docking receptor → exper
    rec_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not rec_pdb.exists():
        return {"target": tid, "status": "no docking receptor"}
    pred_ca = parse_ca(rec_pdb)
    ref_ca = parse_ca(tdir / "protein_aligned.pdb")
    common = sorted(set(pred_ca) & set(ref_ca))
    if len(common) < 50:
        return {"target": tid, "status": "too few common CAs"}
    P = np.array([pred_ca[r] for r in common])
    Q = np.array([ref_ca[r] for r in common])
    R, t, _ = kabsch(P, Q)

    # Build pose pool
    pool = build_pose_pool(run_dir, template)
    if not pool:
        return {"target": tid, "status": "empty pose pool"}

    # Score table
    analysis = run_dir / "outputs" / "analysis"
    ba_by_name: dict[str, float] = {}
    rmsd_by_name: dict[str, tuple[float, float]] = {}
    for ba_tsv in analysis.glob("ba_pred_*.tsv"):
        ba_by_name.update(parse_ba_tsv(ba_tsv))
    for r_tsv in analysis.glob("rmsd_pred_*.tsv"):
        rmsd_by_name.update(parse_rmsd_tsv(r_tsv))

    # Compute actual RMSD once per pose (transform → rmsd)
    enriched = []
    for p in pool:
        p_mol_t = transform_mol(p["mol"], R, t, inplace=False)
        actual = pose_rmsd(p_mol_t, ref_lig)
        prmsd, prob = rmsd_by_name.get(p["name"], (None, None))
        enriched.append({
            "name": p["name"],
            "tool": p["tool"],
            "actual_rmsd": round(actual, 3) if actual == actual else None,
            "pRMSD": prmsd,
            "prob": prob,
            "pKd": ba_by_name.get(p["name"]),
        })

    scored = [e for e in enriched if e["pRMSD"] is not None and e["actual_rmsd"] is not None]

    def pick(sort_key) -> list[dict]:
        return sorted(scored, key=sort_key)[:5]

    top5_prmsd = pick(lambda e: (e["pRMSD"], e["prob"]))
    top5_lscore = pick(lambda e: (e["prob"], e["pRMSD"]))  # prob asc = LSCORE desc
    # Product score: pRMSD * prob (both "lower is better"), tie-break prob asc
    top5_prod = pick(lambda e: (e["pRMSD"] * e["prob"], e["prob"]))

    best_overall = min((e["actual_rmsd"] for e in enriched if e["actual_rmsd"] is not None), default=None)

    def summarize(top5):
        if not top5:
            return None, None, [], []
        t1 = top5[0]["actual_rmsd"]
        best5 = round(min(e["actual_rmsd"] for e in top5), 3)
        return t1, best5, [e["actual_rmsd"] for e in top5], [e["name"] for e in top5]

    pt1, pb5, pactual, pnames = summarize(top5_prmsd)
    lt1, lb5, lactual, lnames = summarize(top5_lscore)
    xt1, xb5, xactual, xnames = summarize(top5_prod)

    return {
        "target": tid,
        "n_pool": len(enriched),
        "n_scored": len(scored),
        "overall_best_rmsd": round(best_overall, 3) if best_overall is not None else None,
        "prmsd_top1": pt1, "prmsd_best5": pb5,
        "prmsd_top5_actual": pactual, "prmsd_top5_names": pnames,
        "lscore_top1": lt1, "lscore_best5": lb5,
        "lscore_top5_actual": lactual, "lscore_top5_names": lnames,
        "prod_top1": xt1, "prod_best5": xb5,
        "prod_top5_actual": xactual, "prod_top5_names": xnames,
    }


def main():
    table = load_smiles_and_exper_affinity()
    results = []
    header = (f"{'target':<8} | {'pool':>4} | {'best':>5} | "
              f"{'pR t1':>6} {'t5':>5} | {'LS t1':>6} {'t5':>5} | {'PxL t1':>6} {'t5':>5}")
    print(header)
    print("-" * len(header))
    for tid in sorted(table.keys()):
        r = evaluate_target(tid, table[tid]["smiles"])
        results.append(r)
        if "status" in r:
            print(f"{tid:<8} | {r['status']}")
            continue
        print(f"{tid:<8} | {r['n_pool']:>4} | {r['overall_best_rmsd']:>5} | "
              f"{r['prmsd_top1']:>6} {r['prmsd_best5']:>5} | "
              f"{r['lscore_top1']:>6} {r['lscore_best5']:>5} | "
              f"{r['prod_top1']:>6} {r['prod_best5']:>5}")

    ok = [r for r in results if "overall_best_rmsd" in r]
    def count_under(key, thr):
        return sum(1 for r in ok if r[key] is not None and r[key] < thr)
    print()
    for key, label in [
        ("prmsd_top1", "pRMSD      top1"),
        ("prmsd_best5", "pRMSD      best5"),
        ("lscore_top1", "LSCORE     top1"),
        ("lscore_best5", "LSCORE     best5"),
        ("prod_top1", "pRMSD*prob top1"),
        ("prod_best5", "pRMSD*prob best5"),
        ("overall_best_rmsd", "overall best"),
    ]:
        print(f"{label} < 2Å: {count_under(key, 2.0)}/{len(ok)}")

    out = ROOT / "compare_selection.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
