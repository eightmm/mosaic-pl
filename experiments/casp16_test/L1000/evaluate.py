"""Evaluate L1000 (CASP16 chymase) LG submissions against experimental crystals.

For each target with a submission file:
  - Extract per-MODEL protein CA + ligand (MDL block) from the LG file
  - Kabsch-align predicted protein onto experimental protein (common residues)
  - Apply the transform to the predicted ligand
  - Reassign bond orders on both predicted and experimental ligands from the
    SMILES template (CASP target SMILES) so ``GetBestRMS`` sees matching
    atom graphs — without this RDKit's ``PDBBlock`` / ``MolBlock`` paths
    leave ligands with inconsistent bond orders and the RMS match fails.
  - Compute symmetry-aware heavy-atom RMSD to the experimental ligand
  - Attach experimental IC50 / binding_affinity for comparison
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

# Make src/casp17 importable without a package install (the L1000
# experiments folder is not installed; we simply add the repo's src/
# directory onto sys.path).
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import (  # noqa: E402
    kabsch,
    mol_from_mdl_body,
    parse_ca,
    pose_rmsd,
    reassign_bonds,
    transform_mol,
)
from casp17.lg_format import parse_lg  # noqa: E402

ROOT = Path(__file__).parent
EXPER_DIR = ROOT / "L1000_prepared"
SUB_DIR = ROOT.parents[1] / "submissions"
AFFINITY_CSV = ROOT / "L1000_exper_affinity.csv"


# ---------- table loaders ----------

def load_affinity_table() -> dict[str, dict]:
    table: dict[str, dict] = {}
    with AFFINITY_CSV.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tid = row["Target ID"].strip()
            try:
                ic50 = float(row["CHYMASE_IC50_COLI_NH"])
            except ValueError:
                ic50 = math.nan
            try:
                ba_kcal = float(row["binding_affinity"])
            except ValueError:
                ba_kcal = math.nan
            table[tid] = {
                "ic50_uM": ic50,
                "ba_kcal_mol": ba_kcal,
                "smiles": row.get("ligand_smiles", "").strip(),
            }
    return table


# ---------- experimental ligand loader ----------

def load_experimental_ligand(pdb_path: Path, template: Chem.Mol) -> Chem.Mol | None:
    """Load a CASP-prepared crystal ligand PDB and reassign bond orders
    from the SMILES template. ``sanitize=False`` tolerates the
    non-standard valences that show up in some crystal ligands; we then
    strip hydrogens manually because RDKit's ``removeHs`` flag is a
    no-op when sanitization is disabled."""
    raw = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if raw is None:
        return None
    try:
        raw = Chem.RemoveHs(raw, sanitize=False)
    except Exception:
        return None
    return reassign_bonds(raw, template)


# ---------- per-target evaluation ----------

def evaluate_target(target: str, exper_dir: Path, lg_path: Path, smiles: str) -> dict:
    out: dict = {"target": target}
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        out["error"] = "bad SMILES template"
        return out

    prot_pdb = exper_dir / "protein_aligned.pdb"
    lig_pdb_candidates = sorted(exper_dir.glob("ligand_*.pdb"))
    if not lig_pdb_candidates:
        out["error"] = "no exper ligand"
        return out
    ref_lig = load_experimental_ligand(lig_pdb_candidates[0], template)
    if ref_lig is None:
        out["error"] = "exper ligand bond reassign failed"
        return out
    ref_ca = parse_ca(prot_pdb)

    lg = parse_lg(lg_path)
    out["affnty"] = lg.get("affnty")
    out["n_models"] = len(lg["models"])
    out["models"] = []

    for m in lg["models"]:
        entry: dict = {
            "idx": m["idx"],
            "name": m["name"],
            "lscore": m["lscore"],
        }
        pred_ca = parse_ca(m["atom_lines"])
        common = sorted(set(pred_ca) & set(ref_ca))
        if len(common) < 50:
            entry["error"] = f"only {len(common)} common CAs"
            out["models"].append(entry)
            continue
        P = np.array([pred_ca[r] for r in common])
        Q = np.array([ref_ca[r] for r in common])
        R, t, prot_rmsd = kabsch(P, Q)
        entry["protein_ca_rmsd"] = round(prot_rmsd, 3)
        entry["n_aligned_residues"] = len(common)

        pred_raw = mol_from_mdl_body(m["mdl_text"])
        pred_mol = reassign_bonds(pred_raw, template)
        if pred_mol is None:
            entry["error"] = "ligand parse/reassign failed"
            out["models"].append(entry)
            continue
        transform_mol(pred_mol, R, t)
        try:
            entry["ligand_rmsd"] = round(pose_rmsd(pred_mol, ref_lig), 3)
        except Exception as e:
            entry["error"] = f"rmsd err: {e}"
        out["models"].append(entry)

    rmsds = [m.get("ligand_rmsd") for m in out["models"] if isinstance(m.get("ligand_rmsd"), float)]
    out["top1_rmsd"] = out["models"][0].get("ligand_rmsd") if out["models"] else None
    out["best_rmsd_in_top5"] = round(min(rmsds), 3) if rmsds else None
    return out


def main():
    aff = load_affinity_table()
    targets = sorted(EXPER_DIR.glob("L*"))
    results = []
    for tdir in targets:
        tid = tdir.name
        lg = SUB_DIR / f"{tid}_input.lg"
        if not lg.exists():
            results.append({"target": tid, "status": "no submission"})
            continue
        smi = aff.get(tid, {}).get("smiles", "")
        if not smi:
            results.append({"target": tid, "status": "no SMILES in affinity table"})
            continue
        try:
            res = evaluate_target(tid, tdir, lg, smi)
        except Exception as e:
            res = {"target": tid, "error": f"crash: {e!r}"}
        if tid in aff:
            res["exper_ic50_uM"] = aff[tid]["ic50_uM"]
            res["exper_ba_kcal_mol"] = aff[tid]["ba_kcal_mol"]
        results.append(res)
        print(f"{tid}: top1={res.get('top1_rmsd')} best5={res.get('best_rmsd_in_top5')} affnty={res.get('affnty')}")

    out_path = ROOT / "evaluation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
