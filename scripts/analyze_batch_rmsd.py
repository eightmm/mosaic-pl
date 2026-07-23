#!/usr/bin/env python3
"""Per-target pose-pool RMSD analysis against native binding.

For every target run (``<run-dir>/<target>_input/<target>_input/``) this:

1. Loads the cofold reference cif (first protenix ``*_aligned.cif`` found).
2. Pulls the crystal mmCIF + candidate-ligand CCDs from the RCSB index DB.
3. USalign-fits crystal chain A onto the cofold protein → R, t transform.
4. Materialises each native ligand instance through gemmi (no hand-rolled
   PDB writer) → ``MolFromPDBFile`` → ``AssignBondOrdersFromTemplate`` to
   recover bond orders from the input SMILES.
5. Walks every staged pose SDF in ``outputs/analysis/poses/`` and computes
   ``rdMolAlign.CalcRMS`` against each native ligand (CalcRMS handles
   symmetric atom permutations natively — no naive heavy-atom RMSD).
6. Reports per-target counts + zone-rolled summary.

Replaces an ad-hoc inline analysis whose hand-written PDB writer broke
``MolFromPDBFile`` ("Cannot determine element / Problem with residue
number") for ~half the batch. The gemmi-based writer here uses the same
PDB serializer the docking pipeline uses, so RDKit always parses it.

Usage::

    .venv/bin/python scripts/analyze_batch_rmsd.py \\
        --runs-dir experiments/novel2025_runs/runs \\
        --targets 7hqq 9av3 9e72 \\
        --output-tsv /tmp/rmsd.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

_REPO_ROOT = Path(__file__).resolve().parent.parent
USALIGN = _REPO_ROOT / ".local" / "bin" / "USalign"
RCSB_DB = Path("/home/jaemin/DB/RCSB/processed/rcsb_index.db")


def _usalign_xform(crystal_cif: Path, cofold_ref_cif: Path, td: Path):
    """Return (R, t) so that crystal-chain-A coords map into the cofold
    protein frame: ``cofold = R @ crystal + t``. ``None, None`` on failure."""
    s = gemmi.read_structure(str(crystal_cif))
    ns = gemmi.Structure(); ns.name = s.name
    m = gemmi.Model("1")
    has = False
    for ch in s[0]:
        if ch.name != "A":
            continue
        nc = gemmi.Chain(ch.name)
        for r in ch:
            if r.entity_type == gemmi.EntityType.Polymer:
                nc.add_residue(r)
        if len(nc) > 0:
            m.add_chain(nc); has = True
    if not has:
        return None, None
    ns.add_model(m); ns.write_pdb(str(td / "crystal.pdb"))
    gemmi.read_structure(str(cofold_ref_cif)).write_pdb(str(td / "cofold.pdb"))
    subprocess.run(
        [str(USALIGN), str(td/"crystal.pdb"), str(td/"cofold.pdb"),
         "-mol", "prot", "-m", str(td/"m.txt")],
        capture_output=True, text=True, timeout=60,
    )
    if not (td/"m.txt").exists():
        return None, None
    mat = (td/"m.txt").read_text().splitlines()
    R = np.eye(3); t = np.zeros(3)
    for i, ln in enumerate(mat):
        if ln.strip().startswith("m") and "u[m]" in ln:
            for j in range(3):
                p = mat[i + 1 + j].split()
                t[j] = float(p[1])
                R[j, 0] = float(p[2]); R[j, 1] = float(p[3]); R[j, 2] = float(p[4])
            break
    return R, t


def _residue_to_pdb_via_gemmi(residue, R, t, out_pdb: Path) -> Path | None:
    """Write a single residue to PDB through gemmi's serializer.

    Hand-rolling PDB ATOM records is fragile (column alignment, element
    inference, residue-number formatting all break ``MolFromPDBFile``).
    Routing through ``gemmi.Structure.write_pdb`` produces a parser-friendly
    PDB every time.
    """
    mini = gemmi.Structure(); mini.name = residue.name
    mm = gemmi.Model("1")
    mc = gemmi.Chain("A")
    new_res = gemmi.Residue()
    new_res.name = residue.name
    new_res.seqid = gemmi.SeqId(1, ' ')
    new_res.entity_type = gemmi.EntityType.NonPolymer
    for a in residue:
        if a.element.name == "H":
            continue
        p = np.asarray([a.pos.x, a.pos.y, a.pos.z]) @ R.T + t
        na = gemmi.Atom()
        na.name = a.name
        na.element = a.element
        na.pos = gemmi.Position(float(p[0]), float(p[1]), float(p[2]))
        na.occ = 1.0
        na.b_iso = 0.0
        new_res.add_atom(na)
    if len(new_res) == 0:
        return None
    mc.add_residue(new_res)
    mm.add_chain(mc)
    mini.add_model(mm)
    mini.write_pdb(str(out_pdb))
    return out_pdb


def _get_target_smiles(target: str, pipeline_dir: Path) -> list[str]:
    yml = pipeline_dir / f"{target}_input.yaml"
    if not yml.exists():
        return []
    with open(yml) as f:
        data = yaml.safe_load(f)
    return [
        s["ligand"]["smiles"]
        for s in data.get("sequences", [])
        if isinstance(s, dict) and "ligand" in s and s["ligand"].get("smiles")
    ]


def _get_native_mols(target: str, cofold_ref_cif: Path, smiles_list: list[str]):
    """Return list of (ccd_code, RDKit Mol with bond orders) in cofold frame."""
    conn = sqlite3.connect(str(RCSB_DB))
    try:
        row = conn.execute(
            "SELECT cif_path FROM entries WHERE pdb_id = ?", (target.lower(),),
        ).fetchone()
        if not row:
            return []
        crystal_cif = Path(row[0])
        ccds_rows = conn.execute(
            "SELECT DISTINCT ccd_code FROM ligand_instances "
            "WHERE pdb_id=? AND is_candidate=1",
            (target.lower(),),
        ).fetchall()
    finally:
        conn.close()
    target_ccds = {r[0].upper() for r in ccds_rows}
    if not target_ccds or not crystal_cif.exists():
        return []

    td = Path(tempfile.mkdtemp(prefix=f"rmsd_{target}_"))
    R, t = _usalign_xform(crystal_cif, cofold_ref_cif, td)
    if R is None:
        return []

    templates = []
    for smi in smiles_list:
        try:
            tmpl = Chem.MolFromSmiles(smi)
            if tmpl is not None:
                templates.append(tmpl)
        except Exception:
            pass
    if not templates:
        return []

    natives = []
    crystal = gemmi.read_structure(str(crystal_cif))
    for ch in crystal[0]:
        for r in ch:
            ccd = r.name.upper()
            if ccd not in target_ccds:
                continue
            res_pdb = td / f"{ccd}_{ch.name}_{r.seqid.num}.pdb"
            if _residue_to_pdb_via_gemmi(r, R, t, res_pdb) is None:
                continue
            try:
                raw_mol = Chem.MolFromPDBFile(
                    str(res_pdb), removeHs=True, sanitize=False,
                    proximityBonding=True,
                )
            except Exception:
                continue
            if raw_mol is None:
                continue
            for tmpl in templates:
                if raw_mol.GetNumHeavyAtoms() != tmpl.GetNumHeavyAtoms():
                    continue
                try:
                    assigned = AssignBondOrdersFromTemplate(tmpl, raw_mol)
                    if assigned is not None:
                        natives.append((ccd, assigned))
                        break
                except Exception:
                    continue
    return natives


def _calc_rms(pose_mol, native_mol) -> float | None:
    """Symmetric-aware RMSD via RDKit (handles ring/aromatic permutations)."""
    try:
        return float(rdMolAlign.CalcRMS(pose_mol, native_mol))
    except Exception:
        return None


def analyze_one(target: str, run_root: Path, pipeline_dir: Path) -> dict | None:
    run = run_root / f"{target}_input" / f"{target}_input"
    if not (run / "outputs").exists():
        return None
    cofold_ref = next((run / "outputs/protenix").rglob("*_aligned.cif"), None)
    if cofold_ref is None:
        return None

    cluster_file = run / "outputs/template_pockets/template_pocket_clusters.json"
    n_clusters = 0
    if cluster_file.exists():
        try:
            cd = json.loads(cluster_file.read_text())
            cls = cd["clusters"] if isinstance(cd, dict) and "clusters" in cd else cd
            n_clusters = len(cls)
        except Exception:
            pass

    smis = _get_target_smiles(target, pipeline_dir)
    natives = _get_native_mols(target, cofold_ref, smis)
    if not natives:
        return {
            "target": target, "n_natives": 0, "n_clusters": n_clusters,
            "n_poses": 0, "n_compared": 0,
            "lt2": 0, "lt5": 0, "lt10": 0, "best_rmsd": None,
        }

    all_rmsds: list[float] = []
    n_poses = 0
    for sdf in sorted((run / "outputs/analysis/poses").glob("*.sdf")):
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True):
            if mol is None:
                continue
            try:
                mol.GetConformer()
            except ValueError:
                continue
            n_poses += 1
            best = None
            for _ccd, nat in natives:
                if mol.GetNumHeavyAtoms() != nat.GetNumHeavyAtoms():
                    continue
                rms = _calc_rms(mol, nat)
                if rms is None:
                    continue
                if best is None or rms < best:
                    best = rms
            if best is not None:
                all_rmsds.append(best)
    return {
        "target": target,
        "n_natives": len(natives),
        "n_clusters": n_clusters,
        "n_poses": n_poses,
        "n_compared": len(all_rmsds),
        "lt2": sum(1 for x in all_rmsds if x < 2),
        "lt5": sum(1 for x in all_rmsds if x < 5),
        "lt10": sum(1 for x in all_rmsds if x < 10),
        "best_rmsd": min(all_rmsds) if all_rmsds else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path,
                   default=_REPO_ROOT / "experiments/novel2025_runs/runs")
    p.add_argument("--pipeline-dir", type=Path,
                   default=_REPO_ROOT / "experiments/novel2025_test/pipeline",
                   help="Directory holding ``<target>_input.yaml`` files")
    p.add_argument("--targets", nargs="+",
                   help="Specific targets (no _input suffix). Default = all *_input dirs in runs-dir.")
    p.add_argument("--output-tsv", type=Path, default=None)
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    if args.targets:
        targets = list(args.targets)
    else:
        targets = sorted(
            p.name.replace("_input", "")
            for p in args.runs_dir.iterdir()
            if p.is_dir() and p.name.endswith("_input")
        )
    print(f"[analyze] {len(targets)} targets in {args.runs_dir}")

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(analyze_one, t, args.runs_dir, args.pipeline_dir): t
            for t in targets
        }
        for f in as_completed(futs):
            t = futs[f]
            try:
                r = f.result()
                if r is not None:
                    results[t] = r
            except Exception as e:
                print(f"  {t}: ERROR {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n{'target':<7} {'n_lig':>5} {'n_clst':>6} {'n_poses':>7} {'comp':>5} "
          f"{'<2Å':>4} {'<5Å':>5} {'<10Å':>5} {'best':<6}")
    print("-" * 60)
    T = {"n_poses": 0, "n_compared": 0, "lt2": 0, "lt5": 0, "lt10": 0,
         "n_clusters": 0, "n_natives": 0}
    for t in targets:
        r = results.get(t)
        if not r:
            continue
        best = f"{r['best_rmsd']:.2f}" if r["best_rmsd"] is not None else "n/a"
        print(f"{t:<7} {r['n_natives']:>5} {r['n_clusters']:>6} {r['n_poses']:>7} "
              f"{r['n_compared']:>5} {r['lt2']:>4} {r['lt5']:>5} {r['lt10']:>5} {best:<6}")
        for k in T:
            T[k] += r.get(k, 0)
    print("-" * 60)
    print(f"{'TOTAL':<7} {T['n_natives']:>5} {T['n_clusters']:>6} {T['n_poses']:>7} "
          f"{T['n_compared']:>5} {T['lt2']:>4} {T['lt5']:>5} {T['lt10']:>5}")

    if args.output_tsv:
        with open(args.output_tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["target", "n_natives", "n_clusters", "n_poses",
                        "n_compared", "lt2", "lt5", "lt10", "best_rmsd"])
            for t in targets:
                r = results.get(t)
                if not r:
                    continue
                w.writerow([t, r["n_natives"], r["n_clusters"], r["n_poses"],
                            r["n_compared"], r["lt2"], r["lt5"], r["lt10"],
                            f"{r['best_rmsd']:.4f}" if r["best_rmsd"] is not None else ""])
        print(f"  → wrote {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
