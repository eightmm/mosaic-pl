#!/usr/bin/env python3
"""Random 50-target frame-alignment audit.

For each sampled target, verify that **all coordinate consumers share one
protein-based frame**:

  1. ``receptor.pdb`` (Track 1 docking receptor) ↔ ``cofolding_structure``
     (the *_aligned.cif picked by prep). Their polymer-Cα centroids should
     agree within a tight tolerance.
  2. Each pose source family (cofold_{af3,boltz2,boltz2x,protenix},
     vina_*, autodock_gpu_*, template_*) — sample one pose, take heavy-atom
     centroid, distance from receptor centroid must be ≤ receptor_extent + 30 Å.
  3. The four cofold models' ``*_aligned.cif`` polymer-Cα centroids vs the
     reference (the one named in ``alignment_summary.json``) — pairwise gap
     should be small (Kabsch aligned).

Outputs a TSV plus stdout summary. Failures grouped by category.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def _polymer_ca(path: Path) -> np.ndarray | None:
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        return None
    pts: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.entity_type != gemmi.EntityType.Polymer:
                    continue
                for atom in res:
                    if atom.name.strip() == "CA":
                        pts.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def _heavy_atoms(path: Path) -> np.ndarray | None:
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        return None
    pts: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.entity_type != gemmi.EntityType.Polymer:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    pts.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def _sdf_centroid(path: Path) -> np.ndarray | None:
    try:
        sup = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = next((m for m in sup if m is not None), None)
        if mol is None:
            return None
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z]
                           for i in range(mol.GetNumAtoms())])
        return coords.mean(axis=0)
    except Exception:
        return None


def audit_target(base: Path, max_padding: float = 30.0) -> dict:
    """One target's audit."""
    rep: dict = {"target": base.name, "issues": [], "stats": {}}

    prep_path = base / "inputs" / "docking" / "docking_prep_summary.json"
    if not prep_path.exists():
        rep["issues"].append("F: docking_prep_summary missing")
        return rep
    prep = json.loads(prep_path.read_text())

    # --- 1. receptor.pdb ↔ cofolding_structure
    rec_pdb = base / "inputs" / "docking" / "receptor.pdb"
    rec_ca = _polymer_ca(rec_pdb)
    if rec_ca is None:
        rep["issues"].append("R: receptor.pdb missing CA")
        return rep
    rec_ca_centroid = rec_ca.mean(axis=0)
    rec_heavy = _heavy_atoms(rec_pdb)
    if rec_heavy is None:
        rep["issues"].append("R: receptor.pdb missing heavy atoms")
        return rep
    rec_heavy_centroid = rec_heavy.mean(axis=0)
    rec_extent = float(np.linalg.norm(rec_heavy - rec_heavy_centroid, axis=1).max())

    cof_struct_path = prep.get("cofolding_structure")
    if cof_struct_path:
        cof_ca = _polymer_ca(Path(cof_struct_path))
        if cof_ca is None:
            rep["issues"].append("R: cofolding_structure missing CA")
        else:
            cof_ca_centroid = cof_ca.mean(axis=0)
            dist = float(np.linalg.norm(rec_ca_centroid - cof_ca_centroid))
            rep["stats"]["rec_vs_cofold_ca_dist"] = round(dist, 3)
            if dist > 0.5:
                rep["issues"].append(f"M: receptor.pdb ↔ cofolding_structure CA centroid = {dist:.2f}Å (>0.5)")

    # --- 2. Sample pose centroids per family
    pose_root = base / "outputs"
    max_allowed = rec_extent + max_padding
    family_recipes = [
        ("cofold_af3",      ("rglob", "alphafold3", "*_aligned.cif")),
        ("cofold_boltz2",   ("rglob", "boltz2", "*_aligned.cif")),
        ("cofold_boltz2x",  ("rglob", "boltz2x", "*_aligned.cif")),
        ("cofold_protenix", ("rglob", "protenix", "*_aligned.cif")),
        ("vina_cof",        ("glob", ".", "vina_cofolding_1/seed_42/ligand_*/docked.sdf")),
        ("adg_cof",         ("glob", ".", "autodock_gpu_cofolding_1/seed_42/ligand_*/docking.sdf")),
        ("vina_tc1",        ("glob", ".", "vina_template_consensus_1/seed_42/ligand_*/docked.sdf")),
        ("adg_tc1",         ("glob", ".", "autodock_gpu_template_consensus_1/seed_42/ligand_*/docking.sdf")),
        ("template_dock",   ("glob", ".", "template_docking/*/lig_align/*.sdf")),
    ]
    family_centroids = {}
    pose_outliers: list[dict] = []
    for fam, recipe in family_recipes:
        mode, sub, patt = recipe
        sub_path = pose_root if sub == "." else (pose_root / sub)
        if not sub_path.exists():
            continue
        if mode == "rglob":
            candidates = sorted(sub_path.rglob(patt))
        else:
            candidates = sorted(sub_path.glob(patt))
        if not candidates:
            continue
        # take the first file
        f = candidates[0]
        if f.suffix.lower() == ".cif":
            arr = _polymer_ca(f)
            if arr is None: continue
            c = arr.mean(axis=0)
        else:
            c = _sdf_centroid(f)
            if c is None: continue
        d = float(np.linalg.norm(c - rec_heavy_centroid))
        family_centroids[fam] = (round(d, 2), str(f.relative_to(base)))
        if d > max_allowed:
            pose_outliers.append({"fam": fam, "dist": d, "max": max_allowed, "file": str(f.relative_to(base))})

    rep["stats"]["family_centroid_dist"] = {k: v[0] for k, v in family_centroids.items()}
    rep["stats"]["rec_extent"] = round(rec_extent, 2)
    rep["stats"]["max_allowed"] = round(max_allowed, 2)
    if pose_outliers:
        for o in pose_outliers:
            rep["issues"].append(f"P: {o['fam']} centroid {o['dist']:.1f}Å > {o['max']:.0f}Å ({o['file']})")

    # --- 3. Cofold model pairwise CA centroid agreement
    cofold_aligned: dict[str, np.ndarray] = {}
    for fam, sub in (
        ("af3", "alphafold3"),
        ("boltz2", "boltz2"),
        ("boltz2x", "boltz2x"),
        ("protenix", "protenix"),
    ):
        candidates = sorted((pose_root / sub).rglob("*_aligned.cif")) if (pose_root / sub).exists() else []
        if not candidates:
            continue
        arr = _polymer_ca(candidates[0])
        if arr is not None:
            cofold_aligned[fam] = arr.mean(axis=0)

    if len(cofold_aligned) >= 2:
        keys = list(cofold_aligned.keys())
        pairs = [(a, b, float(np.linalg.norm(cofold_aligned[a] - cofold_aligned[b])))
                 for i, a in enumerate(keys) for b in keys[i+1:]]
        worst = max(pairs, key=lambda x: x[2])
        rep["stats"]["cofold_max_pair_dist"] = round(worst[2], 2)
        rep["stats"]["cofold_worst_pair"] = f"{worst[0]}↔{worst[1]}"
        if worst[2] > 1.0:
            rep["issues"].append(f"C: cofold model CA centroids diverge: {worst[0]}↔{worst[1]} = {worst[2]:.2f}Å (>1.0)")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("experiments/msa_e2e_test/runs"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-tsv", type=Path,
                    default=Path("/tmp/frame_audit_50.tsv"))
    args = ap.parse_args()

    targets = sorted(p for p in args.runs_dir.iterdir()
                     if p.is_dir() and p.name.endswith("_input"))
    targets = [p / p.name for p in targets if (p / p.name).exists()]
    random.seed(args.seed)
    sample = random.sample(targets, min(args.n, len(targets)))
    print(f"[audit] sampling {len(sample)}/{len(targets)} targets (seed={args.seed})")

    reports = []
    for i, t in enumerate(sample):
        r = audit_target(t)
        reports.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sample)}] last={t.name}", flush=True)

    # Write TSV
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target", "rec_vs_cofold_ca", "rec_extent",
                    "cofold_max_pair_dist", "cofold_worst_pair",
                    "family_centroid_dist", "n_issues", "issues"])
        for r in reports:
            s = r["stats"]
            w.writerow([
                r["target"],
                s.get("rec_vs_cofold_ca_dist", ""),
                s.get("rec_extent", ""),
                s.get("cofold_max_pair_dist", ""),
                s.get("cofold_worst_pair", ""),
                json.dumps(s.get("family_centroid_dist", {})),
                len(r["issues"]),
                " | ".join(r["issues"]),
            ])

    # Summary
    n_clean = sum(1 for r in reports if not r["issues"])
    print()
    print(f"=== summary === {n_clean}/{len(reports)} clean")
    cat = Counter()
    for r in reports:
        for issue in r["issues"]:
            cat[issue.split(":")[0]] += 1
    for k, v in cat.most_common():
        print(f"  {v:>3}× {k}-class issues")
    print()
    print("worst 10:")
    for r in sorted(reports, key=lambda x: -len(x["issues"]))[:10]:
        if not r["issues"]:
            break
        print(f"  {r['target']}: {r['stats']}")
        for issue in r["issues"][:3]:
            print(f"    - {issue}")
    print()
    print(f"tsv: {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
