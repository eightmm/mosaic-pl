#!/usr/bin/env python3
"""Tier B prototype: do a clash gate + symmetry-aware cross-family consensus
recover poses the scalar selector misses?

Sample = recoverable targets (sub-2A pose exists, rrf_scalar misses) with
staged data, from tier_b_sample.txt. Baseline rrf MISSES all of them by
construction, so any strategy that flips a target to a sub-2A pick is a net
gain. Reports #recovered / N per strategy, by zone.

Coords: parsed directly from staged pose SDF V2000 atom blocks (robust, no
sanitize). Symmetry RMSD: RDKit GetBestRMS on centroid-prefiltered
cross-family pairs, falling back to index-RMSD when a mol won't load.

Join: CSV pose_name `<src>_seed_<N>_<k>` -> file `<src>_seed_<N>.sdf` rec k.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import ablate_pose_selection as A

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "novel2025_runs" / "runs"
HIT = 2.0
CLASH_DIST = 2.0          # heavy-atom overlap distance (A)
CLASH_MAX = 4             # >this many receptor atoms overlapping -> bad pose
CONS_THRESH = 2.0         # cross-family agreement RMSD (A)
CONS_WEIGHT = 0.1         # production value (compute_submission_scores)

_ELEM_H = {"H", "D"}


def _read_sdf_records(path: Path):
    """Yield (heavy_coords ndarray, elements list) per record. Direct V2000 parse."""
    if not path.is_file():
        return
    text = path.read_text(errors="replace")
    for block in text.split("$$$$"):
        lines = block.splitlines()
        if len(lines) < 4:
            continue
        counts = lines[3]
        try:
            natoms = int(counts[:3])
        except ValueError:
            continue
        coords = []
        elems = []
        for ln in lines[4:4 + natoms]:
            if len(ln) < 34:
                continue
            try:
                x, y, z = float(ln[0:10]), float(ln[10:20]), float(ln[20:30])
            except ValueError:
                continue
            el = ln[31:34].strip()
            if el in _ELEM_H:
                continue
            coords.append((x, y, z))
            elems.append(el)
        if coords:
            yield np.asarray(coords, float), elems


def _receptor_heavy(path: Path):
    pts = []
    for ln in path.read_text(errors="replace").splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            el = ln[76:78].strip() or ln[12:16].strip()[:1]
            if el in _ELEM_H:
                continue
            try:
                pts.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
    return np.asarray(pts, float) if pts else None


def _build_mol_loader(target, rows):
    """name -> RDKit mol (record k of its SDF), cached. None if unparseable."""
    try:
        from rdkit import Chem
    except ImportError:
        return lambda name: None
    supplier_cache: dict[Path, list] = {}
    mol_cache: dict[str, object] = {}

    def load(name):
        if name in mol_cache:
            return mol_cache[name]
        pf, k = _pose_file(target, name)
        if k is None or not pf.is_file():
            mol_cache[name] = None
            return None
        if pf not in supplier_cache:
            supplier_cache[pf] = list(Chem.SDMolSupplier(str(pf), sanitize=False, removeHs=True))
        mols = supplier_cache[pf]
        m = mols[k] if k < len(mols) else None
        mol_cache[name] = m
        return m

    return load


def _best_rms(m1, m2, fallback):
    """Symmetry-aware IN-FRAME heavy-atom RMSD via rdMolAlign.CalcRMS:
    minimises over symmetry-equivalent atom mappings but does NOT align
    (no translation/rotation), so two poses in different pockets stay far
    apart. GetBestRMS would superimpose first and corrupt the consensus
    signal. Returns fallback (index-RMSD) when a mol is missing / errors."""
    if m1 is None or m2 is None or m1.GetNumAtoms() != m2.GetNumAtoms():
        return fallback
    # CalcRMS enumerates symmetry automorphisms; for large/highly-symmetric
    # mols this can blow up. Cap to keep it bounded (large ligands rarely have
    # the small-symmetry ambiguity CalcRMS is meant to fix anyway).
    if m1.GetNumAtoms() > 28:
        return fallback
    try:
        from rdkit.Chem import rdMolAlign
        return rdMolAlign.CalcRMS(m1, m2, maxMatches=1000)
    except Exception:
        return fallback


def _pose_file(target, pose_name):
    stem, _, k = pose_name.rpartition("_")
    return (RUNS / f"{target}_input" / f"{target}_input" / "outputs" / "analysis"
            / "poses" / f"{stem}.sdf"), int(k) if k.isdigit() else None


def _clash(pose_xyz, rec_xyz):
    if rec_xyz is None:
        return 0
    # count receptor atoms within CLASH_DIST of ANY pose atom
    n = 0
    for a in pose_xyz:
        d2 = ((rec_xyz - a) ** 2).sum(axis=1)
        n += int((d2 < CLASH_DIST * CLASH_DIST).sum())
    return n


def main():
    sample = set(t.strip() for t in (ROOT / "tier_b_sample.txt").read_text().split() if t.strip())
    # single CSV pass: keep ONLY sample-target rows (not all 1.4M).
    rows_by_target = defaultdict(dict)
    zone = {}
    with open(A.CSV) as f:
        for r in csv.DictReader(f):
            t = r["target"]
            if t not in sample:
                continue
            rows_by_target[t][r["pose_name"]] = r
            zone.setdefault(t, r["seq_zone"])
    sample = [t for t in sample if t in rows_by_target]
    print(f"loaded {len(sample)} sample targets", flush=True)

    results = defaultdict(lambda: defaultdict(int))  # strat -> zone -> recovered
    counts = defaultdict(int)

    results = defaultdict(lambda: defaultdict(int))  # strat -> zone -> recovered
    counts = defaultdict(int)

    for ti, t in enumerate(sample, 1):
        z = zone[t]
        counts[z] += 1
        print(f"[{ti}/{len(sample)}] {t} ({z})", flush=True)
        rec_xyz = _receptor_heavy(RUNS / f"{t}_input" / f"{t}_input"
                                  / "inputs" / "docking" / "receptor.pdb")
        # load coords for each pose_name
        file_cache: dict[Path, list] = {}
        poses = []
        for pose_name, r in rows_by_target[t].items():
            tr = A._f(r["true_rmsd"])
            if tr is None:
                continue
            pf, k = _pose_file(t, pose_name)
            if k is None:
                continue
            if pf not in file_cache:
                file_cache[pf] = list(_read_sdf_records(pf))
            recs = file_cache[pf]
            if k >= len(recs):
                continue
            xyz, _ = recs[k]
            poses.append({
                "name": pose_name, "fam": A._family(r["source"]),
                "lscore": A._f(r["lscore"]), "plddt": A._f(r["plddt"]),
                "iptm": A._f(r["iptm"]), "ptm": A._f(r["ptm"]), "conf": A._f(r["conf"]),
                "is_cofold": A._family(r["source"]) in A.COFOLD_FAMS,
                "true_rmsd": tr, "xyz": xyz, "cen": xyz.mean(axis=0),
            })
        if not poses:
            continue

        # clash count per pose
        for p in poses:
            p["clash"] = _clash(p["xyz"], rec_xyz)

        # --- cross-family consensus support: index-RMSD vs symmetry-aware ---
        # index = production's method (direct positional RMSD, atom-order
        # assumed). symmetry = RDKit GetBestRMS (handles atom-order +
        # topological symmetry); fallback to index when a mol won't load.
        rdkit_mol = _build_mol_loader(t, rows_by_target[t])
        for p in poses:
            p["support_idx"] = 0
            p["support_sym"] = 0
        # CalcRMS (symmetry, in-frame) only minimises vs index, so it can only
        # ADD agreements. It only matters when poses already overlap spatially
        # (centroids close) — for far pairs symmetry can't rescue. Gate the
        # expensive CalcRMS on centroid overlap to keep it O(overlapping pairs).
        for i in range(len(poses)):
            pi = poses[i]
            for j in range(i + 1, len(poses)):
                pj = poses[j]
                if pi["fam"] == pj["fam"]:
                    continue
                cd = abs(pi["cen"] - pj["cen"]).max()
                if cd > CONS_THRESH + 1:
                    continue
                if pi["xyz"].shape != pj["xyz"].shape:
                    continue
                r_idx = math.sqrt(((pi["xyz"] - pj["xyz"]) ** 2).sum(axis=1).mean())
                if r_idx < CONS_THRESH:
                    pi["support_idx"] += 1
                    pj["support_idx"] += 1
                # symmetry refinement only where centroids genuinely overlap
                r_sym = _best_rms(rdkit_mol(pi["name"]), rdkit_mol(pj["name"]), r_idx) \
                    if cd <= CONS_THRESH else r_idx
                if r_sym < CONS_THRESH:
                    pi["support_sym"] += 1
                    pj["support_sym"] += 1

        # --- rrf scalar scores (reuse logic) ---
        def rrf_scores(pool):
            sc = defaultdict(float)
            kk = 60

            def acc(ranked):
                for rnk, p in enumerate(ranked):
                    sc[id(p)] += 1.0 / (kk + rnk + 1)
            acc(sorted([p for p in pool if p["lscore"] is not None], key=lambda p: -p["lscore"]))
            cof = [p for p in pool if p["is_cofold"]]
            byf = defaultdict(list)
            for p in cof:
                if p["plddt"] is not None:
                    byf[p["fam"]].append(p)
            for ps in byf.values():
                acc(sorted(ps, key=lambda p: -p["plddt"]))
            for a in ("iptm", "ptm", "conf"):
                acc(sorted([p for p in cof if p[a] is not None], key=lambda p: -p[a]))
            return sc

        def pick(pool, support_key):
            if not pool:
                return None
            sc = rrf_scores(pool)
            best, bv = None, -1
            for p in pool:
                v = sc[id(p)]
                if support_key:
                    v *= 1.0 + CONS_WEIGHT * math.log1p(p[support_key])
                if v > bv:
                    bv, best = v, p
            return best

        gated = [p for p in poses if p["clash"] <= CLASH_MAX] or poses
        strategies = {
            "baseline_rrf":      pick(poses, None),
            "consensus_idx":     pick(poses, "support_idx"),   # = production method
            "consensus_sym":     pick(poses, "support_sym"),   # proposal #2
            "clash+consensus_sym": pick(gated, "support_sym"),
        }
        for sname, p in strategies.items():
            if p and p["true_rmsd"] < HIT:
                results[sname][z] += 1
        # running totals (partial-safe if interrupted)
        run = {s: sum(results[s].values()) for s in strategies}
        print(f"    running: " + "  ".join(f"{s}={run[s]}" for s in strategies), flush=True)

    print(f"\nsample: {dict(counts)}  total={sum(counts.values())}")
    print(f"(baseline rrf misses ALL by construction — these are recoverable targets)\n")
    print(f"{'strategy':20s} {'novel':>7s} {'remote':>7s} {'related':>8s} {'TOTAL':>7s}")
    for sname in ("baseline_rrf", "consensus_idx", "consensus_sym", "clash+consensus_sym"):
        rec = results[sname]
        tot = sum(rec.values())
        print(f"{sname:18s} {rec['novel']:7d} {rec['remote']:7d} {rec['related']:8d} {tot:7d}")


if __name__ == "__main__":
    main()
