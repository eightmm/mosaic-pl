#!/usr/bin/env python3
"""Collapse every donor crystal's solvent into one consensus site list.

R2386 asks for 500 solvent positions per model, and only those nearest a
well-ordered residue are scored (2026-08-07 amendment, `docs/casp17_ts_solvent.md`
§1b). Before deciding how to fill 500 slots it has to be known how many distinct
sites the experimental record actually supports — which means pooling all 41
usable donors, not the 12 the submission pipeline currently reads, and merging
them on a criterion that reflects what can physically coexist.

Why van der Waals rather than a flat cutoff
-------------------------------------------
The shipped pipeline merges same-kind sites within a fixed 1.0 A. That number is
neither the resolution limit nor a chemical distance: at 1.0 A two waters are
already interpenetrating (an O–O hydrogen bond is 2.6–3.0 A), so distinct
"sites" that no structure could show simultaneously survive as separate
predictions. Here two observations are the same site when they are closer than
the shortest real separation for that species pair — the same
``MIN_SOLVENT_CONTACT`` floors the submission builder enforces, which are set
just under the shortest observed coordination/H-bond distance. Merging is
therefore the exact complement of the clash filter: whatever survives clustering
can be written out together without violating the floors.

Clustering is greedy over observations sorted by donor quality, so the
best-resolved crystals seed the sites and weaker ones accrete. Each cluster
carries its supporting donor set, which is the evidence weight the model
selection needs, and its species vote — a position called Mg by one crystal and
water by another is one site with a contested identity, not two sites.

Usage:
    uv run python scripts/cluster_solvent_donors.py \
      --templates 'data/solvent_templates/R2386/*.cif' \
      --sequence inputs/R2386.fasta \
      --frame 3G78 --min-coverage 80 \
      --out data/solvent_templates/R2386/consensus_sites.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from scipy.spatial import cKDTree  # noqa: E402

from make_solvent_ts_submission import (  # noqa: E402
    MIN_RNA_CONTACT,
    MIN_SOLVENT_CONTACT,
    SOLVENT_ALIASES,
    SOLVENT_NAMES,
    _solvent_floor,
    load_template,
    superpose,
)


def read_sequence(path: Path) -> str:
    return "".join(ln.strip() for ln in path.read_text().splitlines()
                   if ln.strip() and not ln.startswith(">"))


def donor_solvent(tpl) -> list[tuple[str, np.ndarray]]:
    """(species, xyz) for every requested solvent atom, Tl read as K."""
    out = []
    for name, x, y, z, _b in tpl.solvent:
        kind = SOLVENT_ALIASES.get(name, name)
        if kind in SOLVENT_NAMES:
            out.append((kind, np.array([x, y, z], dtype=float)))
    return out


def repair_overlaps(sites: list[dict], merge_scale: float,
                    max_rounds: int = 50) -> int:
    """Merge clusters that drifted inside each other's contact floor.

    The greedy pass compares each observation against a *running mean*: a
    cluster's centre moves every time it accretes. Two clusters that were more
    than a floor apart when they were seeded can therefore be pulled together by
    later observations, and nothing in the greedy loop ever looks back. On R2386
    that left 106 pairs — 155 of 829 sites — closer than the species pair
    physically allows, which contradicts the one property the merge is supposed
    to guarantee.

    This closes the loop: while any pair sits below its floor, merge the closest
    such pair (weighted by observation count, votes and donors pooled) and look
    again. Merging moves the survivor, so the check has to repeat until a round
    finds nothing — a single pass would leave new violations behind it.
    """
    merged = 0
    for _ in range(max_rounds):
        # Carry the original index: a site dict holds numpy arrays, so
        # ``list.index`` compares them elementwise and raises.
        live = [(k, s) for k, s in enumerate(sites) if s is not None]
        if len(live) < 2:
            break
        alive = [s for _, s in live]
        pos = np.array([s["xyz"] for s in alive])
        tree = cKDTree(pos)
        worst = max(MIN_SOLVENT_CONTACT.values()) * merge_scale
        bad = []
        for i, j in tree.query_pairs(r=worst):
            cut = merge_scale * _solvent_floor(alive[i]["kind"], alive[j]["kind"])
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < cut:
                bad.append((d, i, j))
        if not bad:
            break
        bad.sort()
        used: set[int] = set()
        for _d, i, j in bad:
            if i in used or j in used:
                continue          # its partner moved; settle it next round
            used |= {i, j}
            ki, kj = live[i][0], live[j][0]
            a, b, drop = alive[i], alive[j], kj
            if a["n"] < b["n"]:
                a, b, drop = alive[j], alive[i], ki
            total = a["n"] + b["n"]
            a["xyz"] = (a["xyz"] * a["n"] + b["xyz"] * b["n"]) / total
            a["n"] = total
            a["votes"] += b["votes"]
            a["pdbs"] |= b["pdbs"]
            a["kind"] = a["votes"].most_common(1)[0][0]
            sites[drop] = None
            merged += 1
    else:
        print(f"  WARNING: overlap repair hit {max_rounds} rounds and stopped")
    sites[:] = [s for s in sites if s is not None]
    return merged


def count_overlaps(sites: list[dict], merge_scale: float) -> int:
    """Pairs of sites closer than the floor their species pair allows."""
    if len(sites) < 2:
        return 0
    pos = np.array([s["xyz"] for s in sites])
    tree = cKDTree(pos)
    worst = max(MIN_SOLVENT_CONTACT.values()) * merge_scale
    return sum(
        1 for i, j in tree.query_pairs(r=worst)
        if np.linalg.norm(pos[i] - pos[j])
        < merge_scale * _solvent_floor(sites[i]["kind"], sites[j]["kind"]))


def rna_min_distance(points: np.ndarray, rna: np.ndarray, chunk: int = 4000):
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        block = points[i:i + chunk]
        out[i:i + chunk] = np.linalg.norm(block[:, None, :] - rna[None, :, :],
                                          axis=-1).min(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", required=True,
                    help="glob of donor mmCIFs (quote it)")
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frame", default="3G78",
                    help="PDB id whose RNA frame everything is placed in")
    ap.add_argument("--min-coverage", type=float, default=80.0,
                    help="drop donors covering less than this %% of the target")
    ap.add_argument("--max-rmsd", type=float, default=3.0,
                    help="reject a donor whose P-atom superposition onto the frame "
                         "exceeds this; a construct that is not the same molecule "
                         "still shares P numbering and lands tens of A off")
    ap.add_argument("--max-coverage", type=float, default=105.0,
                    help="reject donors with more modelled residues than the target "
                         "(a longer construct, not this intron)")
    ap.add_argument("--merge-scale", type=float, default=1.0,
                    help="multiply the species-pair contact floor used as the "
                         "merge distance (1.0 = merge exactly what could not "
                         "coexist)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seq = read_sequence(args.sequence)
    paths = sorted(Path(p) for p in glob.glob(args.templates))
    if not paths:
        raise SystemExit(f"no templates matched {args.templates!r}")

    frame = None
    donors = []
    for p in paths:
        try:
            tpl = load_template(p)
        except Exception as exc:  # a few entries are EM maps / odd assemblies
            print(f"  skip {p.stem.upper()}: {type(exc).__name__} {exc}")
            continue
        cov = 100.0 * len(tpl.p_map()) / max(len(seq), 1)
        if tpl.pdb_id == args.frame.upper():
            frame = tpl
        if not (args.min_coverage <= cov <= args.max_coverage) or not donor_solvent(tpl):
            continue
        donors.append((cov, tpl))
    if frame is None:
        raise SystemExit(f"frame {args.frame} not among the templates")
    donors.sort(key=lambda t: -t[0])
    print(f"donors kept: {len(donors)} of {len(paths)} "
          f"(coverage >= {args.min_coverage}%, solvent > 0)")

    # Pool every observation into the frame. Sorting by superposition quality
    # first means the crystals that land most confidently seed the clusters.
    obs: list[dict] = []
    kept: list[str] = []
    for cov, tpl in donors:
        if tpl.pdb_id == frame.pdb_id:
            rot, trans, rmsd, nshared = np.eye(3), np.zeros(3), 0.0, len(tpl.p_map())
        else:
            try:
                rot, trans, rmsd, nshared = superpose(tpl, frame)
            except ValueError as exc:
                print(f"  skip {tpl.pdb_id}: {exc}")
                continue
        if rmsd > args.max_rmsd:
            print(f"  reject {tpl.pdb_id}: P-RMSD {rmsd:.2f} A > {args.max_rmsd} "
                  f"(cov {cov:.0f}%) — not this molecule's frame")
            continue
        for kind, xyz in donor_solvent(tpl):
            obs.append({"kind": kind, "xyz": xyz @ rot + trans,
                        "pdb": tpl.pdb_id, "rmsd": rmsd})
        kept.append(tpl.pdb_id)
        print(f"  {tpl.pdb_id:6} cov {cov:5.1f}%  P-RMSD {rmsd:5.2f} A "
              f"({nshared} shared)  solvent {len(donor_solvent(tpl)):4d}")
    obs.sort(key=lambda o: o["rmsd"])
    print(f"\ndonors contributing: {len(kept)}")
    print(f"pooled observations: {len(obs)}  "
          f"({dict(Counter(o['kind'] for o in obs))})")

    # Greedy merge: an observation joins the first existing site it could not
    # physically coexist with. Sites keep a running mean so a cluster's position
    # is the consensus of its donors rather than whichever crystal came first.
    sites: list[dict] = []
    grid: dict[tuple[int, int, int], list[int]] = {}
    CELL = 4.0

    def cell_of(p):
        return (int(p[0] // CELL), int(p[1] // CELL), int(p[2] // CELL))

    for o in obs:
        best = None
        cx, cy, cz = cell_of(o["xyz"])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for idx in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        s = sites[idx]
                        d = float(np.linalg.norm(s["xyz"] - o["xyz"]))
                        cut = args.merge_scale * _solvent_floor(s["kind"], o["kind"])
                        if d < cut and (best is None or d < best[1]):
                            best = (idx, d)
        if best is None:
            sites.append({"xyz": o["xyz"].copy(), "kind": o["kind"],
                          "votes": Counter([o["kind"]]), "pdbs": {o["pdb"]}, "n": 1})
            grid.setdefault(cell_of(o["xyz"]), []).append(len(sites) - 1)
            continue
        s = sites[best[0]]
        s["n"] += 1
        s["xyz"] += (o["xyz"] - s["xyz"]) / s["n"]
        s["votes"][o["kind"]] += 1
        s["pdbs"].add(o["pdb"])
        s["kind"] = s["votes"].most_common(1)[0][0]

    before = len(sites)
    overlaps = count_overlaps(sites, args.merge_scale)
    merged = repair_overlaps(sites, args.merge_scale)
    left = count_overlaps(sites, args.merge_scale)
    print(f"greedy pass: {before} sites, {overlaps} pair(s) below their floor "
          f"(centroids drift as clusters accrete)")
    print(f"repair pass: merged {merged} — {len(sites)} sites, {left} pair(s) left")
    if left:
        raise SystemExit("overlap repair did not converge; refusing to write a "
                         "site list that cannot be written out together")

    print(f"consensus sites: {len(sites)}")
    sup = Counter(len(s["pdbs"]) for s in sites)
    print("\nsupport (distinct donor crystals per site):")
    for k in sorted(sup):
        cum = sum(v for kk, v in sup.items() if kk >= k)
        print(f"  >= {k:2d} donors: {cum:5d} sites   (exactly {k}: {sup[k]})")

    print("\nspecies of the consensus sites:", dict(Counter(s["kind"] for s in sites)))
    contested = [s for s in sites if len(s["votes"]) > 1]
    print(f"sites whose donors disagree on the species: {len(contested)}")
    if contested:
        ex = sorted(contested, key=lambda s: -len(s["pdbs"]))[:5]
        for s in ex:
            print(f"  {dict(s['votes'])} across {len(s['pdbs'])} donors")

    # Distance to the frame's RNA: a site burrowed into the backbone is a
    # transfer artefact, and the assessors only score sites near ordered residues.
    rna = np.array([xyz for atoms in frame.residues.values() for xyz in atoms.values()])
    pts = np.array([s["xyz"] for s in sites])
    dmin = rna_min_distance(pts, rna)
    floor = np.array([min(MIN_RNA_CONTACT.get((s["kind"], e), 2.4)
                          for e in ("O", "N", "C", "P")) for s in sites])
    print(f"\nsites closer to RNA than the species floor: {(dmin < floor).sum()}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([
            {"xyz": [round(float(v), 3) for v in s["xyz"]], "kind": s["kind"],
             "n_obs": s["n"], "n_donors": len(s["pdbs"]),
             "donors": sorted(s["pdbs"]), "votes": dict(s["votes"]),
             "rna_min": round(float(d), 2)}
            for s, d in zip(sites, dmin)], indent=1))
        print(f"\nwrote {args.out} ({len(sites)} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
