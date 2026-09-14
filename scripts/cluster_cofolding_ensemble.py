#!/usr/bin/env python3
"""Cluster a target's cofolding predictions and say whether they hold >1 state.

Written for CASP17 T2451, where the experimentalists reported that the protein
"crystallizes in two distinct conformations" and the submission has to spend
MODELs 1-5 on one and 6,7,8,9,0 on the other. The question that decides how to
fill those slots is whether *our own* sampling ever produced a second state —
and answering it by eye over 100 mmCIFs is how you end up shipping ten models of
one conformation.

Three things this gets right that an ad-hoc script usually does not:

- **`*_aligned.cif` copies are dropped.** `align_cofolding_outputs.py` writes an
  aligned twin of every prediction; counting both doubles the ensemble and
  plants a fake 2-member cluster at distance 0.
- **Chain assignment is optimised per comparison.** For a homo-oligomer the
  chain labels are arbitrary, so a dimer RMSD that assumes A↔A is measuring
  label noise on top of geometry.
- **Quaternary and tertiary are reported separately.** A pair of structures can
  be 2 A apart as assemblies and 0.3 A apart as protomers; those two numbers
  mean completely different things for "is there a second conformation".

Usage:
    uv run python scripts/cluster_cofolding_ensemble.py \
      --run-dir experiments/CASP17/T2451/run/T2451 \
      --source alphafold3 --source protenix --source boltz_rerun \
      --cutoffs 0.4 0.6 0.8 1.0 1.5 2.0 \
      --compare-lg experiments/CASP17/T2451/submissions/T2451_LCDD.lg
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np

_RESIDUE_INFO = gemmi.find_tabulated_residue


def chain_ca(path: Path, minlen: int) -> list[dict[int, np.ndarray]]:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = []
    for model in st:
        for chain in model:
            d = {}
            for res in chain:
                tab = _RESIDUE_INFO(res.name)
                atom = res.find_atom("CA", "*")
                if tab and tab.is_amino_acid() and atom:
                    d[res.seqid.num] = np.array(atom.pos.tolist())
            if len(d) >= minlen:
                out.append(d)
        break
    return out


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    ac, bc = a - a.mean(0), b - b.mean(0)
    v, _, wt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1, 1, d]) @ wt
    return float(np.sqrt((((ac @ rot) - bc) ** 2).sum() / len(a)))


def assembly_rmsd(x: list[dict], y: list[dict]) -> float:
    """Best CA RMSD over chain assignments — labels are arbitrary in a homomer."""
    if len(x) != len(y):
        return float("nan")
    best = float("inf")
    for perm in itertools.permutations(range(len(y))):
        a, b = [], []
        for i, j in enumerate(perm):
            keys = sorted(set(x[i]) & set(y[j]))
            a += [x[i][k] for k in keys]
            b += [y[j][k] for k in keys]
        if not a:
            continue
        best = min(best, kabsch_rmsd(np.array(a), np.array(b)))
    return best


def single_link(dist: np.ndarray, cutoff: float) -> Counter:
    lab = list(range(len(dist)))
    for i in range(len(dist)):
        for j in range(i + 1, len(dist)):
            if dist[i, j] <= cutoff:
                a, b = lab[i], lab[j]
                if a != b:
                    lab = [a if x == b else x for x in lab]
    return Counter(lab), lab


def lg_model_chains(path: Path) -> dict[str, list[dict[int, np.ndarray]]]:
    models: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    cur = name = None
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            name, cur = line.split()[1], {}
        elif line.startswith("ATOM") and cur is not None and line[12:16].strip() == "CA":
            cur.setdefault(line[21], {})[int(line[22:26])] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("END") and cur is not None:
            models[name] = cur
            cur = None
    return {k: list(v.values()) for k, v in models.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source", action="append", default=[],
                    help="subdirectory of <run-dir>/outputs to scan, repeatable")
    ap.add_argument("--min-residues", type=int, default=100)
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    ap.add_argument("--compare-lg", type=Path,
                    help="also report each LG MODEL's distance to the ensemble")
    ap.add_argument("--pick", type=int, default=0, metavar="N",
                    help="print N maximally-separated structures, restricted to "
                         "--pick-source when given; use them as arrangement templates")
    ap.add_argument("--pick-source", default=None,
                    help="limit --pick to structures under this source subdirectory")
    args = ap.parse_args()

    outputs = args.run_dir / "outputs"
    files: list[Path] = []
    for src in args.source or ["alphafold3", "protenix", "boltz2", "boltz2x"]:
        found = [p for p in sorted((outputs / src).rglob("*.cif"))
                 if not p.name.endswith("_aligned.cif")]
        print(f"  {src}: {len(found)} primary cif")
        files += found
    if not files:
        raise SystemExit("no cofolding structures found")

    structures, names = [], []
    for f in files:
        ch = chain_ca(f, args.min_residues)
        if ch:
            structures.append(ch)
            names.append(str(f.relative_to(outputs)))
    sizes = Counter(len(s) for s in structures)
    print(f"\nloaded {len(structures)} structures; chains per structure: {dict(sizes)}")
    keep = sizes.most_common(1)[0][0]
    if len(sizes) > 1:
        print(f"  restricting to the {sizes[keep]} with {keep} chain(s)")
        pairs = [(s, n) for s, n in zip(structures, names) if len(s) == keep]
        structures, names = [p[0] for p in pairs], [p[1] for p in pairs]

    n = len(structures)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = assembly_rmsd(structures[i], structures[j])
    iu = np.triu_indices(n, 1)
    print(f"\nassembly CA RMSD: median={np.median(dist[iu]):.2f}  "
          f"p95={np.percentile(dist[iu], 95):.2f}  max={dist.max():.2f} A")

    protomers = [c for s in structures for c in s]
    shared = sorted(set.intersection(*[set(p) for p in protomers]))
    mono = [np.array([p[k] for k in shared]) for p in protomers]
    step = max(1, len(mono) // 60)
    md = [kabsch_rmsd(mono[i], mono[j])
          for i in range(0, len(mono), step) for j in range(i + step, len(mono), step)]
    print(f"protomer CA RMSD ({len(mono)} chains, {len(shared)} shared residues): "
          f"median={np.median(md):.2f}  max={max(md):.2f} A")

    print("\nsingle-link clustering of assemblies:")
    for cut in args.cutoffs:
        counts, _ = single_link(dist, cut)
        print(f"  {cut:>4.1f} A -> {len(counts):3d} clusters, "
              f"sizes {sorted(counts.values(), reverse=True)[:8]}")

    # A real second state is a *populated* cluster that stays separate; a lone
    # structure at the rim is sampling noise. Report both so they are not confused.
    counts, lab = single_link(dist, args.cutoffs[-2] if len(args.cutoffs) > 1
                              else args.cutoffs[0])
    populated = [k for k, v in counts.items() if v >= max(2, int(0.05 * n))]
    print(f"\npopulated clusters (>=5% of the ensemble) at "
          f"{args.cutoffs[-2] if len(args.cutoffs) > 1 else args.cutoffs[0]} A: "
          f"{len(populated)}")
    if len(populated) > 1:
        reps = []
        for k in populated:
            mem = [i for i, x in enumerate(lab) if x == k]
            rep = min(mem, key=lambda i: dist[np.ix_([i], mem)].mean())
            reps.append((k, len(mem), rep))
        for k, size, rep in reps:
            print(f"  cluster {k}: n={size}  representative {names[rep]}")
        for (_, _, a), (_, _, b) in itertools.combinations(reps, 2):
            print(f"  separation {names[a]} <-> {names[b]}: {dist[a, b]:.2f} A")
    else:
        med = np.median(dist, axis=1)
        o = int(np.argmax(med))
        print(f"  one state only. Most peripheral structure: {names[o]} "
              f"(median {med[o]:.2f} A, max {dist[o].max():.2f} A)")

    if args.compare_lg:
        print(f"\ndistance from each MODEL of {args.compare_lg.name} to the ensemble:")
        for num, chains in lg_model_chains(args.compare_lg).items():
            if len(chains) != keep:
                continue
            d = np.array([assembly_rmsd(chains, s) for s in structures])
            print(f"  MODEL {num:>2}: nearest {d.min():5.2f} A ({names[int(d.argmin())]})"
                  f"   median {np.median(d):5.2f} A")

    if args.pick:
        pool = [i for i, nm in enumerate(names)
                if args.pick_source is None or nm.startswith(args.pick_source)]
        if not pool:
            raise SystemExit(f"--pick-source {args.pick_source!r} matched nothing")
        # Farthest-point sampling: seed with the pool's medoid, then repeatedly
        # add whichever candidate is furthest from everything already chosen.
        # Picking the top-N by spread instead would return one extreme and its
        # near-twins.
        sub = dist[np.ix_(pool, pool)]
        chosen = [int(np.argmin(sub.mean(1)))]
        while len(chosen) < min(args.pick, len(pool)):
            rest = [k for k in range(len(pool)) if k not in chosen]
            chosen.append(max(rest, key=lambda k: min(sub[k, c] for c in chosen)))
        print(f"\n{args.pick} representatives"
              f"{' from ' + args.pick_source if args.pick_source else ''} "
              f"(farthest-point, medoid first):")
        for rank, k in enumerate(chosen, 1):
            i = pool[k]
            sep = min((sub[k, c] for c in chosen if c != k), default=0.0)
            print(f"  {rank}. {names[i]}")
            print(f"       separation from the other picks: {sep:.2f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
