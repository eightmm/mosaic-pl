#!/usr/bin/env python3
"""Does per-fragment holo cofolding actually move the receptor / binding site?

Motivating question: we run the whole pipeline (cofold + dock) once per fragment.
If the predicted receptor is essentially identical across fragments, that cofold
cost is wasted and we could fold ONE receptor and only dock each fragment.

Measures, over a sample of completed holo runs (one consistent cofolding model
per fragment, default boltz2x/seed_101/model_0):

  1. global CA RMSD  — each holo receptor vs a reference receptor, after optimal
     (Kabsch) superposition. "How different is the fold/backbone?"
  2. pocket CA RMSD  — same, restricted to the reference pocket residues.
     "How different is the binding site itself?"
  3. ligand centroid — where the fragment sits, expressed in the reference frame
     (i.e. after superposing that fragment's receptor onto the reference). Spread
     of these centroids answers "do fragments even bind the same site?"
  4. per-residue pocket displacement — which pocket residues move the most.

Reference: the apo run's receptor when available (``--reference apo``), else the
first sampled holo receptor. Comparing holo-vs-apo directly answers "does adding
the ligand change the receptor?".

Output: a CSV of per-fragment metrics + a printed summary with percentiles.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    import gemmi
except ImportError:  # pragma: no cover
    sys.exit("gemmi is required")

REPO = Path(__file__).resolve().parents[1]

# Cofolding model to compare. One consistent choice per fragment, otherwise we'd
# be measuring model/seed differences instead of ligand-induced differences.
REL_BOLTZ2X = "outputs/boltz2x/seed_101/boltz_results_boltz_input/predictions/boltz_input/boltz_input_model_0.cif"


def load_structure(path: Path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def _is_aa(res) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info and info.is_amino_acid())


def _is_water(res) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info and info.is_water())


def ca_coords(st) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """CA coordinates of polymer residues + their (chain, seqid) labels."""
    xyz, labels = [], []
    for model in st:
        for chain in model:
            for res in chain:
                if not _is_aa(res):
                    continue
                at = res.find_atom("CA", "*")
                if at is None:
                    continue
                xyz.append([at.pos.x, at.pos.y, at.pos.z])
                labels.append((chain.name, res.seqid.num))
        break  # first model only
    return np.asarray(xyz, dtype=float), labels


def ligand_heavy_coords(st) -> np.ndarray:
    """Heavy atoms of non-polymer, non-water residues (the docked fragment)."""
    xyz = []
    for model in st:
        for chain in model:
            for res in chain:
                if _is_aa(res) or _is_water(res):
                    continue
                for at in res:
                    if at.element == gemmi.Element("H"):
                        continue
                    xyz.append([at.pos.x, at.pos.y, at.pos.z])
        break
    return np.asarray(xyz, dtype=float)


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Optimal rotation+translation taking `mobile` onto `target`."""
    mc, tc = mobile.mean(0), target.mean(0)
    P, Q = mobile - mc, target - tc
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R, mc, tc


def apply_tf(x: np.ndarray, R, mc, tc) -> np.ndarray:
    return (x - mc) @ R + tc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def pocket_mask(labels, ca_xyz, lig_xyz, cutoff=8.0) -> np.ndarray:
    """Residues whose CA is within `cutoff` of any ligand heavy atom.

    CA-based (not any-heavy-atom) because we compare CA positions; 8 A on CA is
    the usual proxy for the 5 A heavy-atom pocket definition.
    """
    if lig_xyz.size == 0:
        return np.zeros(len(ca_xyz), dtype=bool)
    d = np.linalg.norm(ca_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1)
    return d.min(axis=1) <= cutoff


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holo-root", type=Path,
                    default=REPO / "experiments/ligand_series/L01/holo")
    ap.add_argument("--apo-cif", type=Path,
                    default=REPO / "experiments/ligand_series/L01/L01_apo" / REL_BOLTZ2X)
    ap.add_argument("--rel-model", default=REL_BOLTZ2X,
                    help="cofolding model path relative to a fragment run dir")
    ap.add_argument("--sample", type=int, default=100, help="number of fragments (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pocket-cutoff", type=float, default=8.0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/ligand_series/holo_receptor_variation.csv")
    args = ap.parse_args()

    runs = sorted(p for p in args.holo_root.glob("L0*") if (p / args.rel_model).is_file())
    if not runs:
        sys.exit(f"no fragment runs with {args.rel_model} under {args.holo_root}")
    if args.sample and args.sample < len(runs):
        random.Random(args.seed).shuffle(runs)
        runs = sorted(runs[: args.sample])
    print(f"[variation] {len(runs)} fragments sampled from {args.holo_root}")

    # ---- reference ---------------------------------------------------------- #
    ref_kind = "apo"
    if args.apo_cif.is_file():
        ref_st = load_structure(args.apo_cif)
    else:
        ref_kind = "first_holo"
        ref_st = load_structure(runs[0] / args.rel_model)
    ref_ca, ref_labels = ca_coords(ref_st)
    ref_index = {lab: i for i, lab in enumerate(ref_labels)}
    print(f"[variation] reference = {ref_kind}, {len(ref_ca)} CA")

    # Reference pocket: defined from the FIRST holo fragment's ligand, mapped
    # onto the reference numbering (the apo model has no ligand of its own).
    first_st = load_structure(runs[0] / args.rel_model)
    first_ca, first_labels = ca_coords(first_st)
    first_lig = ligand_heavy_coords(first_st)
    R0, mc0, tc0 = kabsch(_common(first_ca, first_labels, ref_index),
                          _ref_common(ref_ca, first_labels, ref_index))
    lig_in_ref = apply_tf(first_lig, R0, mc0, tc0) if first_lig.size else first_lig
    pmask = pocket_mask(ref_labels, ref_ca, lig_in_ref, args.pocket_cutoff)
    print(f"[variation] reference pocket: {int(pmask.sum())} residues "
          f"(CA within {args.pocket_cutoff} A of the first fragment's ligand)")

    rows = []
    pocket_disp_acc = np.zeros(int(pmask.sum()))
    n_acc = 0
    for run in runs:
        try:
            st = load_structure(run / args.rel_model)
            ca, labels = ca_coords(st)
            lig = ligand_heavy_coords(st)
            mob = _common(ca, labels, ref_index)
            tgt = _ref_common(ref_ca, labels, ref_index)
            if len(mob) < 20:
                continue
            R, mc, tc = kabsch(mob, tgt)
            mob_fit = apply_tf(mob, R, mc, tc)
            g_rmsd = rmsd(mob_fit, tgt)

            # pocket subset, in reference indexing
            idx = np.array([ref_index[l] for l in labels if l in ref_index])
            in_pocket = pmask[idx]
            p_rmsd = rmsd(mob_fit[in_pocket], tgt[in_pocket]) if in_pocket.any() else np.nan
            if in_pocket.any() and in_pocket.sum() == pocket_disp_acc.size:
                pocket_disp_acc += np.linalg.norm(mob_fit[in_pocket] - tgt[in_pocket], axis=1)
                n_acc += 1

            lig_c = apply_tf(lig, R, mc, tc).mean(0) if lig.size else np.full(3, np.nan)
            rows.append({
                "fragment": run.name,
                "n_ca": len(mob),
                "global_ca_rmsd": g_rmsd,
                "pocket_ca_rmsd": p_rmsd,
                "lig_cx": lig_c[0], "lig_cy": lig_c[1], "lig_cz": lig_c[2],
            })
        except Exception as exc:
            print(f"  skip {run.name}: {type(exc).__name__}: {exc}")

    if not rows:
        sys.exit("no fragments could be compared")

    import pandas as pd
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- summary ------------------------------------------------------------ #
    def pct(s):
        s = s.dropna()
        return {k: round(float(np.percentile(s, q)), 3)
                for k, q in (("p05", 5), ("median", 50), ("p95", 95), ("max", 100))}

    lig_xyz = df[["lig_cx", "lig_cy", "lig_cz"]].to_numpy()
    lig_ok = lig_xyz[~np.isnan(lig_xyz).any(axis=1)]
    lig_center = lig_ok.mean(0) if len(lig_ok) else np.full(3, np.nan)
    lig_spread = (np.linalg.norm(lig_ok - lig_center, axis=1) if len(lig_ok)
                  else np.array([np.nan]))

    summary = {
        "reference": ref_kind,
        "n_fragments": len(df),
        "pocket_residues": int(pmask.sum()),
        "global_ca_rmsd_A": pct(df["global_ca_rmsd"]),
        "pocket_ca_rmsd_A": pct(df["pocket_ca_rmsd"]),
        "ligand_centroid_spread_A": {
            "median": round(float(np.median(lig_spread)), 3),
            "p95": round(float(np.percentile(lig_spread, 95)), 3),
            "max": round(float(lig_spread.max()), 3),
            "n_with_ligand": int(len(lig_ok)),
        },
    }
    if n_acc:
        worst = np.argsort(pocket_disp_acc / n_acc)[::-1][:10]
        plabels = [ref_labels[i] for i in np.where(pmask)[0]]
        summary["worst_moving_pocket_residues"] = [
            {"residue": f"{plabels[i][0]}{plabels[i][1]}",
             "mean_disp_A": round(float(pocket_disp_acc[i] / n_acc), 3)}
            for i in worst
        ]
    print(json.dumps(summary, indent=2))
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[variation] wrote {args.out}")
    return 0


def _common(ca: np.ndarray, labels, ref_index) -> np.ndarray:
    """Rows of `ca` whose residue label exists in the reference."""
    keep = [i for i, l in enumerate(labels) if l in ref_index]
    return ca[keep]


def _ref_common(ref_ca: np.ndarray, labels, ref_index) -> np.ndarray:
    """Reference rows matching `labels`, in the same order as `_common`."""
    keep = [ref_index[l] for l in labels if l in ref_index]
    return ref_ca[keep]


if __name__ == "__main__":
    raise SystemExit(main())
