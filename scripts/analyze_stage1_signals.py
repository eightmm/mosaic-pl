#!/usr/bin/env python3
"""Score every CASP17 ligand-series stage-1 signal against the revealed truth.

The stage-2 reveal publishes the real binder labels as a `binding` column in the
target's SMILES CSV:

    curl -s "https://predictioncenter.org/casp17/target.cgi?target=L01&view=smiles" \
      -o inputs/ligand_series/L01/ligands_truth.csv

This script joins those labels to every per-fragment signal we can produce —
the submitted `bind.txt` score, Boltz-2 affinity, the pose rescorers
(AKScore2 / GenScore / EquiScore) and plain RDKit descriptors — and reports
AUC, BEDROC(alpha=20) and enrichment factors, plus a permutation baseline so the
BEDROC numbers are interpretable.

    uv run python scripts/analyze_stage1_signals.py --target L01
    uv run python scripts/analyze_stage1_signals.py --target L02 --restrict-common

Results as of 2026-07-29 are written up in docs/ligand_series_stage1_postmortem.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Per-pose score directions. `ak_ens` is energy-like (lower is better) and is
# summed with `ak_dock` to form `ak_score`; see scripts/score_poses.py.
POSE_COLS = {
    "ak_score": True,
    "ak_ens": True,
    "ak_dock": True,
    "gen_score": False,
    "equi_pred": False,
}

DESCRIPTORS = {
    "mol.heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "mol.MW": None,
    "mol.cLogP": None,
    "mol.TPSA": None,
    "mol.HBD": None,
    "mol.HBA": None,
    "mol.rot_bonds": None,
    "mol.rings": None,
    "mol.arom_rings": None,
    "mol.fracCsp3": None,
    "mol.QED": None,
    "mol.formal_chg": None,
}

GROUPS = [
    ("submitted", ["submitted"]),
    ("boltz", ["boltz_prob_mean", "boltz_prob_max", "boltz_aff_min", "boltz_aff_mean", "boltz_LE"]),
    ("akscore2", ["ak_score_best", "ak_score_mean", "ak_ens_best", "ak_ens_mean",
                  "ak_dock_best", "ak_dock_mean"]),
    ("genscore", ["gen_score_best", "gen_score_p90", "gen_score_mean"]),
    ("equiscore", ["equi_pred_best", "equi_pred_p90", "equi_pred_mean"]),
    ("molprop", list(DESCRIPTORS)),
]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def avg_ranks_desc(s: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score; ties share the average rank."""
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def bedroc_from_ranks(ranks: np.ndarray, y: np.ndarray, alpha: float = 20.0) -> float:
    """Truchon & Bayly (2007) BEDROC."""
    n_tot, n_act = len(ranks), int(y.sum())
    ra = n_act / n_tot
    ri = ranks[y] / n_tot
    rie_num = np.exp(-alpha * ri).sum() / n_act
    rie_den = (1 / n_tot) * (1 - math.exp(-alpha)) / (math.exp(alpha / n_tot) - 1)
    rie = rie_num / rie_den
    return (rie * ra * math.sinh(alpha / 2)
            / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1 / (1 - math.exp(alpha * (1 - ra))))


def metrics(scores, labels, alpha: float = 20.0):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n_tot, n_act = len(s), int(y.sum())
    if n_act in (0, n_tot):
        return None
    ranks = avg_ranks_desc(s)
    rr = n_tot + 1 - ranks
    out = {
        "N": n_tot,
        "n": n_act,
        "AUC": (rr[y].sum() - n_act * (n_act + 1) / 2) / (n_act * (n_tot - n_act)),
        "BEDROC": bedroc_from_ranks(ranks, y, alpha),
    }
    order = np.argsort(-s, kind="mergesort")
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(round(n_tot * frac)))
        hits = int(y[order[:k]].sum())
        pct = int(frac * 100)
        out[f"EF{pct}"] = hits / (n_act * k / n_tot)
        out[f"hits{pct}"] = hits
    return out


def random_bedroc(n_tot: int, n_act: int, alpha: float = 20.0, trials: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = np.empty(trials)
    for t in range(trials):
        s = rng.random(n_tot)
        y = np.zeros(n_tot, bool)
        y[rng.choice(n_tot, n_act, replace=False)] = True
        vals[t] = bedroc_from_ranks(avg_ranks_desc(s), y, alpha)
    return float(vals.mean()), float(np.percentile(vals, 95))


# --------------------------------------------------------------------------- #
# signal collection
# --------------------------------------------------------------------------- #
def read_truth(path: Path):
    rows = list(csv.DictReader(open(path)))
    if "binding" not in rows[0]:
        raise SystemExit(f"{path} has no `binding` column — stage-2 truth not published yet?")
    return {r["CASP ID"]: (r["binding"].upper() == "TRUE", r["canonical_smiles"]) for r in rows}


def read_submitted(path: Path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")[0].split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def boltz_signals(run: Path):
    probs, vals = [], []
    for jf in run.glob("outputs/boltz*/seed_*/**/affinity_*.json"):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        p, v = d.get("affinity_probability_binary"), d.get("affinity_pred_value")
        if isinstance(p, (int, float)):
            probs.append(float(p))
        if isinstance(v, (int, float)):
            vals.append(float(v))
    out = {}
    if probs:
        out["boltz_prob_mean"] = float(np.mean(probs))
        out["boltz_prob_max"] = float(np.max(probs))
    if vals:  # pred_value is lower-is-better -> flip so every signal is higher-is-better
        out["boltz_aff_min"] = -float(np.min(vals))
        out["boltz_aff_mean"] = -float(np.mean(vals))
    return out


def pose_signals(run: Path):
    cs = run / "outputs/scoring/pose_scores.csv"
    if not cs.exists():
        return {}
    cols = {c: [] for c in POSE_COLS}
    with open(cs) as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                v = row.get(c)
                if v in (None, "", "NaN", "nan"):
                    continue
                try:
                    cols[c].append(float(v))
                except ValueError:
                    pass
    out = {"n_poses": len(cols["gen_score"])}
    for c, lower in POSE_COLS.items():
        v = np.asarray(cols[c], float)
        if v.size == 0:
            continue
        s = -v if lower else v          # normalise to higher-is-better
        out[f"{c}_best"] = float(s.max())
        out[f"{c}_p90"] = float(np.percentile(s, 90))
        out[f"{c}_mean"] = float(s.mean())
    return out


def descriptor_signals(smiles: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    fns = {
        "mol.heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
        "mol.MW": Descriptors.MolWt,
        "mol.cLogP": Crippen.MolLogP,
        "mol.TPSA": rdMolDescriptors.CalcTPSA,
        "mol.HBD": rdMolDescriptors.CalcNumHBD,
        "mol.HBA": rdMolDescriptors.CalcNumHBA,
        "mol.rot_bonds": rdMolDescriptors.CalcNumRotatableBonds,
        "mol.rings": rdMolDescriptors.CalcNumRings,
        "mol.arom_rings": rdMolDescriptors.CalcNumAromaticRings,
        "mol.fracCsp3": rdMolDescriptors.CalcFractionCSP3,
        "mol.QED": QED.qed,
        "mol.formal_chg": Chem.GetFormalCharge,
    }
    m = Chem.MolFromSmiles(smiles)
    out = {}
    for k, fn in fns.items():
        try:
            out[k] = float(fn(m)) if m is not None else np.nan
        except Exception:
            out[k] = np.nan
    return out


def legacy_run_dir(target: str, fid: str) -> str:
    """Map an official ligand id to the stage-1 run directory that holds it.

    The stage-1 `ligands.csv` for L01 was built from a scrape that deduplicated
    the official list's one repeated compound (L010356 and L010357 carry the
    same SMILES), so it had 1209 rows and every id from L010357 on was shifted
    down by one: our dir `L010357` actually contains official L010358's
    molecule. Official L010357 itself is chemically identical to L010356, so its
    signals come from that dir.

    Only meaningful for stage-1 artifacts. Phase-2 runs are generated straight
    from the truth CSV and need no remap.
    """
    if target != "L01":
        return fid
    n = int(fid[3:])
    if n <= 356:
        return fid
    if n == 357:
        return "L010356"
    return f"L01{n - 1:04d}"


def collect(target: str, holo_root: Path, truth_csv: Path, bind_txt: Path,
            legacy_id_shift: bool = False):
    truth = read_truth(truth_csv)
    submitted = read_submitted(bind_txt)
    rows = []
    for fid, (label, smi) in truth.items():
        r = {"frag": fid, "label": label, "submitted": submitted.get(fid, np.nan)}
        r.update(descriptor_signals(smi))
        run = holo_root / (legacy_run_dir(target, fid) if legacy_id_shift else fid)
        if run.is_dir():
            r.update(boltz_signals(run))
            r.update(pose_signals(run))
        heavy = r.get("mol.heavy_atoms") or 0
        if heavy and not np.isnan(r.get("boltz_aff_min", np.nan)):
            r["boltz_LE"] = r["boltz_aff_min"] / heavy      # ligand efficiency
        rows.append(r)
    return rows


def print_table(target: str, rows, restrict_key: str | None = None, alpha: float = 20.0):
    y = [r["label"] for r in rows]
    keep = None
    title = f"{target}  N={len(rows)}  binders={sum(y)}"
    if restrict_key:
        keep = [not np.isnan(r.get(restrict_key, np.nan)) for r in rows]
        title += f"  (restricted to the {sum(keep)} fragments with `{restrict_key}`)"
    print(f"\n### {title}")
    n_tot = sum(keep) if keep else len(rows)
    n_act = sum(1 for i, r in enumerate(rows) if r["label"] and (not keep or keep[i]))
    rb_mean, rb_p95 = random_bedroc(n_tot, n_act, alpha)
    print(f"random baseline: AUC 0.500 | BEDROC(a={alpha:g}) mean {rb_mean:.3f}, "
          f"95th pct {rb_p95:.3f} | EF 1.00")
    print("\n| group | signal | n | AUC | BEDROC | EF1% | EF5% | EF10% | hits@5% |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for gname, keys in GROUPS:
        for k in keys:
            vals = [r.get(k, np.nan) for r in rows]
            if all(isinstance(v, float) and np.isnan(v) for v in vals):
                continue
            if keep:
                vals = [v if keep[i] else np.nan for i, v in enumerate(vals)]
            m = metrics(vals, y, alpha)
            if m is None:
                continue
            flag = " *" if m["BEDROC"] > rb_p95 else ""
            print(f"| {gname} | {k} | {m['N']} | {m['AUC']:.3f} | {m['BEDROC']:.3f}{flag} | "
                  f"{m['EF1']:.2f} | {m['EF5']:.2f} | {m['EF10']:.2f} | {m['hits5']}/{m['n']} |")
    print("\n`*` = BEDROC above the 95th percentile of the random permutation baseline.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["L01", "L02"])
    ap.add_argument("--truth-csv", type=Path, default=None)
    ap.add_argument("--bind-txt", type=Path, default=None)
    ap.add_argument("--holo-root", type=Path, default=None)
    ap.add_argument("--group", default="129")
    ap.add_argument("--alpha", type=float, default=20.0, help="BEDROC alpha")
    ap.add_argument("--restrict-common", action="store_true",
                    help="also print a table restricted to fragments that have Boltz output")
    ap.add_argument("--legacy-id-shift", action="store_true",
                    help="stage-1 artifacts only: remap official ids onto the L01 run "
                         "dirs, which are shifted by one from L010357 on (see "
                         "legacy_run_dir)")
    args = ap.parse_args()

    T = args.target
    truth = args.truth_csv or REPO / f"inputs/ligand_series/{T}/ligands_truth.csv"
    bind = args.bind_txt or REPO / f"experiments/ligand_series/{T}LG{args.group}.bind.txt"
    holo = args.holo_root or REPO / f"experiments/ligand_series/{T}/holo"
    if not truth.exists():
        raise SystemExit(
            f"missing {truth}\n  curl -s "
            f'"https://predictioncenter.org/casp17/target.cgi?target={T}&view=smiles" -o {truth}')

    rows = collect(T, holo, truth, bind, legacy_id_shift=args.legacy_id_shift)
    print_table(T, rows, alpha=args.alpha)
    if args.restrict_common:
        print_table(T, rows, restrict_key="boltz_prob_mean", alpha=args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
