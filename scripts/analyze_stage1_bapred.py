#!/usr/bin/env python3
"""Score the ligand-series stage-1 fragments with BA-Pred / RMSD-Pred output.

The stage-1 submission never used the post-analysis GNNs as a *binder* signal —
`bind.txt` was built from Boltz-2 affinity plus the docking scores. But every
holo run already carries per-pose BA-Pred (pKd) and RMSD-Pred (pRMSD +
P(RMSD > 2 A)) tables under ``outputs/analysis/``, so the counterfactual "what
if we had ranked the library by BA-Pred?" can be answered from artifacts on
disk — no GPU re-run needed.

This aggregates those per-pose tables into per-fragment scalars, joins the
stage-2 truth labels and reports AUC / BEDROC(alpha=20) / EF against the same
permutation baseline used by ``analyze_stage1_signals.py``.

    uv run python scripts/analyze_stage1_bapred.py --target L01 --legacy-id-shift
    uv run python scripts/analyze_stage1_bapred.py --target L02

`--csv` appends a long-format row per signal so the deck figures can consume it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_stage1_signals import (  # noqa: E402
    descriptor_signals,
    legacy_run_dir,
    metrics,
    random_bedroc,
    read_truth,
)

# Higher-is-better after the transforms applied in `pose_signals`.
SIGNALS = [
    ("bapred_pkd_best", "BA-Pred pKd (best pose)"),
    ("bapred_pkd_p90", "BA-Pred pKd (p90)"),
    ("bapred_pkd_mean", "BA-Pred pKd (mean)"),
    ("bapred_pkd_top10", "BA-Pred pKd (top-10 mean)"),
    ("bapred_pkd_reliable", "BA-Pred pKd (best reliable pose)"),
    ("bapred_at_best_lscore", "BA-Pred pKd @ pipeline-selected pose"),
    ("bapred_x_lscore", "BA-Pred pKd x LSCORE (best)"),
    ("bapred_LE", "BA-Pred ligand efficiency"),
    ("lscore_best", "LSCORE (best pose)"),
    ("lscore_mean", "LSCORE (mean)"),
    ("lscore_frac_reliable", "fraction of poses with LSCORE >= 0.5"),
    ("n_poses_scored", "pose count (control)"),
]


def pose_signals(run: Path) -> dict:
    """Aggregate every BA-Pred / RMSD-Pred pose row of one fragment run."""
    adir = run / "outputs" / "analysis"
    if not adir.is_dir():
        return {}

    pkd: dict[str, float] = {}
    for tsv in adir.glob("ba_pred_*.tsv"):
        with open(tsv) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                name = (row.get("Name") or "").strip()
                try:
                    pkd[name] = float(row["pKd"])
                except (KeyError, TypeError, ValueError):
                    continue

    lsc: dict[str, float] = {}
    for tsv in adir.glob("rmsd_pred_*.tsv"):
        with open(tsv) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                name = (row.get("Name") or "").strip()
                try:
                    # LSCORE, exactly as compute_submission_scores.PoseScore does it.
                    lsc[name] = max(0.0, min(1.0, 1.0 - float(row["Is_Above_2A"])))
                except (KeyError, TypeError, ValueError):
                    continue

    if not pkd and not lsc:
        return {}

    out: dict[str, float] = {}
    if pkd:
        v = np.fromiter(pkd.values(), float)
        out["bapred_pkd_best"] = float(v.max())
        out["bapred_pkd_p90"] = float(np.percentile(v, 90))
        out["bapred_pkd_mean"] = float(v.mean())
        out["bapred_pkd_top10"] = float(np.sort(v)[-10:].mean())
        out["n_poses_scored"] = float(v.size)
    if lsc:
        s = np.fromiter(lsc.values(), float)
        out["lscore_best"] = float(s.max())
        out["lscore_mean"] = float(s.mean())
        out["lscore_frac_reliable"] = float((s >= 0.5).mean())

    shared = [n for n in pkd if n in lsc]
    if shared:
        pairs = [(pkd[n], lsc[n]) for n in shared]
        best_l = max(pairs, key=lambda t: t[1])
        out["bapred_at_best_lscore"] = float(best_l[0])
        out["bapred_x_lscore"] = float(max(p * l for p, l in pairs))
        reliable = [p for p, l in pairs if l >= 0.5]
        if reliable:
            out["bapred_pkd_reliable"] = float(max(reliable))
    return out


def collect(target: str, holo_root: Path, truth_csv: Path, legacy_id_shift: bool):
    truth = read_truth(truth_csv)
    rows = []
    for fid, (label, smi) in truth.items():
        r = {"frag": fid, "label": label}
        run = holo_root / (legacy_run_dir(target, fid) if legacy_id_shift else fid)
        r.update(pose_signals(run))
        heavy = descriptor_signals(smi).get("mol.heavy_atoms") or 0
        if heavy and "bapred_pkd_best" in r:
            r["bapred_LE"] = r["bapred_pkd_best"] / heavy
        rows.append(r)
    return rows


def report(target: str, rows, alpha: float, csv_out: Path | None):
    y = [r["label"] for r in rows]
    covered = sum(1 for r in rows if "bapred_pkd_best" in r or "lscore_best" in r)
    print(f"\n### {target}  N={len(rows)}  binders={sum(y)}  "
          f"post-analysis coverage={covered}/{len(rows)}")

    keep = [("bapred_pkd_best" in r) or ("lscore_best" in r) for r in rows]
    n_tot = sum(keep)
    n_act = sum(1 for i, r in enumerate(rows) if r["label"] and keep[i])
    rb_mean, rb_p95 = random_bedroc(n_tot, n_act, alpha)
    print(f"scored subset: N={n_tot} binders={n_act} | random BEDROC(a={alpha:g}) "
          f"mean {rb_mean:.3f}, 95th pct {rb_p95:.3f}")

    print("\n| signal | n | AUC | BEDROC | EF1% | EF5% | EF10% | hits@5% |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    written = []
    for key, label in SIGNALS:
        vals = [r.get(key, np.nan) for r in rows]
        if all(isinstance(v, float) and np.isnan(v) for v in vals):
            continue
        m = metrics(vals, y, alpha)
        if m is None:
            continue
        flag = " *" if m["BEDROC"] > rb_p95 else ""
        print(f"| {label} | {m['N']} | {m['AUC']:.3f} | {m['BEDROC']:.3f}{flag} | "
              f"{m['EF1']:.2f} | {m['EF5']:.2f} | {m['EF10']:.2f} | {m['hits5']}/{m['n']} |")
        written.append((key, label, m))
    print("\n`*` = BEDROC above the 95th percentile of the random permutation baseline.")

    if csv_out:
        new = not csv_out.exists()
        with open(csv_out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["target", "signal", "label", "n", "n_act",
                            "auc", "bedroc", "ef1", "ef5", "ef10"])
            for key, label, m in written:
                w.writerow([target, key, label, m["N"], m["n"], m["AUC"],
                            m["BEDROC"], m["EF1"], m["EF5"], m["EF10"]])
        print(f"wrote {csv_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["L01", "L02"])
    ap.add_argument("--truth-csv", type=Path, default=None)
    ap.add_argument("--holo-root", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--legacy-id-shift", action="store_true",
                    help="stage-1 L01 artifacts only — see analyze_stage1_signals.legacy_run_dir")
    args = ap.parse_args()

    T = args.target
    truth = args.truth_csv or REPO / f"inputs/ligand_series/{T}/ligands_truth.csv"
    holo = args.holo_root or REPO / f"experiments/ligand_series/{T}/holo"
    if not truth.exists():
        raise SystemExit(f"missing {truth}")

    rows = collect(T, holo, truth, legacy_id_shift=args.legacy_id_shift)
    report(T, rows, args.alpha, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
