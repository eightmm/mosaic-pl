"""Re-aggregate per_pose_scores.csv with proper per-scorer pose pools.

Each scorer is meaningful only on a subset of poses:

  - rmsd_pred / lscore / ba_pred  → ALL poses (every pose gets BA-Pred + RMSD-Pred)
  - iptm / ptm / plddt / conf     → COFOLD poses only (per-pose confidence).
       Docking & template poses inherit the docking-anchor cofold's confidence,
       which is a CONSTANT and degenerates ranking. Excluded from those scorers.
  - boltz_aff                     → COFOLD-BOLTZ poses only (boltz2 + boltz2x);
       per-pose affinity exists only for the cofold-Boltz pose pool.
  - oracle                        → ALL poses (best-possible RMSD)

For each scorer, pick top-1 / best-of-5 inside its proper pool, then aggregate.
Reads `per_pose_scores.csv`; writes `per_metric_summary_proper.txt`.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "per_pose_scores.csv"
OUT_PATH = ROOT / "per_metric_summary_proper.txt"


def _f(v: str):
    return float(v) if v not in ("", "None") else None


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                "target": r["target"],
                "zone": r["seq_zone"],
                "source": r["source"],
                "pose_name": r["pose_name"],
                "ba_pred_pkd": _f(r["ba_pred_pkd"]),
                "prmsd": _f(r["prmsd"]),
                "lscore": _f(r["lscore"]),
                "iptm": _f(r["iptm"]),
                "ptm": _f(r["ptm"]),
                "plddt": _f(r["plddt"]),
                "conf": _f(r["conf"]),
                "boltz_aff": _f(r["boltz_aff_log10_kd_nM"]),
                "boltz_prob": _f(r["boltz_binder_prob"]),
                "true_rmsd": _f(r["true_rmsd"]),
            })

    is_cofold = lambda r: r["source"].startswith("cofold_")
    is_cofold_boltz = lambda r: r["source"].startswith(("cofold_boltz2_", "cofold_boltz2x_"))

    BIG = 1e9
    NEG = -1e9

    scorers = [
        # (id, label, pool_filter, sort_key)
        ("rmsd_pred", "Lower pRMSD",        None,            lambda r: (r["prmsd"] if r["prmsd"] is not None else BIG)),
        ("lscore",    "Higher LSCORE",      None,            lambda r: -(r["lscore"] if r["lscore"] is not None else NEG)),
        ("ba_pred",   "Higher pKd",         None,            lambda r: -(r["ba_pred_pkd"] if r["ba_pred_pkd"] is not None else NEG)),
        ("iptm",      "Higher ipTM",        is_cofold,       lambda r: -(r["iptm"] if r["iptm"] is not None else NEG)),
        ("ptm",       "Higher pTM",         is_cofold,       lambda r: -(r["ptm"] if r["ptm"] is not None else NEG)),
        ("plddt",     "Higher pLDDT",       is_cofold,       lambda r: -(r["plddt"] if r["plddt"] is not None else NEG)),
        ("conf",      "Higher conf",        is_cofold,       lambda r: -(r["conf"] if r["conf"] is not None else NEG)),
        ("boltz_aff", "Lower log10(Kd_nM)", is_cofold_boltz, lambda r: (r["boltz_aff"] if (r["boltz_aff"] is not None and (r["boltz_prob"] or 0) >= 0.5) else BIG)),
        ("oracle",    "Lowest TRUE RMSD",   None,            lambda r: (r["true_rmsd"] if r["true_rmsd"] is not None else BIG)),
    ]

    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)

    lines = [f"Per-scorer evaluation, proper pose pools (489 targets, {len(rows)} pose rows)\n"]
    header = (f"  {'scorer':<11}  {'label':<20}  {'pool':<14}  "
              f"{'zone':<8} {'n':>4} {'pool_size':>9} {'top1<2':>7} {'best5<2':>8} {'top1<1':>7} {'med_t1':>7} {'med_b5':>7}")

    for sc_id, label, pool_filter, key in scorers:
        # Per-zone aggregation
        zone_stats = {}
        all_pool_sizes = []
        for tgt, prs in by_target.items():
            pool = prs if pool_filter is None else [p for p in prs if pool_filter(p)]
            valid = [p for p in pool if p["true_rmsd"] is not None]
            if not valid:
                continue
            ordered = sorted(valid, key=key)
            top1 = ordered[0]["true_rmsd"]
            best5 = min(p["true_rmsd"] for p in ordered[:5])
            zone = prs[0]["zone"]
            d = zone_stats.setdefault(zone, {
                "n": 0, "pool_sizes": [],
                "top1_lt2": 0, "best5_lt2": 0, "top1_lt1": 0,
                "top1_vals": [], "best5_vals": [],
            })
            d["n"] += 1
            d["pool_sizes"].append(len(valid))
            all_pool_sizes.append(len(valid))
            if top1 < 2:  d["top1_lt2"] += 1
            if top1 < 1:  d["top1_lt1"] += 1
            if best5 < 2: d["best5_lt2"] += 1
            d["top1_vals"].append(top1)
            d["best5_vals"].append(best5)
        # Total
        all_d = {"n": 0, "pool_sizes": all_pool_sizes,
                 "top1_lt2": 0, "best5_lt2": 0, "top1_lt1": 0,
                 "top1_vals": [], "best5_vals": []}
        for d in zone_stats.values():
            for k in ("n", "top1_lt2", "best5_lt2", "top1_lt1"):
                all_d[k] += d[k]
            all_d["top1_vals"].extend(d["top1_vals"])
            all_d["best5_vals"].extend(d["best5_vals"])

        pool_str = (
            "all"           if pool_filter is None and sc_id != "oracle" else
            "cofold-only"   if pool_filter is is_cofold else
            "boltz-cofold"  if pool_filter is is_cofold_boltz else
            "all (oracle)"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for zone in ("novel", "remote", "related"):
            d = zone_stats.get(zone)
            if not d or d["n"] == 0:
                continue
            avg_pool = sum(d["pool_sizes"]) / len(d["pool_sizes"])
            mt = sorted(d["top1_vals"])[len(d["top1_vals"]) // 2]
            mb = sorted(d["best5_vals"])[len(d["best5_vals"]) // 2]
            lines.append(
                f"  {sc_id:<11}  {label:<20}  {pool_str:<14}  "
                f"{zone:<8} {d['n']:>4} {avg_pool:>9.1f} "
                f"{d['top1_lt2']:>7} {d['best5_lt2']:>8} {d['top1_lt1']:>7} "
                f"{mt:>7.2f} {mb:>7.2f}"
            )
        if all_d["n"]:
            avg_pool = sum(all_d["pool_sizes"]) / len(all_d["pool_sizes"])
            mt = sorted(all_d["top1_vals"])[len(all_d["top1_vals"]) // 2]
            mb = sorted(all_d["best5_vals"])[len(all_d["best5_vals"]) // 2]
            lines.append(
                f"  {sc_id:<11}  {label:<20}  {pool_str:<14}  "
                f"{'TOTAL':<8} {all_d['n']:>4} {avg_pool:>9.1f} "
                f"{all_d['top1_lt2']:>7} {all_d['best5_lt2']:>8} {all_d['top1_lt1']:>7} "
                f"{mt:>7.2f} {mb:>7.2f}"
            )
        lines.append("")
    out = "\n".join(lines)
    print(out)
    OUT_PATH.write_text(out + "\n")


if __name__ == "__main__":
    main()
