"""Two analyses bundled:

  Q1. cofold model 안에서 같은 모델이 25개 (5 seeds × 5 samples) pose 를
      얼마나 같은 곳에 붙이나? — per (target, model) 의 true_rmsd spread.
  Q2. 전체 pose 의 true_rmsd 누적분포 (CDF) — family 별 + overall.

읽기: per_pose_scores.csv
출력: stdout 표 + analysis_pose_diversity.png (CDF plot)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, mean

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "per_pose_scores.csv"
OUT_PNG = ROOT / "analysis_pose_diversity.png"


def _f(v):
    return float(v) if v not in ("", "None") else None


def source_family(src: str) -> str:
    if src.startswith("cofold_boltz2x_"):  return "cofold_boltz2x"
    if src.startswith("cofold_boltz2_"):   return "cofold_boltz2"
    if src.startswith("cofold_protenix_"): return "cofold_protenix"
    if src.startswith("cofold_af3_"):      return "cofold_af3"
    if src.startswith("vina_cofolding"):       return "vina_cofold"
    if src.startswith("vina_p2rank"):          return "vina_p2rank"
    if src.startswith("vina_swinsite"):        return "vina_swinsite"
    if src.startswith("autodock_gpu_cofolding"): return "adg_cofold"
    if src.startswith("autodock_gpu_p2rank"):    return "adg_p2rank"
    if src.startswith("autodock_gpu_swinsite"):  return "adg_swinsite"
    if src.startswith("protenix_dock"):          return "pxdock"
    if "template_" in src and "_lig_align_" in src: return "template_lig_align"
    if "template_" in src and "_vina_" in src:      return "template_vina"
    if "template_" in src and "_adg_"  in src:      return "template_adg"
    return src


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            t = _f(r["true_rmsd"])
            if t is None:
                continue
            rows.append({
                "target": r["target"],
                "fam": source_family(r["source"]),
                "true_rmsd": t,
            })
    print(f"valid pose rows: {len(rows)}")

    # ============= Q1: cofold intra-model spread =============
    print("\n=== Q1. Cofold intra-model pose spread ===")
    print("    (per (target, model) 의 25 pose true_rmsd 분포)")
    cofold_fams = ["cofold_boltz2", "cofold_boltz2x", "cofold_protenix", "cofold_af3"]
    by_tf = defaultdict(list)
    for r in rows:
        if r["fam"] in cofold_fams:
            by_tf[(r["target"], r["fam"])].append(r["true_rmsd"])

    print(f"  {'family':<18} {'n_(t,m)':>8} {'med_n':>6} {'med_std':>8} {'med_range':>10} {'med_min':>8} {'med_iqr':>8}")
    print("  " + "-" * 72)
    for fam in cofold_fams:
        groups = [vals for (t, m), vals in by_tf.items() if m == fam]
        if not groups:
            continue
        ns = [len(g) for g in groups]
        stds = [float(np.std(g, ddof=0)) for g in groups if len(g) > 1]
        ranges = [(max(g) - min(g)) for g in groups if len(g) > 1]
        mins = [min(g) for g in groups]
        # IQR 도 함께
        iqrs = [float(np.percentile(g, 75) - np.percentile(g, 25)) for g in groups if len(g) > 1]
        print(f"  {fam:<18} {len(groups):>8} {int(median(ns)):>6} "
              f"{median(stds):>8.2f} {median(ranges):>10.2f} "
              f"{median(mins):>8.2f} {median(iqrs):>8.2f}")

    # 보충: 같은 모델의 pose 가 "거의 같은 자리" 에 붙는 비율
    #   IQR < 1 Å 인 (target, model) 비율 — 25 pose 중 중간 50% 가 1 Å 안에 모여있음
    print()
    print("  * std=0 means all 25 poses identical RMSD; tiny IQR means tightly clustered")
    print()
    print(f"  family            n_(t,m)  IQR<1Å  IQR<2Å  range<2Å  range<5Å")
    print("  " + "-" * 65)
    for fam in cofold_fams:
        groups = [(t, vals) for (t, m), vals in by_tf.items() if m == fam and len(vals) > 1]
        n = len(groups)
        if n == 0:
            continue
        iqr1 = sum(1 for _, g in groups if (np.percentile(g, 75) - np.percentile(g, 25)) < 1.0)
        iqr2 = sum(1 for _, g in groups if (np.percentile(g, 75) - np.percentile(g, 25)) < 2.0)
        rng2 = sum(1 for _, g in groups if (max(g) - min(g)) < 2.0)
        rng5 = sum(1 for _, g in groups if (max(g) - min(g)) < 5.0)
        print(f"  {fam:<18} {n:>5}  {iqr1:>5}({100*iqr1/n:.0f}%)  "
              f"{iqr2:>5}({100*iqr2/n:.0f}%)  {rng2:>5}({100*rng2/n:.0f}%)  "
              f"{rng5:>5}({100*rng5/n:.0f}%)")

    # ============= Q2: CDF plot =============
    print("\n=== Q2. true_rmsd CDF plot ===")
    families_order = [
        ("cofold_protenix", "tab:blue"),
        ("cofold_af3",      "tab:cyan"),
        ("cofold_boltz2",   "tab:orange"),
        ("cofold_boltz2x",  "tab:red"),
        ("vina_cofold",     "tab:green"),
        ("vina_swinsite",   "tab:olive"),
        ("vina_p2rank",     "tab:brown"),
        ("adg_cofold",      "tab:purple"),
        ("template_lig_align", "magenta"),
        ("pxdock",          "gray"),
    ]
    by_fam: dict[str, list[float]] = defaultdict(list)
    all_vals: list[float] = []
    for r in rows:
        by_fam[r["fam"]].append(r["true_rmsd"])
        all_vals.append(r["true_rmsd"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: family CDF
    for fam, color in families_order:
        v = sorted(by_fam.get(fam, []))
        if not v:
            continue
        ys = np.arange(1, len(v) + 1) / len(v)
        axes[0].plot(v, ys, label=f"{fam} (n={len(v)})", color=color, linewidth=1.4)
    # overall
    v = sorted(all_vals)
    ys = np.arange(1, len(v) + 1) / len(v)
    axes[0].plot(v, ys, label=f"ALL (n={len(v)})", color="black", linewidth=2.0, linestyle="--")
    axes[0].axvline(2.0, color="red", linestyle=":", alpha=0.5, label="2 Å")
    axes[0].axvline(1.0, color="orange", linestyle=":", alpha=0.5, label="1 Å")
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.1, 100)
    axes[0].set_xlabel("true RMSD (Å, log)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Per-pose CDF by source family (lower-left = better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    # Right: per-target BEST pose per family CDF — closer to oracle behavior
    by_target_fam: dict[tuple, float] = {}
    for r in rows:
        key = (r["target"], r["fam"])
        if key not in by_target_fam or r["true_rmsd"] < by_target_fam[key]:
            by_target_fam[key] = r["true_rmsd"]

    for fam, color in families_order:
        v = sorted(t for (_, m), t in by_target_fam.items() if m == fam)
        if not v:
            continue
        ys = np.arange(1, len(v) + 1) / len(v)
        axes[1].plot(v, ys, label=f"{fam} (n_tgt={len(v)})", color=color, linewidth=1.4)
    # Per-target ALL-family best (oracle-like)
    target_best: dict[str, float] = {}
    for r in rows:
        if r["target"] not in target_best or r["true_rmsd"] < target_best[r["target"]]:
            target_best[r["target"]] = r["true_rmsd"]
    v = sorted(target_best.values())
    ys = np.arange(1, len(v) + 1) / len(v)
    axes[1].plot(v, ys, label=f"oracle ALL-family (n_tgt={len(v)})",
                 color="black", linewidth=2.2, linestyle="--")
    axes[1].axvline(2.0, color="red", linestyle=":", alpha=0.5, label="2 Å")
    axes[1].axvline(1.0, color="orange", linestyle=":", alpha=0.5, label="1 Å")
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.1, 100)
    axes[1].set_xlabel("min true RMSD per target (Å, log)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Per-target BEST pose CDF by family (oracle bound)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"  → wrote {OUT_PNG}")

    # 추가 텍스트 요약: 각 family 의 P10/P25/P50/P75 raw + per-target-best
    print()
    print(f"  {'family':<20} {'pose_P10':>8} {'pose_P25':>8} {'pose_P50':>8} {'best_P10':>8} {'best_P25':>8} {'best_P50':>8}")
    print("  " + "-" * 72)
    for fam, _ in families_order:
        pose_v = by_fam.get(fam, [])
        best_v = [t for (_, m), t in by_target_fam.items() if m == fam]
        if not pose_v:
            continue
        pp = np.percentile(pose_v, [10, 25, 50])
        bp = np.percentile(best_v, [10, 25, 50]) if best_v else (0, 0, 0)
        print(f"  {fam:<20} {pp[0]:>8.2f} {pp[1]:>8.2f} {pp[2]:>8.2f} "
              f"{bp[0]:>8.2f} {bp[1]:>8.2f} {bp[2]:>8.2f}")


if __name__ == "__main__":
    main()
