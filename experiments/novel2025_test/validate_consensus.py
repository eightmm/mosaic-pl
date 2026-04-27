"""Step 3 (consensus → native) 가설 빠른 검증.

per_pose_scores.csv 만으로 pose-pose RMSD 없이 빠르게 ROI 추정.

핵심 프록시:
  같은 타겟에서 여러 family 가 비슷한 true_rmsd 영역의 pose 를 만들면
  → 그들은 crystal 근처에서 합의(consensus)했다고 본다.
  family 간 합의 강도 = "min_true_rmsd 가 best ± δ Å 이내인 family 수".

검증할 것:
  Q1. native (<2Å) pose 가 존재하는 타겟 중,
      N_fam_consensus ≥ 2 인 비율은?  (consensus 신호의 가용성)
  Q2. 현재 best 스코어러(lscore) 가 top-1 을 틀린 타겟 중에서,
      그 중 native 가 존재하면서 N_fam_consensus ≥ 2 인 타겟 비율?
      → 이게 consensus 로 회수 가능한 잠재 타겟 수.
  Q3. native 가 없을 때(타겟 자체가 어려움) consensus 가 잘못된 신호 줄 위험:
      non-native (true_rmsd ≥ 2) 가 N_fam_consensus ≥ 2 인 비율도 측정.

판정:
  Q1, Q2 ≥ 30% 면 consensus 도입 가치 있음.
  Q3 가 Q1 보다 훨씬 낮아야 (decoy basin 거짓양성 적어야) 안전.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "per_pose_scores.csv"


def _f(v):
    return float(v) if v not in ("", "None") else None


def source_family(src: str) -> str:
    if src.startswith("cofold_boltz2x_"): return "cofold_boltz2x"
    if src.startswith("cofold_boltz2_"):  return "cofold_boltz2"
    if src.startswith("cofold_protenix_"): return "cofold_protenix"
    if src.startswith("cofold_af3_"):      return "cofold_af3"
    if src.startswith("vina_cofolding"):    return "vina_cofold"
    if src.startswith("vina_p2rank"):       return "vina_p2rank"
    if src.startswith("vina_swinsite"):     return "vina_swinsite"
    if src.startswith("autodock_gpu_cofolding"): return "adg_cofold"
    if src.startswith("autodock_gpu_p2rank"):    return "adg_p2rank"
    if src.startswith("autodock_gpu_swinsite"):  return "adg_swinsite"
    if src.startswith("protenix_dock"): return "pxdock"
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
                "zone": r["seq_zone"],
                "fam": source_family(r["source"]),
                "lscore": _f(r["lscore"]),
                "true_rmsd": t,
            })
    print(f"valid pose rows: {len(rows)}")

    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)
    n_targets = len(by_target)
    print(f"targets: {n_targets}\n")

    # 각 타겟마다 family 별 best (min) true_rmsd
    fam_best = {}
    for tgt, prs in by_target.items():
        fb = defaultdict(lambda: 1e9)
        for p in prs:
            if p["true_rmsd"] < fb[p["fam"]]:
                fb[p["fam"]] = p["true_rmsd"]
        fam_best[tgt] = dict(fb)

    # Helper: consensus = N_families with best_rmsd within δ of global best
    def consensus_count(tgt: str, delta: float, only_native: bool = False) -> int:
        fb = fam_best[tgt]
        if not fb:
            return 0
        gmin = min(fb.values())
        if only_native and gmin >= 2.0:
            return 0
        return sum(1 for v in fb.values() if v <= gmin + delta)

    # ---- Q1. consensus 가용성 ----
    print("=== Q1. native (<2Å) 보유 타겟의 consensus 분포 ===")
    print(f"  {'δ(Å)':>5}  {'N_fam≥2':>8}  {'N_fam≥3':>8}  {'N_fam≥4':>8}  {'N_fam(median)':>14}")
    native_targets = [t for t, fb in fam_best.items() if min(fb.values()) < 2.0]
    print(f"  ({len(native_targets)} targets with native pose, of {n_targets} total)")
    for delta in (0.5, 1.0, 1.5, 2.0):
        nfams = [consensus_count(t, delta) for t in native_targets]
        n2 = sum(1 for x in nfams if x >= 2)
        n3 = sum(1 for x in nfams if x >= 3)
        n4 = sum(1 for x in nfams if x >= 4)
        med = sorted(nfams)[len(nfams) // 2]
        print(f"  {delta:>5.1f}  {n2:>8}  {n3:>8}  {n4:>8}  {med:>14}")
    print()

    # ---- Q2. lscore 가 틀린 타겟 중, consensus 로 회수 가능한 비율 ----
    # 현재 lscore top-1 (전체 pose pool) 이 native 인지
    print("=== Q2. lscore top-1 이 틀린 타겟에서 consensus 회수 가능성 ===")
    lscore_top1 = {}
    for tgt, prs in by_target.items():
        valid = [p for p in prs if p["lscore"] is not None]
        if not valid:
            continue
        valid.sort(key=lambda p: -p["lscore"])
        lscore_top1[tgt] = valid[0]["true_rmsd"] < 2.0

    miss_targets = [t for t, hit in lscore_top1.items() if not hit and min(fam_best[t].values()) < 2.0]
    n_miss = sum(1 for hit in lscore_top1.values() if not hit)
    print(f"  lscore top-1 fail: {n_miss}/{len(lscore_top1)} targets")
    print(f"  그 중 native pose 가 존재하는 타겟 (회수 후보): {len(miss_targets)}")
    print()
    print(f"  {'δ(Å)':>5}  {'cand':>5}  {'N_fam≥2':>8}  {'recovery_potential':>20}")
    for delta in (0.5, 1.0, 1.5, 2.0):
        rec = sum(1 for t in miss_targets if consensus_count(t, delta) >= 2)
        rec_pct = 100 * rec / len(miss_targets) if miss_targets else 0
        # gross uplift: recovered targets / total targets
        gross = 100 * rec / n_targets
        print(f"  {delta:>5.1f}  {len(miss_targets):>5}  {rec:>8}  {rec_pct:>5.1f}% (≈+{gross:>4.1f}% SR)")
    print()

    # ---- Q3. non-native(decoy) consensus 거짓양성 ----
    print("=== Q3. native pose 가 없는 타겟에서도 consensus 가 강한가? (false alarm) ===")
    nonnative_targets = [t for t, fb in fam_best.items() if min(fb.values()) >= 2.0]
    print(f"  ({len(nonnative_targets)} targets with NO native pose)")
    print(f"  {'δ(Å)':>5}  {'N_fam≥2':>8}  {'rate':>6}")
    for delta in (0.5, 1.0, 1.5, 2.0):
        n2 = sum(1 for t in nonnative_targets if consensus_count(t, delta) >= 2)
        rate = 100 * n2 / len(nonnative_targets) if nonnative_targets else 0
        print(f"  {delta:>5.1f}  {n2:>8}  {rate:>5.1f}%")
    print()

    # ---- Q4. consensus 강도와 best-true-rmsd 의 monotone 상관 ----
    # pose 단위가 아니라 타겟 단위로: N_fam_consensus 가 높을수록 global best rmsd 가 낮은가?
    print("=== Q4. δ=1.0 Å 기준, N_fam_consensus 와 global best RMSD 분포 ===")
    delta = 1.0
    bins = defaultdict(list)
    for t, fb in fam_best.items():
        nf = consensus_count(t, delta)
        bins[nf].append(min(fb.values()))
    print(f"  {'N_fam':>6}  {'n_tgt':>6}  {'med_best':>9}  {'%native':>7}")
    for nf in sorted(bins):
        vals = bins[nf]
        med = sorted(vals)[len(vals) // 2]
        n_nat = sum(1 for v in vals if v < 2.0)
        print(f"  {nf:>6}  {len(vals):>6}  {med:>9.2f}  {100*n_nat/len(vals):>6.1f}%")


if __name__ == "__main__":
    main()
