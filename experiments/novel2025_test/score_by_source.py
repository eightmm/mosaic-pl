"""소스(트랙)별 분석:
  - 각 소스가 평균적으로 얼마나 좋은 RMSD를 만드나?
  - 각 스코어러의 top-1 픽이 어느 소스에서 왔나?
  - lig_align(MCS) vs vina/adg/cofold 비교

읽기: per_pose_scores.csv, 출력: stdout 표.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Run outputs + per_pose_scores.csv live in the sibling data dir
# (experiments/novel2025_runs), not in this harness dir.
CSV_PATH = ROOT.parent / "novel2025_runs" / "per_pose_scores.csv"


def _f(v):
    return float(v) if v not in ("", "None") else None


def source_family(src: str) -> str:
    """소스 키를 family 라벨로."""
    if src.startswith("cofold_boltz2x_"):
        return "cofold_boltz2x"
    if src.startswith("cofold_boltz2_"):
        return "cofold_boltz2"
    if src.startswith("cofold_protenix_"):
        return "cofold_protenix"
    if src.startswith("cofold_af3_"):
        return "cofold_af3"
    if src.startswith("vina_cofolding"):
        return "vina_cofolding"
    if src.startswith("vina_p2rank"):
        return "vina_p2rank"
    if src.startswith("vina_swinsite"):
        return "vina_swinsite"
    if src.startswith("autodock_gpu_cofolding"):
        return "adg_cofolding"
    if src.startswith("autodock_gpu_p2rank"):
        return "adg_p2rank"
    if src.startswith("autodock_gpu_swinsite"):
        return "adg_swinsite"
    if src.startswith("protenix_dock"):
        return "protenix_dock"
    if "template_" in src and "_lig_align_" in src:
        return "template_lig_align(MCS)"
    if "template_" in src and "_vina_" in src:
        return "template_vina"
    if "template_" in src and "_adg_" in src:
        return "template_adg"
    return src


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                "target": r["target"],
                "zone": r["seq_zone"],
                "source": r["source"],
                "fam": source_family(r["source"]),
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

    valid = [r for r in rows if r["true_rmsd"] is not None]
    print(f"Total pose rows: {len(rows)}, with true_rmsd: {len(valid)}")
    print()

    # 1) Per-source stats
    print("=== 소스 family 별 pose 통계 ===")
    print(f"  {'family':<22}  {'n_poses':>8}  {'<2A':>5}  {'%<2A':>6}  {'<1A':>5}  {'med_rmsd':>9}  {'min_rmsd':>9}")
    print("  " + "-" * 78)
    fam_stats = defaultdict(list)
    for r in valid:
        fam_stats[r["fam"]].append(r["true_rmsd"])
    for fam in sorted(fam_stats, key=lambda f: -len(fam_stats[f])):
        vals = sorted(fam_stats[fam])
        n = len(vals)
        n2 = sum(1 for v in vals if v < 2)
        n1 = sum(1 for v in vals if v < 1)
        med = vals[n // 2]
        mn = vals[0]
        print(f"  {fam:<22}  {n:>8}  {n2:>5}  {100*n2/n:>5.1f}%  {n1:>5}  {med:>9.2f}  {mn:>9.2f}")
    print()

    # 2) Per-target best pose per source family — 타겟별로 각 family 가 native pose 를 만들 수 있었나?
    print("=== 타겟별 family 가 <2Å pose 를 ‘만들 수 있었던’ 비율 (recall) ===")
    by_target_fam = defaultdict(lambda: defaultdict(list))
    target_zones = {}
    for r in valid:
        by_target_fam[r["target"]][r["fam"]].append(r["true_rmsd"])
        target_zones[r["target"]] = r["zone"]
    n_targets = len(by_target_fam)
    print(f"  (타겟 {n_targets}개 기준)")
    print(f"  {'family':<22}  {'tgt_with_pose':>13}  {'tgt_with_<2A':>13}  {'recall%':>8}")
    print("  " + "-" * 62)
    fam_recall = {}
    for fam in sorted(fam_stats, key=lambda f: -len(fam_stats[f])):
        n_with = sum(1 for tgt in by_target_fam if fam in by_target_fam[tgt])
        n_hit = sum(1 for tgt in by_target_fam if fam in by_target_fam[tgt] and min(by_target_fam[tgt][fam]) < 2)
        rec = 100 * n_hit / n_with if n_with else 0
        fam_recall[fam] = (n_with, n_hit)
        print(f"  {fam:<22}  {n_with:>13}  {n_hit:>13}  {rec:>7.1f}%")
    print()

    # 3) 스코어러별 top-1 픽이 어느 family 에서 나왔나?
    BIG = 1e9
    NEG = -1e9
    is_cofold = lambda r: r["source"].startswith("cofold_")
    is_cofold_boltz = lambda r: r["source"].startswith(("cofold_boltz2_", "cofold_boltz2x_"))
    scorers = [
        ("rmsd_pred", None,            lambda r: (r["prmsd"] if r["prmsd"] is not None else BIG)),
        ("lscore",    None,            lambda r: -(r["lscore"] if r["lscore"] is not None else NEG)),
        ("ba_pred",   None,            lambda r: -(r["ba_pred_pkd"] if r["ba_pred_pkd"] is not None else NEG)),
        ("plddt",     is_cofold,       lambda r: -(r["plddt"] if r["plddt"] is not None else NEG)),
        ("ptm",       is_cofold,       lambda r: -(r["ptm"] if r["ptm"] is not None else NEG)),
        ("iptm",      is_cofold,       lambda r: -(r["iptm"] if r["iptm"] is not None else NEG)),
        ("boltz_aff", is_cofold_boltz, lambda r: (r["boltz_aff"] if (r["boltz_aff"] is not None and (r["boltz_prob"] or 0) >= 0.5) else BIG)),
    ]
    by_target = defaultdict(list)
    for r in valid:
        by_target[r["target"]].append(r)

    print("=== 스코어러별 top-1 픽의 family 분포 + 그 픽의 hit(<2Å) 카운트 ===")
    for sc_id, pool, key in scorers:
        pick_fam = defaultdict(lambda: [0, 0])  # [picks, hits]
        n_t = 0
        for tgt, prs in by_target.items():
            pool_rows = prs if pool is None else [p for p in prs if pool(p)]
            if not pool_rows:
                continue
            ordered = sorted(pool_rows, key=key)
            pick = ordered[0]
            pick_fam[pick["fam"]][0] += 1
            if pick["true_rmsd"] < 2:
                pick_fam[pick["fam"]][1] += 1
            n_t += 1
        total_hits = sum(v[1] for v in pick_fam.values())
        print(f"  scorer={sc_id}  total_picks={n_t}  hits<2Å={total_hits} ({100*total_hits/n_t:.1f}%)")
        print(f"    {'family':<22}  {'picks':>6}  {'hits':>5}  {'pick%':>6}  {'hit_rate':>8}")
        for fam in sorted(pick_fam, key=lambda f: -pick_fam[f][0]):
            picks, hits = pick_fam[fam]
            print(f"    {fam:<22}  {picks:>6}  {hits:>5}  {100*picks/n_t:>5.1f}%  {100*hits/picks:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
