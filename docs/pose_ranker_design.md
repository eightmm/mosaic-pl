# Pose ranker design: RRF + consensus + lig_align bonus

`scripts/compute_submission_scores.py` 의 `select_best_pose` /
`select_diverse_top_k` 가 사용하는 비-ML 랭커 설계. 단일 스코어러
(lscore / pRMSD / ipTM / pLDDT / …) 만 쓸 때의 천장(top-1 ~30 %) 을 넘기 위한 ROI 검증된 조합이다.

## Motivation
novel2025_test 489 타겟에서 단일 스코어러 별 top-1 / best-of-5 native-rate
(true_rmsd<2Å, 풀 = 각 스코어러가 의미를 가지는 pose 만 —
`per_metric_summary_proper.txt`).

best-of-5 = 같은 스코어러로 정렬한 상위 5 개 중 가장 작은 true_rmsd 가 < 2 Å
인 비율 (CASP17 LG 형식이 5 MODEL 까지 허용하므로, 실질적인 "submission 이
적어도 한 번 native 를 맞춤" 비율).

| scorer        | pool                | n   | top-1 < 2 Å | best-5 < 2 Å |
|---------------|---------------------|----:|------------:|------------:|
| plddt         | cofold-only         | 479 | 143  (29.9 %) | 164  (34.2 %) |
| conf          | cofold-only         | 479 | 141  (29.4 %) | 163  (34.0 %) |
| ptm           | cofold-only         | 479 | 130  (27.1 %) | 163  (34.0 %) |
| iptm          | cofold-only         | 479 | 134  (28.0 %) | 160  (33.4 %) |
| boltz_aff     | cofold-Boltz-only   | 367 |  98  (26.7 %) | 116  (31.6 %) |
| lscore        | all                 | 489 | 123  (25.2 %) | 152  (31.1 %) |
| rmsd_pred     | all                 | 489 | 111  (22.7 %) | 138  (28.2 %) |
| ba_pred       | all                 | 489 |  50  (10.2 %) |  83  (17.0 %) |
| **oracle**    | all                 | 489 | **287 (58.7 %)** | 287 (58.7 %) |

→ 단일 스코어러 ceiling top-1 ~30 % / best-5 ~34 % (cofold pLDDT), oracle
천장 58.7 %. top-1 → best-5 로 가도 단일 스코어러는 +4–5 %p 만 회복 (스코어러
top 영역 내부의 pose 다양성이 작음). oracle 은 top-1 = best-5 (per-target 가장
좋은 한 pose 만 보므로 best-of-5 가 동일).

49.3 %p 갭의 의미:
- 489 - 287 = **202 타겟은 풀 안에 < 2 Å pose 자체가 없음** — generation 단계
  한계, ranker 로 회수 불가능.
- 287 - 152 = 135 타겟은 best-of-5(lscore) 에서도 놓치는 회수 가능 분량 →
  ranker 설계 대상.

> NOTE: cofold-only 스코어러는 cofold pose pool 에서만 의미 있다. 같은 스코어러
> 를 docking pose 까지 포함한 전체 풀에서 ranking 하면 docking pose 가 상속
> 받는 docking-anchor 의 constant 값으로 인해 ranking 이 degenerate 한다 (그
> 결과는 `per_metric_summary.txt` 의 raw 값이고, ranker 설계의 기준은
> `per_metric_summary_proper.txt`).

### 489 vs 499 의 갭
원래 input yaml 은 499 타겟. CSV 에 489 만 들어간 이유 (총 10 누락):
- **8 타겟 SIGALRM 타임아웃** (HEM-class organometallic 에서 RDKit `rdFMCS`
  무한루프, `score_per_metric.py` 의 180 s per-target timeout 으로 잘림):
  9emt, 9mel, 9mgt, 9mh5, 9q3k, 9r07, 9r7a, 9whf
- **2 타겟 submission 미완료** (run 은 끝났지만 `experiments/submissions/{tgt}_input.lg`
  파일이 생성되지 않음 — `make_casp_submission` 단계 실패): 9cv8, 9dm5
  → `score_per_metric.py` 가 LG 파일을 iterator 의 시드로 쓰므로 자동 제외.

## Empirical validation of consensus (Step 3 hypothesis check)
`experiments/novel2025_test/validate_consensus.py` — δ=1.0 Å 기준 cross-family consensus 강도 vs `min(true_rmsd)` 분포:

| N_fam (consensus) | n_targets | %native (<2Å) |
|------------------:|----------:|---------------:|
| 1                 | 87        | 23 %           |
| 4                 | 73        | 62 %           |
| 7                 | 54        | 89 %           |
| 11                | 7         | 100 %          |

**결론**: cross-family consensus 는 native 확률과 monotone 한 양의 상관. 단, native 가 없는 타겟의 67 % 도 consensus 형성 → 단독으로 쓰면 false-alarm 큼. 따라서 RRF 위의 modest re-ranker (×(1 + 0.1·log(1+support))) 로 도입.

## Algorithm

```
final(i) = RRF(i) · (1 + 0.1 · log(1 + support(i))) + bonus(i)
```

### RRF
```
RRF(i) = Σ_s 1 / (60 + rank_s(i))
        s ∈ { lscore, plddt_norm, iptm, ptm, conf }
```
- **k = 60**: canonical RRF 하이퍼파라미터, tune 안 함.
- **lscore**: 모든 pose. (전체 풀 ranking)
- **plddt_norm**: cofold-only. **모델 family 안에서만** 순위. 절대값은 비교
  안 함 — boltz2x 의 raw pLDDT 가 protenix/af3 보다 낮은 scale 에 있어
  global rank 로 합치면 boltz2x 가 부당하게 손해 본다 (validation 증거: raw
  pLDDT 로는 boltz2x picks=51 / hit_rate 54.9% 인데도 top-1 못 잡음).
- **iptm / ptm / conf**: cofold-only, global rank.
- **non-cofold pose**: cofold-only 스코어러는 0 기여. 자연스럽게 cofold pose
  가 우위 (실제 hit-rate 분포와 일치).

### Consensus support
```
support(i) = | { j : RMSD_heavy(i, j) < 2 Å, family(j) ≠ family(i) } |
```
- **Receptor frame**: cofold/docking pose 는 각자 cofold/anchor frame 에 있고
  template pose 는 staging 단계에서 cofold frame 으로 정렬된다. 따라서 추가
  rigid alignment 안 함, raw atom-index RMSD.
- **Atom-index RMSD (no symmetry)**: RDKit `CalcRMS` 의 substructure-matching 은
  pair 당 ~ms — 500 pose × 500 pair 면 250s/타겟. numpy `sqrt(((a-b)**2).sum(axis=1).mean())` 로 < 5s. 같은 SMILES 의 다른 conformer 끼리는 atom ordering 일치하므로 정확도 손실 거의 없음 (다른 docking tool 에서 ordering 이 다를 수 있어 ~0.5 Å 노이즈 가능, 2 Å threshold 대비 무시 가능).
- **Centroid pre-filter**: |Δcentroid_x| > 3 Å 면 RMSD ≥ 3 Å 보장 → skip.
- **Family**: `_source_family(source)` 의 14가지 family (cofold_boltz2,
  cofold_protenix, vina_p2rank, adg_swinsite, template_lig_align, …).

### lig_align bonus
```
bonus(i) = 0.2 · lscore(i)        if pose ∈ template_lig_align
        else 0
```
**Why**: template_lig_align (Track 3, MCS-guided template-pose ligand alignment)
의 family-recall 이 24.3 % 로 가장 높은데, lscore / iptm / pLDDT / conf 어느
스코어러도 lig_align pose 를 top-1 으로 안 집는다 (per_pose_scores.csv 분석:
56 타겟이 lig_align <2 Å pose 보유했으나 그 중 0~3개만 회수). 작은 additive
boost 로 일부 회수.

## Pose pool dedup
`collect_pose_scores` 마지막 단계에서 legacy 이중집계 제거: 같은 staging file 이
`tool` (no suffix) 와 `tool_{lig_id}` 양쪽으로 등록된 경우 bare-stem drop. 안
하면 cross-family consensus 가 자기 자신과 매칭해서 부풀어 오른다 (10sl
1214→637 pose).

## Cofold confidence loading
`_attach_pose_confidence(run_dir, pose_scores)` — `_build_cofold_confidence_map`
(boltz2/2x/protenix/af3 의 sibling JSON), `_build_boltz_pose_affinity` (cofold-
Boltz 만), `_docking_anchor_confidence` (docking pose 의 inherited value).
template pose 는 confidence 없이 None.

## Performance
- 10sl (637 pose): RRF 0.00 s, consensus 3.2 s, full ranker 3.5 s.
- numpy 변환 전 (RDKit `CalcRMS`) 32 s → numpy 후 3 s, **~10× speedup**.
- 489 타겟 batch 검증: ~30 분 예상 (single core).

## Ablation entrypoints
- `select_best_pose_pRMSD_legacy(poses)` — pre-RRF (smallest pRMSD, lscore tie-break) 로 비교 가능.
- `_rrf_score`, `_consensus_support`, `_final_ranker_score` 모두 외부에서 직접
  호출 가능 (ablation: weight=0 / lig_align_bonus=0 / consensus 만 / RRF 만 등).

## Validation
`experiments/novel2025_test/validate_new_ranker.py` 가 offline 으로
per_pose_scores.csv 의 true_rmsd 와 매칭하여 lscore baseline / legacy_pRMSD /
new_top1 / new_top5 의 SR 비교. (`--sample-stride N` 으로 빠른 샘플 검증, 또는
전체.)

## Open follow-ups
- **frame mismatch**: docking pose 의 cofold-anchor 가 평가 cofold 와 다를 때
  consensus 가 약해질 가능성. 현재는 majority 가 일치하므로 영향 작음.
- **template_vina / template_adg coordinate frame bug**: recall 2.2 % / 1.5 %,
  RMSD 20–700 Å. lig_align 은 정상. docking box 정의 또는 receptor align 단계
  버그로 추정 — 별도 트랙.
- **lig_align bonus weight tune**: 0.2 는 직관, 0.1 ~ 0.5 sweep 가능.
- **consensus weight tune**: 0.1·log(1+support) 도 직관. validation Q4 에 따르면
  support=10 부근에서 saturation.
