# Pose ranker design: RRF + consensus + lig_align bonus

`scripts/compute_submission_scores.py` 의 `select_best_pose` /
`select_diverse_top_k` 가 사용하는 비-ML 랭커 설계. 단일 스코어러
(lscore / pRMSD / ipTM / pLDDT / …) 만 쓸 때의 천장(top-1 ~24%) 을 넘기 위한 ROI 검증된 조합이다.

## Motivation
novel2025_test 489 타겟에서 단일 스코어러 별 top-1 native-rate (true_rmsd<2Å):

| scorer        | pool                | top-1 |
|---------------|---------------------|------:|
| lscore        | all                 | 24.5% |
| rmsd_pred     | all                 | 22.7% |
| iptm          | cofold-only         | 15.1% |
| ptm           | cofold-only         | 17.9% |
| plddt         | cofold-only         | 17.3% |
| conf          | cofold-only         | 18.6% |
| boltz_aff     | cofold-Boltz-only   | 14.8% |
| ba_pred       | all                 | 10.2% |
| **oracle**    | all                 | **58.7%** |

→ 모든 단일 스코어러 < 25 %, oracle 천장 ~59 %. 30+ % 차이는 "올바른 pose 가 풀 안에 있는데 아무 스코어러도 일관되게 못 집는다" 가 원인.

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
