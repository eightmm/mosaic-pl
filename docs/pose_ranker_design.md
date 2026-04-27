# Pose ranker — 설계 + 검증 로그

`scripts/compute_submission_scores.py` 의 `select_best_pose` /
`select_diverse_top_k` 가 사용하는 **non-ML pose ranker** 의 설계 결정과
novel2025_test (489 타겟) 위에서의 ablation 결과 전부.

## TL;DR

여러 후보 (RRF, cluster, cascaded, …) 를 ablation 한 결과
**`lscore_top5_diverse` 가 명확한 winner**.

**Uniform-denominator 검증 (v4)** — 모든 ranker 가 동일 470 타겟 위에서 비교:

| ranker                   | n   | top-1 < 2 Å | best-5 < 2 Å |
|--------------------------|----:|------------:|-------------:|
| **`lscore_top5_diverse`** ⭐ | 470 | 112 (23.8 %) | **182 (38.7 %)** |
| `cascaded_top5`              | 470 | 112 (23.8 %) | 172 (36.6 %) |
| `cascaded` (top-1 only)      | 470 | 112 (23.8 %) | 112 (23.8 %) |
| `lscore` baseline (top-1)    | 470 | 112 (23.8 %) | 112 (23.8 %) |
| `legacy_pRMSD` (현 prod)     | 470 |  99 (21.1 %) |  99 (21.1 %) |

→ legacy 대비 **+17.6 %p best-5**, oracle (58.7 %) 까지의 갭 33.5 → ~20 %p 회수.

**Per-zone (`lscore_top5_diverse` best-5):**
- novel: 38.2 % (n=68)
- remote: **50.9 %** (n=112)
- related: 34.1 % (n=290)

### 분모 = 470 의 의미
원래 truth map 은 497 타겟. validator 가 모든 ranker 의 top-1 lookup 이 성공한
**교집합** 만 카운트:
- 27 타겟이 lookup 실패로 drop
  - 대부분: cofold dedup mismatch 또는 lig_align "Conformer_N_Rank_M_K" 의
    pose_name 불일치 (CSV 가 일부 dedup 전 단계 entry 보유)
  - 모든 ranker 가 동일하게 영향 받아 ranker-간 비교에는 noise 없음

### 데이터 fix 이력 (v3 → v4)
- v3 (n=419, lscore_top5 = 42.2 %): `_resolve_pose_file` 가 `protenix_dock_L`
  pose 의 SDF 못 찾아서 score_per_metric.py 가 pxdock 행을 CSV 에 안 씀.
  validator 가 cascaded 의 pxdock 픽 lookup 실패시켜 70 타겟 drop.
- v4 (n=470, lscore_top5 = 38.7 %): pose_file 버그 수정 + `supplement_pxdock.py`
  로 380 타겟 / 5656 pxdock 행 보충 + intersection 로직 추가.
- v4 의 SR 가 v3 보다 약간 낮은 이유: intersection 분모가 더 컴 (419 → 470)
  + 추가된 27 dropped 타겟 들이 평균보다 약간 어렵.

## Algorithm

```python
def lscore_top5_diverse(poses, k=5, rmsd_threshold=2.0):
    cand = sorted([p for p in poses if p.lscore is not None],
                  key=lambda p: -p.lscore)
    selected, selected_mols = [], []
    for c in cand:
        m = load_mol(c)
        if not selected:
            selected.append(c); selected_mols.append(m); continue
        if any(rmsd(m, prev) < rmsd_threshold for prev in selected_mols):
            continue   # too close to an already-picked pose
        selected.append(c); selected_mols.append(m)
        if len(selected) == k: break
    return selected
```

핵심 신호 = **lscore (RMSD-Pred 의 1 − P(>2Å))**, 다른 신호는 안 섞는다.
top-1 부터 시작해서 각 후속 pick 은 **모든 이전 pick 과 ≥ 2 Å heavy-atom RMSD**
로 떨어져야 함. RDKit `CalcRMS` 비싼 substructure matching 대신 numpy raw
`sqrt(((a−b)²).sum(axis=1).mean())` 사용.

## Single-scorer ceiling (출발점)

`per_metric_summary_proper.txt` — 각 스코어러를 의미 있는 풀에서만 ranking:

| scorer        | pool                | n   | top-1 < 2 Å | best-5 < 2 Å |
|---------------|---------------------|----:|------------:|-------------:|
| plddt         | cofold-only         | 479 | 143  (29.9 %) | 164  (34.2 %) |
| conf          | cofold-only         | 479 | 141  (29.4 %) | 163  (34.0 %) |
| iptm          | cofold-only         | 479 | 134  (28.0 %) | 160  (33.4 %) |
| ptm           | cofold-only         | 479 | 130  (27.1 %) | 163  (34.0 %) |
| boltz_aff     | cofold-Boltz-only   | 367 |  98  (26.7 %) | 116  (31.6 %) |
| lscore        | all                 | 489 | 123  (25.2 %) | 152  (31.1 %) |
| rmsd_pred     | all                 | 489 | 111  (22.7 %) | 138  (28.2 %) |
| ba_pred       | all                 | 489 |  50  (10.2 %) |  83  (17.0 %) |
| **oracle**    | all                 | 489 | **287 (58.7 %)** | 287 (58.7 %) |

> **NOTE**: cofold-only 스코어러를 docking pose 까지 포함한 전체 풀에서 ranking
> 하면 docking pose 가 상속 받는 docking-anchor 의 constant 값으로 ranking 이
> degenerate 한다 (그 결과는 `per_metric_summary.txt` 의 raw 값). proper-pool
> 기준이 의미 있다.

49.3 %p 갭 분해:
- 489 - 287 = **202 타겟은 풀 안에 < 2 Å pose 자체가 없음** — generation
  단계 한계, ranker 회수 불가능.
- 287 - 152 = 135 타겟이 best-of-5 (lscore 기준) 에서도 놓치는 회수 가능
  분량 → ranker 설계 대상.

### 489 vs 499
원래 input 499. CSV 에 489 만 (10 누락):
- **8 SIGALRM 타임아웃** (HEM-class organometallic 에서 RDKit `rdFMCS` 무한루프,
  `score_per_metric.py` 180 s timeout 으로 잘림): `9emt 9mel 9mgt 9mh5 9q3k
  9r07 9r7a 9whf`
- **2 submission 미완료** (run 끝났지만 `submissions/{tgt}_input.lg` 파일
  없음 — `make_casp_submission` 단계 실패): `9cv8 9dm5`

## Ablation: 시도해본 ranker 들과 패배 이유

### 1) RRF + consensus + lig_align bonus (실패)

가설:
```
final(i) = RRF(i) · (1 + 0.1·log(1+support(i))) + 0.2·lscore(i)·is_lig_align(i)
RRF(i) = Σ 1 / (60 + rank_s(i)),  s ∈ {lscore, plddt_norm, iptm, ptm, conf}
support(i) = |{j : RMSD(i,j) < 2Å, family(j) ≠ family(i)}|
```

stride=10 sample 결과: **22.5 %** top-1 (vs lscore 37 %). **-15 %p**.

실패 원인:
- **scorer correlation**: plddt / conf / iptm / ptm 4개 모두 같은 cofold model
  의 confidence 변형 → 독립 vote 가 아닌데 4 표로 중복 가중. lscore 가 정확히
  잡은 docking pose 를 cofold pose 로 강제로 밀어버림.
- **per-pose consensus 신호 약함**: validate_consensus.py Q3 에서 native 가
  없는 타겟의 67 % 도 multi-family consensus 형성 → false-positive 큼.

### 2) Cluster-then-pick (실패)

가설: pose 들을 2 Å single-link 클러스터링, cluster quality 로 top-5 선택.
representative = cluster 안 min(pRMSD).

stride=10 cluster scoring 6 변형 ablation:

| variant                 | top-1   | best-5  |
|-------------------------|--------:|--------:|
| `n_fam` (순수 consensus) |  24.5 % |  28.6 % |
| `lscore_x_fam` (현재)    |  28.6 % |  36.7 % |
| `size`                   |  28.6 % |  38.8 % |
| `size_x_lscore`          |  28.6 % |  40.8 % |
| `max_lscore`             |  30.6 % |  38.8 % |
| `sum_lscore`             |  30.6 % |  40.8 % |

- 순수 consensus (`n_fam`) 이 worst — Q3 false-positive 가 직격
- lscore 가 들어간 변형이 회복하지만, 어느 변형도 lscore_top5_diverse (37 % /
  44 %) 못 이김

실패 원인 (특히 max_lscore / sum_lscore):
- 옳은 cluster 는 골랐지만 cluster representative 가 `min(pRMSD)` 라
  lscore-best 와 어긋남
- representative 도 lscore-best 로 바꾸면 top-1 은 lscore baseline 과 동일,
  best-5 다양성은 lscore_top5 와 비슷 → 새로 얻는 게 없음

### 3) Cascaded filter (lscore 보존, 약간 손해)

가설: lscore top-50 만 후보 → plddt tie-break + lig_align margin promotion.

full 489 결과: **25.8 % top-1 / 39.0 % best-5** (lscore_top5 27.0 % / 42.2 % 보다
조금씩 손해).

원인: lig_align promotion 과 plddt tie-break 이 평균에서 약간의 noise 를 추가.
lscore 단독이 더 안정적.

## 왜 lscore 가 강한가

1. **RMSD-Pred 의 직접 학습 목표가 "이 pose 가 native (<2 Å) 인가"** — lscore =
   1 − P(>2 Å) 는 그 확률을 직접 재는 신호. proxy 가 아닌 직접 신호.
2. **모든 source family 에 적용 가능** — cofold / docking / template 모두 같은
   인터페이스. 풀 sub-segmentation 없이 동일 ranking.
3. **scaling 단순** — calibrated 확률 [0, 1] 이라 추가 normalization 없음.
4. **다른 신호들은 indirect proxy**:
   - pLDDT / iptm / ptm / conf: "전체 구조가 자신감 있게 예측됐나"
     ≠ "ligand pose 가 옳은가" (모델은 자신 있는데 ligand 가 wrong basin 에 있을
     수 있음)
   - ba_pred (pKd): binding affinity 만 — wrong pose 도 강한 affinity 가질 수
     있음
   - boltz_aff: cofold-Boltz pool 만 → coverage 부족

따라서 위 신호들을 RRF / cluster 로 섞으면 **lscore 의 강한 직접 신호를 indirect
proxy 로 희석시킨다** — 이게 모든 mixing approach 가 lscore 단독을 못 이긴
이유.

## Empirical validation: cross-family consensus signal

`experiments/novel2025_test/validate_consensus.py` — δ=1.0 Å 기준 cross-family
consensus 강도 vs 그 타겟 풀의 `min(true_rmsd)`:

| N_fam (consensus)  | n_targets | %native (<2 Å) |
|-------------------:|----------:|---------------:|
| 1                  | 87        | 23 %           |
| 4                  | 73        | 62 %           |
| 7                  | 54        | 89 %           |
| 11                 | 7         | 100 %          |

✓ Cross-family consensus 는 native 확률과 monotone 한 양의 상관.
✗ But: native 가 없는 타겟의 67 % 도 consensus 형성 → 단독으로는 false-alarm
큼. ablation 에서 confirmed (cluster `n_fam` 24.5 %, RRF 22.5 %).

→ 결론: consensus 는 *상관* 은 있지만 *예측력* 으로는 lscore 한 개를 못 이김.

## Cofold pose intra-model spread (참고)

같은 cofold 모델이 25 pose (5 seed × 5 sample) 를 얼마나 좁게 vs 넓게 만드나?
(`experiments/novel2025_test/analyze_pose_diversity.py`):

| family          | (target, model) IQR<1 Å | IQR<2 Å | range<2 Å |
|-----------------|------------------------:|--------:|----------:|
| cofold_boltz2   | 56 % | 69 % | 37 % |
| cofold_boltz2x  | **61 %** | **73 %** | **43 %** |
| cofold_protenix | 49 % | 65 % | 30 % |
| cofold_af3      | 49 % | 62 % | 29 % |

→ Boltz2x 가 가장 좁게 모음 (mode-collapse 강), Protenix/AF3 가 가장 다양.
30-40 % 케이스에서는 의미있는 conformational diversity 존재. CDF 그림은
`experiments/novel2025_test/analysis_pose_diversity.png`.

## Ablation entrypoints (코드)

`scripts/compute_submission_scores.py` 에 ablation 용 함수 그대로 보존:

| 함수 | 역할 |
|---|---|
| `select_best_pose_pRMSD_legacy(poses)` | 이전 production: min pRMSD, lscore tie |
| `select_best_pose_cluster(poses)` | cluster-then-pick top-1 (옵션 A) |
| `select_top_k_cluster_by(poses, quality)` | cluster 변형 6종 ablation |
| `select_best_pose_cascaded(poses)` | cascaded filter top-1 (옵션 B) |
| `select_top_k_cascaded(poses, k)` | cascaded top-5 |
| RRF helpers (`_rrf_score`, `_consensus_support`, `_final_ranker_score`) | RRF 변형 직접 호출 |

미래에 추가 ablation 이 필요하면 이 entrypoints 위에서 진행 가능.

## Validation pipeline

```
per_pose_scores.csv  (true_rmsd 사전 계산, USalign + RDKit symmetric heavy-atom RMSD)
        │
        ├─ score_per_metric_proper.py     →  per-scorer baseline (proper pool)
        ├─ score_by_source.py             →  per-family hit/recall
        ├─ validate_consensus.py          →  consensus 가설 검증
        ├─ analyze_pose_diversity.py      →  intra-model spread + family CDF
        ├─ cluster_ablation.py            →  cluster scoring 6 변형 (target 당 1번 cluster)
        └─ validate_new_ranker.py         →  ranker 끼리 직접 비교 (top-1 / best-5 SR)
```

각 스크립트는 stride 옵션으로 sample 검증 또는 full batch 둘 다 가능.

## Performance

- Full 489 batch (5 ranker, mol_cache 공유): ~28 분 single core
  (대상: nohup'd `validate_new_ranker.py`, 14:42-15:35 walltime, ~3.4 s/target avg)
- 10sl smoke (637 pose): RRF 0.0 s, consensus 3.2 s, cluster build 1.5 s
- numpy direct RMSD vs RDKit `CalcRMS`: 32 s → 3 s (~10 ×)

## Production code path

`select_best_pose(poses)` 와 `select_diverse_top_k(poses, k=5)` 가
`make_casp_submission.py` 에서 호출됨. 위 ablation 결과를 반영해
**lscore_top5_diverse 동작** 을 default 로 둔다.

## Open follow-ups

- **template_vina / template_adg coordinate frame bug**: family recall 2.2 % /
  1.5 %, RMSD 20–700 Å. lig_align (24.3 % recall) 정상. docking box 정의
  또는 receptor align 단계 버그로 추정 — generation 단계 개선이라 ranker 와
  분리.
- **8 SIGALRM 타임아웃 타겟 회수**: HEM-class organometallic 에서 `rdFMCS` 무한루프.
  `timeout=` 명시 또는 organometallic detector 추가하면 graceful skip 가능.
- **2 submission-missing 타겟 (9cv8, 9dm5)**: `make_casp_submission` 단계
  실패 원인 트레이스 필요.
- **frame mismatch 위험**: docking pose 의 cofold-anchor 가 평가 cofold 와 다를
  때 consensus 가 약해질 가능성. ablation 에서는 majority 가 일치해 영향
  미미.
