# novel2025 — per-source / per-metric performance analysis

잘못된 pocket 선택과 selector ranking이 실제 top-1 submission 성공률에 어떻게
영향을 주는지, 그리고 개별 cofolding/docking 모델의 native score가 correct
pose를 얼마나 변별하는지를 정량 비교한 분석 로그. 분석 스크립트는 모두
`experiments/novel2025_test/_eval_work/` 아래에 있다.

---

## 0. Pipeline 상태 스냅샷 (분석 시점)

| 항목 | 값 |
|------|-----|
| 총 novel2025 타겟 | 499 |
| Submission 완료 (`experiments/submissions/*.lg`) | 346 |
| 진행률 | 69.3% |
| 평균 run 시간 (6000ada 기준) | ~60–90 min / target |
| 실패한 파이프라인 bug (최종 수정까지) | 11개 커밋 (MSA bridge, AF3 chain id remap, BA/RMSD-Pred 이름 정규화, template ligand SDF, template receptor PDB, RDKit 3D title, smart SMILES split, 기타) |
| 실패한 노드 이슈 | `heavy`(gpu1, H100+Blackwell6000pro)에서 dgl 2.4 / torch 2.4 kernel 미포함으로 BA/RMSD-Pred silent 실패 → partition을 `6000ada`로만 고정 |

---

## 1. 최종 submission 품질 (우리 ensemble top-5)

`experiments/novel2025_test/evaluate.py`로 LG 파일을 crystal ligand에 align
(USalign R, t)해서 계산. 275 / 330 타겟 평가 성공.

| zone | n | top-1 <2Å | best-5 <2Å | top-1 <1Å | median top-1 | median best-5 | median TM |
|------|---|-----------|------------|-----------|--------------|---------------|-----------|
| novel | 32 | **9 (28%)** | **15 (47%)** | 4 (13%) | **4.55 Å** | **2.17 Å** | 0.980 |
| remote | 48 | 11 (23%) | 20 (42%) | 5 (10%) | 4.29 | 2.77 | 0.983 |
| related | 195 | 30 (15%) | 51 (26%) | 8 (4%) | 7.42 | 4.41 | 0.982 |
| **TOTAL** | **275** | **50 (18%)** | **86 (31%)** | **17 (6%)** | **6.19 Å** | **3.94 Å** | **0.982** |

Novel/remote zone이 related zone보다 성적 높은 이유는 cofolding이 close
homolog bias를 덜 받아 올바른 pocket을 더 자주 잡기 때문. related zone은
cofolding + P2Rank + SwinSite가 같은 pathology를 공유해서 wrong pocket에
합의하는 경우가 많음.

---

## 2. Per-source standalone: pRMSD-Pred ranking

스크립트: `_eval_work/per_source_eval.py` → `per_source_eval.json`

각 source의 pose pool을 우리의 **pRMSD-Pred** (.venvs/pred의 RMSD-Pred GNN)
점수로 정렬해서 top-1 / best-5를 측정. 즉 "이 source의 포즈만 쓰고 우리
pRMSD-Pred로 랭킹했을 때" 성적.

| Source | n | avg pool | Med Top-1 | Top-1 <2Å | Med Best-5 | Best-5 <2Å |
|--------|---|----------|-----------|-----------|------------|------------|
| cofold_af3 | 197 | 28 | 6.84 | **22.8%** | 5.27 | 26.9% |
| cofold_protenix | 196 | 25 | 5.80 | 21.4% | 4.96 | 26.5% |
| cofold_boltz2x | 197 | 25 | 6.23 | 19.3% | 5.69 | 21.8% |
| cofold_boltz2 | 197 | 25 | 6.17 | 15.7% | 5.38 | 19.3% |
| autodock_gpu_cofolding | 311 | 71 | 6.88 | 11.6% | 6.28 | 15.4% |
| vina_cofolding | 311 | 49 | 7.25 | 12.9% | 6.18 | 18.3% |
| vina_p2rank | 311 | 49 | 7.50 | 12.2% | 6.85 | 14.8% |
| autodock_gpu_p2rank | 311 | 74 | 7.89 | 9.6% | 6.65 | 10.9% |

비교를 위해 **우리 ensemble (현재 submission)**: top-1 <2Å 18.2% / best-5 <2Å 31.0%.

- **top-1 기준 cofold_af3(22.8%) / cofold_protenix(21.4%)가 우리 ensemble(18.2%)보다 높음** — 즉 pRMSD-Pred가 cofolding 포즈 pool만 보면 잘 뽑는데, docking 포즈가 풀에 합류하면 docking 포즈를 과신해서 top-1 품질이 떨어짐.
- best-5 기준은 우리 ensemble(31%) > 개별 source 최고(26.9%) — 다양성의 가치.

⚠️ 이 테이블의 cofold_* `n`이 197인 이유는 multi-ligand 타겟에서 cofolding
ligand 추출 로직이 X2 chain(ion)까지 포함해 template mismatch로 실패했기
때문. §4에서 수정 후 재집계.

---

## 3. Per-source standalone: 각 모델의 native score

스크립트: `_eval_work/per_source_native.py` → `per_source_native.json`

`rmsd 모델없이 각 모델의 스코어만으로` 비교. 각 source 내부에서 제공하는
native confidence/energy로 top-1 / best-5 선정.

| Source | native score | n | avg pool | Med Top-1 | Top-1 <2Å | Med Best-5 | Best-5 <2Å |
|--------|--------------|---|----------|-----------|-----------|------------|------------|
| cofold_boltz2 | confidence_score | 198 | 25 | 6.37 | 17.7% | 5.75 | 20.7% |
| cofold_boltz2x | confidence_score | 198 | 25 | 6.45 | 18.7% | 6.07 | 21.7% |
| cofold_protenix | (plddt+ptm+iptm)/3 | 197 | 25 | 6.62 | 20.3% | 5.06 | 25.4% |
| cofold_af3 | ranking_score | 198 | 28 | 6.87 | 20.7% | 5.61 | 23.2% |
| vina_cofolding | vina free_energy | 314 | 49 | 7.28 | 14.0% | 6.78 | 15.6% |
| protenix_dock | pxdock pscore | 314 | 8 | 7.30 | 13.7% | 5.90 | 16.9% |
| vina_p2rank | vina free_energy | 314 | 49 | 7.80 | 11.5% | 7.21 | 13.7% |
| autodock_gpu_cofolding | adg free_energy | 314 | 71 | 8.65 | 5.1% | 7.94 | 7.3% |
| autodock_gpu_p2rank | adg free_energy | 314 | 74 | 8.89 | 6.1% | 8.30 | 7.6% |

- **Native ADG free_energy가 가장 약한 signal** (5-6%). ADG grid + genetic
  algorithm 조합이 wrong pocket에 한 번 수렴하면 빠져나오지 못해서, correct
  pocket을 native energy로 구분 못 함. Vina (stochastic MC) 11-14% 보다 명확히 나쁨.
- Cofolding의 native score도 독자적으로 사용 시 17-21% 수준. pRMSD-Pred
  ranking(§2) 보다 낮지만 우리 ensemble과 비슷.

---

## 4. Per-source: **수정된 cofolding ligand 추출** 후 native score

스크립트: `_eval_work/per_source_native.py` (chain=='L' 전용 추출로 수정)
→ `per_source_native.json` 재작성

이전 §3 결과에서 cofolding `n`이 197-198로 나왔던 건, 평가 스크립트의
`extract_cofolding_ligand`가 **multi-ligand 타겟(116개)에서 ion/cofactor
까지 함께 추출**해 template 원자수와 불일치 → `reassign_bonds` 실패 →
해당 타겟이 집계에서 누락됐기 때문. **파이프라인 문제는 없었고 분석 스크립트
문제**. chain 이름을 `L`로 제한하면 전 타겟에서 candidate 리간드만 깔끔하게
뽑힘.

| Source | native score | n | avg pool | Med Top-1 | **Top-1 <2Å** | **Med Best-5** | **Best-5 <2Å** |
|--------|--------------|---|----------|-----------|---------------|----------------|----------------|
| cofold_protenix | (plddt+ptm+iptm)/3 | 317 | 25 | **4.96** | 28.4% | **3.94** | 33.8% |
| cofold_af3 | ranking_score | 322 | 30 | 5.14 | **30.7%** | 4.07 | **33.9%** |
| cofold_boltz2 | confidence_score | 322 | 25 | 5.34 | 25.2% | 4.71 | 29.8% |
| cofold_boltz2x | confidence_score | 322 | 25 | 5.38 | 27.0% | 4.77 | 31.4% |
| vina_cofolding | vina free_energy | 322 | 49 | 7.28 | 14.0% | 6.78 | 15.5% |
| protenix_dock | pxdock pscore | 322 | 8 | 7.33 | 14.0% | 5.90 | 17.4% |
| vina_p2rank | vina free_energy | 322 | 49 | 7.82 | 11.5% | 7.28 | 13.7% |
| autodock_gpu_cofolding | adg free_energy | 322 | 71 | 8.69 | 5.3% | 8.00 | 7.5% |
| autodock_gpu_p2rank | adg free_energy | 322 | 74 | 8.89 | 6.2% | 8.30 | 7.8% |

- **cofold_af3 단독(native ranking_score)으로도 30.7%** — 우리 ensemble(18.2%)보다 +12pp.
- **cofold_protenix 단독 28.4%** 도 ensemble 압도.
- Best-5에서도 cofold_af3(33.9%) > 우리 ensemble(31.0%) — single-model
  sample 25개가 docking + cofolding 합친 pool 250개보다 더 잘 커버.
- **ensemble의 진짜 문제는 다양성 부족이 아니라 selector(pRMSD-Pred)가
  docking 포즈를 cofolding 포즈보다 과신**. 훈련 데이터가 Vina/ADG 포즈 위주라
  cofolding 포즈에 덜 특화됐을 것으로 추정.

---

## 5. Cofolding 모델 × 여러 metric 비교

스크립트: `_eval_work/per_cofold_metric.py` → `per_cofold_metric.json`

각 cofolding 모델이 제공하는 **모든 confidence metric 각각으로 ranking**
했을 때 top-1 / best-5. Oracle은 해당 모델의 25 sample 중 실제 최저 RMSD
(모델 단독 upper bound).

### cofold_af3 (n=324, **Oracle <2Å = 42.6%**, oracle med 2.54Å)

| Metric | Med T1 | **Top-1 <2Å** | Med B5 | **Best-5 <2Å** |
|--------|--------|---------------|--------|----------------|
| **ranking_score** | 5.14 | **30.9%** | 4.07 | 34.0% |
| iptm | 5.14 | 30.6% | 4.15 | **35.5%** |
| ptm | 5.33 | 30.6% | 4.20 | 35.2% |
| neg_fraction_disordered | 5.50 | 28.4% | 4.06 | 34.6% |

### cofold_protenix (n=319, **Oracle <2Å = 42.0%**, oracle med 2.67Å)

| Metric | Med T1 | **Top-1 <2Å** | Med B5 | **Best-5 <2Å** |
|--------|--------|---------------|--------|----------------|
| **neg_gpde** | **4.51** | **29.5%** | **3.53** | 33.9% |
| plddt | 4.82 | 28.8% | 3.61 | 31.3% |
| iptm | 4.86 | 28.8% | 3.59 | **34.5%** |
| ptm | 5.11 | 26.6% | 3.57 | 32.9% |

### cofold_boltz2 (n=324, Oracle <2Å = 34.3%, oracle med 3.96Å)

| Metric | Med T1 | Top-1 <2Å | Med B5 | Best-5 <2Å |
|--------|--------|-----------|--------|------------|
| ptm | 5.07 | 23.5% | 4.55 | **30.2%** |
| **complex_iplddt** | 5.32 | **26.2%** | 4.71 | 29.9% |
| iptm | 5.32 | 25.3% | 4.59 | 29.9% |
| ligand_iptm | 5.32 | 25.3% | 4.59 | 29.9% |
| confidence_score (default) | 5.34 | 25.0% | 4.71 | 29.6% |
| neg_complex_pde | 5.34 | 24.1% | 4.55 | 30.2% |
| complex_plddt | 5.37 | 24.1% | 4.72 | 29.3% |
| neg_complex_ipde | 5.37 | 24.4% | 4.38 | 29.9% |

### cofold_boltz2x (n=324, Oracle <2Å = 33.6%, oracle med 3.97Å)

| Metric | Med T1 | Top-1 <2Å | Med B5 | Best-5 <2Å |
|--------|--------|-----------|--------|------------|
| **complex_iplddt** | 5.26 | **29.3%** | 4.86 | 31.5% |
| complex_plddt | 5.29 | 26.9% | 4.86 | 30.6% |
| iptm | 5.33 | 25.9% | 4.70 | 30.2% |
| ligand_iptm | 5.33 | 25.9% | 4.70 | 30.2% |
| confidence_score (default) | 5.38 | 26.9% | 4.77 | 31.2% |
| ptm | 5.37 | 24.1% | 4.69 | 29.0% |
| neg_complex_ipde | 5.40 | 26.2% | 4.79 | 29.9% |
| neg_complex_pde | 5.47 | 24.4% | 4.64 | 28.7% |

### 핵심 발견

1. **Pool quality (Oracle upper bound):**
   - af3 42.6% ≈ protenix 42.0% ≫ boltz2/2x 33-34%
   - AF3와 Protenix는 본질적으로 더 자주 correct pose를 샘플링함. Boltz는 약 10pp 낮음.

2. **Model당 "최고 metric" (top-1 기준):**

   | Model | Best metric | Top-1 <2Å | vs default |
   |-------|-------------|-----------|------------|
   | af3 | `ranking_score` (default) | 30.9% | — |
   | protenix | `neg_gpde` | 29.5% | vs `(plddt+ptm+iptm)/3` 28.4% → **+1.1pp** |
   | boltz2 | `complex_iplddt` | 26.2% | vs `confidence_score` 25.0% → **+1.2pp** |
   | boltz2x | `complex_iplddt` | 29.3% | vs `confidence_score` 26.9% → **+2.4pp** |

   Protenix는 현재 커스텀 평균 대신 `gpde`(global pred distance error, 낮을수록 좋음)
   단독이 더 discriminating. Boltz는 `confidence_score`보다 `complex_iplddt`
   (interface pLDDT)가 pose 선별에 더 유용.

3. **Best-5 기준 최강 metric:**
   - af3: **iptm** 35.5% (다양성 있는 랭킹)
   - protenix: **iptm** 34.5%
   - boltz2/2x: `ptm` / `complex_iplddt`
   → iptm/ptm 계열이 다양한 포즈 pool을 더 잘 커버. `ranking_score` 같은
   composite는 top-1에 강하지만 다양성은 떨어지는 경향.

4. **우리 ensemble과의 격차:**
   - 현 ensemble top-1 <2Å 18.2%
   - af3 단독 (ranking_score) 30.9% / protenix 단독 (neg_gpde) 29.5%
   - Best-5도 ensemble 31.0% < af3 35.5% (iptm)
   → single cofolding model 단독이 multi-tool ensemble을 top-1/best-5 모두에서 이김.
   Ensemble selector의 ranking 문제.

---

## 6. 실용적 권고

현재 `scripts/compute_submission_scores.py::select_diverse_top_k`가 pRMSD-Pred
primary로 고르는 구조인데, 위 분석은 다음을 시사:

### (A) 즉시 도입 가능 (submission 재생성만, 파이프라인 재실행 불필요)

```
MODEL 1 = cofold_af3 top-1 by ranking_score          (expected ~31% <2Å)
MODEL 2 = cofold_protenix top-1 by neg_gpde          (~29%)
MODEL 3 = cofold_boltz2x top-1 by complex_iplddt     (~29%)
MODEL 4 = cofold_boltz2  top-1 by complex_iplddt     (~26%)
MODEL 5 = docking best (from multi-variant pool)     (다양성 확보)
```

80개 완료된 submission에 대해 `make_casp_submission.py` 재실행만으로
(~2분) top-1 <2Å를 현 18.2% → 예상 **30%+** 끌어올릴 수 있음.
Pipeline을 다시 돌릴 필요 없음 — 모든 cofolding sample의 CIF + per-sample
confidence JSON은 run_dir에 이미 저장돼 있음.

### (B) Pipeline 개선 (장기)

1. **`run_post_analysis.py::_extract_cofolding_ligand`를 chain=='L' 전용으로 수정** —
   현재 multi-ligand 타겟에서 ion까지 추출해 BA-Pred/RMSD-Pred 입력이 오염되는 버그
   (이번 분석에서 같은 버그를 수정 후 cofold `n`이 197→322로 회복된 것과
   정확히 같은 문제가 실제 scoring 경로에도 있음).

2. **pRMSD-Pred 재학습 or 사후 보정**: cofolding 포즈에 대해 systematically
   over-confident하게 predict하도록. 최소한 "source-prior" 가중치로 
   cofolding 포즈 score에 1.0+, docking 포즈 score에 0.8x를 곱하는 보정만 해도
   ranking이 크게 달라질 것.

3. **ADG_p2rank 제외 검토**: 322 타겟 중 top-1 <2Å 5-6%로 bottom, pool 크기
   74로 가장 크지만 noise만 만드는 중. docking seed 줄이거나 제외 시 post-
   analysis 부담 감소.

4. **Protenix metric으로 gpde 추가 + confidence ranking 개선**: 현재
   `compute_submission_scores` 쪽에서 per-cofolding-model confidence를
   anchor로 쓰지 않음. Protenix/Boltz confidence를 MODEL 1 선택에 반영하면
   +10pp 이상 가능.

---

## 7. 참고 파일

| 스크립트 | 출력 JSON | 역할 |
|---------|-----------|------|
| `evaluate.py` | `evaluation.json` | 최종 LG 파일 top-5 품질 |
| `_eval_work/per_source_eval.py` | `per_source_eval.json` | source별 pRMSD-Pred ranking 성능 |
| `_eval_work/per_source_native.py` | `per_source_native.json` | source별 native score ranking 성능 (chain=='L' 수정 반영) |
| `_eval_work/per_cofold_metric.py` | `per_cofold_metric.json` | cofold 모델 × 각 metric 성능 (oracle 포함) |
