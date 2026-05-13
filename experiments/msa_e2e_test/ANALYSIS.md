# Novel2025 새 파이프라인 batch — 분석 (2026-05-13)

대상: `experiments/msa_e2e_test/runs/<target>_input/<target>_input/` 의 **499 novel2025 타깃** (`experiments/novel2025_test/cluster_targets_100.csv`).
새 파이프라인 = MSA pipeline + mmseqs∪foldseek 템플릿 + consensus pocket (top-10) + Multi-track docking (Vina + ADG + Track 2/3) + RMSD/BA-Pred + LG submission.

---

## 1. Batch overview

| 항목 | 값 |
|---|---:|
| 입력 타깃 | 499 |
| 디스크 사이즈 | 1.3 TB |
| Cofold 모델 | Boltz2 + Boltz2x + Protenix + AlphaFold3 (5 seeds × 5 samples) |
| 도킹 도구 | Vina + AutoDock-GPU (PxDock off per `b7dd9d5`) |
| Binding-site 소스 | cofolding_{1..3}, p2rank_{1..3}, swinsite_{1..3}, **template_consensus_{1..10}** (mmseqs∪foldseek) |
| Track 2 / 3 | template-based box docking, lig-align (MCS≥0.5) |
| Pose pool 평균 | 2,894 pose/target (median 1,476; max 21,513) |

Pose family breakdown:

| family | total | % | avg/target | median | max |
|---|---:|---:|---:|---:|---:|
| ADG | 845,797 | 62.7% | 1,815 | 515 | 17,436 |
| Vina | 395,524 | 29.3% | 849 | 696 | 3,573 |
| Cofold | 63,216 | 4.7% | 136 | 101 | 505 |
| Template (Track 2/3) | 44,240 | 3.3% | 95 | 76 | 591 |
| PxDock | 0 | 0% | — | — | (꺼둠) |

총 **1,348,777 pose × 14 score column** (true RMSD 포함) 적재됨 — `per_pose_scores.csv` (~120 MB).

---

## 2. Oracle SR (best-pose ceiling)

499 중 **466 타깃**에 대해 풀에서 best RMSD 선택. 4가지 카운팅 방식:

| 방식 | n | <2Å | <1Å | 의미 |
|---|---:|---:|---:|---|
| A. ANY-ligand (예전 출력) | 466 | **317 (68.0%)** | 197 (42.3%) | 모든 ligand 통합 best, cofactor만 맞춰도 hit |
| B. ALL-ligand (strict) | 466 | **276 (59.2%)** | 155 (33.3%) | 타깃의 모든 ligand 가 sub-2Å 일 때만 hit |
| C. PRIMARY '_L' only | 450 | **295 (65.6%)** | 176 (39.1%) | L (관심 ligand) 만 체크 |
| D. PER-(target, ligand) | 628 | **367 (58.4%)** | 229 (36.5%) | PoseBusters/PLINDER 식 |

**Zone 분포 (방식 C: PRIMARY)**

| zone | n | <2Å | <1Å |
|---|---:|---:|---:|
| novel | 59 | 31 (52.5%) | 17 (28.8%) |
| remote | 101 | 81 (80.2%) | 52 (51.5%) |
| related | 290 | 183 (63.1%) | 107 (36.9%) |

멀티-ligand 타깃 분포: 1ℓ=350 / 2ℓ=83 / 3ℓ=22 / 4ℓ=9 / 5ℓ=2 (총 25% multi-cofactor 타깃).

**리포트 권장 헤드라인**: 방식 **C (Primary L) 65.6%** 또는 **D (per-lig) 58.4%** — CASP-comparable. 방식 A 는 cofactor 인플레이션 때문에 부적합.

---

## 3. Per-scorer top-1 SR

각 scorer로 풀에서 top-1 / top-5 골라낸 결과 (TOTAL n=466, <2Å):

| scorer | top1<2Å | top5<2Å | top1<1Å | med top1 | med best5 |
|---|---:|---:|---:|---:|---:|
| **iptm** | **148 (31.8%)** | 172 (36.9%) | 69 (14.8%) | 4.78 | 3.36 |
| **conf** | 143 (30.7%) | 169 (36.3%) | 70 (15.0%) | 4.96 | 3.61 |
| **plddt** | 142 (30.5%) | 181 (38.8%) | 75 (16.1%) | 4.47 | 3.12 |
| **ptm** | 133 (28.5%) | 171 (36.7%) | 57 (12.2%) | 4.96 | 3.39 |
| boltz_aff | 121 (26.0%) | 154 (33.0%) | 56 (12.0%) | 5.52 | 4.19 |
| lscore | 114 (24.5%) | 145 (31.1%) | 67 (14.4%) | 7.57 | 5.27 |
| rmsd_pred | 97 (20.8%) | 133 (28.5%) | 53 (11.4%) | 8.49 | 5.28 |
| ba_pred | 37 (7.9%) | 76 (16.3%) | 11 (2.4%) | 11.59 | 7.21 |
| **oracle** | **317 (68.0%)** | 317 (68.0%) | 197 (42.3%) | 1.26 | 1.26 |

핵심 관찰:
- **ranker 가 병목** — pose pool oracle 68% 인데 best single scorer (iptm) 가 31.8%, gap **36 pp**.
- cofold confidence 기반 scorer (iptm/conf/plddt/ptm) > 도킹 affinity 기반 scorer (boltz_aff/lscore/ba_pred/rmsd_pred).
- 가장 약한 ba_pred 7.9% — affinity 모델이 pose 선택용으로 부적합한 신호.

---

## 4. LG submission MODEL 1 / 1-5 SR

`make_casp_submission.py` 가 LG-format 으로 top-5 MODEL 골라내는 결과 (`evaluate.py` 평가).

| metric | OLD (5/2 LG) | NEW (5/13 regen) | Δ |
|---|---:|---:|---:|
| MODEL 1 <2Å | 116/498 (23.3%) | 117/498 (23.5%) | +1 (+0.2pp) |
| MODEL 1 <1Å | 50/498 (10.0%) | **65/498 (13.1%)** | **+15 (+3.1pp)** |
| MODEL 1-5 <2Å | 182/498 (36.5%) | 183/498 (36.7%) | +1 (+0.2pp) |
| median TM (cofold↔crystal) | 0.979 | 0.979 | — |

**Zone 분포 (NEW, MODEL 1 <2Å)**

| zone | n | top1 | best5 |
|---|---:|---:|---:|
| novel | 71 | 8 (11.3%) | 20 (28.2%) |
| remote | 116 | 35 (30.2%) | 61 (52.6%) |
| related | 311 | 74 (23.8%) | 102 (32.8%) |

**해석**
- LG regen 후 top1<2Å 변화 미미 (+1) — LG ranker(lscore) 가 새 pose 풀에서도 비슷한 1위 뽑음.
- top1<1Å +15 — "맞춘 케이스의 정확도" 끌어올림 (multi-ligand-aware RMSD + 새 ADG TC 포즈).
- 2Å 트랜지션 케이스는 ranker 약함이 직접 병목.

---

## 5. Frame consistency (50 타깃 random sample)

| 검증 | 결과 |
|---|---|
| `receptor.pdb` ↔ `cofolding_structure` (anchor) CA centroid | **50/50 == 0.0000 Å** (동일 좌표계 확정) |
| Cofold 4-model Kabsch 후 CA centroid 최대 pair 거리 | median 0.0 Å, max 4.7 Å — 2/50 만 >1Å |
| Pose family heavy-atom centroid vs receptor centroid | 모두 (rec_extent + 30 Å) 안 |
| vina/adg/template — frame 일관성 | ✅ 모두 receptor frame |

→ **도킹된 모든 시스템이 단백질 기준 같은 frame에 정렬됨**. 4-cofold model 사이 작은 conformational divergence는 있지만 frame 버그 아님.

---

## 6. 알려진 버그 / 수정 이력 (이번 batch 동안)

| 날짜 | 버그 | 영향 | 수정 |
|---|---|---|---|
| 5/4 (`018b78f`) | template-bridge가 alignment 전에 먼저 fire → template_consensus centroid가 unaligned frame 에 박힘 | template_consensus pocket 무용지물 | 브릿지 순서 align → template → prep 로 |
| 5/11 (`prepare_docking_inputs.py:1153`) | receptor_pdbqt 상대경로로 기록 → autogrid 가 cwd=grid_dir 에서 찾다 실패 → ADG template_consensus_* 사실상 0개 pose | 217/462 타깃의 ADG TC 누락 | `args.output_dir.resolve()` 강제 절대화 + adapter runner 에서 `Path(...).resolve()` 안전망 + autogrid 에러 stderr+FAILED 마커 |
| 5/11 (`src/casp17/geometry.py`) | `pose_rmsd` 가 `GetSubstructMatches(uniquify=False)` enumeration — symmetric ligand 에서 hang (SIGALRM 무시) | score job 12h timeout | `rdMolAlign.CalcRMS` 로 교체. `docs/per_pose_rmsd_method.md` ⚠ 블록 추가 (GetBestRMS 금지) |
| 5/12 (`score_per_metric.evaluate_run`) | candidate_ccd 첫 번째만 ref_lig 로 로드 → 멀티-cofactor 타깃 L2/L3 pose 가 wrong-ref vs MCS fallback 무한 루프 → 31s/pose × 1000 pose | 멀티-ligand 타깃 전체 hang (8z15 등) | prep['ligands'] 의 SMILES 로 candidate_ccd 매칭 → `lig_id_to_ref` 맵 → pose source `_L`/`_L2` suffix 로 ref 선택 |

---

## 7. 진행 중 — Cat 1 / 2 / 3 재실행

`per_pose_scores.csv` 에 466/499 만 들어간 이유. 33 missing 의 카테고리별 원인:

### Cat 1 (19) — `no_prep` — inputs/docking 통째로 빔
SLURM 로그상 5/3-5/4 에 정상 실행 + LG 생성 끝났는데 `outputs/{boltz2,protenix,af3,vina_*,adg_*,...}` 와 `inputs/docking/` 통째로 사라짐 (msa_pipeline + template_search dir만 남음). 어딘가의 cleanup 이 데이터 날림.
- 처리: 각 타깃의 `scripts/run_wrapper.sbatch.sh` 재제출 → full pipeline GPU 재실행
- 잡 ID: 35289-35307

### Cat 2 (5) — `no_staged_poses` — analysis/ 비어 있지만 raw docking 살아있음
8pvw, 8rpa, 8rx6, 8s7n, 8w9s. cofold cif + vina/adg dlg 있지만 `outputs/analysis/poses/*.sdf` 사라짐.
- 처리: post-analysis + LG 만 재실행
- 잡 ID: 35308

### Cat 3 (9) — `score_skipped` — 180s timeout
9dsv (U6A), 9emt (5NG), 9ifw (FAD), 9mgt (COA), 9mh5 (COA), 9n1b (B12), 9uo2 (76F), 9vjx (3VV), 9zno (A1C3G). 전부 큰 cofactor/긴 사슬/organometallic.
- 처리: per-target timeout 180s → 1800s 로 올려서 score_per_metric 재실행
- 잡 ID: 35309

---

## 8. 우선순위 다음 작업

1. **Cat 1/2/3 결과 합쳐서 466 → 499 완전 커버리지** (이번 진행 중)
2. **새 ranker** — pose pool oracle 68% 와 best scorer 32% 사이 36pp gap. iptm/plddt/conf 앙상블 또는 학습 기반 ranker 필요.
3. **ADG cofactor receptor type 문제 (54/462 타깃)** — `c65797b` 후 retained metals 가 autogrid4 force-field 에 없는 type (CG/Co/Hg/K/Na/Ni). prep 에서 이 atom 들 처리 결정 필요 (drop vs map).
4. **multi-ligand 평가 컨벤션 확정** — 보고용 Oracle 헤드라인을 Primary(C) 로 갈지 PER-lig(D) 로 갈지 의사결정.

---

## 부록: 데이터 위치

- Pose pool: `experiments/msa_e2e_test/per_pose_scores.csv` (1,348,777 rows)
- Per-scorer SR: `experiments/msa_e2e_test/per_metric_summary.txt`
- LG submissions: `experiments/msa_e2e_test/submissions/*.lg` (499)
- LG-based eval: `experiments/msa_e2e_test/evaluation.json` (498/499 평가)
- Frame audit: `/tmp/frame_audit_50.tsv` (50 random sample)
- Cluster CSV: `experiments/novel2025_test/cluster_targets_100.csv` (499 → 246 unique seq enzymes at 100%)
- Sample scripts: `experiments/msa_e2e_test/{rerun_adg_tc.sh, regen_lg.sh, rescore_cat3.py, rerun_cat2.sh}`
- Frame audit tool: `scripts/audit_frame_50.py`, `scripts/check_frame_consistency.py`
- pose RMSD 정의: `docs/per_pose_rmsd_method.md` (⚠ GetBestRMS 금지 규칙 박혀 있음)
