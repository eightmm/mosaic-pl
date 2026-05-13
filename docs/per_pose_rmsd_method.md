# Per-pose true-RMSD 계산 (novel2025 evaluation)

`experiments/novel2025_test/per_pose_scores.csv` 의 `true_rmsd` 컬럼은
파이프라인이 생성한 **모든 도킹 pose** 를 **crystal ligand** 와 비교한 heavy-atom
RMSD 다. 이 문서는 그 RMSD 가 어떻게 산출되었는지를 정확히 적어둔다 — 이후
ranker / 분석 / regression test 에서 RMSD 의 의미를 다시 추적할 필요가 있을 때
참고용이다.

대상 코드:
- `experiments/novel2025_test/score_per_metric.py` — per-pose RMSD 계산 + CSV 작성
- `experiments/novel2025_test/evaluate.py` — submission 평가용 동일 파이프라인
- `src/casp17/geometry.py` — `pose_rmsd`, `kabsch`, `transform_mol`

## 입력
1. **Pose 후보**
   `outputs/analysis/poses/*.sdf` 에 staging 된 모든 conformer.
   각 SDF 한 record = 한 pose. 출처:
   - `cofold_{boltz2,boltz2x,protenix,af3}_{lig}_{n}.sdf` — cofold model 의
     `*_aligned.cif` 에서 ligand 만 추출, `sorted(rglob("*_aligned.cif"))` 순서로 staging.
   - `vina_*_seed_{S}.sdf`, `autodock_gpu_*_seed_{S}.sdf`, `protenix_dock/poses.sdf`
     — Track 1 docking.
   - `template_{TID}_{lig_align,vina,adg}_{lig}.sdf` — Track 2/3 template-based.
2. **Crystal ligand**
   RCSB mmCIF (`rcsb_index.db` 가 가리키는 경로) → 첫번째 candidate ligand atom
   block. `load_crystal_ligand` 이 ATOM/HETATM 블록과 SMILES 템플릿을 받아
   `_largest_template_matching_fragment` + `AssignBondOrdersFromTemplate` 으로
   bond order 를 재구성한다. 실패 시 `_sanitize_loose` (kekulize 생략) 폴백.

## 좌표계 정렬
모든 cofold model / docking tool / template track 은 **각자의 receptor frame** 에
ligand 를 만들지만, 우리는 RMSD 를 **crystal frame** 에서 계산한다. 전환은
USalign 의 4x3 rigid transform 한 번으로 처리:

1. **cofold receptor → crystal protein** USalign:
   ```
   crystal ≈ R · pred + t      (R: 3x3 rotation, t: 3 translation)
   ```
   `dump_crystal_protein_pdb(cif, crystal_pdb)` 로 crystal mmCIF 의 protein-only
   PDB 를 만들고 `run_usalign(cofold_pdb, crystal_pdb)` 호출. 결과 (R, t) 와
   `tm_score` (USalign 의 Structure_2-normalized) 를 캐시.
2. 각 pose 의 모든 heavy-atom 좌표에 동일한 (R, t) 를 적용하여 crystal frame
   으로 옮긴다 (`transform_mol`).
   - Track 2/3 template pose 도 staging 단계에서 cofold frame 으로 정렬되어
     있으므로 동일한 cofold→crystal transform 이 그대로 적용된다.

> Cofold pose 는 자기 자신의 cofold frame 에 있으므로 cofold→crystal 정렬이
> 자명하게 맞다. Docking pose 는 docking-anchor cofold (= `docking_prep_summary.json`
> 의 `cofolding_structure`) frame 에 있다. anchor 가 boltz2x 인데 평가 cofold
> PDB 는 protenix 라면 frame mismatch 가 발생한다 — 현재 score_per_metric.py 는
> 첫번째로 발견된 cofold (sorted dir 우선순위) 를 사용한다. 이게 anchor 와 다른
> 경우 docking pose 의 RMSD 는 의미가 약해질 수 있다 (TODO: anchor 기준으로
> 정렬하도록 수정 가능).

## RMSD 자체
crystal frame 으로 옮긴 pose `pred` 와 reference `ref` 사이에서:
1. **Primary** — `pose_rmsd(pred, ref)` (`src/casp17/geometry.py`)
   - heavy-atom 만, 동일 분자식 (atom count 일치) 가정.
   - `Chem.MolToSmiles` round-trip 으로 atom ordering 정규화 후 RDKit
     `rdMolAlign.CalcRMS` (symmetry-corrected, no alignment).

> ### ⚠ 규칙: RMSD 는 반드시 `rdMolAlign.CalcRMS` — `GetBestRMS` 금지
> - **GetBestRMS** 는 내부적으로 Kabsch 정렬을 수행한 뒤 RMSD 를 반환한다.
>   → ligand 가 pocket 밖으로 떨어졌는데도 "잘 정렬된 작은 RMSD" 가 나와
>      placement 오류를 가린다. CASP pose evaluation 에는 부적합.
> - **CalcRMS** 는 정렬 없이 현재 좌표에서 symmetry-aware atom mapping 만으로
>   RMSD 를 계산한다. 우리는 receptor 단위로만 정렬하고 ligand 좌표는 그대로
>   보고 싶으므로 이게 맞다.
> - 추가로, 직접 `GetSubstructMatches(..., uniquify=False)` 로 permutation 을
>   enumerate 하는 패턴도 금지 — 2026-05-11 batch 에서 symmetric ligand
>   하나가 `score_per_metric.py` 를 1시간 넘게 hang 시켰음. C-extension 내부라
>   SIGALRM 180s timeout 가 안 먹혔다. CalcRMS 가 내부적으로 cap 을 둔다.
> - 신규 평가/분석 코드에서도 동일: `from rdkit.Chem import rdMolAlign;
>   rdMolAlign.CalcRMS(prb, ref)` 로 통일.
2. **Fallback** — `_mcs_rmsd(pred, ref)`
   - primary 가 NaN / atom-count mismatch / sanitization 실패 시 호출.
   - `rdFMCS.FindMCS(elementsMatch, anyBondMatch, ringMatchesRingOnly=True,
     completeRingsOnly=True)` 로 최대 공통 substructure 찾고, 그 atom mapping 만
     으로 RMSD 평균.
   - 정렬은 안 한다 — 이미 crystal frame 으로 옮겨진 좌표 위에서의 raw 거리.
   - HEM/NAG 같은 organometallic / glycan 에서 primary 가 실패할 때 살린다.

## 실패 / 제외
다음 경우 `true_rmsd` 가 `None` 이 되어 CSV 에 빈 칸으로 남고 SR 집계에서
제외된다:
- crystal CIF 가 `rcsb_index.db` 에 없거나 CCD 코드 mismatch
- ligand SMILES 가 `Chem.MolFromSmiles` 로 파싱 실패 (현재 8개 타겟:
  `9emt 9mel 9mgt 9mh5 9q3k 9r07 9r7a 9whf` — 대부분 HEM-class 유기금속,
  `_mcs_rmsd` 가 무한루프에 빠져 SIGALRM 180s 타임아웃으로 잘라냄)
- USalign 실패 (cofold receptor 가 너무 짧거나 chain 매칭 실패)
- pose SDF 가 staging 단계에서 누락

## 검증 / 디버깅
- 한 pose 단위로 다시 보고 싶을 땐:
  ```python
  from experiments.novel2025_test.score_per_metric import _conformer_at, _compute_rmsd
  pred = _conformer_at(sdf_path, record_idx)        # transform 전
  pred_xfm = transform_mol(pred, R, t)               # crystal frame 으로
  rmsd = _compute_rmsd(pred_xfm, crystal_lig, eff_template)
  ```
- `_per_metric_work/` 캐시는 USalign 결과 + crystal pdb 를 타겟별로 저장.
  반복 분석 시 그 캐시를 그대로 재사용한다.

## 왜 이 정의인가
- **rigid alignment 안 함** — 우리가 만들고 싶은 건 "binding pocket 안에서
  실제로 native pose 를 맞췄나?" 다. 추가로 ligand 만 따로 align 하면 (예:
  Kabsch on ligand atoms) 항상 작은 RMSD 가 나오므로 평가 의미가 사라진다.
  대신 receptor 단위로만 정렬한 뒤 그 위에서 ligand 좌표를 그대로 본다.
- **symmetric heavy-atom RMSD** — RDKit 의 `CalcRMS` 는 분자 대칭을
  내부적으로 처리해 가장 작은 atom-mapping RMSD 를 반환한다 (정렬 X). C2 대칭
  분자 (예: symmetric biphenyl) 에서 atom-index mismatch 만으로 잘못된 RMSD 가
  나오는 걸 막는다. `GetBestRMS` 는 절대 쓰지 말 것 — 위 ⚠ 규칙 참조.
- **MCS fallback** — bond order 재구성 실패는 평가 단계에서도 흔하다 (HEM,
  NAG, 변환된 protonation 상태 등). MCS 매핑은 element 만 보고 최대 공통
  substructure 의 거리 평균을 내므로 partial match 에서도 합리적 거리를 준다.

## 한계 / 알려진 이슈
- **frame mismatch 위험**: docking pose 가 docking-anchor cofold 가 아닌
  다른 cofold model 의 frame 으로 잘못 정렬될 수 있다 (위 NOTE 참조). 현재까지의
  분석에서는 vast majority 가 cofold-anchor (= protenix or boltz2 둘 중 하나)
  로 일치하므로 큰 문제는 아니지만, 새 anchor 선택 로직이 들어가면 재검토 필요.
- **template_vina / template_adg RMSD 가 비정상적으로 큼**: 근본 원인은 RMSD
  계산이 아니라 template-based docking box 정의 / receptor alignment 단계의
  좌표계 mismatch 로 추정 (recall 2.2%/1.5%, RMSD 20-700Å). lig_align 은 동일
  pipeline 위에서도 24.3% recall 로 정상.
- **8개 타임아웃 타겟**: HEM 등 유기금속에서 `rdFMCS` 가 폴리노미알 시간을
  넘김. 향후 `timeout=` 파라미터 명시 또는 organometallic detection 으로 graceful skip.
