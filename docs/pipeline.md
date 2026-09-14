# CASP17 Protein-Ligand Pipeline

End-to-end pipeline: **단백질 서열 + 리간드 SMILES 의 unified YAML 한 장 → CASP17 LG 제출 파일**.
4 개 co-folding 모델, 3 개 docking 도구, 2 개 binding-site predictor + template-consensus pocket,
2 개 post-analysis GNN, mmseqs ∪ foldseek union template search 가 한 번의 SLURM job 으로 완주한다.

---

## 1. Overview

```mermaid
flowchart TB
    INPUT["Unified YAML\n(protein FASTA + ligand SMILES)"]

    MM["1a. mmseqs2\nrcsb_seqDB (488k seqs)\n→ mmseqs_hits.tsv"]

    subgraph S2["2. Co-folding (5 seeds × 5 samples = 25/model)"]
        direction LR
        B2["Boltz-2"] --- B2X["Boltz-2x"] --- PTX["Protenix"] --- AF3["AF3"]
    end

    FS["1b. foldseek\nrcsb_structDB (251k structs)\nquery = best cofold cif\n→ foldseek_hits.tsv"]

    subgraph SB["Template bridges (auto)"]
        direction LR
        F1["Union filter\n(pdb_id, chain_id)\nNO MCS gate"] --> F2["Pocket extract\n(USalign per hit)"] --> F3["Top-K consensus\ntemplate_consensus_1..10"]
    end

    AL["2.5 Frame alignment\nKabsch CA → *_aligned.cif"]
    PREP["3. Docking prep\n≤3 cofold + ≤6 predictors + ≤10 consensus"]

    subgraph S5["4. Docking (multi-track)"]
        direction LR
        T1["Track 1\ncofold-based\nVina/ADG × ≤19 × 5\n(PxDock opt-in)"]
        T2["Track 2\ntemplate-box\n(any template)"]
        T3["Track 3\nlig-align\n(MCS ≥ 0.5)"]
    end

    ION["5. Ion placement\n(conditional)"]
    POST["6. Post-analysis\nBA-Pred + RMSD-Pred"]
    LG["7. CASP17 LG submission\nMODEL 1..5"]

    INPUT --> MM
    INPUT --> S2
    S2 -->|"best cofold cif"| FS
    MM --> SB
    FS --> SB
    S2 --> AL
    SB --> PREP
    AL --> PREP
    PREP --> S5
    S5 --> ION
    ION --> POST
    POST --> LG

    style MM fill:#4fc3f7,color:#000
    style FS fill:#4fc3f7,color:#000
    style F3 fill:#ffd54f,color:#000
    style LG fill:#66bb6a,color:#000
```

> 위 다이어그램은 main trunk (실행 순서) 만 표시합니다. Track 2/3 은 template bridges (`filtered_hits.tsv`) 와 frame-aligned cofold cif 를 둘 다 읽고, Ion placement 도 마찬가지 (cofold cif + template hits) — 자세한 입력 의존성은 각 stage 본문 참조.

**Wrapper execution order** (`build_wrapper_shell_script` 가 emit):

```
template-search-sequence (mmseqs)
  → co-folding (Boltz-2 → Boltz-2x → Protenix → AF3) with MSA cross-tool reuse
    └── bridge: AF3 unified MSA (jackhmmer + hmmsearch + templates) → Boltz / Protenix / AF3 모두 동일 a3m + cif 공유 (msa_pipeline.enabled=true)
  → template-search-structure (foldseek, query = best cofold cif)
    └── bridge: union filter → pocket extraction → pocket clustering
  → bridge: align cofolding outputs (Kabsch to common frame) → *_aligned.cif
  → bridge: docking prep (≤3 cofold cluster + ≤6 predictor top-K + ≤10 template-consensus = 최대 19 binding-site sources)
  → docking (Track 1: Vina/ADG × ≤19 sources × 5 seeds; PxDock opt-in via `protenix_dock.enabled=true`)
  → multi-track docking (Track 2 + Track 3, conditional)
  → ion placement (conditional)
  → post-analysis (BA-Pred + RMSD-Pred, staged poses)
  → CASP17 LG submission (top-5 diverse MODEL)
```

---

## 1.1 Runtime environments

각 외부 모델은 격리된 venv (또는 micromamba env) 에서 동작한다. SLURM job 이 시작될 때 wrapper 가 자동으로 `module load cuda/12.8` + `LD_LIBRARY_PATH`(CUDA targets + per-venv nvidia + boost) + `.local/bin` PATH 를 세팅한다.

| 도구 | 위치 | Python | 핵심 의존성 | 주요 SLURM 파티션 |
|---|---|---|---|---|
| Boltz-2 / Boltz-2x | `.venvs/boltz` | 3.12 | torch 2.11 + CUDA 13.0 + cuequivariance | `6000ada`, `heavy` |
| Protenix | `.venvs/protenix` | 3.12 | torch 2.7 + CUDA 12.6 + cuequivariance | `6000ada`, `heavy` |
| AlphaFold3 | `.venvs/alphafold3` | 3.12 | JAX + 자체 CUDA `LD_LIBRARY_PATH` runner (`run_alphafold3.sh`) | `6000ada` (sm_86 ~ sm_89) |
| Protenix-Dock + Vina | `.venvs/protenix-dock` | 3.11 micromamba | ambertools / tleap (conda-only) + pxdock | `6000ada` |
| AutoDock-GPU + autogrid4 | `.local/bin/` | — | CUDA 바이너리 + C++ 바이너리 (별도 venv 없음) | `6000ada` |
| SwinSite + BA-Pred + RMSD-Pred | `.venvs/pred` | 3.12 | torch 2.4 + dgl 2.4 + openbabel | `6000ada` (BA/RMSD-Pred 는 sm_90+ 미지원) |
| P2Rank | `.local/bin/prank` | JDK 21 | Java wrapper | CPU |
| MMseqs2 + Foldseek + USalign | `.local/bin/` | — | static binaries | CPU (`cpu_only`, `test`) |

> **하드웨어 호환성**: BA-Pred / RMSD-Pred 의 사전 빌드된 CUDA kernel 은 sm_90 (H100) 이후 카드를 지원하지 않아 `heavy` 파티션 (H100 / Blackwell) 에서 silent 빈 결과를 만든다. 새로 가드 추가 (`run_post_analysis.py`) — sm_90+ 디바이스에서 명시적 거부.

> **Master 노드 제약**: GPU 없음 (147.46.139.205). YAML 파싱 / manifest / validation 만 master 에서, 실제 모델 추론은 SLURM compute 노드에서.

---

## 2. Stages

### Stage 1 — Template search

서열 (mmseqs2) 과 구조 (foldseek) 두 검색을 **둘 다 돌리고** `(pdb_id, chain_id)` 키로 union-merge.
가능한 한 많은 template 을 모은 다음, 각 template 의 bound-ligand 위치를 cofold frame 으로 align 해서
**consensus pocket** 을 만든다. **Tanimoto/MCS 는 metadata only** — gating 은 Track 3 (lig-align) 에서만.

#### 1-1. mmseqs2 sequence search

| | |
|---|---|
| Tool | `mmseqs easy-search` against `data/search_dbs/sequence/rcsb_seqDB` (488k seqs, preindexed) |
| Params | `min_seq_id=0.3`, `min_cov=0.7`, `sens=7.5`, `max_hits=500` |
| Output | `outputs/template_search_sequence/mmseqs_hits.tsv` (14-col) |
| Time | ~3 s |

#### 1-2. foldseek structure search (query = best cofold cif)

| | |
|---|---|
| Tool | `foldseek easy-search` against `data/search_dbs/structure/rcsb_structDB` |
| Query | `query_from_cofolding=true`; priority `[alphafold3, boltz2x, boltz2, protenix]` 로 첫 cif 자동 선택. Stage-dependency validator 가 cofold 이전 실행을 차단 |
| Params | `--alignment-type 1` (3Di+AA), `-s 9.5`, `--max-seqs 2000`, format `query,target,evalue,bits,alntmscore,qtmscore,ttmscore,prob` |
| Output | `outputs/template_search_structure/foldseek_hits.tsv` |
| Time | ~30 s ~ 2 min/타겟 |

> Foldseek 의 `qtmscore` 는 **estimate** 라 pre-filter 의미가 약함 → ordering 힌트로만 사용. 진짜 quality gate 는 다음 단계 (pocket extraction) 에서 USalign 이 결정. `qtmscore_min=0.0` (default, disabled).

#### 1-3. Union filter (mmseqs ∪ foldseek)

- **Script**: `scripts/run_template_filter.py`, module `src/casp17/template_filter.py`
- **Logic**: 두 TSV 를 source 태깅 → `(pdb_id, chain_id)` dedup (한 row 가 양쪽 다 있으면 두 metric 모두 보존, `in_mmseqs=in_foldseek=1`) → `rcsb_index.db` 에서 candidate ligand (`ligand_type ∈ {small_molecule, cofactor, metabolite, nucleotide_like, peptide_like}`) 조회 → **Tanimoto (Morgan FP r=2 / 2048 bit) 만 metadata 로 저장** — gating 안 함. MCS 는 filter 에서 계산하지 않고 (`best_mcs_coverage = 0.0` placeholder), 실제 사용처인 Track 3 (lig-align) 가 picked template 한정으로 lazy 계산.
- **Sort key**: `(in_mmseqs+in_foldseek 합 ↓, qtmscore ↓, pident ↓, n_ligands ↓, tanimoto ↓, mcs ↓)`
- **Output**: `filtered_hits.tsv` — 14 컬럼 + `in_mmseqs / in_foldseek / qtmscore / ttmscore / alntmscore / prob` 6 컬럼 append
- **Time-split**: `template_search_sequence.max_deposition_date: "YYYY-MM-DD"` — held-out 벤치 (e.g. novel2025) 에서 leakage 차단

**성능 구조**:

- **RCSB sqlite 커넥션 재사용** — `filter_hits_with_ligands` 가 한 filter run 동안 단 한 번 `sqlite3.connect()` 하고 끝. `lookup_ligands` / `lookup_deposition_date` 둘 다 connection-or-path 오버로드라 호출 측이 커넥션을 넘김. 같은 `pdb_id` 가 여러 chain hit 으로 반복되는 경우를 위해 `ligand_cache` / `dep_cache` dict 도 함수 안에 둠 — pdb 당 sqlite query 1 회.
- **Pre-baked Morgan FP cache** — `data/processed/ligand_fp_cache.db` (~12 MB sqlite, 49k CCDs). `casp17.ligand_fp_cache.LigandFPCache.get_fp(ccd_code)` 가 in-process dict → sqlite → SMILES on-demand 순으로 lookup. Miss 일 때 SMILES 로 계산해 `INSERT OR REPLACE` 하므로 새 CCD 가 들어와도 자동 흡수. Target 쪽 FP 는 `_compute_target_fp(target_smiles)` 가 filter run 시작 시점에 한 번 계산하고 모든 hit 비교에 재사용. RDKit FP 직렬화는 `ExplicitBitVect.ToBinary()` ↔ `ExplicitBitVect(bytes)` 라운드트립 — `CreateFromBinaryText` 는 다른 포맷이라 nBits 가 silent 192 로 깨지므로 사용 금지.
- **FP cache 빌드 스크립트**: `.venv/bin/python scripts/build_ligand_fp_cache.py [--rcsb-db PATH] [--cache-db PATH]`. RCSB `ccd_components` 의 SMILES 모두 순회해 Morgan FP (radius=2, nBits=2048) 한 줄씩 적재 (~18 s). Idempotent — 동일 (ccd_code, fp_radius, fp_nbits) 키는 skip, 새/변경된 SMILES 만 갱신.

#### 1-4. Template pocket extraction

- **Script**: `scripts/extract_template_pockets.py`
- **Alignment 도구**: `casp17.usalign.run_usalign` (`.local/bin/USalign`). 구조 기반 (TM-align) 이라 foldseek 이 hit 을 찾은 view 와 동일. distant homolog 도 정확히 align — gemmi 는 sequence-anchored 라 같은 fold 라도 sequence 멀면 matched residue <50 으로 떨어져 RMSD 가 폭발 → ligand centroid 가 엉뚱한 위치로 transform 됨
- **Chain-specific alignment** (`_extract_chain_pdb`): foldseek/mmseqs hit 은 `(pdb_id, chain_id)` 쌍으로 오고 `chain_id` 가 query fold 에 매칭된 protomer. multi-chain template 의 chain-mapping ambiguity 를 없애기 위해 host chain 의 polymer 원자만 단일-chain PDB 로 추출해 USalign reference 로 사용. PDB 포맷이 1-char chain id 만 허용하므로 추출 시 `"A"` 로 rename (`8qrt_CCC` 같은 multi-char asym id 도 동일하게 처리됨). 반환되는 R/t 는 그 protomer frame 으로 보장. Chain 이 CIF 에 없으면 (synthetic asym id, missing chain 등) whole-CIF alignment 로 fallback.
- **Host-chain ligand 만 보존**: 정렬 후 ligand centroid 도 `lig.chain == host_chain` 인 것만 남김. 다른 protomer 의 ligand 는 다른 R/t 가 필요하므로 별도 hit row 로 들어옴. Host filter 가 모두 비우는 케이스 (single-chain CIF + synthetic asym id) 만 fallback 으로 ligand 전체 통과 — downstream cluster 단계의 surface-margin filter 가 잘못 transform 된 것을 정리.
- **Output**: `outputs/template_pockets/template_pockets.json` (flat list — 한 row = 한 ligand-instance pocket point. homotetramer 는 4 record). 필드: `template_pdb_id, template_chain, ligand_ccd, ligand_chain, ligand_n_heavy, centroid_(x|y|z)` (cofold frame) `, alignment_tmscore, alignment_rmsd, in_mmseqs, in_foldseek, pident, qtmscore, best_tanimoto, best_mcs_coverage` + 최상단에 `reference_cif` (cluster 단계의 surface-margin filter 입력)
- **Quality gate**: `--min-tmscore 0.4` (default). canonical 0.5 보다 약간 낮춰서 foldseek `qtmscore_min=0.5` 통과 hit 을 이중 penalise 하지 않음
- **Default `--max-templates 2000`** — USalign 실측 ~0.5 s/template (300 aa 기준) × 2000 ≈ 17 min/타겟. foldseek `max_hits=2000` 와 매칭. 보통 단백질에서 TM ≥ 0.5 통과 template 50–300 개라 대부분은 게이트에서 reject — pool 확대해도 cluster 결과 안정적

#### 1-5. Pocket clustering (top-K consensus)

- **Script**: `scripts/cluster_template_pockets.py`
- **Surface-margin pre-filter** (`--surface-margin`, default **10 Å**): cluster 직전에 cofold receptor 의 모든 heavy-atom 을 `scipy.spatial.cKDTree` 에 적재하고, 각 pocket centroid 의 nearest-protein-atom 거리를 query. `> margin` 이면 drop. 이는 multi-chain template 에서 USalign 이 한 protomer 만 align 했을 때 다른 protomer 의 ligand centroid 가 50–200 Å 떠 있는 케이스를 잡아냄. KDTree query 라 단순 axis-aligned bbox 가 못 보는 elongated fold 의 dead corner 에서도 작동. `0` 이면 비활성화. Reference CIF 는 `template_pockets.json` 의 최상단 `reference_cif` 필드에서 읽음.
- **Algorithm**: hierarchical agglomerative single-link (`scipy.cluster.hierarchy.linkage(method="single")` + `fcluster(t=cutoff, criterion="distance")`). Pairwise pocket distance matrix 한 번 만들고, 멤버 페어 최단거리가 `cutoff` 미만인 두 cluster 를 모두 merge. **결과 cluster 들 사이의 멤버 페어 거리는 보장된 `> cutoff`** — 같은 binding site 가 cluster 간 redundancy 로 갈리지 않음.
- **Cutoff**: `--cutoff` default **5.0 Å** (druglike pocket 직경 ~10–15 Å)
- **Per-pocket weight** (`_pocket_weight`): `sources + struct_sim + lig_sim` — 범위 ≈ [0, 4].
  - `sources = in_mmseqs + in_foldseek` (0 ~ 2)
  - `struct_sim = max(alignment_tmscore, qtmscore, pident/100)` (0 ~ 1) — mmseqs-only hit 도 USalign actual TM 으로 평가됨
  - `lig_sim = max(best_tanimoto, best_mcs_coverage)` (0 ~ 1) — query SMILES 와 chemical similarity. Tanimoto 는 fingerprint global 유사도, MCS coverage (filter 가 0 으로 두므로 cluster 단계에선 사실상 Tanimoto 만) 는 scaffold 공유 — `max` 로 작은 fragment-MCS 와 큰 분자 Tanimoto 둘 다 펜로 당겨짐. Lig-sim 부스트의 의미: 동일 fold 라도 ligand 가 query 와 chemically 닮은 cluster 가 진짜 active site 일 가능성이 큼.
- **Cluster centroid**: weighted mean (`Σ w·xyz / Σ w`, w=0 fallback 시 unweighted)
- **Cluster `evidence_score`**: `Σ weight`. 정렬 후 top-K (`--top-k 5` default) 보존
- **Per-cluster metadata**: `n_members, n_unique_pdb, evidence_score, spread_angstrom, in_both_sources, in_mmseqs_only, in_foldseek_only, best_alignment_tmscore, best_qtmscore, best_pident, best_tanimoto, best_mcs_coverage, best_ligand_similarity`
- **Output**: `outputs/template_pockets/template_pocket_clusters.json` — 다음 단계 (`prepare_docking_inputs.py`) 가 읽어 `template_consensus_{1..10}` binding-site source 를 등록

---

### Stage 2 — Co-folding

4 개 모델이 순차 실행. 동일한 unified YAML 입력을 각 모델 포맷으로 변환 후 GPU 추론.
**Multi-seed**: 각 모델을 5 seed × 5 diffusion samples = **25 구조/모델** (총 100 구조).

| Model | venv | 25 structs (single GPU) | 특징 |
|---|---|---:|---|
| Boltz-2 | `.venvs/boltz` | ~10 min | 구조 + confidence + MSA + affinity |
| Boltz-2x | `.venvs/boltz` | ~12 min | + `use_potentials=true` (constraints) |
| Protenix | `.venvs/protenix` | ~8 min | 구조 + confidence |
| AlphaFold3 | `.venvs/alphafold3` | ~6 min | + ranking, native multi-seed (loop 불필요) |

> RTX 6000 Ada 단일 GPU, CASP16 L2001/L2002 기준. `cofolding_seeds=[42,101,202,303,404]`.

#### Input adaptation (`src/casp17/adapters.py`)

- 리간드가 있으면 `properties.affinity` 자동 추가 (Boltz 전용)
- Boltz: `prepare_boltz()` → `[boltz2 (use_potentials=false), boltz2x (use_potentials=true)]` 리스트 반환 → orchestrator 가 `*` 로 unpack
- **AF3 chain id remap**: AF3 schema 가 `^[A-Z]+$` 만 허용. `L2`/`X2` 같은 숫자 포함 id 는 단일 letter 우선 → 미사용 letter (A..Z → AA..ZZ 순) 로 배정. `bondedAtomPairs` 의 chain 참조도 동일 remap. Boltz/Protenix 는 YAML id 그대로
- **AF3 `templates` 필드 처리 (msa_pipeline 인지)**: `msa_pipeline.enabled=true` (default) 이면 wrapper 가 AF3 data pipeline (`run_alphafold.py --run_data_pipeline=true`) 을 먼저 돌려 hmmsearch 까지 실행 → 그 결과가 input JSON 의 `templates` 에 들어와야 함. 따라서 adapter 는 protein block 에 `templates` 키를 **세팅하지 않고 비워둠** (AF3 파서가 missing 키를 `None` 으로 읽음 → `pipeline.py` 에서 `run_template_search=not has_templates` 가 `True` 가 되어 hmmsearch 가 실제로 돈다). Legacy mode (`msa_pipeline.enabled=false`) 에서는 hmmsearch 가 안 도므로 adapter 가 `templates=[]` 로 명시적 빈 리스트를 박아 AF3 가 자체 fetch 시도하지 않게 함

#### MSA 재사용 (cross-seed + cross-model)

- **Cross-seed 캐시**: Boltz 첫 seed 만 MSA 서버 fetch. 이후 seed + Boltz-2x 는 생성된 `boltz_results_*/msa/` 디렉토리를 wrapper 가 `_boltz_msa_cache` 변수로 보관 → 다음 seed 출력 경로로 `cp -r`. MSA 는 서열-결정적 (diffusion seed 와 무관) 이라 한 번의 fetch 가 모든 seed × Boltz-2 + Boltz-2x 커버
- **Boltz → Protenix**: `script_builder` 인라인 heredoc 이 Boltz `uniref.a3m` 을 찾아 Protenix JSON 의 각 `proteinChain` 에 `unpairedMsaPath` 주입 + `pairedMsa=""`. cache miss 시 graceful fallback
- **Boltz → AF3**: `scripts/bridge_boltz_msa_to_af3.py` — MSA CSV → A3M 변환 + AF3 JSON 패치 (`pairedMsa=""`, `templates=[]`)

**Boltz affinity output** (자동, 리간드 있을 때):
```json
{ "affinity_pred_value": 2.62,             // log10(IC50 µM) — lower = stronger
  "affinity_probability_binary": 0.41 }    // binder probability [0-1]
```

---

### Stage 2.5 — Frame alignment (cofolding → common coordinate frame)

각 cofolding 모델은 자체 좌표계에서 출력. 이 bridge 가 모든 CIF 를 단일 reference frame 으로 Kabsch 정렬 →
downstream (docking, post-analysis, submission, reference comparison) 전체가 동일 좌표계에서 동작.

- **Script**: `scripts/align_cofolding_outputs.py`
- **Reference 우선순위**:
  1. **Template CIF** — top-1 template 의 RCSB mmCIF (Track 2/3 와 자연스럽게 동일 프레임)
  2. **Best pLDDT cofold model** — template 없을 때 fallback
- **Algorithm**:
  1. Reference CIF 에서 chain 별 CA + residue name sequence 추출
  2. 각 query CIF 에서 sliding offset 으로 residue name 매칭 (결정학적 번호 차이 허용)
  3. `gemmi.superpose_positions(ref_cas, query_cas)` → rotation + translation
  4. Query 의 **모든 원자** (protein + ligand) 에 동일 rigid transform 적용
  5. `*_aligned.cif` 로 원본 옆에 저장
- **Downstream 참조**: `find_best_cofolding_structure()` (docking prep) 와 `find_best_cofolding_cif()` (submission) 가 `_aligned.cif` 우선

---

### Stage 3 — Docking preparation

Cofolding 출력에서 docking 입력을 자동 생성. `scripts/prepare_docking_inputs.py`.

#### Best model selection

- 각 cofolding 모델의 mean pLDDT 비교 (Boltz npz × 100, Protenix scalar, AF3 `atom_plddts` mean — 모두 [0, 100] 정규화)
- **수정 전 버그**: Boltz [0, 1] 을 AF3 [0, 100] 그대로 비교 → 항상 AF3 픽됨. 정규화 후 pLDDT 가 정상 비교됨

#### Receptor preparation

| Step | Tool | Output | 용도 |
|---|---|---|---|
| CIF → PDB | gemmi (selective: dockable ligand chain + water 만 제거, **metal/cofactor retain**) | `receptor.pdb` | 기본 |
| PDB → PQR | pdb2pqr `--ff=AMBER` | (intermediate) | 수소 + 전하 |
| PQR → protonated PDB | 자체 변환 | `receptor_protonated.pdb` | Protenix-Dock |
| PQR → PDBQT | AD4 atom mapping + metal HETATM 재첨부 | `receptor.pdbqt` | Vina + AutoDock-GPU |

**Per-ligand "ligand-aware" receptors (multi-ligand 타겟)**:

Multi-ligand 타겟 (`len(dockable_chains) > 1`) 에서는 위 apo receptor 외에 ligand 별로 **다른 dockable ligand 의 cofold pose 를 정적 원자로 박은 receptor** 를 추가 생성:

- 각 dockable ligand chain `<id>` 에 대해 `cif_to_pdb(structure, dockable_ligand_chains={id})` — 자기 chain 만 strip 하고 나머지 ligand chain 은 cofold 좌표 그대로 retain → `receptor_excl_<id>.pdb` + `.pdbqt`
- `usable_ligands[i]` 의 `receptor_pdbqt` / `receptor_pdb` 필드에 그 ligand 전용 경로가 들어감
- Vina/ADG 런타임 스크립트가 `lig.get('receptor_pdbqt') or receptor_pdbqt` 로 per-ligand receptor 를 우선 참조 — single-ligand 타겟은 `None` 이라 자동으로 top-level apo receptor 로 fallback
- ADG 측은 추가로 `_parse_rec_types(lig_receptor)` 로 receptor atom type 도 ligand 별로 재파싱 (다른 ligand 가 박힌 receptor 라 새 AD4 type 이 추가될 수 있음)
- 효과: ligand i 를 docking 할 때 ligand j 의 cofold pose 가 binding site 의 일부로 보여서 두 ligand 가 같은 pocket 으로 몰려가는 site competition error 가 자연 제거됨
- Per-ligand receptor 빌드 실패 시 (gemmi/pdb2pqr edge case) 해당 ligand 만 apo receptor 로 fallback, 다른 ligand 는 그대로 진행

**RNA/DNA receptor 자동 분기** (commit `efb9255`+):

`pdb_to_pdbqt` 가 receptor PDB 의 nucleotide residue (A/U/G/C, DA/DT/DG/DC, RA/RU/RG/RC, T, DI/I) 감지 → `pdb2pqr` 우회하고 **`obabel -p 7.4 --partialcharge gasteiger -xr`** 단일 단계로 PDBQT 생성. 이유: pdb2pqr 의 AMBER FF 가 nucleotide parameterize 못 함 → 기존 path 면 RNA/DNA chain 통째로 silent drop.

- obabel binary: `.venvs/pred/bin/obabel` (이미 설치)
- Protein-only receptor 는 default `pdb2pqr` path 그대로 — 회귀 없음
- Verified atom types on synthetic DNA: `{P, OA, NA, HD, C}` (모두 표준 AD4 type)
- Protein + RNA/DNA hetero-multimer receptor 도 nucleotide 한 residue 만 감지되면 obabel 분기

**Metal/cofactor retain** (commit `c65797b`+):

이전엔 `gemmi.remove_ligands_and_waters()` 가 receptor 의 모든 non-polymer 를 strip → metalloprotein 의 active-site coordination 좌표 손실. Fix:

- `cif_to_pdb(dockable_ligand_chains={"L", ...})` — input YAML 의 SMILES-bearing ligand chain id 만 추출해 그 chain 의 non-polymer 만 제거. Metal/cofactor 는 receptor 에 retain
- `pdb_to_pdbqt` 가 pdb2pqr 후에 metal HETATM 을 manual 재첨부 (`+2.000 Mg` 같은 AD4 atom type + formal charge — pdb2pqr/AMBER 가 bare metal parameterize 못함)
- 지원: Mg²⁺, Zn²⁺, Ca²⁺, Fe³⁺, Mn²⁺, Cu²⁺, Ni²⁺, Co²⁺, Cd²⁺, Hg²⁺, Ba²⁺, Sr²⁺, Al³⁺, K⁺, Na⁺, Cl⁻
- Verified on 21ii_input (Mg²⁺ pyrophosphatase): `receptor.pdb` HETATM `0 → 1`, PDBQT 에 `+2.000 Mg` 라인
- Track 2 receptor (RCSB template) 도 동일 logic 적용 (organic ligand ≥6 heavy atoms 만 strip)
- 한계: AutoDock-GPU runtime GPF 는 ligand atom type 만 보고 grid map 생성. Receptor metal 은 elec/dsolv map 에 contribute (대부분 metal-coordinating docking 의 dominant signal). Ligand 자체에 metal 들어있는 케이스는 unhandled

#### Ligand preparation

- SMILES → 3D conformer: RDKit `EmbedMolecule(ETKDGv3)` + `MMFFOptimizeMolecule(maxIters=500)`
- SDF → PDBQT: meeko `MoleculePreparation`

#### Binding-site sources

Track 1 docking 의 box center 후보를 **3 카테고리 = 최대 19 개** 로 fan-out:
- **A. Cofold clusters** (`cofolding_1/2/3`) — 조건부 1 ~ 3 개
- **B. Pocket predictors top-K** (`swinsite_1/2/3`, `p2rank_1/2/3`) — 조건부 0 ~ 6 개
- **C. Template consensus** (`template_consensus_1..10`) — 조건부 0 ~ 10 개

모든 source 의 metadata 에는 `nearest_protein_chain` (가까운 protein chain id) 와 `nearest_ca_distance` (Å) 가 부착됨 — multi-chain receptor 에서 chain A 의 active site vs chain B 의 동등 site vs A-B interface 를 사후 분석에서 구분 가능.

각 source 는 **독립된 docking variant** 로 실행되어 한 source 가 잘못된 pocket 을 잡아도 다른 source 가 backup.
어떤 source 가 winner 인지는 사후 ranker (`compute_submission_scores.py`) 가 판정.

**A. Cofold clusters — 조건부 (1 ~ 3 개)**

`cofolding_1`, `cofolding_2`, `cofolding_3` 은 **모든 4 모델 × 25 seeds = 100 placements** 의 heavy-atom ligand centroid 를 5 Å greedy first-match clustering (running mean centroid) 한 top-K 결과.
**Stage 2.5 frame alignment 가 100 cif 를 단일 reference frame 으로 align 한 다음**에 클러스터링하므로 좌표 비교가 의미를 가짐 (`align_cofolding_outputs.py::main` 이 `rglob("*.cif")` 으로 4 모델 × per-seed 모든 cif 를 align).

등록 조건 (`prepare_docking_inputs.py::_extract_cofolding_ligand_clusters`):
- **Dockable ligand chain 만 평균** — input YAML 에서 SMILES-bearing ligand chain id (`L`, `L2` 등) 만 추려 그 chain 의 heavy atom centroid 로 placement 계산. metal/cofactor 같은 ccd-only entry 는 centroid 오염원이라 제외
- **Dynamic min_members**: `max(2, n_placements // 20)` (= 5 % of placements). default 100 placement 면 5 ; seed × sample 변경 시 자동 scale. 함수가 `(clusters, diagnostics={"n_placements", "min_members"})` 를 반환해 `source_status` 에 기록됨
- **`cutoff = 5.0 Å`** (= `COFOLD_CLUSTER_CUTOFF`) — template-pocket cluster 와 동일
- **최대 3 개** (= `COFOLD_CLUSTER_TOP_K`) — `n_members` desc
- 실제 등록 수:
  - 잘 수렴된 타겟 (단일 binding pocket) → 1 cluster (`n_members ≈ 100`) → `cofolding_1` 만
  - Multi-pocket / inter-model 불일치 → 2-3 cluster → `cofolding_1/2/3`

각 entry 의 metadata: `n_members`, `n_unique_models`, `models` (어떤 모델들이 기여했는지), `cluster_rank`.

**B. Pocket predictors top-K — 조건부 (0 ~ 6 개)**

각 predictor 마다 score 내림차순 top-3 (`POCKET_PREDICTOR_TOP_K = 3`) 을 등록.
Multi-chain receptor 에서 chain B/C/D 의 동등 active site 가 자연스럽게 rank 2/3 으로 잡힘.

| Source | 기원 | Skip 조건 |
|---|---|---|
| `swinsite_<rank>` | Swin-Unet ML pocket predictor (GPU, `.venvs/pred`) | predictor 가 K 미만 pocket 만 찾으면 잔여 rank 의 variant 만 `sys.exit(0)` |
| `p2rank_<rank>` | Surface-based geometric (JDK 21) | 위와 동일 |

각 entry 의 metadata: `score` (predictor 자체 confidence), `rank` (1=best), `nearest_protein_chain`, `nearest_ca_distance`.

**C. Template-consensus sources — 조건부 (0 ~ 10 개)**

Stage 1-5 에서 만든 `template_pocket_clusters.json` 의 top-K cluster centroid 를 `template_consensus_1..10`
이름으로 등록. 등록 조건 (`prepare_docking_inputs.py::_add_template_consensus_sources`):

- **`n_members ≥ 2`** (= `TEMPLATE_CONSENSUS_MIN_MEMBERS`) — singleton cluster 는 alternate/spurious site 일 가능성이 커서 제외
- **최대 10 개** (= `TEMPLATE_CONSENSUS_TOP_K`) — `evidence_score` desc
- 실제 등록 수는 타겟별로 다름:
  - Template hit 없거나 모든 cluster 가 singleton → 0 개
  - Template hit 풍부 (e.g. 잘 알려진 fold) → 5–10 개

**Variant fan-out (`src/casp17/adapters.py::_DOCKING_BOX_SOURCES`)**

`prepare_vina` / `prepare_autodock_gpu` 가 위 15-source 튜플을 순회하며 각각 `PreparedModelRun` 을 emit.
각 variant 는 생성 시점에 `BOX_SOURCE='cofolding_1'|...|'template_consensus_10'` 이 runner script 에 baked-in →
런타임에 `summary['binding_site_predictions'][BOX_SOURCE]['center']` 를 읽어 box 설정.
**해당 source 가 missing 이면 `sys.exit(0)` clean skip** (실패가 아니라 정상 종료) — 그래서 자료 부족한 타겟이라도 파이프라인이 멈추지 않음.

**기타**:
- **Adaptive box size** (commit `5fb2039`+) — `_adaptive_box_size()` 가 ligand SDF 들의 heavy-atom XYZ extent 를 계산해 `max(extent) + 8 Å` padding 적용. Floor 22.5 Å (legacy default — 일반 druglike 에선 변화 없음), cap 40 Å. Macrocyclic / peptide ligand 는 자동으로 더 큰 box, ion-only 처럼 작은 ligand 는 22.5 floor 유지. Grid spacing 0.375 Å 고정 (Vina/ADG/PxDock 공통)
- **Fallback box pick** — PxDock 처럼 단일-box 만 받는 도구용 priority: `cofolding_1 > cofolding_2 > cofolding_3 > template_consensus_N (n_unique_pdb ≥ 2 인 strong consensus) > swinsite_1/2/3 > p2rank_1/2/3 > weak consensus`
- **Output**: `inputs/docking/docking_prep_summary.json`. 핵심 필드:
  - `binding_site_predictions` — key = source name, value = `{center, size, metadata}`
  - **`source_status`** (commit `5fb2039`+) — observability 용 진단 블록. 어느 카테고리의 source 가 등록 0 이었는지 (silent skip 추적), `n_cofold_placements_total`, `cofold_min_members_floor`, `n_dropped_ligands_at_prep`, `dockable_chains_from_yaml` 모두 한 곳. RNA target 처럼 swinsite/p2rank 가 비어 있는 케이스 즉시 파악 가능

> **Design rationale**:
> 1. **Cofold clusters** — co-fold 25 seeds × 4 models 가 단일 frame 에서 어디에 ligand 를 놓는지가 가장 직접적 신호. 단일 best cifold centroid 만 쓰던 옛 design 은 multi-pocket / inter-model 불일치를 잡지 못함. `cluster_template_pockets.py` 와 동일 greedy first-match centroid 알고리즘을 재사용해 0 cost 에 가까운 확장.
> 2. **Template-consensus 가 backup** — 한 cofold cluster 가 잘못된 pocket 을 잡아도 (e.g. CASP16 L2001: 100 placements 가 모두 35 Å off — 같은 fold 의 모든 모델이 동일하게 빗나감) RCSB 의 실험적으로 검증된 binding pose 좌표가 독립 backup 으로 제공됨.

---

### Stage 4 — Docking (multi-track)

3 개 트랙. Track 1 항상 실행, Track 2 는 template hit 있으면, Track 3 은 MCS ≥ 0.5 일 때.

#### Track 1 — Cofolding-based docking (항상)

Cofolding best model 을 receptor, **위에서 등록된 binding-site source (≤3 cofold cluster + ≤6 predictor (3 swinsite + 3 p2rank) + ≤10 consensus = 최대 19) 각각을 독립 box 로** 사용.
Vina + AutoDock-GPU 가 **최대 19 × 2 = 38 variant** 로 병렬 실행 (실제 수 = 등록된 source 수). 각 variant 는 `docking_seeds` (default 5) 만큼 반복 → 최대 190 run/target.
PxDock 은 **default 비활성화** (`protenix_dock.enabled=false` since commit `d04e2c9`+) — single run 이지만 ~5–30 min/타겟 소요로 docking wall-clock 의 ~77 % 차지함. novel2025 batch 의 native rate 기여도 ~10 % vs cost ~77 % 라 ROI 낮음. 명시적으로 enable 한 target 에서만 실행 (cache-map 재생성 비용 때문에 fan-out 없이 priority-picked single box).

| Tool | Variants | Type | Time/seed | Output |
|---|---|---|---:|---|
| Vina | `vina_<source>` × ≤19 | Python API (CPU) | ~3 s | `outputs/vina_<source>/seed_<seed>/docked.pdbqt` |
| AutoDock-GPU | `autodock-gpu_<source>` × ≤19 | CUDA binary | ~10 s | `outputs/autodock_gpu_<source>/seed_<seed>/docked.dlg` |
| Protenix-Dock | (single, no fan-out) | CPU force field | ~5–30 min | `outputs/protenix_dock/poses_*.sdf + *_out.json` |

- 모든 tool 은 `docking_prep_summary.json` 에서 receptor / ligand / box 를 runtime 에 읽음
- AutoDock-GPU 래퍼는 추가로 **런타임에 ligand pdbqt + receptor pdbqt 둘 다 파싱**해서 `ligand_types` + `receptor_types` + grid map 동적 구성. F/Cl/Br/P/I/Si 등 비표준 ligand atom 자동 대응 + RNA/DNA receptor 의 phosphate (P) + retain 된 metal (Mg/Zn/...) 가 receptor_types 에 자동 포함되어 autogrid4 의 `WARNING: receptor type X not in list` silent drop 방지
- AutoDock-GPU 하드 캡: `nrun=100`, **`nev=1500000`**, **`--ngen=27000`** (max LGA generations), `--heuristics=1`, `--autostop=1`. autostop 이 막혀도 nev/ngen 이 worst-case wall-time 를 한정. config 키: `autodock_gpu.nev`, `autodock_gpu.ngen`
- 멀티-리간드: 위 wrapper 는 `usable_ligands` 를 순회하며 `lig.get('receptor_pdbqt') or receptor_pdbqt` 로 per-ligand "ligand-aware" receptor 를 우선 사용 (Stage 3 참조)
- PxDock 활성화 시 docking 시간의 ~77 % 차지 → default 비활성화
- **Config**: `docking_seeds=[42,101,202,303,404]`. variant 자동

#### Track 2 — Template-based box docking (any template, no MCS gate)

Template search 가 찾은 hit (candidate ligand 가 하나라도 있는 모든 PDB) 의 실험 구조를 receptor,
template 리간드 위치를 docking box 로. **Template ligand 가 query 와 닮을 필요 없음** — pocket geometry 만 빌리니 어떤 ligand 든 OK.

- **Script**: `scripts/prepare_template_docking.py` + `scripts/run_multi_track_docking.py`
- **Logic**:
  1. **Cluster-aware template selection**: `template_pockets.json` + `template_pocket_clusters.json` 의 각 cluster 마다 evidence-best representative template 1 개 픽 (`select_cluster_representative_templates`, `--max-templates 10` 까지). Track 1 `vina_template_consensus_1..10` 과 **같은 cluster set 을 cover** 하되 receptor 만 *experimental template* 으로 교체. pockets json 없을 때만 sort top-N fallback.
  2. **PDB-id dedup** (`run_multi_track_docking.py`): cluster representative 가 같은 PDB 를 두 cluster 에서 동시에 가리키는 경우 (인접 pocket 이 같은 구조로 cover 되는 케이스) 두 번째부터는 skip. `multi_track_summary.json` 에 중복 row 가 박히는 것을 방지. dedup 후 남은 unique template 만 Track 2/3 진행
  3. RCSB CIF → receptor PDB/PDBQT (gemmi + pdb2pqr, **metal/cofactor retain** — Stage 3 와 동일 logic. Strip 규칙은 CCD-based — `ligand_codes` 의 CCD 만 strip, HEM/NAD/FAD 같은 cofactor 는 retain. RNA/DNA template 은 obabel 분기)
  4. **pdb2pqr 실패 시 obabel fallback**: pdb2pqr 가 nonstandard residue / missing atom / unusual altloc 으로 죽는 template (4v8w, 6gjc 류) 에서 `obabel -p 7.4 --partialcharge gasteiger -xr` 로 PDBQT 직접 생성. obabel 까지 실패하면 `RuntimeError` raise → caller (`prepare_template_docking.main`) 가 그 template 만 skip (`Skipping Track 2 docking for template <pdb>`). 거대 biological assembly (예: 5BP4 의 ~260 k 원자) 처럼 어떤 도구도 못 처리하는 케이스에 silent broken PDBQT 가 만들어지지 않도록 명시적으로 raise 하는 구조
  5. **USalign template → cofold frame transform** 계산 → template CIF 의 모든 atom 에 적용 → receptor 와 box 좌표가 cofold receptor 와 같은 좌표계
  6. Template 리간드 bound-pose SDF 추출 (`extract_template_ligand_sdf()`)
  7. Target SMILES → SDF/PDBQT (RDKit + meeko)
  8. **Multi-ligand expansion** — 각 dockable target ligand (input YAML 의 SMILES-bearing entry) 별로 dock 실행. ccd-only entry (metal/ion 등) 는 prep 단계에서 skip. Output 명명: `outputs/template_docking/<pdb_id>/<tool>/ligand_<lig_id>/...`. Vina + ADG 실행 (PxDock 은 default 비활성화)
- **MCS 게이트 없음**: `check_template_hits` 는 `num_ligands > 0` 만 검사
- **Frame**: docked pose 는 USalign template→cofold transform 으로 cofold receptor 좌표계에 들어옴 → BA-Pred 입력 / 최종 MODEL block 모두 동일 frame. Track 1 의 `vina_template_consensus_*` 와 box 좌표 매칭

> **Track 1 vina_template_consensus_N ↔ Track 2 cluster N representative 의 차이**: box 는 동일 (cluster N centroid 근처), receptor 만 다름. Track 1 = cofold (predicted) receptor, Track 2 = experimental template receptor (USalign-transformed to cofold frame). Cofold protein 이 정확하면 Track 1 의 cofold receptor 가 valid, cofold 가 wrong fold 면 Track 2 의 experimental conformation 이 backup. novel2025 batch 에서 Track 2 family 의 native rate 가 낮아 (~2.2 %) ROI 회의적이지만 multi-chain receptor 에서 cluster-aware 가 효과 있을지 측정 대기.

#### Track 3 — lig-align (MCS ≥ 0.5)

Template 리간드의 결합 포즈를 MCS anchor 로 활용해 target 리간드의 3D 포즈를 직접 생성.

- **Library**: `lig_align.run_pipeline()` — **hub `.venv` 에 설치**. (이전엔 `protenix-dock` venv 에서 import 시도 → silent ImportError 로 모든 lig_align 출력이 비어 있었음. 현재는 `subprocess` 로 hub venv 의 python 호출)
- **Production params**: `num_confs=500`, `mcs_mode="auto"`, `optimize=False`, `weight_preset="vina"`, `top_k=5` — template 당 ~90 s, 3 templates / target 안에서 12 h wrapper cap 잘 들어감 (이전 `num_confs=1000, optimize=True, top_k=10` 은 90+ min/pocket — 실 prod 에서 한번도 완주 못함)
- **Input**: template receptor PDB + template ligand SDF (bound-pose) + target SMILES
- **Output**: `outputs/template_docking/<pdb_id>/lig_align/` (ranked SDF poses)

#### Multi-track decision logic

```mermaid
flowchart TB
    F["filtered_hits.tsv\n(union, no MCS gate)"]
    F --> ANY{"any hit\nnum_ligands > 0?"}
    ANY -->|No| SKIP["Track 2+3 skip\n(Track 1 only)"]
    ANY -->|Yes| T2_PREP["Track 2 prep\n(cluster-aware,\n1 representative\nper cluster, ≤10)"]
    T2_PREP --> T2["Track 2: vina/adg/pxdock\n(template box)"]
    T2_PREP --> MCS{"per-template\nbest_mcs_coverage\n>= mcs_threshold?\n(default 0.5)"}
    MCS -->|Yes| T3["Track 3: lig-align"]
    MCS -->|No| T3SKIP["Track 3 skip\nfor this template"]
    T2 & T3 --> SUM["multi_track_summary.json"]

    style MCS fill:#ffd54f,color:#000
```

| Track | Receptor | Box source | Method | Entry |
|---|---|---|---|---|
| 1 | cofold best (`_aligned`) | ≤19 (≤3 cofold cluster + ≤6 predictor top-K + ≤10 consensus) | Vina + ADG (+ PxDock opt-in) | always |
| 2 | template PDB (RCSB, USalign→cofold frame, **cluster-aware top-1 per cluster**) | template ligand centroid | Vina + ADG (+ PxDock opt-in) | `num_ligands > 0` |
| 3 | template PDB (same as Track 2) | MCS anchor alignment | lig-align | per-template `best_mcs_coverage ≥ mcs_threshold` |

---

### Stage 5 — Ion / metal placement (conditional, post-hoc)

Input YAML 에 ion CCD 엔티티 (ZN/MG/CA/FE 등) 가 있으면 자동 실행. **Stage 3 의 metal retain (cofold 가 놓은 metal 좌표를 docking receptor 에 보존) 과는 별개의 post-hoc 분석** — template alignment 기반으로 *대안* 위치 후보 cluster.

- **Script**: `scripts/collect_template_ions.py`
- **Logic**: target ion 보유 template (rcsb_index.db) → gemmi CA superposition (template → cofold) → rotation/translation 을 ion 좌표에 적용 → distance clustering (default 2.0 Å) → confidence 그룹별 (high `pident≥70%`, medium 50–70 %, low 30–50 %) 리포트
- **Reference cofold CIF**: `find_best_cofolding_structure` 가 우선 `*_aligned.cif` (Stage 2.5 출력) 을 픽 → 모든 ion 좌표가 docking pose / 최종 MODEL 과 동일 frame 으로 들어감. Aligned CIF 없을 때만 unaligned cofold 출력으로 fallback (안전망)
- **Output**: `outputs/ion_placement/ion_placement_summary.json`
- **Auto skip**: input 에 ion 없으면 미실행
- **Receptor 와의 관계**: 이 stage 는 docking 결과에 직접 inject 안 됨. Cofold 가 metal 위치 잘 잡았으면 Stage 3 의 metal retain 으로 docking 이 metal coordinator 인식. Cofold metal 위치가 의심스러울 땐 이 ion placement 결과로 alternate 좌표 검토 가능

---

### Stage 6 — Post-analysis

모든 docking + cofold 포즈에 대해 binding affinity 와 pose RMSD 를 GNN 으로 예측.

- **Script**: `scripts/run_post_analysis.py` (GPU node)
- **venv**: `.venvs/pred` (torch 2.4 + dgl 2.4 + openbabel + rdkit) — **sm_90/sm_100 (H100, Blackwell 6000pro) kernel 미포함** → `heavy` partition 절대 금지 (`no kernel image` 즉시 종료, task 는 exit 0 처럼 보이지만 TSV 비어있음). `test` / `6000ada` 만 사용
- **Variant-aware 자동 발견**: `find_ligand_files` 가 `outputs/vina_*/seed_*` / `outputs/autodock_gpu_*/seed_*` 를 glob 으로 스캔 → 각 source 별로 독립 tool key 로 staging. TSV 도 `ba_pred_vina_cofolding.tsv` 식으로 분리

#### Pose staging

| 입력 | 처리 | Output stem |
|---|---|---|
| `outputs/vina_*/seed_*/docked.pdbqt` | meeko `mk_export.py` 로 SDF 복제 | `analysis/poses/{tool}_seed_{N}.{pdbqt,sdf}` |
| `outputs/autodock_gpu_*/seed_*/docked.dlg` | meeko `mk_export.py` 로 SDF 복제 | `analysis/poses/{tool}_seed_{N}.{dlg,sdf}` |
| `outputs/protenix_dock/*_out.json` | `_pxdock_json_to_sdf` (RDKit multi-record) — **`|coord| > 200 Å` outlier drop** (PxDock numerical blow-up 방어, novel2025 의 ~1.85 % poses 가 1e7 Å 까지 폭발) | `outputs/protenix_dock/poses_<lid>.sdf` |
| `outputs/{boltz2,boltz2x,protenix,af3}/**/*_aligned.cif` | 리간드 (chain `L` 또는 `LIG*/UNK/UNL` residue) 추출 → 입력 docking ligand SDF 를 템플릿으로 `AssignBondOrdersFromTemplate` (CIF 의 결합차수 정보 소실 보강) | `analysis/poses/cofold_{model}.sdf` |

> **입력 ligand SDF (`inputs/docking/ligand_*.sdf`) 는 post-analysis 제외**. RDKit embedding 좌표가 receptor 와 정렬 안 됨 → BA-Pred 의 "8 Å 이내 단백질 원자 추출" 로직이 빈 mol 반환 → `mol_to_graph` 단계에서 `AttributeError`. Docking + cofold aligned 포즈만 대상.

**Track 2/3 multi-ligand pose staging** (commit `5fb2039`+):

Post-analysis 가 두 layout 모두 인식:
- 새 (multi-ligand): `outputs/template_docking/<pdb_id>/<tool>/ligand_<lig_id>/{docked.pdbqt|docked.dlg|*.sdf}` → `template_<pdb_id>_<tool>_<lig_id>` tool key
- 옛 (single-ligand legacy): `outputs/template_docking/<pdb_id>/<tool>/{docked.pdbqt|...}` → `template_<pdb_id>_<tool>_<primary_lig>` tool key (fallback)

Per-template USalign(template_receptor → cofold_receptor) transform 을 staged SDF 에 적용해 BA-Pred / RMSD-Pred / 최종 LG MODEL 모두 cofold receptor 좌표계.

**PxDock pose path** (commit `28bd99f`+): `outputs/protenix_dock/poses_<lig_id>.sdf` 로 통일. 이전엔 multi-ligand 의 경우 `protenix_dock/ligand_<id>/poses.sdf` 와 `compute_submission_scores._resolve_pose_file` 의 `poses_<id>.sdf` 가 mismatch → PxDock pose 가 LG diversity check 에서 silent drop 되던 문제 해결.

#### Naming normalization

BA-Pred 와 RMSD-Pred 는 SDF `_Name` 처리 규칙이 다름 (BA 는 raw, RMSD 는 record idx 한 번 더 append).
`compute_submission_scores.aggregate::_canonicalize` 가 pose name 을 part 단위 walk down 하면서 staged 파일 존재하는 최장 prefix 찾아 단일 record-index 형태로 축소 → BA/RMSD TSV 가 canonical key 에서 1-1 join.

#### Output

- `outputs/analysis/poses/` — staged pose 파일 (pdbqt/dlg + sdf 쌍)
- `outputs/analysis/{vina,autodock_gpu}_poses.txt` — per-tool 입력 리스트
- `outputs/analysis/ba_pred_<tool>.tsv` — per-pose pKd
- `outputs/analysis/rmsd_pred_<tool>.tsv` — per-pose pRMSD, P(>2 Å)
- `outputs/analysis/summary.json` — `ba_pred` / `rmsd_pred` 성공 source map + **`ba_pred_failed` / `rmsd_pred_failed` 명시적 source 리스트** (long run 에서 silent skip 안 잃기 위한 second line of defense)

`run_post_analysis.py` 가 BA-Pred / RMSD-Pred 마무리 직후 `=== Post-analysis tally: BA-Pred N/M ok, RMSD-Pred N/M ok ===` 라인을 찍고, 실패 source 가 있으면 sorted 리스트 출력. 전형적 실패 모드: docking source 의 box centroid 가 protein 바깥에 떨어져 ligand 가 100 Å 이상 떠 → BA-Pred 의 `mol_to_graph` 가 `None` 반환 → `AttributeError`. Cluster surface-margin filter 가 1 차로 막지만 두 번째 방어선으로 여기에서 명시적으로 로깅.

---

### Stage 7 — CASP17 LG submission

#### 7-1. Score aggregation

- **Script**: `scripts/compute_submission_scores.py`
- **LSCORE** (per pose): `1 - P(RMSD > 2 Å)` (RMSD-Pred 직접 출력. 높을수록 native 에 가까울 것으로 예측됨)
- **AFFNTY** (per complex): log-space ensemble of BA-Pred median + Boltz median (filtered by `binder_prob ≥ 0.5`). `10^(avg log10(Kd nM))` 로 변환

#### 7-2. Top-5 diversity-aware pose selection (`select_diverse_top_k`)

CASP LG 포맷은 MODEL 1..5 허용. 단순 top-5 는 매우 유사한 포즈가 반복 → **greedy diversity selection**:

1. **Sort by lscore desc** (lscore 없으면 BA-Pred pKd desc fallback)
2. Top-1 무조건 선택
3. 나머지 walk: 이미 선택된 포즈와의 heavy-atom RMSD ≥ `--diversity-rmsd 2.0` (default) 이면 선택, 아니면 skip
4. k=5 채우거나 후보 소진 시 종료

> **lscore-primary 로 변경된 이유**: novel2025 489-target 풀 ablation (RRF + consensus, cluster-then-pick, cascaded filter 등 다양한 ranker 비교) 에서 단순 lscore 가 가장 좋은 top-1/top-5 SR. pRMSD 는 회귀 값이지만 per-tool family 별 calibration 다름 → cross-family 비교 시 lscore (정규화된 확률) 가 더 안정적. 자세한 ablation 결과는 `docs/pose_ranker_design.md`. `select_best_pose_pRMSD_legacy` 는 호출 가능하게 보존
- **Heavy-atom RMSD**: numpy 직접 계산 (same SMILES different conformer 라 atom ordering 일관 → 2 Å 임계값 대비 sub-Å 노이즈는 무시 가능. RDKit `CalcRMS` 보다 10× 빠름)
- **MDL title**: `pose_to_mdl(file, idx, title=pose.pose_name)` → `mol.SetProp("_Name", title)` 후 `MolToMolBlock`. RDKit 기본 `"     RDKit          3D"` 가 MDL 첫 줄 자리 차지하던 옛 동작 제거 → LG 파일에 `vina_p2rank_seed_202_5` / `cofold_protenix_17` 식으로 source 그대로 기록 (evaluator 호환)
- **Pose pool**: `vina_<≤19src>`, `autodock_gpu_<≤19src>`, `protenix_dock`, `template`, `lig_align`, `cofold_{boltz2,boltz2x,protenix,af3}` — 타겟당 200–500 pose 정도

#### 7-3. LG format assembly

- **Script**: `scripts/make_casp_submission.py`
- **Spec**: `docs/casp17_lg_format.md`
- **Multi-MODEL 규칙**: 각 MODEL 은 한 complete snapshot — receptor + 모든 ligand 의 LIGAND block + per-ligand LSCORE + MDL body + per-MODEL AFFNTY. **multi-ligand 타겟이라도 한 MODEL 에 모든 ligand 가 다 들어감**. MODEL 1 = primary (best aggregate LSCORE)
- **Diversity**: 각 ligand 별로 5 MODEL 의 포즈가 heavy-atom RMSD ≥ 2 Å (per-ligand)
- **Cofold-ligand fallback**: 어떤 ligand 가 docking 에서 0 pose 면 `extract_cofolded_ligand_mdl` 이 cofold CIF 에서 직접 MDL 추출 → 빈 LIGAND block 방지

```
PFRMAT LG
TARGET <id>
AUTHOR <casp-code>
METHOD <description>
METHOD ----
MODEL 1
REMARK <protein_model + pose_sources>
PARENT <template_pdb_or_N/A>
ATOM ... (receptor, B-factor = pLDDT)
TER
LIGAND 001 <name>
LSCORE 0.994
<MDL V2000 block ending in M  END>
LIGAND 002 <name>          # 같은 MODEL 안의 다른 ligand
LSCORE 0.882
<MDL block>
M  END
AFFNTY 362.000 aa          # per-MODEL, optional
MODEL 2
...
END
```

#### Usage

```bash
python scripts/make_casp_submission.py \
    --run-dir experiments/runs/L2001_input \
    --target-id L2001 \
    --ligand-name 761 \
    --author <casp-code> \
    --method "Boltz-2x + Vina/ADG ensemble" \
    --include-affinity \
    --top-k 5 \
    --diversity-rmsd 2.0 \
    --output experiments/CASP17/submissions/L2001.lg
```

CLI knobs: `--top-k 5`, `--diversity-rmsd 2.0`, `--pose-source auto|vina|autodock_gpu|protenix_dock|template|lig_align`, `--include-affinity`, `--lscore N` / `--affinity-nM N` (manual override, MODEL 1 만).

#### Wrapper config

```yaml
post_analysis:
  enabled: true             # default on
  device: cuda

submission:
  enabled: true             # default off; set true to auto-generate .lg
  author: "XXXX-XXXX-XXXX"
  method: "Boltz-2x + ensemble ..."
  include_affinity: true    # default on (PA tasks like L1000); false for pose-only P tasks (L2000)
  parent: "N/A"
  ligand_number: 1
```

**GPU allocation stamp** — wrapper 시작 직후 자동 출력 (CUDA kernel 호환 디버깅용):

```bash
echo "--- GPU allocation ---"
echo "SLURM_NODELIST=${SLURM_NODELIST}  SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
echo "----------------------"
```

---

## 3. Reference

### 3-1. Bridges (auto-inserted between stages)

| Bridge | 위치 | Script | 역할 |
|---|---|---|---|
| **AF3 unified MSA pipeline** | cofolding 직전 (msa_pipeline.enabled=true 시) | `script_builder.py::_emit_msa_pipeline_bridge` + AF3 `run_alphafold.py --run_data_pipeline=true` | jackhmmer + hmmsearch 한 번 실행, ~30-90 분/타겟 (CPU) |
| **Distribute MSA + templates** | AF3 data pipeline 직후 | `bridge_distribute_msa_templates.py` | unpairedMsa/pairedMsa → `shared_msa/{chain}_{kind}.a3m`; inline `templates[*].mmcif` → `shared_msa/templates/<pdb>.cif`; Boltz YAML / Protenix JSON / AF3 input 셋 모두 동일 a3m + cif 가리키게 패치 |
| Boltz MSA cross-seed *(legacy)* | Boltz seed 1 → 이후 seed + Boltz-2x | `script_builder.py` (인라인) | msa_pipeline OFF 일 때만; seed 1 의 `msa/` 디렉토리 재사용 |
| Boltz MSA → Protenix *(legacy)* | Boltz → Protenix | `script_builder.py` (heredoc) | msa_pipeline OFF fallback — Boltz `uniref.a3m` → Protenix `unpairedMsaPath` |
| Boltz MSA → AF3 *(legacy)* | Boltz → AF3 | `bridge_boltz_msa_to_af3.py` | msa_pipeline OFF fallback — CSV → A3M + JSON 패치 (`pairedMsa=""`, `templates=[]`) |
| **Template filter (union)** | mmseqs + foldseek 끝난 후 | **`run_template_filter.py`** | mmseqs ∪ foldseek by `(pdb_id, chain_id)`, FP-cache backed Tanimoto |
| **Template pocket extraction** | filter 직후 | **`extract_template_pockets.py`** | chain-only USalign per hit → host-chain bound-ligand centroid → cofold frame |
| **Template pocket clustering** | extraction 직후 | **`cluster_template_pockets.py`** | surface-margin (KDTree) pre-filter → scipy single-link agglomerative (5 Å) → top-K → `template_consensus_*` source |
| **Frame alignment** | cofolding 끝, docking 직전 | **`align_cofolding_outputs.py`** | Kabsch CA → `*_aligned.cif` |
| Docking prep | alignment → docking | `prepare_docking_inputs.py` | 모델 자동 선택 + 최대 19 binding-site source (≤3 cofold cluster + ≤6 predictor top-K + ≤10 consensus) 등록 + 파일 변환 |
| Multi-track docking | docking 직후 (조건부) | `run_multi_track_docking.py` | Track 2 (any template) + Track 3 (MCS ≥ 0.5). Multi-char chain id 단일 letter 정규화 |
| Ion placement | multi-track 직후 (조건부) | `collect_template_ions.py` | template alignment → ion 위치 cluster |
| Score aggregation | post-analysis 후 | `compute_submission_scores.py` | BA/RMSD/Boltz 집계 + diversity-aware top-5 |
| CASP submission | 최종 | `make_casp_submission.py` | LG format `.lg` (MODEL 1..5) |
| Reference analysis | post-hoc (manual) | `analyze_reference.py` | 정답 crystal vs predicted poses RMSD |

### 3-2. Tool ecosystem

```mermaid
graph TB
    subgraph VENVS[".venvs/ (isolated environments)"]
        BOLTZ["boltz\nPy3.12 · torch 2.11 · CUDA 13.0"]
        PROTENIX["protenix\nPy3.12 · torch 2.7 · CUDA 12.6"]
        AF3["alphafold3\nPy3.12 · JAX"]
        PXDOCK["protenix-dock\nmicromamba Py3.11 · ambertools + tleap"]
        PRED["pred\nPy3.12 · torch 2.4 · dgl 2.4 + openbabel\n(❌ heavy partition)"]
    end

    subgraph BINS[".local/bin/"]
        MMSEQS["mmseqs"]
        FOLDSEEK["foldseek"]
        USALIGN["USalign"]
        AGRID["autogrid4"]
        ADG["autodock_gpu_128wi"]
        PRANK["prank (JDK 21)"]
    end

    subgraph DBS["Search databases"]
        SEQDB["sequence/rcsb_seqDB\n488k seqs · 2.0 GB"]
        STRUCTDB["structure/rcsb_structDB\n251k structs · 7.7 GB"]
        RCSBDB["rcsb_index.db\n251k PDBs · 2.5M ligands"]
    end

    subgraph HUB[".venv (hub)"]
        HUBPY["Py3.12 · gemmi + rdkit\nlig_align (Track 3)"]
    end

    style VENVS fill:#42a5f5,color:#000
    style BINS fill:#ab47bc,color:#000
    style DBS fill:#66bb6a,color:#000
```

### 3-3. Timing — CASP16 L2001/L2002 (single RTX 6000 Ada)

```mermaid
gantt
    title Pipeline execution timeline (L2001, 53:26 total)
    dateFormat X
    axisFormat %Mm

    section Search
    MMseqs2 + filter            :0, 13

    section Co-folding
    Boltz-2 (5×5)                :13, 635
    Boltz-2x (5×5)               :635, 1375
    Protenix (5×5)            :1375, 1867
    AF3 (25 structs)             :1870, 2223

    section Docking
    Vina (5 seeds)               :2228, 2482
    AutoDock-GPU (5 seeds)       :2482, 2524
    Protenix-Dock (1 run)        :2524, 2983

    section Post + submit
    BA-Pred + RMSD-Pred          :2987, 3150
    LG submission                :3155, 3206
```

| Stage | L2001 | L2002 | 비고 |
|---|---:|---:|---|
| Template search (mmseqs + filter) | 13 s | 3 s | foldseek 추가 시 +30 s ~ 2 min |
| Boltz-2 (25 structs) | 622 s | 600 s | + affinity |
| Boltz-2x (25 structs) | 740 s | 680 s | + affinity + potentials |
| Protenix (25 structs) | 492 s | 484 s | |
| AlphaFold3 (25 structs) | 353 s | 393 s | |
| Docking prep bridge | <5 s | <5 s | |
| Vina (5 seeds × N variants) | 254 s | 130 s | |
| AutoDock-GPU (5 seeds × N variants) | 42 s | 42 s | runtime GPF |
| Protenix-Dock (single, opt-in) | 459 s | 513 s | enable 시 docking 시간의 ~77 %; default 비활성화 (commit `b7dd9d5`+) |
| Track 2/3, ion placement | skip | skip | (이 타겟들은 미발화) |
| BA-Pred + RMSD-Pred | ~1–2 min | ~1–2 min | |
| Score aggregation + LG | <10 s | <10 s | |
| **Total (PxDock 활성)** | **53:26** | **52:08** | Track 2/3 + ion 모두 활성 시 +8–20 min 추가 |
| **Total (PxDock skip, default)** | ~46 min | ~44 min | PxDock ~7-9 min 절약. Track 2/3 active 시 + 5-15 min |

> Cofolding 합산 36 분 ≈ 전체 ~67 %. Track 1 docking ~13 분 ≈ ~25 %.
> Source fan-out (최대 19 개) 의 wall-clock 영향은 작음 — Vina 3 s × 19 src × 5 seed ≈ 285 s, ADG 10 s × 19 × 5 ≈ 950 s. 전체 docking 시간은 PxDock 이 결정.

### 3-4. Shared utility modules

여러 스크립트에 중복되던 helper 들을 정리한 공통 모듈.

| Module | API | 용도 |
|---|---|---|
| `src/casp17/geometry.py` | `parse_ca`, `kabsch`, `transform_mol`, `reassign_bonds`, `mol_from_mdl_body`, `pose_rmsd` (symmetry-aware RDKit `CalcRMS`) | 포즈 align / RMSD |
| `src/casp17/lg_format.py` | `parse_lg` (multi-MODEL LG 파서 — ATOM/HETATM/TER + MDL body + `LSCORE`/`AFFNTY`/`LIGAND` 분리) | LG 파일 read 측 |
| `src/casp17/template_filter.py` | `parse_mmseqs_hits`, `parse_foldseek_hits`, `filter_hits_with_ligands` | template union filter |
| `src/casp17/ligand_fp_cache.py` | `LigandFPCache(db_path, radius=2, nbits=2048).get_fp(ccd_code, smiles=None)` + `tanimoto(fp_a, fp_b)` | sqlite 백업 Morgan FP 캐시 (template_filter 의 hot path 단축) |
| `src/casp17/usalign.py` | `run_usalign` (`.local/bin/USalign` 래퍼) | template / receptor align |

**Standalone analysis tools** (manual, 파이프라인이 자동으로 호출하지 않음):

| Script | 용도 |
|---|---|
| `scripts/analyze_batch_rmsd.py` | 배치 단위 pose-pool RMSD 분석. 각 타겟의 `outputs/analysis/poses/` 모든 staged SDF vs 결정구조 native ligand 의 symmetric RMSD (`rdMolAlign.CalcRMS` + `AssignBondOrdersFromTemplate` — atom permutation 자동 처리). USalign 으로 crystal chain A → cofold reference 정렬, gemmi 로 native ligand instance 를 PDB 직렬화 후 RDKit 에 다시 로드 (hand-rolled PDB writer 의 RDKit-parser breakage 회피). zone 별 generation success rate (oracle <2 Å, <1 Å) 와 ranker gap 계산용 |
| `scripts/build_ligand_fp_cache.py` | RCSB `ccd_components` 의 SMILES 49k 개 → Morgan FP cache sqlite 빌드 (~18 s). RCSB DB 갱신 시 재실행 |

### 3-5. Held-out benchmarks

| Dir | 타겟 수 | 용도 | Time-split |
|---|---:|---|---|
| `experiments/novel2025_test/` | 499 (RCSB 2025-01-01 이후 non-redundant) | 시간 분할 held-out 벤치 | `template_search_sequence.max_deposition_date: "2025-01-01"` |

`evaluate.py` 가 최종 `.lg` 파일과 ground-truth 결정구조를 매칭해 ligand RMSD / pKd 오차 집계.
입력 컬럼 매핑 / ligand 선정 규칙 / cluster-aware SR 분석은 `experiments/novel2025_test/README.md` 참조.

**Build-time SMILES 파서 주의점** (`build_inputs.py::_smart_split_smiles`): RCSB index TSV 가 `|` 를
(a) 다중-ligand separator, (b) HEM 같은 분자 내 segment separator (`[Fe]5|6|...`) 둘 다로 씀.
naive `split("|")` 시 HEM-carrying 타겟 16 개가 깨진 SMILES 로 채워져 AF3 가 죽음 → helper 가 `|` 로 쪼갠 후
RDKit `MolFromSmiles` 로 chunk validate → invalid chunk 를 greedy-concat 으로 재조립.
CCD 개수를 ground truth 로 써서 불일치 시 naive split fallback.

**Partition 제약**: `heavy` (gpu1, H100 + Blackwell 6000pro) 사용 금지. 이유는 위 Stage 6 (Post-analysis)
에 적힘 — `.venvs/pred` 의 dgl 2.4 / torch 2.4 빌드가 sm_90/sm_100 kernel 미포함. `novel2025_config.yaml::slurm.partition`
및 `run_array.sbatch.sh` 의 `#SBATCH --partition` 은 **`6000ada` 로만 고정**.

### 3-6. Environment variables

새 환경 setup 시 RCSB DB 경로를 config 마다 override 안 해도 되도록 env var 지원
(commit `5fb2039`+):

| 변수 | Default | 용도 |
|---|---|---|
| `CASP17_RCSB_DIR` | `~/DB/RCSB/raw/mmCIF_data` | `template_search_sequence.rcsb_dir` 기본값 |
| `CASP17_RCSB_INDEX_DB` | `~/DB/RCSB/processed/rcsb_index.db` | `template_search_sequence.rcsb_db_path` 기본값 |

YAML config 에서 `rcsb_dir` / `rcsb_db_path` 를 명시하면 그쪽이 우선. 즉
**해석 순서**: explicit YAML > env var > hardcoded default.

### 3-7. Output tree

```
experiments/runs/<target>/
├── inputs/
│   ├── boltz_input.yaml                            # Boltz unified YAML (+ affinity)
│   ├── protenix_input.json                         # Protenix JSON
│   ├── alphafold3_input.json                       # AF3 JSON (+ MSA from Boltz bridge)
│   ├── docking/
│   │   ├── docking_prep_summary.json                 # receptor / ligand / box paths + binding_site_predictions (≤19 sources) + source_status (silent-skip diagnostic)
│   │   ├── receptor.pdb / .pdbqt / receptor_protonated.pdb  (metal/cofactor retained, apo)
│   │   ├── receptor_excl_<chain>.pdb / .pdbqt        # per-ligand "ligand-aware" receptor (multi-ligand only — others' cofold poses retained as static atoms)
│   │   ├── ligand_<chain>.sdf / .pdbqt
│   │   ├── p2rank/                                   # P2Rank pocket predictions
│   │   └── swinsite/                                 # SwinSite pocket predictions
│   └── template_docking/                           # Track 2 inputs (any template, no MCS gate)
│       ├── template_docking_summary.json
│       └── template_<pdb_id>/
│           ├── <pdb_id>.cif                          # extracted template CIF
│           ├── receptor_aligned.pdb / .pdbqt         # USalign-transformed to cofold frame
│           ├── template_ligand_<CCD>.sdf             # bound-pose ligand (for Track 3)
│           ├── ligand_L.sdf / .pdbqt                 # target ligand (from SMILES)
│           └── docking_prep_summary.json
├── outputs/
│   ├── template_search_sequence/                   # Stage 1-1
│   │   ├── mmseqs_hits.tsv                           # raw mmseqs
│   │   └── filtered_hits.tsv                         # union (mmseqs ∪ foldseek), scored
│   ├── template_search_structure/                  # Stage 1-2
│   │   └── foldseek_hits.tsv                         # raw foldseek
│   ├── template_pockets/                           # Stage 1-4/1-5
│   │   ├── template_pockets.json                     # per-instance pocket centers
│   │   ├── template_pocket_clusters.json             # top-K consensus centroids
│   │   └── _extract_work/                            # extracted CIFs cache
│   ├── boltz2/ boltz2x/ protenix/ alphafold3/      # Stage 2: 25 structs/model
│   │   └── seed_42/ seed_101/ ...                    # per-seed (AF3 는 native multi-seed)
│   ├── vina_<source>/                              # Stage 4 Track 1: ≤19 variants × 5 seeds (registered sources only)
│   │   └── seed_42/ ... ligand_L/docked.pdbqt
│   ├── autodock_gpu_<source>/                      # Stage 4 Track 1: ≤19 variants × 5 seeds
│   │   └── seed_42/ ... ligand_L/docked.dlg
│   ├── protenix_dock/                              # Stage 4 Track 1: single, priority-picked (default off)
│   │   ├── poses_<lid>.sdf                           # multi-pose SDF (per-ligand path, post-fix `28bd99f`)
│   │   └── *_out.json                                # raw atom-mapped output
│   ├── template_docking/                           # Stage 4 Track 2+3
│   │   ├── multi_track_summary.json
│   │   └── <pdb_id>/{vina,autodock_gpu,protenix_dock,lig_align}/ligand_<lig_id>/  (per-ligand sub-dir; legacy single-ligand layout still readable)
│   ├── ion_placement/                              # Stage 5 (if ion in input)
│   │   └── ion_placement_summary.json                # clustered positions by confidence
│   ├── analysis/                                   # Stage 6
│   │   ├── poses/                                    # staged pose files (pdbqt/dlg + sdf)
│   │   ├── ba_pred_<tool>.tsv                        # per-pose pKd
│   │   ├── rmsd_pred_<tool>.tsv                      # per-pose pRMSD, P(>2 Å)
│   │   └── summary.json
│   └── submission_scores.json                      # Stage 7: aggregated + ensemble
├── scripts/                                        # generated runner scripts
├── run_manifest.json
└── wrapper_manifest.json

experiments/CASP17/submissions/                     # Stage 7 output (CASP submission home)
├── <target>_LCDD.lg                                 # CASP17 LG format (MODEL 1..5)
└── ...
```
