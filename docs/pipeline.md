# CASP17 Protein-Ligand Pipeline

## Overview

```mermaid
flowchart TB
    INPUT["Protein Sequence + Ligand SMILES\n(unified YAML)"]

    subgraph S1["1. Template Search"]
        B1["MMseqs2"] --> B2["Template Filter\n(Tanimoto + MCS)"]
    end

    subgraph S2["2. Co-folding"]
        direction LR
        C1["Boltz-2"] --- C2["Boltz-2x"] --- C3["Protenix"] --- C4["AF3"]
    end

    subgraph S3["3. Structure Search"]
        D1["Foldseek x4"] --> D2["Consensus"]
    end

    subgraph S4["4. Docking Prep"]
        E1["Best Model\n+ Binding Site\n+ File Conversion"]
    end

    subgraph S5["5. Docking"]
        direction TB
        subgraph T1["Track 1: Cofolding-based"]
            direction LR
            F1["Vina"] --- F2["ADG"] --- F3["PxDock"]
        end
        subgraph T23["Track 2+3: Template-guided"]
            direction LR
            F4["Template-based\nBox Docking"] --- F5["lig-align"]
        end
    end

    subgraph S6["6. Post-analysis"]
        direction LR
        G1["BA-Pred"] --- G2["RMSD-Pred"]
    end

    INPUT --> S1 & S2
    S1 -->|"MCS >= 0.5"| T23
    S2 --> S3
    S2 --> S4 --> T1
    S5 --> S6
```

---

## Stage 1: Template Search (Sequence)

단백질 서열로 RCSB PDB에서 유사 구조를 찾고, 리간드 정보를 매칭한다.

### Step 1-1: MMseqs2 Sequence Search

```mermaid
flowchart LR
    A["Protein FASTA"] --> B["MMseqs2\neasy-search"]
    B --> C["mmseqs_hits.tsv"]
    style C fill:#4fc3f7,color:#000
```

- **Tool**: `mmseqs easy-search` (488k RCSB 서열 DB, preindexed)
- **Parameter**: `min_seq_identity=0.3`, `min_coverage=0.7`, `sensitivity=7.5`, `max_hits=200`
- **Output**: `outputs/template_search_sequence/mmseqs_hits.tsv`
- **소요 시간**: ~3초

### Step 1-2: Template Filter (Ligand + MCS)

```mermaid
flowchart LR
    A["mmseqs_hits.tsv"] --> B["rcsb_index.db\nSQLite Lookup"]
    B --> C["Ligand CCD Code\n+ SMILES\n+ Category"]
    C --> D["Tanimoto\n(Morgan FP)"]
    C --> E["MCS Coverage\n(rdFMCS)"]
    D & E --> F["filtered_hits.tsv"]
    style B fill:#ce93d8,stroke:#333,color:#000
    style F fill:#ffd54f,color:#000
```

- **Script**: `scripts/run_template_filter.py` (wrapper에서 자동 삽입)
- **Module**: `src/casp17/template_filter.py`
- **Input**: mmseqs_hits.tsv + target ligand SMILES + rcsb_index.db
- **Logic**:
  1. 각 PDB hit에 대해 `rcsb_index.db`에서 리간드 조회 (candidate 리간드만: small_molecule, cofactor, metabolite 등)
  2. Target SMILES와 template 리간드 간 **Tanimoto similarity** 계산 (Morgan FP, radius=2, 2048 bits)
  3. **MCS coverage** 계산 (`rdFMCS.FindMCS`, timeout=5s, `MCS_atoms / min(target, template) heavy atoms`)
  4. 결과를 `best_mcs_coverage → best_tanimoto → pident` 순으로 정렬
- **Output**: `outputs/template_search_sequence/filtered_hits.tsv`
- **핵심 결정**: `best_mcs_coverage >= 0.5`이면 Stage 5에서 Track 2+3 활성화
- CCD 분류 기반으로 drug-like 리간드만 필터링 (ion, 결정화 보조제, 당류 등 제외)

---

## Stage 2: Co-folding

4개 모델이 순차 실행. 동일한 unified YAML 입력을 각 모델 포맷으로 변환 후 GPU 추론.

### Step 2-1: Input Adaptation

```mermaid
flowchart LR
    YAML["Unified YAML\n(Boltz format)"] --> A1["adapters.py"]
    A1 --> B1["boltz_input.yaml\n(Boltz-2 / 2x)"]
    A1 --> B2["protenix_input.json"]
    A1 --> B3["alphafold3_input.json"]
```

- **Module**: `src/casp17/adapters.py`
- 리간드가 있으면 `properties.affinity` 자동 추가 (Boltz 전용)
- Boltz-2: `use_potentials=false`, Boltz-2x: `use_potentials=true` (별도 run)
- `prepare_boltz()` → `[boltz2, boltz2x]` 리스트 반환

### Step 2-2: Model Inference

```mermaid
flowchart TB
    B2["Boltz-2\n(no potentials)\n+ affinity"]
    B2X["Boltz-2x\n(use_potentials)\n+ affinity"]
    PX["Protenix v2"]
    B2 -->|MSA CSV| BRIDGE["bridge_boltz_msa_to_af3.py\nCSV → A3M"]
    BRIDGE --> AF3["AlphaFold3\n(JAX)"]

    B2 --> O1["outputs/boltz2/\nCIF + confidence + affinity"]
    B2X --> O2["outputs/boltz2x/\nCIF + confidence + affinity"]
    PX --> O3["outputs/protenix/\nCIF + confidence"]
    AF3 --> O4["outputs/alphafold3/\nCIF + confidence + ranking"]

    style O1 fill:#4fc3f7,color:#000
    style O2 fill:#4fc3f7,color:#000
    style O3 fill:#4fc3f7,color:#000
    style O4 fill:#4fc3f7,color:#000
```

| Model | venv | 소요 시간 | 특징 |
|-------|------|----------|------|
| Boltz-2 | `.venvs/boltz` | ~50-140s | 구조 + confidence + MSA + affinity |
| Boltz-2x | `.venvs/boltz` | ~50-140s | 위와 동일 + potentials (constraint) |
| Protenix v2 | `.venvs/protenix` | ~80-190s | 구조 + confidence |
| AlphaFold3 | `.venvs/alphafold3` | ~110-130s | 구조 + confidence + ranking (JAX) |

**Bridge: Boltz MSA -> AF3**
- Boltz가 생성한 MSA CSV를 AF3 A3M 포맷으로 변환
- AF3 JSON에 `pairedMsa=""`, `templates=[]` 패치
- Script: `scripts/bridge_boltz_msa_to_af3.py`

**Boltz Affinity Output:**
```json
{
  "affinity_pred_value": 2.62,        // log10(IC50) uM — lower = stronger
  "affinity_probability_binary": 0.41  // binder probability [0-1]
}
```

---

## Stage 3: Structure Search (Foldseek Consensus)

각 cofolding 모델의 출력 구조를 RCSB 구조 DB에서 검색하고, 교차 모델 합의로 순위 매김.

```mermaid
flowchart TB
    B2["Boltz-2 CIF"] --> FS1["Foldseek"]
    B2X["Boltz-2x CIF"] --> FS2["Foldseek"]
    PX["Protenix CIF"] --> FS3["Foldseek"]
    AF3["AF3 CIF"] --> FS4["Foldseek"]

    FS1 & FS2 & FS3 & FS4 --> MERGE["Merge & Consensus"]
    MERGE --> FILTER["Ligand Filter\n(rcsb_index.db)"]
    FILTER --> RESULT["Ranked PDBs\n- num_models found\n- avg TM-score\n- ligand info"]

    style RESULT fill:#ffd54f,color:#000
```

- **Script**: `scripts/run_structure_search.py`
- **Tool**: `foldseek easy-search` (251k RCSB 구조 DB)
- **Logic**:
  1. 4개 모델 각각에서 best CIF 추출
  2. Foldseek으로 RCSB 구조 DB 검색
  3. 교차 모델 합의: 여러 모델에서 공통 발견된 PDB 우선 순위
  4. `rcsb_index.db`로 리간드 보유 여부 필터링
- **Output**: `outputs/structure_search/consensus_summary.json`

---

## Stage 4: Docking Preparation

Cofolding 출력에서 docking 입력 파일을 자동 생성. Wrapper pipeline에서 cofolding → docking 사이에 자동 삽입.

### Step 4-1: Best Model Selection

```mermaid
flowchart LR
    B2["boltz2\npLDDT=?"] & B2X["boltz2x\npLDDT=?"] & PX["protenix\npLDDT=?"] & AF3["af3\npLDDT=?"]
    B2 & B2X & PX & AF3 --> SELECT["pLDDT 비교\nBest 선택"]
    SELECT --> BEST["Best CIF"]
    style BEST fill:#66bb6a,color:#000
```

- 각 모델의 confidence score (pLDDT) 비교 → 최고 점수 모델 자동 선택
- Function: `select_best_model()` in `prepare_docking_inputs.py`

### Step 4-2: Receptor Preparation

```mermaid
flowchart LR
    CIF["Best CIF"] --> GEMMI["gemmi\nremove_ligands_and_waters"]
    GEMMI --> PDB["receptor.pdb"]
    PDB --> PDB2PQR["pdb2pqr\n--ff=AMBER"]
    PDB2PQR --> PROT["receptor_protonated.pdb\n(HIS->HID/HIE/HIP)"]
    PDB2PQR --> PDBQT["receptor.pdbqt\n(AD4 atom types + charges)"]

    style PROT fill:#4fc3f7,color:#000
    style PDBQT fill:#4fc3f7,color:#000
```

| Step | Tool | Output | 용도 |
|------|------|--------|------|
| CIF -> PDB | gemmi | `receptor.pdb` | 기본 구조 |
| PDB -> PQR | pdb2pqr (AMBER) | 중간 파일 | 수소 추가 + 전하 |
| PQR -> protonated PDB | 자체 변환 | `receptor_protonated.pdb` | Protenix-Dock |
| PQR -> PDBQT | AD4 type mapping | `receptor.pdbqt` | Vina + AutoDock-GPU |

### Step 4-3: Ligand Preparation

```mermaid
flowchart LR
    SMILES["Ligand SMILES"] --> RDKIT["RDKit\nETKDGv3 + MMFF"]
    RDKIT --> SDF["ligand_L.sdf"]
    SDF --> MEEKO["meeko\nMoleculePreparation"]
    MEEKO --> PDBQT["ligand_L.pdbqt"]

    style SDF fill:#4fc3f7,color:#000
    style PDBQT fill:#4fc3f7,color:#000
```

- SMILES -> 3D conformer: `AllChem.EmbedMolecule(mol, ETKDGv3())`
- Force field optimization: `MMFFOptimizeMolecule(mol, maxIters=500)`
- SDF -> PDBQT: meeko `MoleculePreparation`

### Step 4-4: Binding Site Prediction

```mermaid
flowchart TB
    PDB["receptor.pdb"] --> SWIN["SwinSite\n(Swin-Unet ML)"]
    PDB --> P2R["P2Rank\n(surface-based)"]
    SDF["ligand.sdf"] --> FALLBACK["Ligand 3D\ncoordinates"]

    SWIN -->|"priority 1"| BOX["Docking Box\n22.5A x 22.5A x 22.5A\nspacing 0.375A"]
    P2R -->|"priority 2"| BOX
    FALLBACK -->|"priority 3"| BOX

    style BOX fill:#66bb6a,color:#000
```

- **우선순위**: SwinSite (ML) > P2Rank (surface) > Ligand coordinates (fallback)
- **Unified box**: 22.5A x 22.5A x 22.5A, grid spacing 0.375A (모든 docking tool 공통)
- **Script**: `scripts/prepare_docking_inputs.py`
- **Output**: `inputs/docking/docking_prep_summary.json` (모든 docking tool이 runtime에 읽음)

---

## Stage 5: Docking (Multi-track)

3개 트랙으로 구성. Track 1은 항상 실행, Track 2+3은 MCS >= threshold일 때 자동 활성화.

### Track 1: Cofolding-based Docking (항상 실행)

Cofolding best model의 구조를 receptor로, SwinSite/P2Rank 예측 위치를 docking box로 사용.

```mermaid
flowchart LR
    subgraph VINA["Vina (Python API)"]
        V1["set_receptor\nset_ligand"] --> V2["compute_vina_maps"] --> V3["dock()"] --> V4["write_poses()"]
    end

    subgraph ADG["AutoDock-GPU (CUDA)"]
        A1["autogrid4\nGPF -> grid maps"] --> A2["autodock_gpu_128wi\n--nrun 100\n--heuristics 1"]
    end

    subgraph PXDOCK["Protenix-Dock"]
        P1["prepare_receptor\n(tleap: +H, solvation)"] --> P2["generate_cache_maps\n(grid caching)"] --> P3["run_docking"]
    end

    style VINA fill:#66bb6a,color:#000
    style ADG fill:#ffb74d,color:#000
    style PXDOCK fill:#f48fb1,color:#000
```

| Tool | Type | 소요 시간 | venv | Output |
|------|------|----------|------|--------|
| Vina | Python API | ~3s | `.venvs/protenix-dock` | `outputs/vina/docked.pdbqt` |
| AutoDock-GPU | CUDA binary | ~10s | `.local/bin/` | `outputs/autodock_gpu/docked.dlg` |
| Protenix-Dock | CPU force field | ~5-30min | `.venvs/protenix-dock` | `outputs/protenix_dock/` |

- 모든 tool은 `docking_prep_summary.json`에서 receptor/ligand/box를 runtime에 읽음
- Protenix-Dock이 전체 시간의 ~77% 차지 (병목)

### Track 2: Template-based Box Docking (MCS >= 0.5)

Template search에서 MCS coverage가 높은 hit의 실험 구조를 receptor로, template 리간드 위치를 docking box로 사용.

```mermaid
flowchart TB
    HITS["filtered_hits.tsv\n(MCS >= 0.5)"] --> PREP["prepare_template_docking.py"]

    subgraph TEMPLATE_PREP["Template Input Preparation"]
        RCIF["Template CIF\n(RCSB mmCIF)"]
        RCIF --> RPDB["Template Receptor\nPDB / PDBQT"]
        RCIF --> TLIG["Template Ligand SDF\n(bound-pose extraction)"]
        RCIF --> QLIG["Target Ligand\nSDF / PDBQT\n(from SMILES)"]
        TLIG --> BOX["Docking Box\n(template ligand\ncentroid)"]
    end

    PREP --> TEMPLATE_PREP

    RPDB & QLIG & BOX --> DOCK["Vina + ADG + PxDock\n(same tools as Track 1)"]
    DOCK --> OUT["outputs/template_docking/&lt;pdb_id&gt;/\nvina/ autodock_gpu/ protenix_dock/"]

    style HITS fill:#ffd54f,color:#000
    style TLIG fill:#ba68c8,color:#000
```

- **Script**: `scripts/prepare_template_docking.py`
- **Logic**:
  1. `filtered_hits.tsv`에서 MCS >= threshold인 상위 3개 template 선택
  2. RCSB CIF 파일에서 receptor PDB/PDBQT 생성 (`gemmi` + `pdb2pqr`)
  3. Template 리간드의 **bound-pose SDF 추출** (`extract_template_ligand_sdf()`)
  4. Template 리간드 좌표 centroid -> docking box center
  5. Target SMILES -> SDF/PDBQT (RDKit + meeko)
  6. 각 template에 대해 Vina + ADG + PxDock 실행
- **장점**: 실험적으로 검증된 리간드 결합 위치를 docking box로 사용 -> 정확도 향상

### Track 3: lig-align (MCS-guided Pose Generation, MCS >= 0.5)

Template 리간드의 결합 포즈를 MCS anchor로 활용해, target 리간드의 3D 포즈를 직접 생성.

```mermaid
flowchart LR
    REF["Template Ligand SDF\n(bound pose)"] --> MCS["MCS Detection\n(rdFMCS)"]
    QUERY["Target SMILES"] --> MCS
    MCS --> ANCHOR["MCS Atom\nAlignment"]
    ANCHOR --> CONF["Conformer Generation\n(1000 conformers\nMCS-constrained)"]
    CONF --> CLUSTER["RMSD Clustering\n(threshold 1.0A)"]
    CLUSTER --> SCORE["Vina Scoring\n(weight_preset=vina)"]
    SCORE --> OPT["Torsion Optimization\n(Adam, 100 steps\nfreeze MCS atoms)"]
    OPT --> TOPK["Top-k Poses\n(SDF output)"]

    style REF fill:#ba68c8,color:#000
    style TOPK fill:#66bb6a,color:#000
```

- **Library**: `lig_align.run_pipeline()` (hub venv에 설치됨)
- **Parameters**:
  - `num_confs=1000`: MCS-constrained conformer 수
  - `mcs_mode="auto"`: single/multi/cross MCS 자동 선택
  - `optimize=True`: gradient-based torsion optimization
  - `weight_preset="vina"`: Vina scoring function weights
  - `top_k=10`: 최종 출력 포즈 수
- **Input**: template receptor PDB + template ligand SDF (bound pose) + target SMILES
- **Output**: `outputs/template_docking/<pdb_id>/lig_align/` (ranked SDF poses)
- **장점**: Docking과 달리 scoring function만 사용, MCS anchor 덕분에 정확한 초기 배치

### Multi-track Decision Logic

```mermaid
flowchart TB
    START["filtered_hits.tsv 확인"]
    START --> CHECK{"best_mcs_coverage\n>= threshold?"}
    CHECK -->|"Yes (>= 0.5)"| PREP["Template-based Box Docking Prep\n(max 3 templates)"]
    CHECK -->|"No"| SKIP["Track 2+3 Skip\n(Track 1 결과만 사용)"]
    PREP --> T2["Track 2: Template-based\nBox Docking\n(Vina + ADG + PxDock)"]
    PREP --> T3["Track 3: lig-align\n(MCS-guided)"]
    T2 & T3 --> SUMMARY["multi_track_summary.json"]

    style CHECK fill:#ffd54f,color:#000
```

- **Orchestrator**: `scripts/run_multi_track_docking.py`
- **Config**: `template_search_sequence.mcs_threshold` (default: `0.5`)
- Wrapper script에서 Track 1 docking 이후 자동 실행

| Track | Receptor | Box Source | Method | 조건 |
|-------|----------|-----------|--------|------|
| Track 1 | Cofolding best model | SwinSite > P2Rank | Vina + ADG + PxDock | 항상 |
| Track 2 | Template PDB (RCSB) | Template ligand centroid | Vina + ADG + PxDock | MCS >= 0.5 |
| Track 3 | Template PDB (RCSB) | MCS anchor alignment | lig-align | MCS >= 0.5 |

---

## Stage 5.5: Ion/Metal Placement (conditional)

Input YAML에 ion/metal CCD 엔티티 (ZN, MG, CA, FE 등)가 있으면 자동 실행. Cofolding은 metal 위치를 정확히 예측하기 어려우므로, template alignment 기반으로 가능한 위치를 수집.

```mermaid
flowchart TB
    INPUT["Input YAML\n(ccd: ZN)"] --> DETECT["Ion 감지\n(known_ions set)"]
    DETECT --> SEARCH["Template Search 결과\n(pident >= 30%)"]
    SEARCH --> FILTER["rcsb_index.db\ntarget ion 보유\ntemplate 필터"]
    FILTER --> ALIGN["gemmi superposition\n(CA atoms)\ntemplate → cofolding"]
    ALIGN --> TRANSFORM["Ion 좌표 변환\n(cofolding frame)"]
    TRANSFORM --> CLUSTER["Distance Clustering\n(threshold 2.0A)"]
    CLUSTER --> REPORT["Confidence 그룹별 리포트"]

    subgraph CONFIDENCE["Confidence Groups"]
        direction LR
        H["High\npident >= 70%"]
        M["Medium\n50-70%"]
        L["Low\n30-50%"]
    end

    REPORT --> CONFIDENCE

    style DETECT fill:#ffd54f,color:#000
    style CLUSTER fill:#66bb6a,color:#000
```

- **Script**: `scripts/collect_template_ions.py`
- **Input**: cofolding best CIF + template search hits + target ion CCD codes
- **Logic**:
  1. Input YAML에서 ion CCD 코드 추출 (ZN, MG, CA, FE 등)
  2. Template search hits 중 해당 ion을 보유한 PDB 필터링 (rcsb_index.db)
  3. 각 template을 cofolding best model에 gemmi CA superposition
  4. Rotation/translation을 ion 좌표에 적용 → cofolding 좌표계로 변환
  5. 거리 기반 클러스터링 (default 2.0A)
  6. Confidence 그룹별 (pident 70%+/50-70%/30-50%) 결과 리포트
- **Output**: `outputs/ion_placement/ion_placement_summary.json`
- **자동 skip**: input에 ion이 없으면 실행하지 않음

**Output example:**
```json
{
  "ions": {
    "ZN": {
      "total_positions": 12,
      "total_templates": 10,
      "clusters": [
        {"centroid": [12.3, 45.6, 78.9], "num_templates": 8, "spread_angstrom": 0.8}
      ],
      "by_confidence": {
        "high": {"pident_range": ">=70%", "num_templates": 3, "clusters": [...]},
        "medium": {"pident_range": "50-70%", "num_templates": 5, "clusters": [...]},
        "low": {"pident_range": "30-50%", "num_templates": 2, "clusters": [...]}
      }
    }
  }
}
```

---

## Stage 6: Post-analysis

모든 docking 결과에 대해 binding affinity와 pose RMSD를 GNN으로 예측.

```mermaid
flowchart TB
    subgraph CONVERT["Format Conversion"]
        VINA_OUT["Vina docked.pdbqt"] --> MK["mk_export.py\n(meeko)"]
        ADG_OUT["ADG docked.dlg"] --> MK
        MK --> SDF_OUT["docked.sdf"]
    end

    subgraph PREDICT["GNN Prediction"]
        SDF_OUT --> BA["BA-Pred\n(Binding Affinity)"]
        SDF_OUT --> RMSD["RMSD-Pred\n(Pose Quality)"]
        REC["receptor.pdb"] --> BA & RMSD
    end

    BA --> BA_OUT["pKd (kcal/mol)\nper model x per tool"]
    RMSD --> RMSD_OUT["pRMSD (>2A prob)\nper model x per tool"]

    style BA_OUT fill:#ffd54f,color:#000
    style RMSD_OUT fill:#ffd54f,color:#000
```

- **Script**: `scripts/run_post_analysis.py` (GPU node에서 실행)
- **venv**: `.venvs/pred` (torch 2.4 + dgl 2.4 + openbabel)
- **Output**:
  - `outputs/analysis/ba_pred_results.tsv` — 각 model x docking tool 조합별 pKd
  - `outputs/analysis/rmsd_pred_results.tsv` — 각 model x docking tool 조합별 pRMSD
  - `outputs/analysis/summary.json` — 최적 조합 선택

---

## Timing (RTX 6000 Ada)

```mermaid
gantt
    title Pipeline Execution Timeline
    dateFormat X
    axisFormat %s

    section Search
    MMseqs2 sequence search     :0, 3
    Template filter (MCS)       :3, 5

    section Co-folding
    Boltz-2 + affinity          :5, 148
    Boltz-2x + affinity         :148, 236
    Protenix v2                 :236, 369
    Boltz MSA -> AF3 bridge     :369, 370
    AlphaFold3                  :370, 499

    section Docking Prep
    Auto-select + SwinSite + P2Rank :499, 509

    section Track 1
    Vina                        :509, 512
    AutoDock-GPU                :512, 522
    Protenix-Dock               :522, 2262

    section Track 2+3 (conditional)
    Template box docking prep   :2262, 2272
    Template Vina + ADG         :2272, 2285
    Template PxDock             :2285, 4025
    lig-align                   :4025, 4035

    section Post-analysis
    BA-Pred + RMSD-Pred         :4035, 4045
```

| Stage | Time | Notes |
|-------|-----:|-------|
| MMseqs2 + template filter | ~5s | sequence search + Tanimoto/MCS scoring |
| Boltz-2 + affinity | ~143s | |
| Boltz-2x + affinity | ~88s | |
| Protenix v2 | ~133s | |
| AlphaFold3 | ~129s | |
| Foldseek x 4 | ~30s | structure search + consensus |
| Docking prep | ~10s | auto-select + SwinSite/P2Rank |
| **Track 1** | **~29min** | cofolding-based (Vina + ADG + PxDock) |
| **Track 2** | **~29min** | template-based box docking (conditional, MCS >= 0.5) |
| **Track 3** | **~10s** | lig-align (conditional, MCS >= 0.5) |
| Post-analysis | ~10s | BA-Pred + RMSD-Pred |
| **Total (Track 1 only)** | **~37min** | |
| **Total (all tracks)** | **~66min** | when template has MCS >= 0.5 |

> Protenix-Dock이 Track 1과 Track 2 모두에서 병목 (~77%).

---

## Bridge Scripts

Wrapper pipeline에서 stage 사이에 자동 삽입되는 bridge step 목록.

| Bridge | 삽입 위치 | Script | 역할 |
|--------|----------|--------|------|
| Boltz MSA -> AF3 | Boltz -> AF3 (cofolding 내부) | `bridge_boltz_msa_to_af3.py` | MSA CSV -> A3M + AF3 JSON 패치 |
| Template Filter | template-search-sequence 직후 | `run_template_filter.py` | Tanimoto + MCS scoring |
| Docking Prep | cofolding -> docking 사이 | `prepare_docking_inputs.py` | 자동 모델 선택 + binding site + 파일 변환 |
| Multi-track Docking | docking 직후 (조건부) | `run_multi_track_docking.py` | Track 2 + Track 3 실행 |
| Ion Placement | multi-track 직후 (조건부) | `collect_template_ions.py` | template alignment → ion 위치 수집 |

---

## Tool Ecosystem

```mermaid
graph TB
    subgraph VENVS[".venvs/ (isolated environments)"]
        BOLTZ["boltz\nPy3.12, torch 2.11\nCUDA 13.0"]
        PROTENIX["protenix\nPy3.12, torch 2.7\nCUDA 12.6"]
        AF3["alphafold3\nPy3.12, JAX"]
        PXDOCK["protenix-dock\nmicromamba, Py3.11\nambertools + tleap"]
        PRED["pred\nPy3.12, torch 2.4\ndgl 2.4 + openbabel"]
    end

    subgraph BINS[".local/bin/"]
        MMSEQS["mmseqs"]
        FOLDSEEK["foldseek"]
        AUTOGRID["autogrid4"]
        ADGPU["autodock_gpu_128wi"]
        PRANK["prank (JDK 21)"]
    end

    subgraph DBS["Search Databases"]
        SEQDB["sequence/rcsb_seqDB\n488k seqs, 2.0GB"]
        STRUCTDB["structure/rcsb_structDB\n251k structs, 7.7GB"]
        RCSBDB["rcsb_index.db\n251k PDBs, 2.5M ligands"]
    end

    BOLTZ ---|"Boltz-2/2x"| COFOLDING["Co-folding"]
    PROTENIX --- COFOLDING
    AF3 --- COFOLDING
    PXDOCK ---|"PxDock + Vina"| DOCKING["Docking"]
    PRED ---|"BA-Pred + RMSD-Pred + SwinSite"| ANALYSIS["Analysis"]
    MMSEQS --- SEARCH["Search"]
    FOLDSEEK --- SEARCH
    ADGPU --- DOCKING
    PRANK --- BINDING["Binding Site"]

    style VENVS fill:#42a5f5,color:#000
    style BINS fill:#ab47bc,color:#000
    style DBS fill:#66bb6a,color:#000
```

---

## Output Structure

```
experiments/runs/<target>/
├── inputs/
│   ├── boltz_input.yaml                    # Boltz unified YAML (+ affinity)
│   ├── protenix_input.json                 # Protenix JSON
│   ├── alphafold3_input.json               # AF3 JSON (+ MSA from Boltz bridge)
│   ├── docking/                            # Track 1 docking inputs
│   │   ├── docking_prep_summary.json         # receptor/ligand/box paths
│   │   ├── receptor.pdb / .pdbqt
│   │   ├── receptor_protonated.pdb
│   │   ├── ligand_L.sdf / .pdbqt
│   │   ├── p2rank/                           # P2Rank pocket predictions
│   │   └── swinsite/                         # SwinSite pocket predictions
│   └── template_docking/                   # Track 2+3 inputs (if MCS >= 0.5)
│       ├── template_docking_summary.json
│       └── template_<pdb_id>/
│           ├── <pdb_id>.cif                  # extracted template CIF
│           ├── receptor.pdb / .pdbqt         # template receptor
│           ├── template_ligand_<CCD>.sdf     # bound-pose ligand (for lig-align)
│           ├── ligand_L.sdf / .pdbqt         # target ligand (from SMILES)
│           └── docking_prep_summary.json
├── outputs/
│   ├── template_search_sequence/           # Stage 1
│   │   ├── mmseqs_hits.tsv                   # raw hits
│   │   └── filtered_hits.tsv                 # scored + ranked
│   ├── boltz2/                             # Stage 2: CIF + confidence + affinity
│   ├── boltz2x/                            # Stage 2: CIF + confidence + affinity
│   ├── protenix/                           # Stage 2: CIF + confidence
│   ├── alphafold3/                         # Stage 2: CIF + confidence + ranking
│   ├── structure_search/                   # Stage 3: Foldseek consensus
│   ├── vina/                               # Stage 5 Track 1
│   ├── autodock_gpu/                       # Stage 5 Track 1
│   ├── protenix_dock/                      # Stage 5 Track 1
│   ├── template_docking/                   # Stage 5 Track 2+3
│   │   ├── multi_track_summary.json
│   │   └── <pdb_id>/
│   │       ├── vina/
│   │       ├── autodock_gpu/
│   │       ├── protenix_dock/
│   │       └── lig_align/
│   ├── ion_placement/                      # Stage 5.5 (if ion in input)
│   │   └── ion_placement_summary.json        # clustered positions by confidence
│   └── analysis/                           # Stage 6
│       ├── ba_pred_results.tsv
│       ├── rmsd_pred_results.tsv
│       └── summary.json
├── scripts/                                # generated runner scripts
├── run_manifest.json
└── wrapper_manifest.json
```
