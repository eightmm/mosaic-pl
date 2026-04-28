# CASP17 Protein-Ligand Hub

Unified Python workspace for CASP17 protein-ligand structure prediction and docking pipelines.
Orchestrates external ML models (Boltz2/2x, Protenix v2, AlphaFold3), **union template search (MMseqs2 + Foldseek)** with cross-source pocket consensus, binding site prediction (P2Rank, SwinSite, **template-consensus pockets**), docking (Vina, AutoDock-GPU, Protenix-Dock), and post-analysis (BA-Pred, RMSD-Pred).

## Full Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: protein sequence + ligand SMILES (unified YAML)        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐  ┌─────────────────────────────┐
    │  Stage 1a: Sequence Search │  │  Stage 1b: Structure Search │
    │  MMseqs2 (488k seqs DB)   │  │  Foldseek (251k struct DB)  │
    │  → mmseqs_hits.tsv        │  │  query = best cofold cif    │
    └─────────────┬─────────────┘  │  → foldseek_hits.tsv        │
                  │                  └─────────────┬───────────────┘
                  │                                │
                  └────────────────┬───────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │  Stage 1c: Template Bridges (auto, after both searches) │
       │  • Union filter — mmseqs ∪ foldseek by (pdb_id, chain)  │
       │    NO MCS gate. Tanimoto/MCS = metadata only.            │
       │  • Pocket extraction — gemmi CA align template→cofold   │
       │    → bound-ligand centroids in cofold frame              │
       │  • Pocket clustering — single-link 5 Å, weighted by     │
       │    (in_mmseqs + in_foldseek) + max(qtmscore, pident/100)│
       │    → top-K consensus centroids                           │
       └───────────────────────────┬────────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────┐
    │  Stage 2: Co-folding (4 models × 5 seeds × 5 samples)  │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
    │  │ Boltz-2  │ │ Boltz-2x │ │ Protenix │ │   AF3    │  │
    │  │(no pot.) │ │(potent.) │ │   v2     │ │  (JAX)   │  │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
    │       └──── BRIDGE: Boltz MSA → AF3 ───────────┘       │
    │  → 25 structures/model × 4 models = 100 structures      │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 4: Docking Prep (automatic)          │
    │  • Auto-select best model (pLDDT score)     │
    │  • 6 binding-site sources →                 │
    │    cofolding · swinsite · p2rank ·           │
    │    template_consensus_{1,2,3}                │
    │  • SMILES → 3D SDF → PDBQT (RDKit + meeko) │
    │  • CIF → PDB → PDBQT (gemmi + pdb2pqr)     │
    │  • Box: 22.5 Å, spacing 0.375 Å            │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 5: Docking (Multi-track, Multi-seed) │
    │                                             │
    │  Track 1 (always) — 6-source fan-out:       │
    │  Vina × 6 + AutoDock-GPU × 6 + PxDock × 1  │
    │                                             │
    │  Track 2 (any template) — pocket borrow:    │
    │  Template-based box docking (Vina+ADG+PxD) │
    │                                             │
    │  Track 3 (MCS ≥ 0.5) — atom-anchor:         │
    │  lig-align (MCS-guided pose generation)     │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────▼──────────────────────────────────────┐
    │  Stage 5.5: Ion Placement (if ion in input) │
    │  • gemmi CA superposition                   │
    │  • Collect ion coords from aligned templates│
    │  • Cluster by distance, group by confidence │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 6: Post-analysis                     │
    │  • PDBQT/DLG → SDF (meeko mk_export.py)    │
    │  • BA-Pred: binding affinity (GNN, per pose)│
    │  • RMSD-Pred: pose RMSD prediction (GNN)    │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 7: CASP17 Submission (LG format)     │
    │  • Aggregate all scores (BA+RMSD+Boltz)    │
    │  • Pick best pose (by LSCORE)               │
    │  • Ensemble AFFNTY (log-space Kd average)   │
    │  • Write .lg file (PDB + MDL + scores)      │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  OUTPUT: experiments/runs/<target>/          │
    │  ├── outputs/boltz2/seed_*/    (25 structs) │
    │  ├── outputs/boltz2x/seed_*/   (25 structs) │
    │  ├── outputs/protenix/seed_*/  (25 structs) │
    │  ├── outputs/alphafold3/       (num_seeds)  │
    │  ├── outputs/vina_{cofolding,swinsite,p2rank,                │
    │  │     template_consensus_{1,2,3}}/seed_*/  (6 × 5 = 30)    │
    │  ├── outputs/autodock_gpu_{... 6 sources}/seed_*/            │
    │  ├── outputs/protenix_dock/    (single)     │
    │  ├── outputs/template_search_sequence/      │
    │  │   ├── mmseqs_hits.tsv  filtered_hits.tsv │
    │  ├── outputs/template_search_structure/     │
    │  │   └── foldseek_hits.tsv                   │
    │  ├── outputs/template_pockets/ (Stage 1c)   │
    │  │   ├── template_pockets.json (per-instance)│
    │  │   └── template_pocket_clusters.json (top-K)│
    │  ├── outputs/template_docking/ (Track 2+3)  │
    │  │   └── <pdb_id>/vina/adg/pxdock/lig_align │
    │  ├── outputs/ion_placement/    (if ion)     │
    │  ├── outputs/analysis/         (BA/RMSD TSVs)│
    │  └── outputs/submission_scores.json         │
    └─────────────────────────────────────────────┘
```

## Setup

### Prerequisites

```bash
sudo apt-get install -y libboost-all-dev autoconf automake libtool
```

### Installation

```bash
git clone <repo-url> CASP17 && cd CASP17
bash scripts/install_external_models.sh              # all tools + models
srun --partition=6000ada --gres=gpu:1 bash scripts/build_autodock_gpu.sh  # GPU build
bash scripts/install_external_models.sh --verify      # verify all
bash scripts/build_search_dbs.sh                      # MMseqs2 + Foldseek DBs
```

### What Gets Installed

| Tool | venv / Location | Purpose |
|------|----------------|---------|
| Boltz2 | `.venvs/boltz` (Py3.12) | Co-folding + affinity |
| Protenix v2 | `.venvs/protenix` (Py3.12) | Co-folding |
| AlphaFold3 | `.venvs/alphafold3` (Py3.12) | Co-folding (JAX) |
| Protenix-Dock | `.venvs/protenix-dock` (micromamba) | Classical docking |
| Vina | `.venvs/protenix-dock` (shared) | Classical docking |
| AutoDock-GPU | `.local/bin/autodock_gpu_128wi` | GPU docking |
| autogrid4 | `.local/bin/autogrid4` | Grid maps for ADG |
| MMseqs2 | `.local/bin/mmseqs` | Sequence search |
| Foldseek | `.local/bin/foldseek` | Structure search |
| P2Rank | `.local/bin/prank` (JDK 21) | Binding site (surface) |
| SwinSite | `.venvs/pred` (shared) | Binding site (ML) |
| BA-Pred | `.venvs/pred` (shared) | Affinity prediction (GNN) |
| RMSD-Pred | `.venvs/pred` (shared) | Pose RMSD prediction (GNN) |

### Search Databases

| DB | Size | Contents |
|----|------|----------|
| MMseqs2 sequence DB | 2.0 GB | 488,841 RCSB protein sequences (preindexed) |
| Foldseek structure DB | 7.7 GB | 251,422 RCSB structures (3Di+AA preindexed) |
| RCSB ligand index | SQLite | 251k entries, 2.5M ligand instances, CCD classification |

## Quick Start

### Full pipeline (one command)

```bash
uv run casp17-pl run-wrapper \
  --input examples/full_pipeline_test.yaml \
  --config examples/full_pipeline_config.yaml \
  --stages template-search-sequence cofolding docking \
  --submit
```

### Step by step

```bash
# 1. Create input
cat > my_target.yaml << 'EOF'
version: 1
seed: 42
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVT...
  - ligand:
      id: L
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
EOF

# 2. Run co-folding (Boltz2 + Boltz2x + Protenix + AF3)
uv run casp17-pl run-all --input my_target.yaml --config examples/full_pipeline_config.yaml --submit

# 3. Structure search on cofolding outputs
uv run python scripts/run_structure_search.py \
  --run-dir experiments/runs/my_target \
  --foldseek-db data/search_dbs/structure/rcsb_structDB \
  --rcsb-db /home/jaemin/DB/RCSB/processed/rcsb_index.db

# 4. Post-analysis (BA-Pred + RMSD-Pred)
srun --partition=6000ada --gres=gpu:1 \
  .venvs/pred/bin/python scripts/run_post_analysis.py \
  --run-dir experiments/runs/my_target --device cuda
```

## Pipeline Stages Detail

### Stage 1: Template Search (Union — sequence + structure)

서열 검색과 구조 검색을 **둘 다 돌려서** 최대한 많은 template 을 모은다. MCS/Tanimoto 는 metadata 로만 보존 — gating 은 lig-MCS-align (Track 3) 에서만.

**1a · Sequence (MMseqs2)** — `easy-search` against `data/search_dbs/sequence/rcsb_seqDB`
- Output: `outputs/template_search_sequence/mmseqs_hits.tsv`

**1b · Structure (Foldseek)** — `easy-search` against `data/search_dbs/structure/rcsb_structDB`
- Query 자동 선택: cofolding 결과 (`query_from_cofolding=true`, priority `[alphafold3, boltz, protenix]`)
- Stage 의존성 검증이 cofold 보다 먼저 도는 순서를 자동 차단
- `--max-seqs 500` (recall 우선 — mmseqs 와 달리 native 한 identity/coverage cut 없음)
- Output: `outputs/template_search_structure/foldseek_hits.tsv`

**1c · Template bridges (auto, after both searches)**

```python
from casp17.template_filter import filter_hits_with_ligands
hits = filter_hits_with_ligands(
    hits_tsv=mmseqs_tsv,
    foldseek_tsv=foldseek_tsv,   # union by (pdb_id, chain_id)
    db_path=rcsb_index_db,
    target_smiles="CCO",
)
# Each hit carries: in_mmseqs, in_foldseek, qtmscore, pident, best_tanimoto, best_mcs_coverage
# Sort key: evidence_breadth (2 sources > 1) > qtmscore > pident > n_ligands > tanimoto > mcs
```

3 단계로 자동 실행:
1. **Union filter** (`run_template_filter.py`) — mmseqs ∪ foldseek dedup, ligand annotation. **MCS 게이트 없음**. **Foldseek-only hit 에 한해** `max(qtmscore, ttmscore) ≥ qtmscore_min` floor 적용 (default 0.5 = Zhang/Skolnick "same fold"). mmseqs hit 과 dual-source hit 은 우회.
2. **Pocket extraction** (`extract_template_pockets.py`) — top `--max-templates 100` (default) hit 을 `gemmi` CA superposition 으로 cofold model 에 align, bound candidate ligand heavy-atom centroid 추출 → `template_pockets.json`.
3. **Pocket clustering** (`cluster_template_pockets.py`) — single-link clustering 5 Å, weight = `(in_mmseqs + in_foldseek) + max(qtmscore, pident/100)`. Top-K (default 5) → `template_pocket_clusters.json`. 각 cluster centroid 가 다음 단계에서 binding-site source 로 등록됨.

**Threshold 정리**:
- `template_search_structure.qtmscore_min` (default `0.5`) — foldseek-only hit 의 fold-similarity 게이트
- `template_search_sequence.mcs_threshold` (default `0.5`) — **Track 3 (lig-align) 만** gating

### Stage 2: Co-folding

4 models run sequentially on a single GPU, each with 5 seeds × 5 diffusion samples = 25 structures per model (100 structures total):

| Model | Output | Time (25 structs) | Features |
|-------|--------|------|----------|
| **Boltz-2** | `outputs/boltz2/seed_*/` | ~10 min (600-620s) | Structure + confidence + MSA + **affinity** |
| **Boltz-2x** | `outputs/boltz2x/seed_*/` | ~12 min (680-740s) | Structure + confidence + MSA + **affinity** (with potentials) |
| **Protenix** | `outputs/protenix/seed_*/` | ~8 min (484-492s) | Structure + confidence |
| **AlphaFold3** | `outputs/alphafold3/` | ~6 min (353-393s) | Structure + confidence + ranking |

Measured on RTX 6000 Ada with `cofolding_seeds=[42,101,202,303,404]`, `diffusion_samples=5` (averaged over two L2000 CASP16 targets).

**Automatic features:**
- Affinity auto-enabled when ligand present (`properties.affinity.binder`)
- Boltz MSA → AF3 bridge (CSV → A3M conversion)
- Boltz uses `--use_msa_server` to fetch MSA from ColabFold
- AF3 runner has dedicated CUDA LD_LIBRARY_PATH for JAX

**Boltz affinity output:**
```json
{
  "affinity_pred_value": 2.62,           // log10(IC50) in μM
  "affinity_probability_binary": 0.41    // binder probability [0-1]
}
```

### Stage 3: (deprecated) Cross-model Foldseek consensus

이전 버전의 cross-model consensus 는 Stage 1b (foldseek easy-search, query=best cofold cif) + Stage 1c (mmseqs ∪ foldseek union filter) 가 대체. 자세한 내용은 위의 Stage 1 참조.

### Stage 4: Docking Preparation (Automatic Bridge)

**Tool**: `scripts/prepare_docking_inputs.py`
Runs automatically between cofolding and docking in the wrapper pipeline.

| Step | Tool | Output |
|------|------|--------|
| Auto-select best model | mean pLDDT comparison | Best CIF |
| CIF → PDB | gemmi | `receptor.pdb` |
| PDB → protonated PDB | pdb2pqr (AMBER) | `receptor_protonated.pdb` (HIS→HID/HIE/HIP) |
| PDB → PDBQT | pdb2pqr + AD4 typing | `receptor.pdbqt` (charges + atom types) |
| SMILES → 3D SDF | RDKit (ETKDG + MMFF) | `ligand_L.sdf` |
| SDF → PDBQT | meeko | `ligand_L.pdbqt` |
| Binding site sources | 6 sources (cofolding + swinsite + p2rank + template_consensus_{1,2,3}) | `binding_site_predictions` dict in summary JSON |

**Binding site sources** (각각 독립 docking variant 로 fan-out — winner-take-all 아님):

1. **Cofolding centroid** — co-folding 모델이 직접 놓은 ligand 위치 (가장 직접적인 신호)
2. **SwinSite** — Swin-Unet ML pocket predictor (`.venvs/pred`)
3. **P2Rank** — surface geometry pocket predictor (Java)
4. **template_consensus_1 / 2 / 3** — `cluster_template_pockets.py` 가 만든 top-3 cluster centroid (mmseqs+foldseek union template 의 bound-ligand 위치 합의)

**Fallback box pick** (`box_method` — protenix-dock 같은 단일-box 도구가 사용): `cofolding > strong consensus (n_unique_pdb ≥ 2) > swinsite > p2rank > weak consensus`.

**Unified box**: 22.5 Å × 22.5 Å × 22.5 Å, grid spacing 0.375 Å.

### Stage 5: Docking (Multi-track)

3 parallel docking tracks, where Track 2+3 activate conditionally:

#### Track 1 (always): Cofolding-based docking — **6-source fan-out**

각 binding-site source 마다 vina/adg variant 를 별도로 돌림 (PxDock 만 priority-picked 단일 run). 한 source 가 pocket 을 못 찾으면 (e.g. SwinSite "no pockets", template hit 없는 타겟의 `template_consensus_*`) 해당 variant 만 `sys.exit(0)` 으로 clean skip.

| Tool | Variant 수 | Type | Output |
|------|-----------|------|--------|
| **Vina** | 6 (`vina_{cofolding,swinsite,p2rank,template_consensus_{1,2,3}}`) | Python API CPU | `outputs/{variant}/seed_<seed>/ligand_<lid>/docked.pdbqt` |
| **AutoDock-GPU** | 6 (`autodock_gpu_{... 6 sources}`) | CUDA GPU | `outputs/{variant}/seed_<seed>/ligand_<lid>/docking.dlg` |
| **Protenix-Dock** | 1 (priority-picked box) | CPU force field | `outputs/protenix_dock/poses_<lid>.sdf` |

ADG wrapper 는 추가로 **런타임에 ligand pdbqt 를 파싱**해서 `ligand_types` / grid map 을 동적 구성 — F/Cl/Br/P/I/Si 등 비표준 원자 타입도 자동 대응. 모든 도구는 `docking_prep_summary.json` 에서 receptor/ligand/box 를 runtime 에 읽음.

#### Track 2 (any template, **no MCS gate**): Template-based Box Docking

Template ligand 가 query 와 닮을 필요 없음 — pocket geometry 만 빌리는 것이 목적. `filtered_hits.tsv` 에서 `num_ligands > 0` 인 상위 3 template (sort key 가 evidence_breadth × similarity 이라 both-source / 높은 qtm 우선).

| Step | Script | Output |
|------|--------|--------|
| Template prep | `prepare_template_docking.py` | receptor PDB/PDBQT + ligand files + box from template ligand |
| Docking | Vina + ADG + PxDock | `outputs/template_docking/<pdb_id>/vina/`, `autodock_gpu/`, `protenix_dock/` |

#### Track 3 (per-template `mcs ≥ mcs_threshold`): lig-align (MCS-guided pose generation)

Uses template ligand bound pose as anchor for MCS-guided conformer generation with Vina scoring.

| Step | Description |
|------|-------------|
| MCS alignment | Align target molecule MCS atoms to template ligand bound pose |
| Conformer generation | 1000 conformers with MCS-constrained torsion sampling |
| Vina scoring | Score + rank poses using Vina energy function |
| Torsion optimization | Gradient-based refinement (optional, default on) |
| Output | Top-k poses as SDF → `outputs/template_docking/<pdb_id>/lig_align/` |

**Orchestrator**: `scripts/run_multi_track_docking.py` — auto-inserted in wrapper after Track 1 docking.

**Config**: `template_search_sequence.mcs_threshold` (default: `0.5`)

#### Multi-seed execution (all docking tools)

Cofolding and Track 1 docking run with multiple seeds for pose diversity:

| Stage | Multi-seed | Details |
|-------|-----------|---------|
| Boltz-2 / Boltz-2x | 5 seeds × 5 samples | 25 structures per model |
| Protenix v2 | 5 seeds × 5 samples | 25 structures |
| AlphaFold3 | native `num_seeds` × `num_diffusion_samples` | 25 structures |
| Vina | 5 seeds | via `DOCK_SEED` env var |
| AutoDock-GPU | 5 seeds | via `DOCK_SEED` env var |
| Protenix-Dock | 1 (single) | too expensive for multi-seed |

**Config**: `cofolding_seeds`, `docking_seeds` lists in `runner_config.yaml`

### Stage 5.5: Ion/Metal Placement (conditional)

Runs automatically when the input YAML contains an ion CCD (ZN, MG, CA, FE, etc.). Cofolding models position ions poorly; this stage collects positions from sequence-similar templates aligned to the cofolding best model.

**Tool**: `scripts/collect_template_ions.py`

```
For each template with target ion:
  1. gemmi CA superposition: template → cofolding best model
  2. Apply transformation to template ion coordinates
  3. Collect transformed positions in cofolding frame
Cluster positions by distance (default threshold 2.0Å)
Group by sequence identity: high (≥70%), medium (50-70%), low (30-50%)
Output: outputs/ion_placement/ion_placement_summary.json
```

### Stage 6: Post-analysis

**Tool**: `scripts/run_post_analysis.py` (runs on GPU node with `.venvs/pred`)

| Step | Tool | Input | Output |
|------|------|-------|--------|
| Pose staging | `find_ligand_files` | `seed_*/docked.pdbqt`, `seed_*/docking.dlg`, `protenix_dock/*_out.json` | `outputs/analysis/poses/{tool}_seed_{N}{ext}` (+ parallel `.sdf` via meeko) |
| PxDock JSON → SDF | RDKit + mapped SMILES | Protenix-Dock `ligand.xyz` | `outputs/protenix_dock/poses.sdf` (multi-record) |
| Affinity prediction | BA-Pred (GNN) | receptor PDB + list of staged SDFs | `ba_pred_<tool>.tsv` (pKd, kcal/mol) |
| Pose RMSD prediction | RMSD-Pred (GNN) | receptor PDB + list of staged SDFs | `rmsd_pred_<tool>.tsv` (pRMSD, >2Å prob) |

Notes:
- Pose files are staged under **unique stems** (e.g. `vina_seed_42.sdf`) because BA-Pred/RMSD-Pred name poses as `{basename}_{record_index}` and would otherwise collide across seeds.
- The **raw input ligand SDF is not run through BA-Pred** — its 3D conformer comes from RDKit's SMILES embedding and is not aligned with the receptor, which makes BA-Pred's 8 Å protein-shell extraction return an empty mol and crash. Docking outputs only.

### Stage 7: CASP17 LG Submission

**Tools**: `scripts/compute_submission_scores.py` + `scripts/make_casp_submission.py`

Aggregates scores from all sources and generates a valid CASP17 LG-format submission file:

```bash
python scripts/make_casp_submission.py \
    --run-dir experiments/runs/L2001_input \
    --target-id L2001 \
    --ligand-name 761 \
    --author <your-casp-code> \
    --method "Boltz-2x + Multi-track ensemble" \
    --include-affinity \
    --output experiments/submissions/L2001.lg
```

**Ensemble scoring**:
- **LSCORE** (per pose): `1 - P(RMSD > 2Å)` from RMSD-Pred
- **AFFNTY** (per submission): log-space ensemble of BA-Pred pKd + Boltz affinity
  - BA-Pred pKd → log10(Kd nM) = `9 - pKd`
  - Boltz `affinity_pred_value` → log10(Kd nM) = `value + 3`
  - Filter Boltz sources with `binder_prob < 0.5`
  - Median across each source → equal-weight average → `10^avg` = Kd (nM)

**Best pose selection**: Pick pose with highest LSCORE (lowest RMSD-Pred prob > 2Å).

**LG format output**:
```
PFRMAT LG
TARGET L2001
AUTHOR <code>
METHOD <description>
MODEL 1
PARENT <template_pdb_or_N/A>
ATOM ... (protein receptor, B-factor = pLDDT 0-100)
TER
LIGAND 001 761
LSCORE 0.850
<MDL V2000 block>
M  END
AFFNTY 12.345 aa   # optional, Kd in nM
END
```

## Configuration

### Hyperparameter Presets

| Preset | Boltz recycling/diffusion | Protenix cycle/step | AF3 recycles |
|--------|--------------------------|---------------------|-------------|
| `fast` | 1 / 1 | 4 / 75 | 3 |
| `balanced` | 3 / 1 | 10 / 200 | 10 |
| `quality` | 6 / 5 | 20 / 400 | 20 |

### Key Config Options

```yaml
preset: fast

# Multi-seed (default: 5 seeds each)
cofolding_seeds: [42, 101, 202, 303, 404]   # 25 structures per model
docking_seeds: [42, 101, 202, 303, 404]     # 5 seeds for Vina/ADG (PxDock: single)

boltz:
  enabled: true
  use_msa_server: true          # fetch MSA from ColabFold
  use_potentials: true          # Boltz-2x mode
  override: true

protenix:
  enabled: true
  model_name: protenix-v2       # or protenix_base_default_v1.0.0
  use_tfg_guidance: false

alphafold3:
  enabled: true
  model_dir: external/alphafold3/models
  run_data_pipeline: false
  run_inference: true

template_search_sequence:
  enabled: true
  database_path: data/search_dbs/sequence/rcsb_seqDB
  rcsb_dir: ~/DB/RCSB/raw/mmCIF_data         # CIF files for pocket extraction + template-based docking
  rcsb_db_path: ~/DB/RCSB/processed/rcsb_index.db  # ligand index
  mcs_threshold: 0.5                          # Track 3 (lig-align) activation threshold ONLY
  max_deposition_date: "2025-01-01"           # optional time-split filter (held-out benches)

template_search_structure:
  enabled: true                               # turn on foldseek for union template search
  database_path: data/search_dbs/structure/rcsb_structDB
  query_from_cofolding: true                  # query auto-resolved from cofold cif
  query_model_priority: [alphafold3, boltz, protenix]
  sensitivity: 9.5
  max_hits: 500                               # wide recall — foldseek has no native id/cov gate
  qtmscore_min: 0.5                           # same-fold floor for foldseek-only hits

vina:
  enabled: true            # fans out into 6 variants per binding-site source
autodock_gpu:
  enabled: true            # fans out into 6 variants per binding-site source
protenix_dock:
  enabled: true            # single run, priority-picked box (no fan-out)

slurm:
  partition: 6000ada
  gpus: 1
  mem: 64G
  time: 06:00:00
```

## Output Structure

```
experiments/runs/<target>/
├── inputs/
│   ├── boltz_input.yaml             # Boltz unified YAML (+ affinity properties)
│   ├── protenix_input.json          # Protenix JSON
│   ├── alphafold3_input.json        # AF3 JSON (+ MSA from Boltz bridge)
│   ├── docking/                     # Track 1 docking inputs
│   │   ├── docking_prep_summary.json  # auto-generated paths + box
│   │   ├── receptor.pdb / .pdbqt      # receptor (protonated, charged)
│   │   ├── receptor_protonated.pdb    # for Protenix-Dock
│   │   ├── ligand_L.sdf / .pdbqt      # ligand (3D from SMILES)
│   │   ├── p2rank/                     # P2Rank pocket predictions
│   │   └── swinsite/                   # SwinSite pocket predictions
│   └── template_docking/            # Track 2 docking inputs (any template — no MCS gate)
│       ├── template_docking_summary.json
│       └── template_<pdb_id>/
│           ├── receptor.pdb / .pdbqt
│           ├── template_ligand_<CCD>.sdf  # bound-pose ligand (Track 3 lig-align: MCS ≥ 0.5)
│           ├── ligand_L.sdf / .pdbqt       # target ligand (from SMILES)
│           └── docking_prep_summary.json
├── outputs/
│   ├── template_search_sequence/    # mmseqs_hits.tsv + filtered_hits.tsv (union)
│   ├── template_search_structure/   # foldseek_hits.tsv (query=best cofold cif)
│   ├── template_pockets/            # per-instance pocket centers + top-K cluster centroids
│   │   ├── template_pockets.json
│   │   └── template_pocket_clusters.json
│   ├── boltz2/                      # structure + confidence + affinity + MSA
│   ├── boltz2x/                     # structure + confidence + affinity (potentials)
│   ├── protenix/                    # structure + confidence
│   ├── alphafold3/                  # structure + confidence + ranking
│   ├── vina_{cofolding,swinsite,p2rank,template_consensus_{1,2,3}}/seed_*/
│   │       └── ligand_<lid>/docked.pdbqt   # Track 1: 6 variants × seeds
│   ├── autodock_gpu_{... 6 sources}/seed_*/
│   │       └── ligand_<lid>/docking.dlg    # Track 1: 6 variants × seeds
│   ├── protenix_dock/               # Track 1: poses_<lid>.sdf + *_out.json (single run)
│   ├── template_docking/            # Track 2+3 results
│   │   ├── multi_track_summary.json   # aggregated results across all templates
│   │   └── <pdb_id>/
│   │       ├── vina/                    # Track 2: template-based box docking
│   │       ├── autodock_gpu/            # Track 2: template-based box docking
│   │       ├── protenix_dock/           # Track 2: template-based box docking
│   │       └── lig_align/               # Track 3: MCS-guided poses (MCS ≥ 0.5 only)
│   ├── ion_placement/               # Ion positions (if ion CCD in input)
│   │   └── ion_placement_summary.json # clustered by confidence (70%/50%/30%)
│   └── analysis/                    # BA-Pred + RMSD-Pred TSVs
├── scripts/                         # generated runner scripts
├── run_manifest.json
└── wrapper_manifest.json
```

## CLI Commands

```bash
# Status
casp17-pl status

# Full pipeline
casp17-pl run-wrapper --input IN --config CFG --stages template-search-sequence cofolding docking --submit

# Individual stages
casp17-pl run-all --input IN --config CFG --submit                    # cofolding only
casp17-pl run-template-search-sequence --input IN --config CFG --submit
casp17-pl run-vina --input IN --config CFG --submit
casp17-pl run-protenix-dock --input IN --config CFG --submit

# Post-pipeline
python scripts/run_structure_search.py --run-dir DIR --foldseek-db DB --rcsb-db DB
python scripts/run_post_analysis.py --run-dir DIR --device cuda

# Validation
casp17-pl validate-run --input IN --config CFG --stages cofolding docking
```

## MCP Server (Claude Code Integration)

```bash
uv run casp17-pl-mcp
```

| Tool | Description |
|------|-------------|
| `status` | Check workspace environment |
| `validate` | Validate pipeline configuration |
| `prepare_cofolding` | Prepare a cofolding run |
| `list_presets` | List hyperparameter presets |
| `add_template` | Add PDB/CIF template to input YAML |
| `create_input` | Create input YAML from natural language |

## Tested End-to-End Results

Full pipeline measured on CASP16 L2000 (cathepsin target, split into L2001 / L2002 sub-runs) using the `casp_submission` config (5 cofolding seeds × 5 samples + 5 docking seeds). Single RTX 6000 Ada per SLURM job, partition `6000ada`. Track 2 / 3 were inactive because no template had `MCS ≥ 0.5`, and no ion entity was present.

| Stage | L2001 (23941) | L2002 (23942) | Status |
|-------|------:|------:|:--:|
| Template search (MMseqs2 + ligand filter) | 13s | 3s | ✓ |
| Boltz-2 (5 seeds × 5 samples) | 622s | 600s | ✓ |
| Boltz-2x (5 seeds × 5 samples) | 740s | 680s | ✓ |
| Protenix v2 (5 seeds × 5 samples) | 492s | 484s | ✓ |
| Bridge: Boltz MSA → AF3 | <1s | <1s | ✓ |
| AlphaFold3 (25 structures) | 353s | 393s | ✓ |
| Bridge: Docking prep (auto-select + SwinSite/P2Rank + SMILES→SDF/PDBQT) | <5s | <5s | ✓ |
| Vina (5 seeds) | 254s | 130s | ✓ |
| AutoDock-GPU (5 seeds) | 42s | 42s | ✓ |
| Protenix-Dock (1 run) | 459s | 513s | ✓ |
| Multi-track docking (Track 2 + 3) | skipped (no MCS≥0.5) | skipped | — |
| Ion/metal placement | skipped (no ion) | skipped | — |
| Post-analysis (BA-Pred + RMSD-Pred, 158 poses × 2) | ~1-2 min | ~1-2 min | ✓ |
| CASP17 LG submission | <5s | <5s | ✓ |
| **Total wall-clock** (`sacct` elapsed) | **00:53:26** | **00:52:08** | ✓ |

Best-pose selection (both runs) used RMSD-Pred `LSCORE = 1 − P(RMSD > 2Å)`:
- **L2001**: `vina/vina_seed_42_7`, pKd=4.88, pRMSD=1.04 Å, LSCORE=0.950, AFFNTY=709 nM
- **L2002**: `vina` pose, pRMSD=2.43 Å, LSCORE=0.851 (updated AFFNTY pending second rerun)

Full run artifacts + slurm logs are preserved under `experiments/runs/archive/2026-04-10_buggy_pipeline/` (first, buggy run) and `experiments/runs/L200{1,2}_input/` (second run with fixed ADG adapter + post-analysis). A detailed incident / fix log lives at `experiments/casp16_test/L2000/DEBUG_LOG.md`.

## Development

```bash
make sync       # uv sync --dev
make test       # pytest (42 tests)
make lint       # ruff check src/
```

## Project Layout

```
src/casp17/
├── models.py            # CommonInput parsing
├── configs.py           # RunnerConfig, presets, model configs
├── adapters.py          # Unified input → model-specific formats
├── orchestrator.py      # Pipeline preparation, script generation
├── script_builder.py    # Shell scripts with banners/timing/env setup + multi-track
├── cli.py               # 15+ subcommands
├── mcp_server.py        # MCP server (6 tools)
├── validation.py        # Pre-flight checks
├── template_filter.py   # Template hit → ligand + Tanimoto/MCS filtering via SQLite
├── ccd/                 # CCD classification (48,965 entries, 10 categories)
│   ├── ccd_categories.py
│   ├── ccd_lookup.py
│   └── ligands.py
├── io_utils.py
├── geometry.py          # Shared pose-eval geometry (Kabsch, RMSD, bond reassignment)
└── lg_format.py         # CASP LG multi-MODEL format parser

scripts/
├── install_external_models.sh     # Full installation + verify
├── build_autodock_gpu.sh          # AutoDock-GPU CUDA build
├── build_search_dbs.sh            # MMseqs2 + Foldseek DB build
├── bridge_boltz_msa_to_af3.py     # Boltz MSA CSV → AF3 A3M
├── prepare_docking_inputs.py      # Auto docking prep + binding site (Track 1)
├── run_template_filter.py         # Template hit filtering (Tanimoto + MCS)
├── prepare_template_docking.py    # Template CIF → receptor/ligand + bound-pose SDF
├── run_multi_track_docking.py     # Multi-track orchestrator (Track 2 + Track 3)
├── collect_template_ions.py       # Ion/metal placement via template alignment
├── run_structure_search.py        # Foldseek consensus across models
├── run_post_analysis.py           # BA-Pred + RMSD-Pred
├── compute_submission_scores.py   # Aggregate scores, ensemble affinity, best pose
└── make_casp_submission.py        # Generate CASP17 LG format submission file
```
