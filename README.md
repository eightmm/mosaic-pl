# CASP17 Protein-Ligand Hub

Unified Python workspace for CASP17 protein-ligand structure prediction and docking pipelines.
Orchestrates external ML models (Boltz2/2x, Protenix v2, AlphaFold3), template search (MMseqs2, Foldseek), binding site prediction (P2Rank, SwinSite), docking (Vina, AutoDock-GPU, Protenix-Dock), and post-analysis (BA-Pred, RMSD-Pred).

## Full Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: protein sequence + ligand SMILES (unified YAML)        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │  Stage 1: Sequence Search │
    │  MMseqs2 (488k seqs DB)   │──→ template hits
    │  + ligand filtering       │    (rcsb_index.db lookup)
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────────────────────────────────┐
    │  Stage 2: Co-folding (4 models parallel)              │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
    │  │ Boltz-2  │ │ Boltz-2x │ │ Protenix │ │   AF3    │ │
    │  │(no pot.) │ │(potent.) │ │   v1/v2  │ │  (JAX)   │ │
    │  │+affinity │ │+affinity │ │          │ │          │ │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
    │       │             │            │             │       │
    │       └──── BRIDGE: Boltz MSA → AF3 ──────────┘       │
    └──────────────────────┬────────────────────────────────-┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 3: Structure Search (Foldseek)       │
    │  Each model output → Foldseek (251k DB)     │
    │  + ligand filtering + cross-model consensus  │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 4: Docking Prep (automatic)          │
    │  • Auto-select best model (pLDDT score)     │
    │  • Binding site: SwinSite > P2Rank          │
    │  • SMILES → 3D SDF → PDBQT (RDKit + meeko) │
    │  • CIF → PDB → PDBQT (gemmi + pdb2pqr)     │
    │  • Box: 22.5Å, spacing 0.375Å              │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 5: Docking (3 tools)                 │
    │  ┌────────┐  ┌─────────────┐  ┌───────────┐│
    │  │  Vina  │  │ AutoDock-GPU│  │Protenix-  ││
    │  │(Py API)│  │  (CUDA)     │  │   Dock    ││
    │  │  ~3s   │  │   ~10s      │  │ ~5-30min  ││
    │  └───┬────┘  └──────┬──────┘  └─────┬─────┘│
    └──────┼──────────────┼───────────────┼──────-┘
           │              │               │
    ┌──────▼──────────────▼───────────────▼──────┐
    │  Stage 6: Post-analysis                    │
    │  • PDBQT/DLG → SDF (meeko mk_export.py)   │
    │  • BA-Pred: binding affinity (GNN)         │
    │  • RMSD-Pred: pose RMSD prediction (GNN)   │
    └──────────────────────┬─────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  OUTPUT: experiments/runs/<target>/          │
    │  ├── outputs/boltz2/     (CIF + affinity)   │
    │  ├── outputs/boltz2x/    (CIF + affinity)   │
    │  ├── outputs/protenix/   (CIF + confidence) │
    │  ├── outputs/alphafold3/ (CIF + confidence) │
    │  ├── outputs/vina/       (docked SDF/PDBQT) │
    │  ├── outputs/autodock_gpu/ (DLG + SDF)      │
    │  ├── outputs/protenix_dock/ (results JSON)  │
    │  ├── outputs/structure_search/ (consensus)  │
    │  └── outputs/analysis/   (BA/RMSD TSVs)     │
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

### Stage 1: Template Search (Sequence)

**Tool**: MMseqs2 (`easy-search` against preindexed RCSB sequence DB)
**Input**: Protein sequence from unified YAML
**Output**: `outputs/template_search_sequence/mmseqs_hits.tsv`
**Post-filter**: `template_filter.py` queries `rcsb_index.db` to find hits with drug-like ligands

```python
from casp17.template_filter import filter_hits_with_ligands
hits = filter_hits_with_ligands(hits_tsv, db_path)
# → hits sorted by ligand presence + sequence identity
```

### Stage 2: Co-folding

4 models run sequentially on GPU:

| Model | Output | Time | Features |
|-------|--------|------|----------|
| **Boltz-2** | `outputs/boltz2/` | ~50-140s | Structure + confidence + MSA + **affinity** |
| **Boltz-2x** | `outputs/boltz2x/` | ~50-140s | Structure + confidence + MSA + **affinity** (with potentials) |
| **Protenix** | `outputs/protenix/` | ~80-190s | Structure + confidence |
| **AlphaFold3** | `outputs/alphafold3/` | ~110-130s | Structure + confidence + ranking |

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

### Stage 3: Structure Search (Foldseek)

**Tool**: `scripts/run_structure_search.py`
**Input**: CIF structures from each cofolding model
**Output**: `outputs/structure_search/consensus_summary.json`

For each model:
1. Run Foldseek against RCSB structure DB
2. Collect all hits, lookup ligands via `rcsb_index.db`
3. Cross-model consensus: PDBs found by multiple models ranked highest
4. Ligand-containing hits highlighted

### Stage 4: Docking Preparation (Automatic Bridge)

**Tool**: `scripts/prepare_docking_inputs.py`
Runs automatically between cofolding and docking in the wrapper pipeline.

| Step | Tool | Output |
|------|------|--------|
| Auto-select best model | pLDDT comparison | Best CIF |
| CIF → PDB | gemmi | `receptor.pdb` |
| PDB → protonated PDB | pdb2pqr (AMBER) | `receptor_protonated.pdb` (HIS→HID/HIE/HIP) |
| PDB → PDBQT | pdb2pqr + AD4 typing | `receptor.pdbqt` (charges + atom types) |
| SMILES → 3D SDF | RDKit (ETKDG + MMFF) | `ligand_L.sdf` |
| SDF → PDBQT | meeko | `ligand_L.pdbqt` |
| Binding site | SwinSite > P2Rank | Box center + size |

**Binding site prediction priority:**
1. **SwinSite** (Swin-Unet ML, `.venvs/pred`) — preferred
2. **P2Rank** (surface-based, Java) — fallback
3. **Ligand coordinates** (RDKit 3D) — last resort

**Unified box**: 22.5Å × 22.5Å × 22.5Å, grid spacing 0.375Å

### Stage 5: Docking

3 docking tools run sequentially:

| Tool | Type | Time | Output |
|------|------|------|--------|
| **Vina** | Python API | ~3s | `outputs/vina/docked.pdbqt` |
| **AutoDock-GPU** | CUDA GPU | ~10s | `outputs/autodock_gpu/docking.dlg` |
| **Protenix-Dock** | CPU force field | ~5-30min | `outputs/protenix_dock/docking_results.json` |

All tools auto-detect receptor/ligand from `docking_prep_summary.json` at runtime.

### Stage 6: Post-analysis

**Tool**: `scripts/run_post_analysis.py` (runs on GPU node with `.venvs/pred`)

| Step | Tool | Input | Output |
|------|------|-------|--------|
| PDBQT → SDF | meeko `mk_export.py` | Vina/ADG output | `docked.sdf`, `docking.sdf` |
| Affinity prediction | BA-Pred (GNN) | receptor PDB + ligand SDF | `ba_pred_*.tsv` (pKd, kcal/mol) |
| Pose RMSD prediction | RMSD-Pred (GNN) | receptor PDB + ligand SDF | `rmsd_pred_*.tsv` (pRMSD, >2Å prob) |

### CCD Ligand Classification

Built-in module (`src/casp17/ccd/`) classifies 48,965 CCD codes into 10 categories:

| Category | is_candidate | Examples |
|----------|-------------|---------|
| small_molecule | ✓ | STI (imatinib), drug-like |
| cofactor | ✓ | ATP, NAD, FAD, HEM |
| metabolite | ✓ | cholesterol, bile acids |
| peptide_like | ✓ | short peptide inhibitors |
| nucleotide_like | ✓ | nucleoside analogs |
| ion | ✗ | ZN, MG, FE, CA |
| crystallization_aid | ✗ | GOL, EDO, PEG, SO4 |
| glycan | ✗ | NAG, MAN, GAL |
| membrane_lipid | ✗ | phospholipids, detergents |
| pigment | ✗ | carotenoids, chlorophylls |

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

vina:
  enabled: true
autodock_gpu:
  enabled: true
protenix_dock:
  enabled: true

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
│   ├── boltz_input.yaml           # Boltz unified YAML (+ affinity properties)
│   ├── protenix_input.json        # Protenix JSON
│   ├── alphafold3_input.json      # AF3 JSON (+ MSA from Boltz bridge)
│   └── docking/
│       ├── docking_prep_summary.json  # auto-generated paths + box
│       ├── receptor.pdb / .pdbqt      # receptor (protonated, charged)
│       ├── receptor_protonated.pdb    # for Protenix-Dock
│       ├── ligand_L.sdf / .pdbqt      # ligand (3D from SMILES)
│       ├── p2rank/                     # P2Rank pocket predictions
│       └── swinsite/                   # SwinSite pocket predictions
├── outputs/
│   ├── template_search_sequence/  # MMseqs2 hits
│   ├── boltz2/                    # structure + confidence + affinity + MSA
│   ├── boltz2x/                   # structure + confidence + affinity (potentials)
│   ├── protenix/                  # structure + confidence
│   ├── alphafold3/                # structure + confidence + ranking
│   ├── structure_search/          # Foldseek consensus across models
│   ├── vina/                      # docked.pdbqt + docked.sdf
│   ├── autodock_gpu/              # docking.dlg + docking.sdf
│   ├── protenix_dock/             # docking_results.json
│   └── analysis/                  # BA-Pred + RMSD-Pred TSVs
├── scripts/                       # generated runner scripts
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

Full pipeline test on RTX 6000 Ada (37 min total):

| Stage | Time | Status |
|-------|------|--------|
| MMseqs2 sequence search | 3s | ✓ |
| Boltz-2 + affinity | 143s | ✓ |
| Boltz-2x + affinity | 88s | ✓ |
| Protenix v1 | 133s | ✓ |
| Bridge: Boltz MSA → AF3 | <1s | ✓ |
| AlphaFold3 | 129s | ✓ |
| Docking prep (P2Rank + auto-select) | 3s | ✓ |
| Vina | 3s | ✓ |
| AutoDock-GPU | 10s | ✓ |
| Protenix-Dock | ~29min | ✓ |
| Foldseek structure search (4 models) | ~30s | ✓ |
| BA-Pred (3 docking tools) | ~10s | ✓ |
| RMSD-Pred (3 docking tools) | ~10s | ✓ |

## Development

```bash
make sync       # uv sync --dev
make test       # pytest (19 tests)
make lint       # ruff check src/
```

## Project Layout

```
src/casp17/
├── models.py            # CommonInput parsing
├── configs.py           # RunnerConfig, presets, model configs
├── adapters.py          # Unified input → model-specific formats
├── orchestrator.py      # Pipeline preparation, script generation
├── script_builder.py    # Shell scripts with banners/timing/env setup
├── cli.py               # 15+ subcommands
├── mcp_server.py        # MCP server (6 tools)
├── validation.py        # Pre-flight checks
├── template_filter.py   # Template hit → ligand filtering via SQLite
├── ccd/                 # CCD classification (48,965 entries, 10 categories)
│   ├── ccd_categories.py
│   ├── ccd_lookup.py
│   └── ligands.py
├── io_utils.py
└── yaml_utils.py

scripts/
├── install_external_models.sh     # Full installation + verify
├── build_autodock_gpu.sh          # AutoDock-GPU CUDA build
├── build_search_dbs.sh            # MMseqs2 + Foldseek DB build
├── bridge_boltz_msa_to_af3.py     # Boltz MSA CSV → AF3 A3M
├── prepare_docking_inputs.py      # Auto docking prep + binding site
├── run_structure_search.py        # Foldseek consensus across models
└── run_post_analysis.py           # BA-Pred + RMSD-Pred
```
