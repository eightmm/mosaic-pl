# CASP17 Protein-Ligand Hub

Unified Python workspace for CASP17 protein-ligand structure prediction and docking pipelines.
Orchestrates external ML models (Boltz2, Protenix v2, AlphaFold3) and template search tools (MMseqs2, Foldseek) with docking via AutoDock Vina, AutoDock-GPU, and Protenix-Dock.

## Setup

### Prerequisites

```bash
# System packages (Ubuntu)
sudo apt-get install -y libboost-all-dev autoconf automake libtool

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# 1. Clone and enter project
git clone <repo-url> CASP17 && cd CASP17

# 2. Install all external models + tools
bash scripts/install_external_models.sh

# 3. Build AutoDock-GPU on a GPU node
srun --partition=6000ada --gres=gpu:1 bash scripts/build_autodock_gpu.sh

# 4. Verify everything
bash scripts/install_external_models.sh --verify

# 5. Check workspace status
uv run casp17-pl status
```

### What Gets Installed

| Tool | venv / Location | Version |
|------|----------------|---------|
| Boltz2 | `.venvs/boltz` (Python 3.12) | latest + cuequivariance |
| Protenix v2 | `.venvs/protenix` (Python 3.12) | v2.0.0 + TFG guidance |
| AlphaFold3 | `.venvs/alphafold3` (Python 3.12) | latest |
| Protenix-Dock | `.venvs/protenix-dock` (Python 3.11) | v0.0.1 |
| AutoDock Vina | `.venvs/protenix-dock` (shared) | 1.2.7 |
| AutoDock-GPU | `.local/bin/autodock_gpu_128wi` | v1.6 (CUDA) |
| autogrid4 | `.local/bin/autogrid4` | 4.2.8 |
| meeko | `.venvs/protenix-dock` (shared) | ligand PDBQT prep |
| gemmi | `.venvs/protenix-dock` (shared) | CIF/PDB conversion |
| pdb2pqr | `.venvs/protenix-dock` (shared) | receptor protonation |

## Quick Start

```bash
# 1. Create input YAML (protein sequence + ligand SMILES)
cat > my_target.yaml << 'EOF'
version: 1
seed: 42
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVT...
      msa: empty
  - ligand:
      id: L
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
EOF

# 2. Run cofolding (Boltz + Protenix)
uv run casp17-pl run-all \
  --input my_target.yaml \
  --config examples/runner_config.example.yaml \
  --submit

# 3. Or run full pipeline (cofolding + auto docking prep + docking)
uv run casp17-pl run-wrapper \
  --input my_target.yaml \
  --config my_config.yaml \
  --stages cofolding docking \
  --submit
```

## Pipeline Stages

```
unified YAML (protein + ligand SMILES)
  |
  v
[template-search-sequence]  MMseqs2 sequence search
[template-search-structure] Foldseek structure search
  |
  v
[cofolding]  Boltz2 / Protenix v2 / AlphaFold3
  |
  |--- [BRIDGE: Boltz MSA -> AF3]  (MSA reuse)
  |
  v
[BRIDGE: Docking Input Prep]
  |  SMILES -> 3D SDF -> PDBQT (RDKit + meeko)
  |  CIF -> PDB -> PDBQT (gemmi + pdb2pqr)
  |  Docking box auto-computed from ligand
  |
  v
[docking]  AutoDock Vina / AutoDock-GPU / Protenix-Dock
  |
  v
experiments/runs/<target>/outputs/
```

### Stage Details

| Stage | Tools | GPU | Description |
|-------|-------|-----|-------------|
| `template-search-sequence` | MMseqs2 | No | Sequence homology search against RCSB DB |
| `template-search-structure` | Foldseek | No | Structure similarity search, can use cofolding output |
| `cofolding` | Boltz2, Protenix v2, AF3 | Yes | Protein-ligand complex structure prediction |
| `docking` | Vina, AutoDock-GPU, Protenix-Dock | GPU optional | Molecular docking with auto-prepared inputs |

### Automatic Bridge Steps

The wrapper pipeline automatically inserts bridge steps:

- **Boltz MSA -> AF3**: When both Boltz (with MSA server) and AF3 are enabled, Boltz's MSA CSV is converted to A3M and injected into AF3 input.
- **Docking Input Prep**: Between cofolding and docking, SMILES are converted to SDF/PDBQT, receptor CIF is converted to PDB/PDBQT with protonation and charges, and the docking box is auto-computed.

## Input Format

Single unified Boltz-style YAML:

```yaml
version: 1
seed: 42
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLQ...
      msa: empty                  # "empty", path to .a3m, or omit for MSA server
  - ligand:
      id: L
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
templates:                         # optional structural templates
  - path: /path/to/template.pdb
    ids: [A]
constraints:                       # optional
  - bond:
      atom1: [A, 1, CA]
      atom2: [L, 1, C1]
properties:                        # optional
  - affinity:
      binder: L
```

Adapters automatically convert this to model-specific formats (Boltz YAML, Protenix JSON, AlphaFold3 JSON).

## Runner Config

Controls all model hyperparameters, tool paths, and SLURM settings.

### Hyperparameter Presets

| Preset | Boltz recycling/diffusion | Protenix cycle/step/sample | AF3 recycles/samples |
|--------|--------------------------|---------------------------|---------------------|
| `fast` | 1 / 1 | 4 / 75 / 1 | 3 / 1 |
| `balanced` | 3 / 1 | 10 / 200 / 5 | 10 / 5 |
| `quality` | 6 / 5 | 20 / 400 / 8 | 20 / 10 |

```bash
uv run casp17-pl write-example-config --preset quality
```

### Key Config Sections

```yaml
preset: balanced

boltz:
  enabled: true
  use_msa_server: false        # true to fetch MSA from ColabFold
  override: true               # overwrite existing results

protenix:
  enabled: true
  model_name: protenix-v2      # new v2 model
  use_tfg_guidance: false      # Training-Free Guidance
  trimul_kernel: cuequivariance
  triatt_kernel: cuequivariance

alphafold3:
  enabled: false               # requires model weights
  model_dir: /path/to/models

vina:
  enabled: true                # auto-detects receptor/ligand from prep bridge

autodock_gpu:
  enabled: true                # requires autogrid4 + GPU
  nrun: 100
  autostop: true

protenix_dock:
  enabled: true                # auto-detects receptor/ligand from prep bridge
  cache_map_spacing: 0.175

slurm:
  partition: 6000ada
  gpus: 1
  mem: 64G
  time: "04:00:00"
```

When using the wrapper pipeline (cofolding + docking), docking tools **auto-detect** receptor/ligand files and box parameters from the docking prep bridge output. No need to manually specify paths.

## CLI Commands

```bash
# Status & config
casp17-pl status
casp17-pl write-example-input
casp17-pl write-example-config --preset balanced

# Prepare (generate scripts without submitting)
casp17-pl prepare-run --input IN --config CFG
casp17-pl prepare-wrapper --input IN --config CFG --stages S1 S2

# Run (prepare + submit)
casp17-pl run-all --input IN --config CFG --submit
casp17-pl run-wrapper --input IN --config CFG --stages cofolding docking --submit
casp17-pl run-vina --input IN --config CFG --submit
casp17-pl run-protenix-dock --input IN --config CFG --submit

# Validate
casp17-pl validate-run --input IN --config CFG --stages cofolding docking
```

## Output Structure

```
experiments/runs/<target>/
  inputs/
    boltz_input.yaml
    protenix_input.json
    alphafold3_input.json
    vina_config.txt
    docking/
      docking_prep_summary.json
      receptor.pdb
      receptor.pdbqt
      ligand_L.sdf
      ligand_L.pdbqt
    autodock_gpu_grid/
      receptor.gpf
      receptor.maps.fld
  outputs/
    boltz/          -> .cif structures, PAE/PDE/pLDDT
    protenix/       -> .cif structures, confidence scores
    alphafold3/     -> .cif structures, confidence .json
    vina/           -> docked.pdbqt, vina.log
    autodock_gpu/   -> docking.dlg, poses
    protenix_dock/  -> docking_results.json
  scripts/
    run_structure.sbatch.sh
    run_docking.sbatch.sh
    run_autodock_gpu.sh
    run_protenix_dock.py
    run_wrapper.sbatch.sh
  run_manifest.json

experiments/logs/
  slurm-<jobid>.out
  slurm-<jobid>.err
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
| `create_input` | Create input YAML from parameters (sequence, SMILES, template) |

## Template Support

Add structural templates via input YAML or MCP:

```yaml
# In unified input YAML
templates:
  - path: /path/to/template.pdb
    ids: [A]  # optional chain filter
```

Templates are automatically passed to:
- **Boltz**: native YAML template section
- **Protenix**: `templatesPath` in proteinChain + `--use_template true`

## Example Workflows

### Cofolding only (Boltz + Protenix v2)

```bash
uv run casp17-pl run-all \
  --input my_target.yaml \
  --config my_config.yaml \
  --submit
```

### Full pipeline (cofolding + docking with auto-prep)

```bash
uv run casp17-pl run-wrapper \
  --input my_target.yaml \
  --config my_config.yaml \
  --stages cofolding docking \
  --submit
```

### Boltz MSA -> AF3 reuse

Enable `use_msa_server: true` in Boltz config + enable AF3. The bridge automatically converts Boltz MSA to A3M and patches AF3 input.

### Template-guided prediction

```yaml
# my_target.yaml
sequences:
  - protein:
      id: A
      sequence: MMAS...
      msa: empty
templates:
  - path: ./templates/7qtb_A.pdb
    ids: [A]
```

## Cluster Notes

- **Master node has no GPU** -- only safe for YAML parsing, manifest generation, validation
- **All inference/CUDA must run on compute nodes** via SLURM
- **Isolated venvs** prevent torch version conflicts (Boltz: 2.11+cu130, Protenix: 2.7+cu126, AF3: JAX)
- Scripts auto-run `module load cuda/12.8` and add venv bins to PATH

## Development

```bash
make sync       # uv sync --dev
make test       # pytest (19 tests)
make lint       # ruff check src/
```

### Adding a New Model

1. Add config dataclass to `configs.py`
2. Create `prepare_<model>()` in `adapters.py` returning `PreparedModelRun`
3. Add to `prepare_docking_run()` in `orchestrator.py`
4. Add CLI subcommand in `cli.py`
5. Add tests in `tests/test_cli.py`

### Project Layout

```
src/casp17_pl_hub/
  models.py          # CommonInput parsing, validation
  configs.py         # RunnerConfig, model configs, presets
  adapters.py        # Unified input -> model-specific formats
  orchestrator.py    # Pipeline preparation, script generation
  script_builder.py  # Shell script generation with banners/timing
  cli.py             # CLI (15+ subcommands)
  mcp_server.py      # MCP server (6 tools)
  validation.py      # Pre-flight checks
  io_utils.py        # File I/O helpers
  yaml_utils.py      # YAML serialization

scripts/
  install_external_models.sh    # Full installation (clone + venv + build)
  build_autodock_gpu.sh         # AutoDock-GPU CUDA build (GPU node)
  bridge_boltz_msa_to_af3.py    # Boltz MSA CSV -> AF3 A3M conversion
  prepare_docking_inputs.py     # SMILES -> SDF/PDBQT, CIF -> PDB/PDBQT
  build_template_search_dbs.sh  # MMseqs2/Foldseek DB construction
```
