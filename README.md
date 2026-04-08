# CASP17 Protein-Ligand Hub

Unified Python workspace for CASP17 protein-ligand structure prediction and docking pipelines.
Orchestrates external ML models (Boltz, Protenix, AlphaFold3) and template search tools (MMseqs2, Foldseek) with final docking via AutoDock Vina or Protenix-Dock.

## Setup

```bash
# Install dependencies
uv sync --dev

# Install external models (Boltz, Protenix, AlphaFold3) into isolated venvs
bash scripts/install_external_models.sh

# Verify installation
bash scripts/verify_external_models.sh

# Check workspace status
uv run casp17-pl status
```

`status` shows: Python version, external model venv status, SLURM partition info, available tools (mmseqs, foldseek, vina).

## Quick Start

```bash
# 1. Generate example input and config files
uv run casp17-pl write-example-input
uv run casp17-pl write-example-config --preset balanced

# 2. Edit the generated files for your target
#    - examples/unified_input.example.yaml  (protein sequence, ligand SMILES)
#    - examples/runner_config.example.yaml   (model settings, paths, SLURM config)

# 3. Validate configuration before submission
uv run casp17-pl validate-run \
  --input examples/unified_input.example.yaml \
  --config examples/runner_config.example.yaml \
  --backend slurm

# 4. Prepare and submit cofolding job
uv run casp17-pl run-all \
  --input examples/unified_input.example.yaml \
  --config examples/runner_config.example.yaml \
  --backend slurm \
  --submit
```

## Pipeline Stages

4 modular stages that can be combined in any order (respecting dependencies):

| Stage | Tool | Purpose |
|-------|------|---------|
| `template-search-sequence` | MMseqs2 | Sequence homology search against RCSB DB |
| `template-search-structure` | Foldseek | Structure similarity search against RCSB DB |
| `cofolding` | Boltz / Protenix / AlphaFold3 | Protein-ligand structure prediction |
| `docking` | AutoDock Vina / Protenix-Dock | Molecular docking |

### Stage Dependencies

- `template-search-structure` can reuse cofolding outputs as query (`query_from_cofolding: true`), in which case it must run AFTER `cofolding`
- `docking` requires prebuilt receptor/ligand files (PDBQT for Vina, PDB+SDF for Protenix-Dock)

## Input Format

Single unified input in Boltz YAML format:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLQ...
      msa: empty                  # "empty" or path to .a3m file
  - ligand:
      id: L
      smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
constraints:
  - bond:
      atom1: [A, 1, CA]
      atom2: [L, 1, C1]
properties:
  - affinity:
      binder: L
seed: 101
```

Adapters automatically convert this to model-specific formats (Boltz YAML, Protenix JSON, AlphaFold3 JSON, Vina config, etc.).

## Runner Config

`runner_config.yaml` controls all model hyperparameters, tool paths, and SLURM settings.

### Hyperparameter Presets

| Preset | Boltz recycling | Boltz diffusion | Protenix cycle/step | AF3 recycles | Use Case |
|--------|----------------|-----------------|---------------------|--------------|----------|
| `fast` | 1 | 1 | 4/75 | 3 | Quick test runs |
| `balanced` | 3 | 1 | 10/200 | 10 | Default production |
| `quality` | 6 | 5 | 20/400 | 20 | Maximum accuracy |

Generate a full config template with:

```bash
uv run casp17-pl write-example-config --preset quality
```

### Key Config Sections

```yaml
preset: balanced              # fast | balanced | quality

boltz:
  enabled: true
  binary: .venvs/boltz/bin/boltz
  recycling_steps: 3
  sampling_steps: 200
  diffusion_samples: 1

protenix:
  enabled: true
  binary: .venvs/protenix/bin/protenix
  cycle: 10
  step: 200
  sample: 5

alphafold3:
  enabled: true
  python_bin: .venvs/alphafold3/bin/python
  script: external/alphafold3/run_alphafold.py
  model_dir: /path/to/alphafold3/models
  run_inference: true

vina:
  enabled: false
  receptor_pdbqt: /path/to/receptor.pdbqt
  ligand_pdbqt: /path/to/ligand.pdbqt
  center_x: 0.0
  center_y: 0.0
  center_z: 0.0
  size_x: 20.0
  size_y: 20.0
  size_z: 20.0

protenix_dock:
  enabled: false
  receptor_pdb: /path/to/receptor.pdb
  ligand_sdf: /path/to/ligand.sdf

slurm:
  partition: 6000ada
  gpus: 1
  cpus_per_task: 8
  mem: 64G
  time: "04:00:00"
```

## CLI Commands

### Status & Configuration

```bash
casp17-pl status                                    # Check environment
casp17-pl write-example-input                       # Generate input template
casp17-pl write-example-config --preset balanced    # Generate config template
```

### Prepare (generate artifacts without execution)

```bash
casp17-pl prepare-run --input IN --config CFG                        # Cofolding
casp17-pl prepare-template-search-sequence --input IN --config CFG   # MMseqs2
casp17-pl prepare-template-search-structure --input IN --config CFG  # Foldseek
casp17-pl prepare-vina --input IN --config CFG                       # Vina docking
casp17-pl prepare-protenix-dock --input IN --config CFG              # Protenix-Dock
casp17-pl prepare-wrapper --input IN --config CFG --stages S1 S2     # Multi-stage
```

### Run (prepare + optional submit)

```bash
casp17-pl run-all --input IN --config CFG --submit                   # Cofolding
casp17-pl run-template-search-sequence --input IN --config CFG --submit
casp17-pl run-template-search-structure --input IN --config CFG --submit
casp17-pl run-vina --input IN --config CFG --submit
casp17-pl run-protenix-dock --input IN --config CFG --submit
casp17-pl run-wrapper --input IN --config CFG --stages S1 S2 --submit
```

### Validate

```bash
casp17-pl validate-run --input IN --config CFG --stages cofolding docking
```

Checks: binary paths, file existence, stage ordering, entity IDs, bond constraints, SLURM config.

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Unified input YAML |
| `--config` | (required) | Runner config YAML |
| `--output-root` | `runs/` | Output directory |
| `--backend` | `slurm` | `slurm` or `local` |
| `--submit` | false | Execute generated script |
| `--stages` | auto | Stage subset for wrapper/validate |

## Output Structure

Each run generates artifacts in `runs/<target>/`:

```
runs/target-name/
├── inputs/
│   ├── boltz_input.yaml
│   ├── protenix_input.json
│   ├── alphafold3_input.json
│   └── vina_config.txt
├── outputs/
│   ├── boltz/
│   ├── protenix/
│   ├── alphafold3/
│   └── vina/
├── scripts/
│   ├── run_structure.sbatch.sh
│   └── run_wrapper.sbatch.sh
└── run_manifest.json
```

## Example Workflows

### Cofolding only (Boltz + Protenix)

```bash
uv run casp17-pl run-all \
  --input my_target.yaml \
  --config my_config.yaml \
  --submit
```

### Full pipeline (template search + cofolding + docking)

```bash
uv run casp17-pl run-wrapper \
  --input my_target.yaml \
  --config my_config.yaml \
  --stages template-search-sequence cofolding docking \
  --submit
```

### Structure search using cofolding outputs

Set `query_from_cofolding: true` in config, then:

```bash
uv run casp17-pl run-wrapper \
  --input my_target.yaml \
  --config my_config.yaml \
  --stages cofolding template-search-structure \
  --submit
```

### Prepare only (dry run)

```bash
# Generate scripts without submitting
uv run casp17-pl prepare-run --input my_target.yaml --config my_config.yaml

# Inspect generated scripts
cat runs/target-name/scripts/run_structure.sbatch.sh

# Submit manually
sbatch runs/target-name/scripts/run_structure.sbatch.sh
```

## MCP Server

For Claude Code integration, an MCP server exposes 4 tools:

```bash
uv run casp17-pl-mcp
```

| Tool | Description |
|------|-------------|
| `status` | Check workspace environment |
| `validate` | Validate pipeline configuration |
| `prepare_cofolding` | Prepare a cofolding run |
| `list_presets` | List available hyperparameter presets |

## Template Search Databases

Build local MMseqs2/Foldseek databases from RCSB data:

```bash
bash scripts/build_template_search_dbs.sh \
  --fasta /path/to/rcsb_pdb_seqres.txt \
  --structures /path/to/rcsb_structures/ \
  --output /path/to/search_dbs/
```

See [docs/template_search.md](docs/template_search.md) for details.

## Cluster Notes

- **Master node has no GPU** -- only safe for YAML parsing, manifest generation, validation
- **All inference/CUDA must run on compute nodes** via SLURM (`--submit` or manual `sbatch`)
- **Isolated venvs** prevent dependency conflicts between models (`.venvs/boltz/`, `.venvs/protenix/`, `.venvs/alphafold3/`)

## Development

```bash
make sync       # uv sync --dev
make test       # pytest
make lint       # ruff check src/
```

### Adding a New Model

1. Add config dataclass to `models.py`
2. Create `prepare_<model>()` in `adapters.py`
3. Add orchestration function in `orchestrator.py`
4. Add CLI subcommand in `cli.py`
5. Add tests in `tests/test_cli.py`

### Project Layout

```
src/casp17_pl_hub/
├── models.py        # Data models, validation, presets
├── adapters.py      # Unified input -> model-specific format
├── orchestrator.py  # Pipeline preparation, script generation
├── cli.py           # CLI entry point (12+ subcommands)
├── mcp_server.py    # MCP server (status, validate, prepare, presets)
└── yaml_utils.py    # YAML serialization utilities
```
