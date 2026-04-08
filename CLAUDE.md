# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CASP17 Protein-Ligand Hub — unified Python workspace for protein structure prediction and ligand docking pipelines. Orchestrates external ML models (Boltz2, Protenix v2, AlphaFold3) and template search tools (MMseqs2, Foldseek) with docking via AutoDock Vina, AutoDock-GPU, and Protenix-Dock.

## Commands

```bash
make sync          # uv sync --dev
make test          # pytest (19 tests)
make lint          # ruff check src/
make build         # uv build
uv run casp17-pl status                    # check CLI status
uv run casp17-pl-mcp                       # start MCP server
uv run pytest tests/test_cli.py -v         # verbose single test file
uv run pytest tests/test_cli.py -k "test_name" -v  # single test
```

## Architecture

### Unified Input -> Model-Specific Adapters

All stages consume a single Boltz-style YAML input (`examples/unified_input.example.yaml`). Adapters translate this to model-specific formats.

**Data flow:** `unified YAML input` + `runner_config YAML` -> `adapters.py` (model-specific conversion) -> `orchestrator.py` (shell script + manifest generation) -> SLURM/local execution

### Core Modules (`src/casp17_pl_hub/`)

| Module | Role |
|--------|------|
| `models.py` | `CommonInput` parsing, schema normalization |
| `configs.py` | `RunnerConfig` with all model configs, presets (fast/balanced/quality), validation |
| `adapters.py` | `prepare_<model>()` functions: unified input -> model-specific formats (YAML, JSON, config, Python scripts) |
| `orchestrator.py` | Pipeline preparation, `PreparedRun` manifest/script generation, stage dependency handling |
| `script_builder.py` | Shell script generation: SBATCH headers, model banners with timing, venv PATH, bridge steps |
| `cli.py` | Argparse CLI with 15+ subcommands delegating to orchestrator |
| `mcp_server.py` | FastMCP server with 6 tools (status, validate, prepare, presets, add_template, create_input) |
| `validation.py` | Pre-flight validation: paths, stage ordering, entity IDs, SLURM config |
| `io_utils.py` | YAML/JSON file I/O |
| `yaml_utils.py` | YAML serialization without external yaml library |

### Pipeline Stages

1. **template-search-sequence** — MMseqs2 sequence search against RCSB DB
2. **template-search-structure** — Foldseek structure search (can use cofolding output as query)
3. **cofolding** — Structure prediction via Boltz2 / Protenix v2 / AlphaFold3
4. **docking** — AutoDock Vina / AutoDock-GPU / Protenix-Dock

### Automatic Bridge Steps (in wrapper pipeline)

- **Boltz MSA -> AF3**: `scripts/bridge_boltz_msa_to_af3.py` converts Boltz MSA CSV to A3M, patches AF3 input JSON
- **Docking Input Prep**: `scripts/prepare_docking_inputs.py` converts SMILES -> SDF -> PDBQT (meeko), CIF -> PDB -> PDBQT (gemmi + pdb2pqr), auto-computes docking box
- Docking adapters auto-detect prep output via `docking_prep_summary.json`

### External Models

Under `external/` (git-cloned, not committed), each with isolated venv in `.venvs/`:

| Model | Source | venv | Python | Key Deps |
|-------|--------|------|--------|----------|
| Boltz2 | `external/boltz/` | `.venvs/boltz` | 3.12 | torch 2.11 + CUDA 13.0 + cuequivariance |
| Protenix v2 | `external/Protenix/` | `.venvs/protenix` | 3.12 | torch 2.7 + CUDA 12.6 + cuequivariance |
| AlphaFold3 | `external/alphafold3/` | `.venvs/alphafold3` | 3.12 | JAX |
| Protenix-Dock | `external/Protenix-Dock/` | `.venvs/protenix-dock` | 3.11 | torch 2.1 + pxdock C++ engine |
| Vina | (pip in protenix-dock venv) | `.venvs/protenix-dock` | 3.11 | pre-built wheel |
| AutoDock-GPU | `external/AutoDock-GPU/` | `.local/bin/` | N/A | CUDA binary |
| autogrid4 | `external/AutoGrid/` | `.local/bin/` | N/A | C++ binary |

Venvs are isolated because torch versions conflict (2.11 vs 2.7 vs 2.1 vs JAX).

### Hyperparameter Presets

`fast` / `balanced` / `quality` presets in `configs.py` control recycling steps, diffusion samples, and sampling parameters across all models.

## Adding a New Model

1. Add config dataclass to `configs.py` with `from_dict()` classmethod
2. Add field to `RunnerConfig` and its `from_dict()`
3. Create `prepare_<model>()` in `adapters.py` returning `PreparedModelRun`
4. Add to `prepare_docking_run()` or `prepare_run()` in `orchestrator.py`
5. Add CLI subcommand in `cli.py`
6. Add tests in `tests/test_cli.py`

## Key Design Decisions

- **Single unified input format** (Boltz YAML) — adapters handle model-specific translation
- **Backend abstraction** — local and SLURM backends produce different shell scripts from the same config
- **Validation before execution** — `validate-run` checks paths and stage dependencies before `--submit`
- **Isolated external model venvs** — prevents dependency conflicts
- **Auto docking prep** — SMILES -> SDF/PDBQT and CIF -> PDB/PDBQT handled automatically in wrapper pipeline
- **Template auto-injection** — PDB/CIF templates automatically added to Boltz YAML and Protenix JSON
- **Output to experiments/** — `experiments/runs/` for results, `experiments/logs/` for SLURM logs
- **Generated scripts include banners** — `[1/N] MODEL_NAME` with per-model timing for easy log reading

## Cluster Notes

- Master node has no GPU — only safe for YAML parsing, manifest generation, import checks
- All inference/CUDA must run on compute nodes via SLURM
- Scripts auto-run `module load cuda/12.8` and add venv bins + `.local/bin` to PATH
- Tests requiring GPU must be submitted via SLURM (see global CLAUDE.md)
