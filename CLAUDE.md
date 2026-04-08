# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CASP17 Protein-Ligand Hub — unified Python workspace for protein structure prediction and ligand docking pipelines. Orchestrates external ML models (Boltz, Protenix, AlphaFold3) and template search tools (MMseqs2, Foldseek) with final docking via AutoDock Vina.

## Commands

```bash
make sync          # uv sync --dev
make test          # pytest
make lint          # ruff check src/
make build         # uv build
uv run casp17-pl status                    # check CLI status
uv run casp17-pl-mcp                       # start MCP server
uv run pytest tests/test_cli.py -v         # verbose single test file
uv run pytest tests/test_cli.py -k "test_name" -v  # single test
```

## Architecture

### Unified Input → Model-Specific Adapters

All stages consume a single Boltz-style YAML input (`examples/unified_input.example.yaml`). Adapters translate this to model-specific formats.

**Data flow:** `unified YAML input` + `runner_config YAML` → `adapters.py` (model-specific conversion) → `orchestrator.py` (shell script + manifest generation) → SLURM/local execution

### Core Modules (`src/casp17_pl_hub/`)

| Module | Role |
|--------|------|
| `models.py` | Data models (`CommonInput`, `RunnerConfig`), schema validation, preset definitions (fast/balanced/quality) |
| `adapters.py` | `prepare_<model>()` functions converting unified input to model-specific formats (JSON, FASTA, config) |
| `orchestrator.py` | Pipeline preparation, `PreparedRun` manifest/script generation, stage dependency validation |
| `cli.py` | Argparse CLI with 12+ subcommands delegating to orchestrator/models |
| `mcp_server.py` | FastMCP server scaffold (placeholder) |
| `yaml_utils.py` | YAML serialization without external yaml library |

### Pipeline Stages

1. **template-search-sequence** — MMseqs2 sequence search against RCSB DB
2. **template-search-structure** — Foldseek structure search (can use cofolding output as query)
3. **cofolding** — Structure prediction via Boltz / Protenix / AlphaFold3
4. **docking** — AutoDock Vina molecular docking

Stage ordering matters: structure search must follow cofolding when `query_from_cofolding=true`.

### External Models

Under `external/` as git submodules, each with isolated venv in `.venvs/`:
- `external/boltz/` → `.venvs/boltz`
- `external/Protenix/` → `.venvs/protenix` (patched for CUDA LayerNorm fallback)
- `external/alphafold3/` → `.venvs/alphafold3`

### Hyperparameter Presets

`fast` / `balanced` / `quality` presets in `models.py` control recycling steps, diffusion samples, and sampling parameters across all models.

## Adding a New Model

1. Add config dataclass to `models.py`
2. Create `prepare_<model>()` in `adapters.py` returning `PreparedModelRun`
3. Add orchestration function in `orchestrator.py`
4. Add CLI subcommand in `cli.py`
5. Add tests in `tests/test_cli.py`

## Key Design Decisions

- **Single unified input format** (Boltz YAML) — adapters handle model-specific translation
- **Backend abstraction** — local and SLURM backends produce different shell scripts from the same config
- **Validation before execution** — `validate-run` checks paths and stage dependencies before `--submit`
- **Isolated external model venvs** — prevents dependency conflicts between Boltz/Protenix/AlphaFold3

## Cluster Notes

- Master node has no GPU — only safe for YAML parsing, manifest generation, import checks
- All inference/CUDA must run on compute nodes via SLURM
- Tests requiring GPU must be submitted via SLURM (see global CLAUDE.md)
