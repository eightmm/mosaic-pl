# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CASP17 Protein-Ligand Hub — unified pipeline for protein structure prediction and ligand docking. Orchestrates 4 co-folding models, 3 docking tools, 2 binding site predictors, 2 post-analysis GNNs, and template search with ligand filtering.

## Commands

```bash
make sync          # uv sync --dev
make test          # pytest (19 tests)
make lint          # ruff check src/
uv run casp17-pl status                    # check CLI status
uv run casp17-pl-mcp                       # start MCP server
bash scripts/install_external_models.sh --verify  # verify all tools
```

## Architecture

### Data Flow

```
unified YAML → adapters.py → model-specific inputs → orchestrator.py → SLURM scripts
                                                                          ↓
template search → cofolding → structure search → docking prep → docking → post-analysis
```

### Core Modules (`src/casp17/`)

| Module | Role |
|--------|------|
| `models.py` | CommonInput parsing, schema normalization |
| `configs.py` | RunnerConfig, 7 model configs, 3 presets (fast/balanced/quality) |
| `adapters.py` | `prepare_<model>()` → model-specific formats. Boltz returns list (boltz2 + boltz2x) |
| `orchestrator.py` | Pipeline preparation, PreparedRun, stage dependency handling |
| `script_builder.py` | Shell scripts: SBATCH headers, model banners with timing, CUDA env setup |
| `cli.py` | 15+ subcommands (prepare-*, run-*, validate-run, status) |
| `mcp_server.py` | FastMCP server: status, validate, prepare, presets, add_template, create_input |
| `validation.py` | Pre-flight checks. Docking pipeline skips receptor/ligand validation (bridge auto-generates) |
| `template_filter.py` | Template hit → ligand filtering via rcsb_index.db SQLite |
| `ccd/` | CCD classification: 48,965 entries, 10 categories (from mmcif-parser) |

### External Tools (isolated venvs)

| Tool | venv | Python | Key Deps |
|------|------|--------|----------|
| Boltz2/2x | `.venvs/boltz` | 3.12 | torch 2.11 + CUDA 13.0 + cuequivariance |
| Protenix v2 | `.venvs/protenix` | 3.12 | torch 2.7 + CUDA 12.6 + cuequivariance |
| AlphaFold3 | `.venvs/alphafold3` | 3.12 | JAX + dedicated CUDA LD_LIBRARY_PATH runner |
| Protenix-Dock + Vina | `.venvs/protenix-dock` | 3.11 (micromamba) | ambertools + tleap + pxdock |
| AutoDock-GPU + autogrid4 | `.local/bin/` | N/A | CUDA binary + C++ binary |
| SwinSite + BA-Pred + RMSD-Pred | `.venvs/pred` | 3.12 | torch 2.4 + dgl 2.4 + openbabel |
| P2Rank | `.local/bin/prank` | JDK 21 | Java wrapper |
| MMseqs2 + Foldseek | `.local/bin/` | N/A | Static binaries |

### Bridge Steps (auto-inserted in wrapper)

- **Boltz MSA → AF3**: `scripts/bridge_boltz_msa_to_af3.py` — CSV→A3M + patch AF3 JSON (pairedMsa="" + templates=[])
- **Docking Prep**: `scripts/prepare_docking_inputs.py` — auto model select + SwinSite/P2Rank + SMILES→SDF/PDBQT + CIF→PDB→PDBQT

### Pipeline Stages

1. **template-search-sequence** — MMseqs2 (488k seqs, preindexed)
2. **cofolding** — Boltz2 + Boltz2x + Protenix + AF3 (affinity auto-enabled)
3. **structure-search** — Foldseek on each cofolding output + cross-model consensus
4. **docking** — Vina + AutoDock-GPU + Protenix-Dock (all read docking_prep_summary.json at runtime)
5. **post-analysis** — BA-Pred + RMSD-Pred via mk_export.py SDF conversion

### Key Design Decisions

- **Boltz2 returns list**: `prepare_boltz()` returns `[boltz2, boltz2x]` — orchestrator unpacks with `*`
- **Docking validation relaxed**: `pipeline="docking"` skips receptor/ligand/box checks (bridge generates at runtime)
- **AF3 dedicated runner**: `run_alphafold3.sh` sets LD_LIBRARY_PATH for JAX CUDA (venv nvidia libs)
- **Protenix-Dock via micromamba**: needs ambertools (tleap) which is conda-only
- **Unified box**: 22.5Å × 22.5Å × 22.5Å, spacing 0.375Å across all docking tools
- **Binding site priority**: SwinSite (ML) > P2Rank (surface) > ligand coords (fallback)
- **Affinity auto-inject**: When ligand present, `properties.affinity` added to Boltz YAML
- **RCSB ligand index**: External SQLite DB (mmcif-parser maintained), queried for template filtering

### Cluster Notes

- Master node: no GPU — YAML parsing, manifest generation, validation only
- All inference/CUDA on compute nodes via SLURM
- Scripts auto-set: `module load cuda/12.8`, LD_LIBRARY_PATH (CUDA targets + venv nvidia + boost), `.local/bin` PATH
- Tests requiring GPU submitted via SLURM
