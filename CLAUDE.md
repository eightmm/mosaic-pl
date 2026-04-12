# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CASP17 Protein-Ligand Hub — unified pipeline for protein structure prediction and ligand docking. Orchestrates 4 co-folding models, 3 docking tools, 2 binding site predictors, 2 post-analysis GNNs, and template search with ligand filtering.

## Commands

```bash
make sync          # uv sync --dev
make test          # pytest (42 tests)
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
template search → template filter (MCS) → cofolding (5 seeds × 5 samples = 25 structs/model)
                                              ↓
                                   structure search → docking prep
                                              ↓
                             Track 1 docking (Vina/ADG 5 seeds, PxDock 1x)
                                              ↓
                              (MCS ≥ 0.5?) → Track 2 template-based box docking + Track 3 lig-align
                                              ↓
                              (ion in input?) → ion placement (template alignment + clustering)
                                              ↓
                                        post-analysis (BA-Pred/RMSD-Pred per pose)
                                              ↓
                              CASP17 LG submission (ensemble LSCORE + AFFNTY)
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
- **Template Filter**: `scripts/run_template_filter.py` — filters mmseqs hits via rcsb_index.db + Tanimoto/MCS scoring
- **Template Docking Prep**: `scripts/prepare_template_docking.py` — template CIF → receptor + ligand files + bound-pose SDF extraction
- **Multi-track Docking**: `scripts/run_multi_track_docking.py` — orchestrates Track 2 (template-based box docking) + Track 3 (lig-align)
- **Ion Placement**: `scripts/collect_template_ions.py` — aligns templates to cofolding model, collects ion positions, clusters by confidence
- **Submission Scoring**: `scripts/compute_submission_scores.py` — aggregates BA-Pred + RMSD-Pred + Boltz affinity, selects best pose, log-space ensemble Kd
- **CASP Submission**: `scripts/make_casp_submission.py` — generates CASP17 LG-format file (protein PDB + ligand MDL + LSCORE + optional AFFNTY)

### Pipeline Stages

1. **template-search-sequence** — MMseqs2 (488k seqs, preindexed) → template filter (ligand + MCS)
2. **cofolding** — Boltz2 + Boltz2x + Protenix + AF3 (affinity auto-enabled)
3. **structure-search** — Foldseek on each cofolding output + cross-model consensus
4. **docking** — Track 1: Vina + AutoDock-GPU + Protenix-Dock (all read docking_prep_summary.json at runtime)
5. **multi-track docking** (auto, MCS ≥ threshold) — Track 2: template-based box docking + Track 3: lig-align (MCS-guided)
6. **ion placement** (auto, if ion CCD in input) — template alignment → ion position clustering by confidence
7. **post-analysis** — BA-Pred + RMSD-Pred on staged multi-seed pose SDFs (input_sdf excluded — unaligned conformer crashes BA-Pred)
8. **CASP17 LG submission** — diversity-aware top-5 pose selection (greedy, ≥2Å pairwise RMSD), multi-MODEL LG file, log-space ensemble AFFNTY

### Key Design Decisions

- **Boltz2 returns list**: `prepare_boltz()` returns `[boltz2, boltz2x]` — orchestrator unpacks with `*`
- **Docking validation relaxed**: `pipeline="docking"` skips receptor/ligand/box checks (bridge generates at runtime)
- **AF3 dedicated runner**: `run_alphafold3.sh` sets LD_LIBRARY_PATH for JAX CUDA (venv nvidia libs)
- **Protenix-Dock via micromamba**: needs ambertools (tleap) which is conda-only
- **Unified box**: 22.5Å × 22.5Å × 22.5Å, spacing 0.375Å across all docking tools
- **Binding site priority**: SwinSite (ML) > P2Rank (surface) > cofolding ligand coords (fallback). Known weakness: P2Rank/SwinSite can miss the correct pocket entirely (CASP16 L2001: 37Å error), future improvement: use cofolding predicted ligand position as primary box center
- **Affinity auto-inject**: When ligand present, `properties.affinity` added to Boltz YAML
- **RCSB ligand index**: External SQLite DB (mmcif-parser maintained), queried for template filtering
- **Multi-track docking**: 3 tracks — Track 1 (always): cofolding→docking, Track 2 (MCS≥threshold): template-based box docking, Track 3 (MCS≥threshold): lig-align. Config: `template_search_sequence.mcs_threshold` (default 0.5)
- **Template ligand SDF**: `prepare_template_docking.py` extracts bound-pose ligand from template CIF for lig-align reference
- **Multi-seed**: `cofolding_seeds` (default 5) × `diffusion_samples` (default 5) = 25 structures per cofolding model. `docking_seeds` (default 5) for Vina/ADG (PxDock single run). Config: `cofolding_seeds`, `docking_seeds` lists in RunnerConfig
- **AF3 templates field**: Adapter always adds `templates=[]` to AF3 protein blocks (required by AF3 schema even when empty)
- **Submission top-5**: Diversity-aware greedy selection — pick by LSCORE desc, accept next only if heavy-atom RMSD ≥ 2Å to all previously selected. Emits MODEL 1..5 blocks in LG file. AFFNTY from log-space ensemble of BA-Pred pKd + Boltz affinity (filtered by binder_prob ≥ 0.5)
- **Model auto-select**: `prepare_docking_inputs.py` picks cofolding model by mean pLDDT normalised to [0,100] scale (Boltz npz ×100, Protenix scalar, AF3 atom_plddts mean). Critical fix: previously compared Boltz [0,1] vs AF3 [0,100] raw → always picked AF3 even when Boltz was superior
- **Post-analysis pose staging**: Multi-seed poses staged under `outputs/analysis/poses/{tool}_seed_{N}.sdf` with unique stems to avoid BA-Pred/RMSD-Pred name collisions. PxDock JSON→SDF via `_pxdock_json_to_sdf`. Input SDF excluded (unaligned conformer)
- **ADG dynamic GPF**: AutoDock-GPU wrapper parses ligand PDBQT at runtime for atom types (F/Cl/Br/P/I/Si support), reads box center from `docking_prep_summary.json` (not compile-time defaults)

### Cluster Notes

- Master node: no GPU — YAML parsing, manifest generation, validation only
- All inference/CUDA on compute nodes via SLURM
- Scripts auto-set: `module load cuda/12.8`, LD_LIBRARY_PATH (CUDA targets + venv nvidia + boost), `.local/bin` PATH
- Tests requiring GPU submitted via SLURM
