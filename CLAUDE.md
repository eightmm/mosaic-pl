# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CASP17 Protein-Ligand Hub — unified pipeline for protein structure prediction and ligand docking. Orchestrates 4 co-folding models, 3 docking tools, 2 binding site predictors + template-consensus pockets, 2 post-analysis GNNs, and union (mmseqs + foldseek) template search with ligand filtering.

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
       template-search-sequence (MMseqs2)         cofolding (5 seeds × 5 samples per model)
                  │                                          │
                  └─────────────────────┐                    │
                                        ↓                    ↓
                            template-search-structure (Foldseek, query=cofold cif)
                                        ↓
                  union filter (mmseqs ∪ foldseek by pdb_id, chain_id) — NO MCS gate
                                        ↓
                  extract template pockets (USalign each → bound-ligand centroids in cofold frame)
                                        ↓
                  cluster pockets (single-link, 5 Å cutoff) → top-K consensus centroids
                                        ↓
                  align cofolding outputs → prepare_docking_inputs
                  (binding sites = cofolding + swinsite + p2rank + template_consensus_{1..3})
                                        ↓
                  Track 1 docking — vina/adg fan out per binding-site source × seeds; pxdock 1×
                                        ↓
                  multi-track docking — Track 2 (template box, any template) +
                                        Track 3 (lig-align, MCS ≥ 0.5 only)
                                        ↓
                  (ion in input?) → ion placement (template alignment + clustering)
                                        ↓
                  post-analysis (BA-Pred / RMSD-Pred per pose)
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
| `template_filter.py` | Union mmseqs + foldseek hits, dedup by (pdb_id, chain_id), annotate with rcsb_index.db ligand info + Tanimoto/MCS metadata |
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

Bridges fire in this order between stages (driven by `script_builder.build_wrapper_shell_script`):

- **Boltz MSA → AF3**: `scripts/bridge_boltz_msa_to_af3.py` — CSV→A3M + patch AF3 JSON (pairedMsa="" + templates=[])
- **Template Filter (union)**: `scripts/run_template_filter.py` — accepts `--hits-tsv` (MMseqs2) and/or `--foldseek-tsv`. Union-merges by `(pdb_id, chain_id)`, annotates each row with candidate-ligand info + Tanimoto/MCS metadata + `in_mmseqs/in_foldseek/qtmscore/ttmscore/alntmscore/prob` columns. **No MCS gate** — every hit with at least one candidate ligand survives. Sort key: `evidence_breadth (2 sources > 1) > qtmscore > pident > n_ligands > tanimoto > mcs`.
- **Template Pocket Extraction**: `scripts/extract_template_pockets.py` — for each filtered hit, gemmi CA superposition (template → best cofolding cif), heavy-atom centroid of every candidate ligand instance, transformed into the cofold frame. Output: flat `template_pockets.json` with provenance (in_mmseqs/in_foldseek/pident/qtmscore). Default `--max-templates 50` to bound wall time on hits>200 targets.
- **Template Pocket Clustering**: `scripts/cluster_template_pockets.py` — greedy single-link clustering by Euclidean distance (default 5 Å). Per-pocket weight = `(in_mmseqs + in_foldseek) + max(qtmscore, pident/100)`; cluster `evidence_score = Σ weight`. Top-K (default 5) saved to `template_pocket_clusters.json` for downstream consumers.
- **Cofolding Alignment**: `scripts/align_cofolding_outputs.py` — aligns every cofold model.cif to a common receptor frame so docking + post-analysis share coords.
- **Docking Prep**: `scripts/prepare_docking_inputs.py` — auto model select (mean pLDDT) + SwinSite/P2Rank + reads `template_pocket_clusters.json` and registers top-3 cluster centroids as `template_consensus_{1,2,3}` sources in `binding_site_predictions` + SMILES→SDF/PDBQT + CIF→PDB→PDBQT.
- **Template Docking Prep**: `scripts/prepare_template_docking.py` — template CIF → receptor + ligand files + bound-pose SDF extraction (any template, no MCS gate).
- **Multi-track Docking**: `scripts/run_multi_track_docking.py` — Track 2 (template-based box docking, **no MCS gate**) + Track 3 (lig-align, **MCS ≥ `template_search_sequence.mcs_threshold`** only, default 0.5).
- **Ion Placement**: `scripts/collect_template_ions.py` — aligns templates to cofolding model, collects ion positions, clusters by confidence.
- **Submission Scoring**: `scripts/compute_submission_scores.py` — aggregates BA-Pred + RMSD-Pred + Boltz affinity, selects best pose (`select_best_pose` = max-`lscore`), log-space ensemble Kd.
- **CASP Submission**: `scripts/make_casp_submission.py` — generates CASP17 LG-format file per spec in `docs/casp17_lg_format.md`: up to 5 `MODEL` blocks per file, each MODEL is one complete snapshot (receptor + `LIGAND nnn` block per ligand + per-ligand `LSCORE` + MDL body ending in `M  END` + optional per-MODEL `AFFNTY`).

### Pipeline Stages

Default resolved order (when both searches enabled): `template-search-sequence → cofolding → template-search-structure → docking`. Stage-dependency validation enforces struct-after-cofold when `query_from_cofolding=true`.

1. **template-search-sequence** — MMseqs2 against `data/search_dbs/sequence/rcsb_seqDB` (488k seqs, preindexed). Writes raw `mmseqs_hits.tsv` (filtering happens later in the union bridge).
2. **cofolding** — Boltz2 + Boltz2x + Protenix + AF3 (affinity auto-enabled)
3. **template-search-structure** — Foldseek `easy-search` against `data/search_dbs/structure/rcsb_structDB`, query auto-resolved from cofolding outputs (`query_from_cofolding=true`, priority `[alphafold3, boltz, protenix]`). Writes raw `foldseek_hits.tsv`.
4. **template bridges (auto)** — union filter → pocket extraction → pocket clustering. Output: `filtered_hits.tsv`, `template_pockets.json`, `template_pocket_clusters.json`.
5. **docking-prep bridge** — align cofolding outputs + `prepare_docking_inputs.py`; consensus pocket centroids land in `binding_site_predictions.template_consensus_{1,2,3}`.
6. **docking** — Track 1: Vina + AutoDock-GPU fan out per binding-site source (now 6: cofolding/swinsite/p2rank + template_consensus_1/2/3) × seeds; Protenix-Dock 1× from priority-picked center. All read `docking_prep_summary.json` at runtime.
7. **multi-track docking** (auto, when template hits exist) — Track 2: template-based box docking (any template); Track 3: lig-align (MCS ≥ `mcs_threshold` only, default 0.5).
8. **ion placement** (auto, if ion CCD in input) — template alignment → ion position clustering by confidence.
9. **post-analysis** — BA-Pred + RMSD-Pred on staged multi-seed pose SDFs (input_sdf excluded — unaligned conformer crashes BA-Pred).
10. **CASP17 LG submission** — up to 5 alternate MODEL snapshots per target. Each MODEL packages: receptor coords + one `LIGAND nnn` block per ligand (all ligands of the target in every MODEL) + per-ligand `LSCORE` + optional per-MODEL `AFFNTY` from log-space ensemble Kd. MODEL 1 = primary prediction. Spec: `docs/casp17_lg_format.md`.

### Key Design Decisions

- **Boltz2 returns list**: `prepare_boltz()` returns `[boltz2, boltz2x]` — orchestrator unpacks with `*`
- **Docking validation relaxed**: `pipeline="docking"` skips receptor/ligand/box checks (bridge generates at runtime)
- **AF3 dedicated runner**: `run_alphafold3.sh` sets LD_LIBRARY_PATH for JAX CUDA (venv nvidia libs)
- **Protenix-Dock via micromamba**: needs ambertools (tleap) which is conda-only
- **Unified box**: 22.5Å × 22.5Å × 22.5Å, spacing 0.375Å across all docking tools
- **Binding site sources** (all dock independently as separate variants — no winner-take-all): `cofolding` (ligand centroid in cofold model), `swinsite` (ML pocket), `p2rank` (geometry), `template_consensus_{1..3}` (top-K cluster centroids from union templates). Fallback box pick priority: `cofolding > strong consensus (n_unique_pdb ≥ 2) > swinsite > p2rank > weak consensus`. The "fallback" only matters for Protenix-Dock and tools that read `box_center` directly; vina/adg variants always use their own dedicated source.
- **Template search union**: mmseqs2 (sequence, `min-seq-id 0.3 / cov 0.7 / sens 7.5 / max-seqs 500`) and foldseek (structure, `easy-search -s 9.5 --max-seqs 2000`) run as **independent stages**, then `template_filter.filter_hits_with_ligands` union-merges by `(pdb_id, chain_id)`. **Tanimoto/MCS are metadata only here** — every hit with a candidate ligand survives. The MCS threshold gates **only** Track 3 (lig-align) where atoms must actually overlay.
- **Single TM authority — USalign post-filter, not foldseek pre-filter**: foldseek's `qtmscore` is an *estimate*; we only use it as a sort hint inside the filter. The authoritative same-fold gate is USalign's actual TM-score, computed during pocket extraction (`extract_template_pockets --min-tmscore`, default 0.5 — Zhang/Skolnick canonical). `template_search_structure.qtmscore_min` defaults to 0.0 (disabled); set > 0 only to filter `filtered_hits.tsv` cosmetically for inspection. Cluster pocket weight `(in_mmseqs + in_foldseek) + max(alignment_tmscore, qtmscore, pident/100)` consumes the actual TM too, so mmseqs-only hits (which carry foldseek qtm=0 in the filter row) get scored on real structural similarity rather than just pident.
- **Template-consensus pockets**: `extract_template_pockets.py` aligns each surviving hit's CIF onto the best cofolding model and emits one heavy-atom centroid per bound candidate ligand. `cluster_template_pockets.py` (single-link, 5 Å) collapses these into top-K consensus centroids weighted by `(in_mmseqs + in_foldseek) + max(qtmscore, pident/100)`. Each centroid becomes a `template_consensus_N` source for vina/adg, and the priority-picked one (when n_unique_pdb ≥ 2) beats swinsite/p2rank as the fallback box.
- **USalign alignment** (not gemmi): `extract_template_pockets.py` uses `casp17.usalign.run_usalign` (the existing USalign binary at `.local/bin/USalign`) instead of `gemmi.calculate_superposition`. Foldseek finds templates by 3Di + structural similarity, so the reciprocal alignment must also be structure-based — gemmi's sequence-anchored superposition collapses on distant homologs (matched-residue count drops to <50, transform RMSD blows up to 15+ Å, ligand centroids land tens of Å off-pocket). USalign reproduces the same "fold view" foldseek used and handles homotetrameric / multi-chain cases via globally optimal chain matching (chain permutation noise then converges in the spatial cluster step). Single safety net: `--min-tmscore 0.5` (canonical Zhang/Skolnick same-fold cutoff). Cost: **~0.5 s/template** measured on 300-aa structures; default `--max-templates 2000` ≈ 17 min/target (matches foldseek `max_hits=2000` so the tail isn't silently dropped).
- **Affinity auto-inject**: When ligand present, `properties.affinity` added to Boltz YAML
- **RCSB ligand index**: External SQLite DB (mmcif-parser maintained), queried for template filtering
- **Multi-track docking**: 3 tracks — Track 1 (always): cofolding→docking, Track 2 (any template): template-based box docking, Track 3 (MCS ≥ `mcs_threshold`): lig-align. Config knob: `template_search_sequence.mcs_threshold` (default 0.5).
- **Template ligand SDF**: `prepare_template_docking.py` extracts bound-pose ligand from template CIF for lig-align reference
- **Multi-seed**: `cofolding_seeds` (default 5) × `diffusion_samples` (default 5) = 25 structures per cofolding model. `docking_seeds` (default 5) for Vina/ADG (PxDock single run). Config: `cofolding_seeds`, `docking_seeds` lists in RunnerConfig
- **AF3 templates field**: Adapter always adds `templates=[]` to AF3 protein blocks (required by AF3 schema even when empty)
- **Submission top-5 MODEL snapshots**: Each MODEL is one complete (receptor + all-ligands + AFFNTY) snapshot. For multi-ligand targets every MODEL contains every ligand (not one ligand per MODEL). For each ligand the 5 MODELs carry 5 alternate poses with heavy-atom RMSD ≥ 2 Å diversity (per-ligand). MODEL 1 is the primary (best aggregate LSCORE) prediction. AFFNTY from log-space ensemble of BA-Pred pKd + Boltz affinity (filtered by binder_prob ≥ 0.5) reported once per MODEL after the last `LIGAND` block
- **Model auto-select**: `prepare_docking_inputs.py` picks cofolding model by mean pLDDT normalised to [0,100] scale (Boltz npz ×100, Protenix scalar, AF3 atom_plddts mean). Critical fix: previously compared Boltz [0,1] vs AF3 [0,100] raw → always picked AF3 even when Boltz was superior
- **Post-analysis pose staging**: Multi-seed poses staged under `outputs/analysis/poses/{tool}_seed_{N}.sdf` with unique stems to avoid BA-Pred/RMSD-Pred name collisions. PxDock JSON→SDF via `_pxdock_json_to_sdf`. Input SDF excluded (unaligned conformer)
- **ADG dynamic GPF**: AutoDock-GPU wrapper parses ligand PDBQT at runtime for atom types (F/Cl/Br/P/I/Si support), reads box center from `docking_prep_summary.json` (not compile-time defaults)
- **Docking-variant fan-out**: `_DOCKING_BOX_SOURCES = (cofolding, swinsite, p2rank, template_consensus_{1,2,3})` in `adapters.py`. `prepare_vina` / `prepare_autodock_gpu` emit one `PreparedModelRun` per source (so the wrapper produces e.g. `vina_template_consensus_1`, `autodock-gpu_template_consensus_2`, …). Runtime variant scripts call `bs_preds.get(BOX_SOURCE)` and `sys.exit(0)` cleanly when the source is missing — targets with no template hits behave like the legacy 3-source pipeline.

### Cluster Notes

- Master node: no GPU — YAML parsing, manifest generation, validation only
- All inference/CUDA on compute nodes via SLURM
- Scripts auto-set: `module load cuda/12.8`, LD_LIBRARY_PATH (CUDA targets + venv nvidia + boost), `.local/bin` PATH
- Tests requiring GPU submitted via SLURM
