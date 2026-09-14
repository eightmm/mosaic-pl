# Mosaic-PL for CASP17

**Mosaic-PL** is LCDD's CASP17 protein-ligand prediction method. Its multi-source workflow preserves complementary hypotheses from co-folding, template transfer, binding-site prediction, and multi-track docking, then ranks diverse poses with RMSD-Pred. LCDD remains the CASP group name.

## At A Glance

- **Co-folding ensemble:** Boltz-2, Boltz-2x, Protenix, and AlphaFold 3.
- **Template-guided pockets:** local RCSB sequence search (MMseqs2) and structure search (Foldseek) are unioned; bound-ligand centroids are transferred only after structural alignment and clustered into consensus pockets.
- **Independent pose generation:** co-folded, P2Rank, SwinSite, and template-consensus pockets drive Vina and AutoDock-GPU tracks; MCS-guided ligand alignment is an additional, conditional track.
- **Pose selection:** receptor-frame RMSD-Pred scoring (`LSCORE = 1 - P(RMSD > 2 A)`) selects up to five diverse models for CASP LG output.

## Quick Start

```bash
git clone https://github.com/eightmm/mosaic-pl.git
cd mosaic-pl
uv sync

# Validate a supplied YAML and prepare the Slurm workflow.
uv run casp17-pl validate-run \
  --input examples/full_pipeline_test.yaml \
  --config examples/full_pipeline_config.yaml \
  --stages template-search-sequence cofolding docking
uv run casp17-pl run-wrapper \
  --input examples/full_pipeline_test.yaml \
  --config examples/full_pipeline_config.yaml \
  --stages template-search-sequence cofolding docking \
  --submit
```

External models, RCSB search databases, and GPU inference are configured separately. See [Setup](#setup), [Pipeline Stages Detail](#pipeline-stages-detail), and [Configuration](#configuration) before submitting a job.

---

## Scope

Mosaic-PL orchestrates external ML models (Boltz2/2x, Protenix, AlphaFold3), **union template search (MMseqs2 + Foldseek)** with cross-source pocket consensus, binding-site prediction (P2Rank, SwinSite, **template-consensus pockets**), docking (Vina, AutoDock-GPU, Protenix-Dock), and post-analysis (BA-Pred, RMSD-Pred).

## Research Workflows

The repository also contains reproducible analysis and validation utilities for co-folding diversity, template-pose transfer, frame consistency, ligand-series scoring, and CASP LG/TS submission checks. See [Research Workflows](docs/research_workflows.md) for the supported analysis paths and their required local data.

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
    │  • up to 19 binding-site sources →          │
    │    cofolding_{1..3} · p2rank_{1..3} ·       │
    │    swinsite_{1..3} ·                        │
    │    template_consensus_{1..10}               │
    │  • SMILES → 3D SDF → PDBQT (RDKit + meeko) │
    │  • CIF → PDB → PDBQT (gemmi + pdb2pqr)     │
    │  • Box: 22.5 Å, spacing 0.375 Å            │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Stage 5: Docking (Multi-track, Multi-seed) │
    │                                             │
    │  Track 1 (always) — per-source fan-out:     │
    │  Vina × up to 19 + AutoDock-GPU × up to 19  │
    │  (PxDock default OFF, b7dd9d5)              │
    │                                             │
    │  Track 2 (any template) — pocket borrow:    │
    │  Template-based box docking (Vina + ADG)    │
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
    │  ├── outputs/vina_{cofolding_{1..3},p2rank_{1..3},           │
    │  │     swinsite_{1..3},template_consensus_{1..10}}/seed_*/   │
    │  ├── outputs/autodock_gpu_{... same 19 sources}/seed_*/      │
    │  ├── outputs/protenix_dock/    (default OFF, b7dd9d5)        │
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
git clone https://github.com/eightmm/mosaic-pl.git mosaic-pl && cd mosaic-pl
bash scripts/install_external_models.sh              # all tools + models
srun --partition=6000ada --gres=gpu:1 bash scripts/build_autodock_gpu.sh  # GPU build
bash scripts/install_external_models.sh --verify      # verify all
bash scripts/build_search_dbs.sh                      # MMseqs2 + Foldseek DBs
```

### What Gets Installed

| Tool | venv / Location | Purpose |
|------|----------------|---------|
| Boltz2 | `.venvs/boltz` (Py3.12) | Co-folding + affinity |
| Protenix | `.venvs/protenix` (Py3.12) | Co-folding |
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

## Detailed Quick Start

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

## Batch (multi-target benchmark)

The **standard workflow** for the 499-target Novel2025 benchmark. The Quick Start above is sufficient for a single target; the following steps are intended for batches of tens or more targets.

```
[bulk YAML generation] → [wrapper array dispatch] → [pose-pool analysis] → [LG evaluation]
```

### 1. Generate input YAML files in bulk

```bash
# Generate experiments/novel2025_test/pipeline/*.yaml from RCSB or a SMILES list.
.venv/bin/python experiments/novel2025_test/build_inputs.py \
    --rcsb-index ~/DB/RCSB/processed/rcsb_index.db \
    --output-dir experiments/novel2025_test/pipeline/ \
    --max-deposition-date 2025-01-01

# Alternatively, build directly from the target list in cluster_targets_100.csv.
# The 499 targets correspond to 246 unique enzymes after 100% MMseqs easy-cluster.
# Report headline SR for both cluster_rep_only and per-target views.
```

### 2. Wrapper array dispatch (SLURM)

`run_wrapper.sbatch.sh` is generated automatically for each target. Submit the batch as follows:

```bash
# One sbatch submission per target (individual GPU jobs).
for t in $(ls experiments/novel2025_runs/runs); do
    sbatch experiments/novel2025_runs/runs/$t/$t/scripts/run_wrapper.sbatch.sh
done

# Or rerun only template search through docking when co-folded structures are available:
sbatch --array=0-498%16 experiments/novel2025_test/rerun_template_to_dock.sbatch.sh
```

### 3. Analyze the pose pool (oracle SR and per-scorer SR)

```bash
# Compute per-pose true RMSD and aggregate top-1/top-5 results for every scorer.
.venv/bin/python experiments/novel2025_test/score_per_metric.py \
    --submissions-dir experiments/novel2025_runs/submissions \
    --runs-dir experiments/novel2025_runs/runs \
    --output-csv experiments/novel2025_runs/per_pose_scores.csv \
    --work-dir experiments/novel2025_runs/_per_metric_work
```

Outputs:
- `per_pose_scores.csv`: one row per pose with every score (~1.3 M rows for 499 targets).
- `per_metric_summary.txt`: per-scorer (`rmsd_pred / lscore / iptm / ptm / plddt / conf / boltz_aff / ba_pred / oracle`) top-1/top-5 SR by zone.

**Four oracle-SR definitions** for multi-cofactor targets:
- `ANY-ligand`: the best result across all ligand pools for a target; inflated by cofactors and **not appropriate**.
- `ALL-ligand`: a hit only when every ligand in the target is below 2 A; the strictest definition.
- `PRIMARY '_L' only`: evaluate only L, the ligand of interest; CASP-comparable and the **recommended headline metric**.
- `PER-(target, ligand)`: one row per `(target, lig_id)` pair, following PoseBusters/PLINDER-style evaluation.

**Important:** `score_per_metric.py` uses `rdMolAlign.CalcRMS` for `pose_rmsd`. Do **not** use `GetBestRMS`, which internally realigns the ligand with Kabsch and can conceal pocket-placement errors. See the warning block in `docs/per_pose_rmsd_method.md` for the full rule.

**Multi-ligand-aware reference matching** (`evaluate_run`): map `_L`/`_L2`/`_L3` pose-name suffixes to prepared ligands, canonicalize SMILES, then select the candidate CCD. This prevents wrong-reference MCS fallback hangs in targets with FAD, NAP, or other cofactors.

### 4. Evaluate the LG submission (MODEL 1 and MODEL 1-5 SR)

```bash
# 1) Regenerate LG files with updated docking results.
sbatch --array=0-498%16 experiments/novel2025_runs/regen_lg.sbatch.sh

# 2) Compare crystal and LG MODEL results: top-1 <2 A / best-of-5 <2 A SR by zone.
.venv/bin/python experiments/novel2025_test/evaluate.py \
    --submissions-dir experiments/novel2025_runs/submissions \
    --output-json experiments/novel2025_runs/evaluation.json
```

`evaluate.py` compares MODEL 1..5 in every `*.lg` file with the crystal ligand and writes a JSON result plus a standard-output summary.

### 5. Audit frame consistency (batch QA)

Verify that every docked pose is aligned to the protein receptor frame:

```bash
# Run once for the complete batch (tens of seconds).
.venv/bin/python scripts/check_frame_consistency.py \
    --runs-dir experiments/novel2025_runs/runs \
    --output-tsv /tmp/frame_check.tsv

# Detailed diagnosis on a random sample of 50 (receptor-to-cofold, cofold pairs, and family centroids).
.venv/bin/python scripts/audit_frame_50.py \
    --runs-dir experiments/novel2025_runs/runs --n 50
```

Issue classes:
- `A: cofold ref unaligned`: preparation received a raw CIF rather than `*_aligned.cif`; this should be zero after fix `018b78f`.
- `B: binding-site source > rec_extent+30 A`: signature of a frame bug.
- `D: cofolding_1 <-> template_consensus_1 > 30 A`: cross-source mismatch signal that may also indicate true multi-pocket behavior.
- `E: alignment rmsd_after > 5 A`: co-folded conformational divergence, not a frame bug.

### 6. Generate the batch analysis report

```bash
# Generate cluster-level and zone-level SR tables plus a Markdown report.
.venv/bin/python scripts/generate_novel2025_report.py \
    --per-pose-csv experiments/novel2025_runs/per_pose_scores.csv \
    --cluster-csv experiments/novel2025_test/cluster_targets_100.csv \
    --output experiments/novel2025_runs/ANALYSIS.md
```

## Pipeline Stages Detail

### Stage 1: Template Search (Union — sequence + structure)

Run **both** sequence and structure searches to recover as many templates as possible. MCS/Tanimoto values are retained only as metadata; gating is applied only in lig-MCS-align (Track 3).

**1a · Sequence (MMseqs2)** — `easy-search` against `data/search_dbs/sequence/rcsb_seqDB`
- Output: `outputs/template_search_sequence/mmseqs_hits.tsv`

**1b · Structure (Foldseek)** — `easy-search` against `data/search_dbs/structure/rcsb_structDB`
- Query selection is automatic from co-folding output (`query_from_cofolding=true`, priority `[alphafold3, boltz, protenix]`).
- Stage-dependency validation prevents structure search from running before co-folding.
- `--max-seqs 500` prioritizes recall; unlike MMseqs2, Foldseek has no native identity/coverage cutoff here.
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

Three steps run automatically:
1. **Union filter** (`run_template_filter.py`): deduplicate the MMseqs2/Foldseek union and annotate ligands. There is **no MCS gate**. Foldseek `qtmscore` is an estimate and is not pre-filtered (default `qtmscore_min=0`); every hit proceeds to USalign for the actual TM-score decision.
2. **Pocket extraction** (`extract_template_pockets.py`): structurally align the top `--max-templates 2000` hits (default) to the co-folded model with **USalign**. Drop hits below `--min-tmscore` (default 0.5, canonical same-fold). Write each bound candidate-ligand heavy-atom centroid to `template_pockets.json`. Typical runtime is **~17 min per target** (~0.5 s/template x 2000 measured with USalign).
3. **Pocket clustering** (`cluster_template_pockets.py`): use **greedy first-match centroid** clustering. Each new pocket joins the first cluster within `--cutoff` A (default 5), or starts a new cluster. This is not single-link clustering; evidence-ordering at extraction makes stronger hits become cluster seeds. Weight is `(in_mmseqs + in_foldseek) + max(alignment_tmscore, qtmscore, pident/100)`. The top-K clusters (default 5; new-pipeline default 10) are written to `template_pocket_clusters.json`, and each centroid becomes a `template_consensus_N` binding-site source.

**Threshold summary**:
- **`extract_template_pockets --min-tmscore 0.5`**: the only quality gate, using the actual USalign TM-score.
- `template_search_sequence.mcs_threshold` (default `0.5`): gates **only Track 3 (lig-align)**.
- `template_search_structure.qtmscore_min` (default `0.0`): disabled; retained for display only.

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

The previous cross-model consensus was superseded by Stage 1b (Foldseek easy-search with the best co-folded CIF as query) and Stage 1c (the MMseqs2/Foldseek union filter). See Stage 1 above for details.

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
| Binding site sources | up to 19 (cofolding_{1..3} + p2rank_{1..3} + swinsite_{1..3} + template_consensus_{1..10}) | `binding_site_predictions` dict in summary JSON |

**Binding-site sources** fan out into independent docking variants; they are not winner-take-all:

1. **Cofolding centroids 1..3**: the top three centroids from co-folded-model pose clusters, computed by `compute_submission_scores` (`4d3ee27`).
2. **P2Rank 1..3**: the top three pockets from the surface-geometry predictor (Java, `6e58e01`).
3. **SwinSite 1..3**: the top three pockets from the Swin-Unet ML predictor (`.venvs/pred`, `6e58e01`).
4. **template_consensus_1..10**: top-K cluster centroids (default 10) produced by `cluster_template_pockets.py` from consensus bound-ligand positions across the MMseqs2/Foldseek template union.

The source count varies by target: a P2Rank or SwinSite source has one variant when it finds one pocket, and `template_consensus_*` has none when no template hit is available. Variants cleanly skip with `sys.exit(0)` when `bs_preds.get(BOX_SOURCE)` is `None`.

**Fixed bridge order since `018b78f` (May 4):** `align_cofolding_outputs → _emit_template_bridges → prepare_docking_inputs`. Running alignment last previously placed `template_consensus_*` centroids in an unaligned frame. Do not change this order.

**Fallback box selection** (`box_method`, used by single-box tools such as Protenix-Dock): `cofolding > strong consensus (n_unique_pdb >= 2) > swinsite > p2rank > weak consensus`.

**Unified box**: 22.5 Å × 22.5 Å × 22.5 Å, grid spacing 0.375 Å.

### Stage 5: Docking (Multi-track)

3 parallel docking tracks, where Track 2+3 activate conditionally:

#### Track 1 (always): Cofolding-based docking — **per-source fan-out**

Run a separate Vina/ADG variant for each binding-site source. If a source has no pocket (for example, P2Rank/SwinSite has only one result or a target has no `template_consensus_*` hit), only that variant cleanly exits with `sys.exit(0)`. PxDock is disabled by default since `b7dd9d5` (`protenix_dock.enabled: false`) and is enabled only for high-evidence clusters.

| Tool | Variants | Type | Output |
|------|----------|------|--------|
| **Vina** | up to 19 (`vina_{cofolding_{1..3},p2rank_{1..3},swinsite_{1..3},template_consensus_{1..10}}`) | Python API CPU | `outputs/{variant}/seed_<seed>/ligand_<lid>/docked.pdbqt` |
| **AutoDock-GPU** | up to 19 (the same 19 sources) | CUDA GPU | `outputs/{variant}/seed_<seed>/ligand_<lid>/docking.dlg` |
| **Protenix-Dock** | Disabled by default | CPU force field | When enabled: `outputs/protenix_dock/poses_<lid>.sdf` |

The ADG wrapper also parses each ligand PDBQT **at runtime** to construct `ligand_types` and grid maps dynamically, including nonstandard atom types such as F, Cl, Br, P, I, and Si. Every tool reads receptor, ligand, and box information from `docking_prep_summary.json` at runtime.

#### Track 2 (any template, **no MCS gate**): Template-based Box Docking

A template ligand need not resemble the query; this track transfers only pocket geometry. It uses the top three templates in `filtered_hits.tsv` with `num_ligands > 0`, prioritizing broader evidence and higher similarity (both-source/high-QTM hits first).

| Step | Script | Output |
|------|--------|--------|
| Template prep | `prepare_template_docking.py` | receptor PDB/PDBQT + ligand files + box from template ligand |
| Docking | Vina + ADG (PxDock default OFF) | `outputs/template_docking/<pdb_id>/vina/`, `autodock_gpu/` |

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
| Protenix | 5 seeds × 5 samples | 25 structures |
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
    --output experiments/CASP17/submissions/L2001.lg
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
  model_name: protenix_base_default_v1.0.0
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
  max_hits: 2000                              # wide recall — foldseek has no native id/cov gate
  qtmscore_min: 0.0                           # 0 = disabled; USalign actual TM is the real gate

vina:
  enabled: true            # fans out into up to 19 variants per binding-site source
autodock_gpu:
  enabled: true            # fans out into up to 19 variants per binding-site source
protenix_dock:
  enabled: false           # default OFF since b7dd9d5 (too slow for marginal SR gain)

slurm:
  partition: 6000ada
  qos: normal
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
│   ├── vina_{cofolding_{1..3},p2rank_{1..3},swinsite_{1..3},template_consensus_{1..10}}/seed_*/
│   │       └── ligand_<lid>/docked.pdbqt   # Track 1: up to 19 variants × 5 seeds
│   ├── autodock_gpu_{... same 19 sources}/seed_*/
│   │       └── ligand_<lid>/docking.dlg    # Track 1: up to 19 variants × 5 seeds
│   ├── protenix_dock/               # default OFF (b7dd9d5)
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
| Protenix (5 seeds × 5 samples) | 492s | 484s | ✓ |
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

## Known Issues / Recent Fixes (2026-05)

The following bugs were found and fixed in the 499-target Novel2025 batch (`experiments/novel2025_runs/`). They are documented to prevent regression.

| Date | Location | Symptom | Fix |
|---|---|---|---|
| 2026-05-04 (`018b78f`) | `src/casp17/script_builder.py` bridge order | `_emit_template_bridges` ran before `align_cofolding_outputs`, so `extract_template_pockets` used an unaligned CIF and placed `template_consensus_*` centroids in the wrong frame. ADG template-consensus variants produced effectively zero poses for 217/462 targets. | Fixed the order as `align -> template-bridges -> docking-prep`, added three fallback warnings, and monitor it with `scripts/check_frame_consistency.py`. **Do not change this order.** |
| 2026-05-11 | `scripts/prepare_docking_inputs.py:1153` | A relative `args.output_dir` wrote a relative `receptor_pdbqt` to `docking_prep_summary.json`; the ADG runner then could not find it when calling AutoGrid with `cwd=lig_grid_dir`. Template-consensus variants failed silently because the exception handler printed "ok". | Force `args.output_dir = args.output_dir.resolve()`, add `Path(receptor).resolve()` as an ADG-runner safeguard, and emit stderr plus a `FAILED` marker on AutoGrid errors. |
| 2026-05-11 | `src/casp17/geometry.py:pose_rmsd` | `GetSubstructMatches(uniquify=False)` enumeration hung on symmetric ligands. SIGALRM could not interrupt the RDKit C extension, causing 12-hour score-job timeouts. | Replaced it with `rdMolAlign.CalcRMS` (symmetry-aware, no alignment, C++ permutation cap). `docs/per_pose_rmsd_method.md` prohibits `GetBestRMS` and `GetSubstructMatches(uniquify=False)`. |
| 2026-05-12 | `experiments/novel2025_test/score_per_metric.py:evaluate_run` | Loading only `candidate_ccd_codes[0]` as `ref_lig` compared L2/L3 poses against the wrong reference in multi-cofactor targets, causing atom-count mismatches and 30-second `FindMCS` fallback timeouts per pose. | Canonically match candidate CCDs to prepared-ligand SMILES, build a `lig_id_to_ref` map, and select references from `_L`/`_L2` suffixes. Validation on 8z15 (FAD+NAP+A1D7V) changed a hang to 223-second completion. |

### Remaining issues

- **AD4 cofactor atom types** (from `c65797b`): `receptor.pdbqt` retains metal/cofactor atoms (CG, Co, Hg, K, Na, Ni) that are not present in the AutoGrid4 force-field mapping table. This produces "Unknown receptor type" errors and silently fails all ADG variants for affected targets, approximately 13% (71/534 Slurm errors). Candidate fixes:
  - Strip only unknown, docking-unsupported cofactor atoms during preparation.
  - Extend the AutoGrid parameter table, for example with a user-level mapping in `~/.autodock/AD4_parameters.dat`.
- **Per-target scoring 180-second timeout:** this still occurs for very large cofactors (FAD, COA, B12, long-chain fatty acids, and large sugars). The nine affected targets (`9dsv 9emt 9ifw 9mgt 9mh5 9n1b 9uo2 9vjx 9zno`) need a separate rerun with a 1800-second timeout.

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
├── extract_template_pockets.py    # USalign-based pocket centroid extraction
├── cluster_template_pockets.py    # Greedy first-match centroid clustering
├── prepare_template_docking.py    # Template CIF → receptor/ligand + bound-pose SDF
├── run_multi_track_docking.py     # Multi-track orchestrator (Track 2 + Track 3)
├── collect_template_ions.py       # Ion/metal placement via template alignment
├── run_structure_search.py        # Foldseek consensus across models
├── run_post_analysis.py           # BA-Pred + RMSD-Pred
├── compute_submission_scores.py   # Aggregate scores, ensemble affinity, best pose
├── make_casp_submission.py        # Generate CASP17 LG format submission file
│
├── # Batch / benchmark tooling
├── collect_unified_poses.py       # Per-target unified pose archive (SDF + manifest)
├── check_frame_consistency.py     # Full-batch frame audit (TSV + summary)
├── audit_frame_50.py              # Detailed 50-sample frame audit (receptor↔cofold, pose families)
├── analyze_binding_site_recall.py # DCC≤4Å recall per binding-site source
├── analyze_sr_by_cluster.py       # SR aggregation by sequence cluster (per_target / cluster_rep)
├── cluster_novel2025_targets.py   # mmseqs easy-cluster on novel2025 inputs
├── generate_novel2025_report.py   # Markdown analysis report from per_pose_scores.csv
└── reprocess_template_pockets_post_align.py  # Re-extract/cluster template pockets without re-docking

experiments/novel2025_test/
├── build_inputs.py                # Bulk yaml generator from RCSB/SMILES list
├── evaluate.py                    # LG submission → MODEL 1/1-5 SR vs crystal
├── score_per_metric.py            # Per-pose true RMSD + scorer aggregation (Oracle/iPTM/etc.)
├── analyze_pose_diversity.py      # Pose pool size + diversity stats
├── score_by_source.py             # SR breakdown by binding-site source family
├── score_per_metric_proper.py     # Per-scorer SR with proper pool filters
└── novel2025_config.yaml          # 499-target batch runner config

docs/
├── pipeline.md                    # Pipeline ground truth (mirrors README's Stage detail)
├── per_pose_rmsd_method.md        # ⚠ CalcRMS rule + RMSD computation spec
├── casp17_lg_format.md            # CASP17 LG format spec mirror
└── pose_ranker_design.md          # Ranker analysis (current bottleneck: 36pp Oracle gap)
```
