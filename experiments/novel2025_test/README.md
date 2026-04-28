# novel2025_test — held-out 2025 benchmark

**Purpose**: Evaluate the CASP17 protein-ligand pipeline on novel structures
deposited on or after **2025-01-01**, using template search restricted to
pre-2025 templates. This is a time-split sanity check — nothing in the
retrieved templates (or downstream MCS-guided docking) should leak from
the target's own crystal structure.

## Data source

- `/home/jaemin/DB/RCSB/processed/seqid_zones_2025_nonredundant.tsv`
  — user-provided non-redundant 2025 RCSB index

## Target counts (revised — multi-chain + multi-ligand supported)

| stage | kept | dropped | reason |
|---|---|---|---|
| **TSV total** | **543** | — | user-provided input |
| `deposition_date >= 2025-01-01` | 543 | 0 | already post-2025 |
| `num_candidate_ligands >= 1` | 543 | 0 | all have candidate ligand |
| `100 <= sum(chain_lengths) <= 900` (GPU budget) | 499 | 44 | too short / too long |
| candidate_smiles[0] non-empty | 499 | 0 | parseability |
| **YAMLs written** → `pipeline/*_input.yaml` | **499** | **44 total** | |

Classification breakdown of the 499 kept targets:

| seq_zone | count | description |
|---|---|---|
| novel | 71 | no homolog in PDB (< ~0.3 max id) |
| remote | 116 | remote homology (0.3–0.5 max id) |
| related | 312 | closer homologs exist (> 0.5 max id) |
| **total** | **499** | |

Chain count: 1-chain 477, 2-chain 20, 3-chain 1, 5-chain 1.
Total length: min 120, median 338, max 849 aa.
Multi-ligand targets (>1 ligand in cofolding YAML): 231 / 499.

## Column mapping (TSV → unified input YAML)

Cofolding vs docking have different needs:

- **Cofolding** is more accurate when the co-complex is fully populated —
  every chain that contacts the ligand, plus every non-aid ligand (metals,
  cofactors, the drug-like candidate).
- **Docking** only cares about the drug-like target ligand we want to
  evaluate — the first `candidate_smiles` entry.

The YAML emits both, with ordering and id conventions that make the
downstream pipeline auto-route things correctly.

### Protein entities

| YAML field | TSV column | processing |
|---|---|---|
| `sequences[].protein.sequence` | `sequences` | `split("\|")` — one entity per chain, ids `A`, `B`, `C`, … |

Multi-chain targets emit one `- protein:` block per chain with sequential
ids. The length filter is applied to the **sum** of all chain lengths.

### Ligand entities

| YAML id | source | processing |
|---|---|---|
| `L` | `candidate_smiles[0]` + `candidate_ccd_codes[0]` | primary drug-like binder, always first. Auto-wired as the affinity `binder` in `adapters.py` and as the Vina/ADG docking target (Vina picks `prep["ligands"][0]`). |
| `L2`, `L3`, … | `candidate_smiles[1:]` | additional drug-like candidates (rare: 87 targets have 2, 15 have 3, 4 have 4, 3 have 5). All still use `smiles:`. |
| `X2`, `X3`, … | `ligand_ccd_codes` + `ligand_smiles` + `ligand_types` minus the candidate set minus crystallization aids | cofactors, metals, inorganic clusters. Metals / ions / metal_clusters use the `ccd:` field (more reliable than free-form SMILES for coordination chemistry); everything else uses `smiles:`. |

Dropped in every case:
- `ligand_types == "crystallization_aid"` (EDO, GOL, PEG, sulfate, …)
  — not part of the functional complex.

Example — 8s02 (novel, 2 chains, 5 raw ligands: BYC, GOL, SF4, BJ8, TRS):

```yaml
sequences:
- protein: {id: A, sequence: ..., msa: empty}       # 447 aa
- protein: {id: B, sequence: ..., msa: empty}       # 379 aa
- ligand:  {id: L,  smiles: '...CoA...'}            # BYC candidate (binder)
- ligand:  {id: L2, smiles: '...Fe4S4 SMILES...'}   # BJ8 candidate
- ligand:  {id: X2, ccd:    SF4}                    # metal cluster
properties:
- affinity: {binder: L}
```

GOL and TRS (crystallization_aid) are omitted.

### Docking target selection

`adapters.py` lines 110–114 auto-set `properties.affinity.binder = L`
because `L` is always the first ligand in our YAML. `prepare_docking_inputs.py`
+ Vina auto-pick `prep["ligands"][0]` as the docking target, which
resolves to `L`. CCD-only ligands are silently skipped by the docking
prep bridge (no SMILES → no SDF/PDBQT possible) so metals never hit
the docking stage.

## Dropped rows (109 total)

- 41 multi-chain targets (the first chain is single protein but the
  `sequences` column contains `|` indicating additional chains we'd
  need to handle — skipped to keep the first pass simple)
- 68 single-chain targets outside the 100–600 aa window

No SMILES parse failures. No post-2025-01-01 leaks.

## Smoke test set (initial end-to-end check)

4 targets originally selected as a smoke test before the full batch:

| id | date | L | max_id | CCD | organism |
|---|---|---|---|---|---|
| 22mj | 2026-03-11 | 322 | 0.36 | A1MDP | Homo sapiens |
| 10sl | 2026-02-25 | 361 | 0.41 | H1N | Trypanosoma cruzi |
| 9ig3 | 2026-03-04 | 327 | 0.40 | A1I30 | Penicillium crustosum |
| 9uad | 2026-03-11 | 315 | 0.41 | A1EMZ | Penicillium simplicissimum |

22mj completed successfully (job 24407, 56m47s) — top-1 ligand RMSD
**0.61 Å** vs RCSB crystal (see FINAL_REPORT §6.2). The other three were
rolled into the main batch.

## Pipeline isolation

- Inputs live under this directory (`pipeline/`, `build_inputs.py`,
  `novel2025_config.yaml`, `batch_prepare_and_submit.sh`)
- Runner config: `novel2025_config.yaml` is the canonical batch config:
  - `template_search_sequence.enabled: true` + `max_deposition_date: 2025-01-01`
  - `template_search_structure.enabled: true` + `query_from_cofolding: true`
    (foldseek `easy-search` against `data/search_dbs/structure/rcsb_structDB`)
  - Both searches run in parallel-ish across stages and their hits are
    union-merged by `(pdb_id, chain_id)` in the auto-inserted template
    bridge. **MCS gating is restricted to Track 3 (lig-align)** — Track 2
    box docking and the new template-consensus pocket extraction use any
    template that has a candidate ligand.
  - Pocket extraction (`extract_template_pockets.py`) + clustering
    (`cluster_template_pockets.py`) run automatically after the filter,
    feeding `template_consensus_{1,2,3}` binding-site sources to vina/adg
    docking variants in addition to cofolding/swinsite/p2rank.
- Outputs land in the shared layout `experiments/runs/<target>_input/`
  with the new `outputs/template_search_structure/` and
  `outputs/template_pockets/` directories alongside the existing tree.
- Submissions go to `experiments/submissions/<target>_input.lg`

## Known issues

- **Protenix fails on gpu1 (RTX PRO 6000 Blackwell Max-Q)** — cuEquivariance
  kernels throw `CUDA error: no kernel image is available for execution on
  the device` on Blackwell (sm_120). The wrapper swallows the Protenix
  failure via `|| echo '(failed, continuing)'`, so the pipeline keeps
  running with Boltz2/2x + AF3 + docking only. Impact: for the ~(4/7) of
  jobs routed to gpu1, the cofolding pose pool drops from ~100 → ~75
  (Protenix's 25 samples are missing). Docking pool is unaffected. Fix
  pending a cuEquivariance upgrade that supports Blackwell.

## Evaluation plan

Once the batch completes, rerun `evaluate.py` / `best_pose.py` /
`compare_selection.py` / `compare_cofold_metrics.py` (from
`experiments/casp16_test/L1000/`) adapted to load experimental reference
from RCSB mmCIF (`/home/jaemin/DB/RCSB/raw/mmCIF_data/<hash>/<pdb>.cif.gz`)
instead of the CASP-style `L1000_prepared/` directory. The 22mj one-off
evaluation already demonstrated the workflow.
