# CASP17 Pipeline — L1000 evaluation + selection study + novel2025 benchmark status

**Run date**: 2026-04-15
**Scope**: L1000 (CASP16 chymase, 17 ligands) end-to-end evaluation, selection
strategy comparison, cofolding-internal-metric study, and setup for the
novel2025 held-out benchmark.

---

## 1. L1000 LG submission performance (current selector)

Ligand RMSD = heavy-atom, symmetry-aware `rdkit GetBestRMS` after SMILES
template bond reassignment. Protein frame from Kabsch alignment on common
CAs (docking receptor → experimental crystal).

| target | top-1 | best-of-5 | AFFNTY (nM) | exper ΔG |
|---|---|---|---|---|
| L1001 | **0.49** ✓ | 0.49 | 193.7 | −12.35 |
| L1002 | **1.80** ✓ | 0.85 | 77.5 | −10.58 |
| L1003 | 2.64 ✗ | 1.40 ✓ | 119.3 | −10.38 |
| L1004 | **0.61** ✓ | 0.61 | 19.9 | −10.72 |
| L1005 | **0.37** ✓ | 0.37 | n/a | −10.45 |
| L1006 | 2.08 ✗ | 0.36 ✓ | 148.0 | −11.06 |
| L1007 | **1.65** ✓ | 1.49 | 1713.5 | −9.12 |
| L1008 | 2.10 ✗ | 0.78 ✓ | 64.4 | −8.78 |
| L1009 | 2.39 ✗ | 1.50 ✓ | 162.5 | −11.36 |
| L1010 | **0.46** ✓ | 0.46 | 1795.9 | −10.58 |
| L1011 | **1.69** ✓ | 1.11 | 204.4 | −10.14 |
| L1012 | 3.07 ✗ | 1.63 ✓ | 141.0 | −10.45 |
| L1013 | **0.35** ✓ | 0.35 | 412.7 | −10.28 |
| L1014 | **0.42** ✓ | 0.42 | n/a | −10.17 |
| L1015 | **1.62** ✓ | 1.62 | 82.7 | −11.45 |
| L1016 | **1.45** ✓ | 0.57 | 100.8 | −10.77 |
| L1017 | **1.79** ✓ | 0.43 | 129.2 | −10.86 |

| metric | count / 17 | rate |
|---|---|---|
| top-1 < 2 Å | **12** | 71 % |
| best-of-5 < 2 Å | **17** | **100 %** |
| overall pool best < 2 Å (generation upper bound) | 17 | 100 % |
| AFFNTY emitted | 15 | 88 % |

The current selector (diversity-aware pRMSD, pool from docking + cofolding)
lands a ≤ 2 Å pose in every target's top-5 — the pipeline is *not*
generation-limited. The 5 top-1 misses are all 2–3 Å.

---

## 2. Pose generation ceiling (exhaustive pool scan)

Per-target scan of every generated pose — vina × 50, autodock-gpu × 50,
protenix-dock × 8, boltz2/2x/protenix/af3 × 25 each. Pool sizes 129–259.

| metric | count / 17 |
|---|---|
| pool has any pose < 2 Å | **17** |
| pool has any pose < 1 Å | 15 |
| pool has any pose < 0.5 Å | 14 |

Per-target minimum RMSD median = **0.35 Å**; best-case L1014 = **0.115 Å**.
Boltz2/2x cofolding routinely produces the pool minimum when the cofolding
protein conformation matches the crystal pocket.

---

## 3. Selection strategy comparison

### 3.1 Docking + cofolding pool (raw, no diversity filter)

| strategy | top-1 < 2 Å | best-of-5 < 2 Å |
|---|---|---|
| pRMSD ↑ (RMSD-Pred primary output) | 9 / 17 | 15 / 17 |
| LSCORE ↓ = `1 − prob(>2Å)` | **11 / 17** | 15 / 17 |
| `pRMSD × prob` product | 10 / 17 | **16 / 17** |
| **diversity-aware pRMSD (LG submission)** | **12 / 17** | **17 / 17** |

`LSCORE` (prob-based) alone outperforms raw `pRMSD` on top-1 because
RMSD-Pred's pRMSD is miscalibrated on some cofolding poses. `pRMSD × prob`
gains 1 on best-of-5 relative to either alone. The diversity filter (≥ 2 Å
pairwise RMSD) is the biggest single improvement — it prevents the ranker
from stacking the top-5 with near-duplicate non-native decoys, forcing one
correct pose into the set for every target.

### 3.2 Cofolding-internal confidence metrics (cofolding pool only) 🆕

For each cofolding sample we load the per-structure confidence JSON
(`boltz`: `confidence_boltz_input_model_N.json`, `protenix`:
`..._summary_confidence_sample_N.json`, `alphafold3`:
`..._summary_confidences.json`). Rank the entire cofolding pool by each
metric (higher = more confident) and report actual top-1 / best-5 RMSD.

| metric | top-1 < 2 Å | best-of-5 < 2 Å |
|---|---|---|
| plddt | 14 / 17 | 16 / 17 |
| **pTM** | **16 / 17 (94 %)** ⭐ | **17 / 17 (100 %)** ⭐ |
| ipTM | 13 / 17 | 16 / 17 |
| ligand_ipTM (boltz only) | 13 / 17 | 16 / 17 |
| confidence_score | 14 / 17 | 17 / 17 |

**pTM alone is the best selector observed anywhere in this study.**

Per-target pTM top-1 RMSD: L1001 0.34 / L1002 0.88 / L1003 1.26 /
L1004 1.55 / L1005 0.72 / L1006 1.28 / L1007 1.30 / L1008 1.99 / L1009 0.85 /
L1010 0.28 / L1011 0.36 / L1012 **2.79** (the only miss) / L1013 0.27 /
L1014 0.21 / L1015 1.12 / L1016 0.40 / L1017 0.73.

### 3.3 Head-to-head summary

| strategy | top-1 < 2 Å | best-5 < 2 Å |
|---|---|---|
| pRMSD raw | 9 / 17 | 15 / 17 |
| LSCORE raw | 11 / 17 | 15 / 17 |
| pRMSD × prob raw | 10 / 17 | 16 / 17 |
| LG diversity-aware pRMSD | 12 / 17 | 17 / 17 |
| **cofold pTM** | **16 / 17** 🏆 | **17 / 17** 🏆 |

**Interpretation**: when the pool is rich in cofolding poses (every L1000
target has ~100 cofold candidates), the cofolding model's own pTM is a
strictly better selector than either the external RMSD-Pred scorer or any
docking-pool heuristic. It shifts top-1 success from 12/17 → 16/17, with
L1012 as the only remaining miss.

**Why pTM wins**: (a) it is the same model's internal confidence about the
entire complex, no cross-model frame / atom-graph mismatch; (b) pTM is a
global structural-quality prior — a cofolding sample with high pTM has a
correct pocket and therefore a correct ligand pose; (c) pTM is reported by
all four cofolding tools with the same definition, so a single scalar
compares poses across models in one pool.

### 3.4 Recommended selector for CASP17

Two paths, both cheap to implement:

1. **Drop-in replacement**: change `select_diverse_top_k` primary key from
   pRMSD to cofold pTM (looked up from the per-pose confidence JSON) with
   diversity filter unchanged. Expected: top-1 16/17, best-5 17/17 on L1000.
2. **Hybrid**: for poses that carry a cofolding provenance, use pTM;
   otherwise fall back to `pRMSD × prob` (best raw metric for docking-only
   poses). Keep diversity filter.

Path 1 is simpler and matches the empirical data.

---

## 4. Affinity prediction vs. experiment

AFFNTY reported by the LG submission = ensemble log-space Kd (nM) from
BA-Pred pKd (all poses) + Boltz affinity_pred_value filtered by
`binder_prob ≥ 0.5`. Of the 17 targets:

- 15 emit AFFNTY (L1005, L1014 fail ensemble due to missing Boltz binder
  components)
- Qualitative binder/non-binder discrimination is usable
- **Quantitative ΔG correlation is weak**: the strongest binder L1001
  (exper ΔG −12.35 = Kd ≈ 1 nM) is predicted at 194 nM, off by ~2 orders
  of magnitude. Predicted Kds cluster in the 20–2000 nM window regardless
  of true potency.

---

## 5. Bugs discovered and fixed

| # | bug | impact | fix |
|---|---|---|---|
| 1 | `SubmissionConfig.include_affinity` defaulted to `False`, yaml didn't override | all LG files shipped without AFFNTY header | default → `True`; yaml flag set; `make_casp_submission.py` unchanged |
| 2 | `evaluate.py` RMSD fallback computed naive index-wise distance on mols without bond orders | reported L1001 top-1 as 5.87 Å (real: 0.575 Å); invalidated first-pass L1000 analysis | replaced with `AssignBondOrdersFromTemplate` + `GetBestRMS`, same logic as `best_pose.py` |
| 3 | `extract_first_smiles` in `run_template_filter.py` broke on block-style YAML (exited ligand mode on the `id: L` line) | every L1000 / novel2025 run passed `None` as target SMILES → MCS/Tanimoto = 0 → **Track 2 (template box docking) and Track 3 (lig-align) never fired**, gated on `mcs_threshold: 0.5` | replaced regex-ish parser with `yaml.safe_load`; 48/48 unit tests still pass |
| 4 | `select_diverse_top_k` primary key was `lscore` desc, but the design doc specified pRMSD asc with LSCORE tie-break | selector never matched its written intent | swapped primary sort; LSCORE remains a reported value on each MODEL |
| 5 | `parse_lg` in `evaluate.py` only captured `AFFNTY` lines *before* MODEL 1, but `make_casp_submission.py` emits AFFNTY just before final `END` | evaluator reported `affnty=None` for all 17 even when the file had it | accept `AFFNTY` anywhere outside a MODEL block |
| 6 | `Chem.MolFromPDBFile(removeHs=True, sanitize=False)` silently keeps hydrogens because `removeHs` depends on sanitization | `evaluate.py` and `best_pose.py` both failed bond reassignment for several experimental ligands (L1010, L1013, L1016) because heavy-atom count didn't match the SMILES template | load with `sanitize=False`, then manual `Chem.RemoveHs(mol, sanitize=False)` |

---

## 6. novel2025 benchmark — status

### 6.1 Pipeline changes (for time-split compatibility)

- `src/casp17/template_filter.py`: added `lookup_deposition_date()` and
  `max_deposition_date: str | None` parameter to `filter_hits_with_ligands`.
  Drops any hit whose RCSB `entries.deposition_date >= cutoff`.
- `scripts/run_template_filter.py`: new `--max-deposition-date` CLI flag.
- `src/casp17/configs.py`: `TemplateSearchSequenceConfig.max_deposition_date`.
- `src/casp17/script_builder.py`: passes the flag through the wrapper
  template-filter invocation when set.
- `experiments/novel2025_test/novel2025_config.yaml`: sets cutoff to
  `2025-01-01`.
- Smoke verification: running the filter on L1001 with cutoff `2000-01-01`
  drops 181/200 hits → pre-2000 survivors only. Running 22mj with cutoff
  `2025-01-01` produces 8 filtered hits, all verified pre-2025.

### 6.2 22mj smoke test (single-target end-to-end)

- **Target**: RCSB `22mj` (chymase-family hydrolase, Homo sapiens,
  deposited 2026-03-11, 322 aa, CCD `A1MDP`, seq_zone=remote, max_id 0.36)
- **Run**: job 24407, 56m47s, exit 0
- **Date filter**: passed — 8 filtered template hits, 0 post-2025 leaks
- **Protein source**: alphafold3 (auto-selected by pLDDT; global CA RMSD
  8.60 Å vs crystal, but pocket-local alignment is clean)
- **AFFNTY**: 3803 nM (no experimental affinity for direct comparison)
- **Ligand pose RMSD** (LG MODELs vs RCSB `22mj.cif.gz` crystal,
  heavy-atom, symmetry-aware):
  - MODEL 1 (vina_seed_101, LSCORE 0.812): **0.61 Å** ✓
  - MODEL 2 (pxdock_pose_1, LSCORE 0.543): 0.48 Å
  - MODEL 3 (vina_seed_101, LSCORE 0.505): 0.64 Å
  - MODEL 4 (vina_seed_42, LSCORE 0.643): 0.83 Å
  - MODEL 5 (vina_seed_202, LSCORE 0.267): 0.77 Å
- **Result**: top-1 **0.61 Å**, best-of-5 **0.48 Å**, 5 / 5 < 1 Å on a
  fully held-out 2025 target with template search restricted to pre-2025.

### 6.3 Full novel2025 target set

Source: `/home/jaemin/DB/RCSB/processed/seqid_zones_2025_nonredundant.tsv`
(user-provided non-redundant 2025 RCSB index).

Curation funnel (`experiments/novel2025_test/build_inputs.py`):

| stage | kept | dropped | reason |
|---|---|---|---|
| **TSV total** | **543** | — | user-provided input |
| `deposition_date >= 2025-01-01` | 543 | 0 | already post-2025 |
| `num_candidate_ligands >= 1` | 543 | 0 | all targets have a candidate |
| single protein chain (no `\|` in sequences) | 502 | 41 | multi-chain complexes |
| `100 <= seq_length <= 600` (GPU budget) | 434 | 68 | out-of-window length |
| candidate SMILES without single-quote | 434 | 0 | YAML safety |
| **YAMLs written** | **434** | **109** total | |

Kept set by `seq_zone`:

| zone | count |
|---|---|
| novel (no homolog) | 57 |
| remote (0.3–0.5 max id) | 104 |
| related (> 0.5 max id) | 273 |

Length: min 120, median 315, max 595 aa.

**Column mapping (TSV → YAML)**:

| YAML field | TSV column | processing |
|---|---|---|
| `protein.sequence` | `sequences` | `split("\|")[0]` (first chain) |
| `ligand.smiles` | `candidate_smiles` | `split("\|")[0]` (first drug-like candidate) |

`candidate_*` is already filtered via `is_candidate=1` in rcsb_index.db, so
crystallization aids (EDO, MLI, PEG, sulfate, …) are excluded automatically.
Example — 22mj raw `EDO|MLI|A1MDP` → candidate = `A1MDP` only.

Caveats:
- 109 multi-candidate targets (2–5 ligands) use **only the first** candidate;
  the rest are silently dropped. Future work: emit one YAML per candidate.
- 41 multi-chain targets were removed earlier at the single-chain filter, so
  every kept target has `sequences == contact_sequences`.

Batch driver: `experiments/novel2025_test/batch_prepare_and_submit.sh`.
Idempotent — skips targets whose wrapper or LG already exists.

**Submission status (as of report update)**:

- 434 wrappers prepared
- 433 sbatch submitted (22mj skipped, LG already exists from §6.2)
- Running: 7 concurrent (gpu1 × 4 on *heavy* partition, gpu3 × 2 and
  gpu4 × 1 on *6000ada*)
- Pending: 427 priority-waiting + 1 resource-waiting
- Partition: each PD job updated to `Partition=heavy,6000ada` so SLURM
  can schedule on whichever has capacity. `gpu1` (H100 + 3 × 6000Pro,
  250 GB RAM) was entirely idle until we added the multi-partition
  routing — it gives +4 concurrent slots on top of the 6000ada queue.
- test partition: considered but rejected (gpu2 has only ~10 GB free RAM
  vs. `--mem=96G` requirement → no capacity even though GPUs are free)
- Revised wall-time estimate: 434 targets / 7 concurrent × ~55 min ≈
  **~57 hours ≈ 2.5 days**, vs. 3–4 days on 6000ada alone.

**10sl / 9ig3 / 9uad** (originally the smoke-test set) are included in
the 433 submissions.

---

## 7. Artifacts

| path | content |
|---|---|
| `experiments/casp16_test/L1000/evaluate.py` | LG → actual RMSD evaluator |
| `experiments/casp16_test/L1000/best_pose.py` | exhaustive pool → per-source RMSD stats |
| `experiments/casp16_test/L1000/compare_selection.py` | pRMSD / LSCORE / pRMSD×prob side-by-side |
| `experiments/casp16_test/L1000/compare_cofold_metrics.py` | cofolding-internal metrics (plddt/pTM/ipTM/ligand_ipTM/confidence) |
| `experiments/casp16_test/L1000/evaluation.json` | per-target LG model table |
| `experiments/casp16_test/L1000/compare_selection.json` | selection strategy comparison raw |
| `experiments/casp16_test/L1000/compare_cofold_metrics.json` | cofolding-metric raw |
| `experiments/casp16_test/L1000/FINAL_REPORT.md` | this document |
| `experiments/novel2025_test/README.md` | benchmark setup description |
| `experiments/novel2025_test/build_inputs.py` | YAML generator from seqid_zones tsv |
| `experiments/novel2025_test/novel2025_config.yaml` | runner config with 2025-01-01 cutoff |
| `experiments/novel2025_test/batch_prepare_and_submit.sh` | wrapper-prep + sbatch driver |
| `experiments/novel2025_test/pipeline/*.yaml` | 434 per-target inputs |
| `experiments/submissions/L10{01..17}_input.lg` | L1000 LG files (with AFFNTY) |
| `experiments/submissions/22mj_input.lg` | novel2025 smoke test LG |

---

## 8. Immediate next steps (unresolved)

1. Batch-prepare + submit the remaining 433 novel2025 wrappers (or a
   curated subset — the hardest 161 = `seq_zone ∈ {remote, novel}` is a
   natural first wave).
2. Consider swapping `select_diverse_top_k` primary key to cofold pTM
   (see §3.4) before the main benchmark run so the novel2025 submissions
   use the better selector.
3. Investigate why AF3 is auto-selected over Boltz for 22mj despite
   having 8.6 Å global CA RMSD — pLDDT ranking probably not calibrated
   across these four models on remote targets.
4. L1005 / L1014 missing AFFNTY: trace ensemble computation to see which
   component failed.
