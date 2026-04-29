# L2000 (CASP16 test target) — Debug & Re-run Log

## Target

- **Source**: CASP16 held-out ligand-protein target L2000 (*cathepsin*)
- **Files**: `experiments/casp16_test/L2000/L2000_cathepsin_README.txt`, `L2000.SMILES.tar.gz`, `L2000_exper_struct.tar.gz`
- **Split**: The target is decomposed into two sub-runs for the pipeline
  - `L2001` → receptor + ligand set 1 (`pipeline/L2001_input.yaml`)
  - `L2002` → receptor + ligand set 2 (`pipeline/L2002_input.yaml`)
- **Goal**: End-to-end validation of the CASP17 protein-ligand hub on a real
  CASP16 ligand case, from co-folding through CASP17 LG submission.

## Environment

| | |
|---|---|
| Timezone | KST (UTC+9) |
| Host | `master` (147.46.139.205), Ubuntu, Linux 6.8.0-90-generic, ~314 GB RAM, no GPU |
| Scheduler | SLURM; jobs placed on partition `6000ada` (RTX 6000 Ada × 8, nodes `gpu3`, `gpu4`) |
| Repo | `/home/jaemin/project/CASP17` |
| Git HEAD (before fixes) | `c64e698` — *feat: integrate post-analysis + CASP submission into wrapper pipeline* |
| Python (all venvs) | 3.12.3 |
| venvs in use | `.venv` (hub), `.venvs/boltz` (Boltz2/2x), `.venvs/protenix` (Protenix), `.venvs/alphafold3` (AF3), `.venvs/protenix-dock` (Vina + PxDock + AutoDock-GPU runner), `.venvs/pred` (BA-Pred + RMSD-Pred) |
| Runner config | `examples/casp_submission_config.yaml` (multi-seed ensemble: 5 cofolding seeds × 5 samples + 5 docking seeds) |
| Job IDs (first run, buggy) | `23937` (L2001), `23938` (L2002) |
| Job IDs (second run, fixed) | `23941` (L2001), `23942` (L2002) |

## Timeline (KST)

### First run — buggy

| Event | Time |
|---|---|
| L2001 queued + started (`23937` on `gpu3`) | 2026-04-10 15:31:46 |
| L2001 completed | 2026-04-10 16:22:06 (elapsed 00:50:20, exit 0:0) |
| L2002 queued | ~2026-04-10 16:00 (PD, Resources) |
| L2002 started (`23938` on `gpu3`) | 2026-04-10 16:22:06 |
| L2002 completed | 2026-04-10 17:13:44 (elapsed 00:51:38, exit 0:0) |
| User flagged "L2000 어디까지됐니?" → debug session opened | 2026-04-10 17:10 |

Both jobs reported exit 0 at the SLURM level, but the contents of
`outputs/` revealed that several downstream stages had failed silently
(only Boltz/Protenix/AF3 co-folding produced usable artifacts).

### Debug session

| Event | Time |
|---|---|
| Root-cause analysis begins | 2026-04-10 17:10 |
| Three classes of bugs identified | 2026-04-10 17:15 |
| Fixes implemented (`adapters.py`, `run_post_analysis.py`, `compute_submission_scores.py`, `make_casp_submission.py`) | 2026-04-10 17:18 |
| ADG adapter smoke-test passes (GPF generated with correct center + F map) | 2026-04-10 17:22 |
| Full pytest suite (48 tests) passes | 2026-04-10 17:24 |
| Buggy run dirs archived → `experiments/runs/archive/2026-04-10_buggy_pipeline/` | 2026-04-10 17:26 |
| Wrappers regenerated via `casp17-pl prepare-wrapper` | 2026-04-10 17:27 |
| L2001 re-submitted (`23941`) | 2026-04-10 17:27:43 |
| L2002 re-submitted (`23942`) | 2026-04-10 17:27:44 |

### Second run — fixed (AF3 receptor, pLDDT scale bug still present)

| Job | Start | End | Elapsed | Exit |
|---|---|---|---|---|
| 23941 (L2001) | 2026-04-10 17:27:43 | 2026-04-10 18:21:09 | 00:53:26 | 0:0 |
| 23942 (L2002) | 2026-04-10 18:21:09 | 2026-04-10 19:13:17 | 00:52:08 | 0:0 |

ADG + Vina + PxDock all succeeded, LG files generated. But `prepare_docking_inputs.py`
auto-selected AlphaFold3 (pLDDT 32.6) because Boltz pLDDT [0,1] was compared raw against
AF3 [0,100]. **All docking poses were in a 20 Å wrong frame.**

### Third run — pLDDT scale fix + Boltz receptor + no PxDock (docking only, cofolding reused)

| Event | Time |
|---|---|
| pLDDT scale bug fixed (`read_confidence_score` × 100 for Boltz) | 2026-04-11 14:15 |
| L2001 docking prep (manual): **boltz2x selected** (pLDDT 91.9) | 2026-04-11 14:18 |
| L2002 docking prep (manual): **boltz2 selected** (pLDDT 92.4) | 2026-04-11 14:18 |
| L2001 submitted (23950, docking only, PxDock disabled) | 2026-04-11 14:19:36 |
| L2002 submitted (23951) | 2026-04-11 14:19:37 |
| L2001 completed | 2026-04-11 14:23:43 (00:04:07) |
| L2002 completed | 2026-04-11 14:23:35 (00:03:56) |
| BA-Pred protein sanitize fix pulled (partial sanitize) | 2026-04-12 13:45 |
| Post-analysis rerun (CPU, BA-Pred + RMSD-Pred) | 2026-04-12 13:47-13:55 |
| LG submissions regenerated (top-5 diverse, AFFNTY ensemble) | 2026-04-12 14:10 |
| Reference analysis (analyze_reference.py) | 2026-04-12 14:14 |

### Final results — third run

**Protein alignment (Kabsch vs 1CGH crystal):**

| Run | Receptor | pLDDT | CA RMSD after Kabsch |
|---|---|---|---|
| L2001 | boltz2x | 91.9 | **1.13 Å** ✅ |
| L2002 | boltz2 | 92.4 | **1.65 Å** ✅ |

**Ligand pose quality:**

| Run | Best actual RMSD | Best by LSCORE | Box↔Crystal dist | Issue |
|---|---|---|---|---|
| L2001 | 38.9 Å ❌ | 40.0 Å | **37.0 Å** | P2Rank predicted wrong pocket |
| L2002 | **2.93 Å** 🟡 | **2.94 Å** | normal | Pocket correct, near-miss on 2Å |

**L2002 correlations** (meaningful — correct pocket):
- pRMSD↔actual Spearman: 0.58
- LSCORE↔actual Spearman: −0.51 (correct sign: higher LSCORE = lower actual RMSD)

**LG submissions (top-5 diverse, ≥ 2Å pairwise RMSD):**

L2001:
```
MODEL 1: vina_seed_42_0   LSCORE=0.994  pKd=5.50  actual=40.0 Å (wrong pocket)
MODEL 2: vina_seed_202_1  LSCORE=0.991  pKd=5.75
MODEL 3: vina_seed_303_7  LSCORE=0.773  pKd=5.02
MODEL 4: vina_seed_404_2  LSCORE=0.708  pKd=6.61
MODEL 5: vina_seed_404_5  LSCORE=0.662  pKd=5.06
AFFNTY: 362 nM
```

L2002:
```
MODEL 1: vina_seed_202_0  LSCORE=0.881  pKd=6.09  actual=2.94 Å
MODEL 2: vina_seed_42_3   LSCORE=0.645  pKd=4.97
MODEL 3: vina_seed_303_3  LSCORE=0.641  pKd=5.95
MODEL 4: vina_seed_101_8  LSCORE=0.594  pKd=5.26
MODEL 5: vina_seed_101_7  LSCORE=0.471  pKd=4.81
AFFNTY: 772 nM
```

## Observed failures in the first run

Evidence gathered from `experiments/logs/slurm-23937.{out,err}` and from
inspecting `experiments/runs/L2001_input/outputs/` before archiving:

1. **AutoDock-GPU — never produced a `.dlg`**
   - `slurm-23937.out`: `Error in setup of Job #1 / Run time of entire job set
     (1 file): 4.249 sec / The job was not successful.` for every seed.
   - `slurm-23937.out` also logged `AutoDock-GPU: center=[0.0, 0.0, 0.0]`
     while `docking_prep_summary.json` had correct `[-6.6527, -4.4797,
     -9.7048]`.
   - Grid generation succeeded (`autogrid4: Successful Completion.`) but the
     generated GPF listed `ligand_types A C HD N NA OA SA` while the ligand
     pdbqt contained `A C F HD N OA` — no `F` map was ever built, so
     AutoDock-GPU aborted in setup.

2. **BA-Pred — `'NoneType' object has no attribute 'GetNumAtoms'`**
   - Stack from `slurm-23937.out`:
     ```
     File ".../bapred/data/data.py", line 277, in mol_to_graph
         n = mol.GetNumAtoms()
     AttributeError: 'NoneType' object has no attribute 'GetNumAtoms'
     ```
   - BA-Pred was called with the *input* ligand SDF
     (`inputs/docking/ligand_L.sdf`), which holds an RDKit-embedded 3D
     conformer that is not aligned with the receptor.
   - Inside BA-Pred, `_process_sdf` returns a valid ligand Mol, then
     `mol_to_graph` builds a protein graph from atoms within 8 Å of the
     ligand (`Chem.MolFromPDBBlock(total_lines)`). Because the ligand
     coordinates are arbitrary, **zero** protein atoms fall inside the 8 Å
     shell, `total_lines` is empty, `MolFromPDBBlock("")` returns `None`, and
     the pipeline explodes on `pmol.GetNumAtoms()`.
   - RMSD-Pred was not affected because it does not do the 8 Å protein-shell
     trick — but its output on the unaligned input was meaningless
     (`pRMSD=6.74`, `LSCORE=0.012`).

3. **CASP17 LG submission — `Unsupported pose file format: .`**
   - `slurm-23937.err`:
     ```
     File "scripts/make_casp_submission.py", line 441, in main
         pose_to_mdl(pose_path, ligand_mdl_file)
     File "scripts/make_casp_submission.py", line 228, in pose_to_mdl
         raise ValueError(f"Unsupported pose file format: {pose_path}")
     ValueError: Unsupported pose file format: .
     ```
   - `compute_submission_scores._find_pose_file` looked for non-multi-seed
     paths (`outputs/vina/docked.pdbqt`, `outputs/autodock_gpu/docking.dlg`,
     `outputs/protenix_dock/*.sdf`) while the real layout uses
     `outputs/vina/seed_*/docked.pdbqt`, `outputs/autodock_gpu/seed_*/docking.dlg`,
     and `outputs/protenix_dock/*_out.json`. Nothing matched, `pose_file`
     fell back to `Path("")`, and `pose_path.suffix` became `""` → the
     error above.
   - As a side effect, the best pose selected was the `input_sdf` entry
     (only source that yielded any scores, via RMSD-Pred), which naturally
     has no file path either.

## Root causes

| # | File | Root cause |
|---|---|---|
| 1 | `src/casp17/adapters.py` | `prepare_autodock_gpu` called `_load_docking_prep_summary(run_dir)` at adapter time (before the docking-prep bridge runs), got `None`, and hard-coded `center = 0.0`. The generated wrapper script had a `if center[0] is None: center = prep[...]` runtime fall-back, but `0.0 is None` is `False`, so the fall-back never fired. |
| 2 | `src/casp17/adapters.py` | `ligand_types` / map list in the generated wrapper were hard-coded to `A C HD N NA OA SA`. Any ligand containing non-standard atom types (F, Cl, Br, P, I, Si, …) aborts AutoDock-GPU in setup because the GPU binary looks up missing atom-type maps. |
| 3 | `scripts/run_post_analysis.py` | `find_ligand_files` only knew the single-seed layout and did not understand Protenix-Dock's JSON-only output. It also registered `inputs/docking/ligand_*.sdf` as a pose, which fed an unaligned conformer to BA-Pred and crashed the 8 Å protein-shell extraction inside BA-Pred. |
| 4 | `scripts/compute_submission_scores.py` | `_find_pose_file(tool)` only looked at flat `outputs/{tool}/docked.*` paths and returned `None` for the multi-seed layout. Downstream code silently used `Path("")`. |
| 5 | `scripts/make_casp_submission.py` | `pose_to_mdl` had no defensive guard for empty / missing `pose_path`, so when the upstream selector returned `Path("")`, the error message was the unhelpful `Unsupported pose file format: .`. It also always picked the first record of any multi-record SDF, so even if the path was correct the best-pose index was dropped on the floor. |

## Fixes applied

All edits are staged in the working tree (uncommitted at log-write time):

```
 M scripts/compute_submission_scores.py
 M scripts/make_casp_submission.py
 M scripts/run_post_analysis.py
 M src/casp17/adapters.py
```

| File | Change |
|---|---|
| `src/casp17/adapters.py` | `prepare_autodock_gpu` no longer calls the prep-summary loader itself. The generated wrapper always re-reads `docking_prep_summary.json` at runtime and loads `receptor_pdbqt`, `ligand_pdbqt`, `box_center`, `box_size`. Explicit runner-config values override the prep summary (via `user_center` / `user_size` arrays). |
| `src/casp17/adapters.py` | The generated wrapper parses the ligand pdbqt at runtime (cols 77-79 of `ATOM`/`HETATM` lines), derives the ligand atom type set, emits matching `ligand_types` + `map receptor.{T}.map` lines, and unions with the standard protein set for `receptor_types`. Any future ligand with F/Cl/Br/P/I/Si now just works. |
| `src/casp17/adapters.py` | Deleted the dead adapter-time GPF write that was shadowed by the per-seed runtime GPF. |
| `scripts/run_post_analysis.py` | `find_ligand_files` rewritten for multi-seed. Each docked pose file is staged under `outputs/analysis/poses/{tool}_{seed}{ext}` with a unique stem (so BA-Pred/RMSD-Pred's `{basename}_{idx}` pose naming does not collide across seeds), and parallel `.sdf` copies are produced via Meeko's `mk_export.py`. A `.txt` list file is passed to the predictors. |
| `scripts/run_post_analysis.py` | Added `_pxdock_json_to_sdf` — converts Protenix-Dock's atom-mapped-SMILES + `ligand.xyz` pose dumps into a multi-record SDF via RDKit. Verified on L2001 real data: 8 poses converted cleanly. |
| `scripts/run_post_analysis.py` | The raw input ligand SDF is **no longer registered** as a post-analysis target. Docking outputs only. |
| `scripts/compute_submission_scores.py` | New `_resolve_pose_file(run_dir, tool, pose_name)` + `_split_pose_name` parses the record index from the pose name (`vina_seed_42_3` → `vina_seed_42`, idx 3) and looks up the concrete staged SDF in `outputs/analysis/poses/`. Falls back to Protenix-Dock's combined `poses.sdf` and to legacy flat layouts. |
| `scripts/make_casp_submission.py` | `pose_to_mdl` gains a `pose_index` parameter. `_sdf_to_mdl` / `_pdbqt_to_mdl` advance the RDKit supplier to the correct record before writing MDL. Empty / missing `pose_path` now raises a helpful error naming the likely cause. `main` derives the index from `best_pose.pose_name` via `_split_pose_name`. |

## Verification

- `uv run python -m pytest tests/ -q` → **48 passed** at HEAD + working tree.
- ADG adapter smoke test (`/tmp/adg_smoke`): generated wrapper re-reads
  `docking_prep_summary.json`, emits a GPF with `gridcenter -6.6527 -4.4797
  -9.7048`, `ligand_types A N F HD C OA`, `receptor_types A C HD N NA OA SA
  F`, and `map receptor.F.map`. `autogrid4` completes cleanly. The final
  `autodock_gpu_128wi` call SIGABRTs as expected on the GPU-less master
  node.
- `find_ligand_files` on the archived L2001 data: registers `vina` as a
  staged 5-seed list of SDFs, and `protenix_dock` as a freshly generated
  `poses.sdf` with 8 records. `autodock_gpu` correctly absent because the
  buggy run produced zero DLG files.
- `aggregate → pose_to_mdl` round-trip on mocked BA/RMSD TSVs over the
  staged L2001 SDFs: 59 PoseScores built, best-pose resolved to a concrete
  file + record index, `ligand.mol` written with 5.4 kB of valid MDL V2000.

## Artifacts preserved

- `experiments/runs/archive/2026-04-10_buggy_pipeline/L2001_input/` — full
  buggy run dir (cofolding outputs are valid, docking/post-analysis/submission
  are the failure evidence).
- `experiments/runs/archive/2026-04-10_buggy_pipeline/L2002_input/` — same
  for the L2002 half.
- `experiments/runs/archive/2026-04-10_buggy_pipeline/slurm-2393{7,8}.{out,err}`
  — copied from `experiments/logs/`.

## Follow-ups

- Commit the fix set as a single `fix(adg/post/submission): …` commit once
  the second run confirms a clean `.lg` submission.
- Once the L2001 / L2002 second run finishes, append the `sacct` rows to the
  timeline table above and note the final BA-Pred / RMSD-Pred / LSCORE /
  AFFNTY values.
- Consider adding a pytest smoke test that regenerates the ADG wrapper for
  a ligand with a non-standard atom type (e.g. fluorine) and asserts that
  the emitted GPF contains the right `ligand_types` / `map` lines. Would
  have caught bug 2 on the bench.
