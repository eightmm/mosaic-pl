# Mosaic-PL Research Workflows

This repository separates reusable workflow code from the local inputs, model
weights, RCSB mirrors, Slurm logs, and CASP submission artifacts needed to run
an experiment. The latter are deliberately not versioned. Every analysis below
therefore records its expected inputs and is intended to be run after the core
pipeline has produced a target run directory.

## Execution Boundary

- The master node is for YAML validation, manifest generation, and lightweight
  inspection. GPU inference, GPU builds, and GPU verification run through
  Slurm compute nodes.
- Validate a target before submitting it with `casp17-pl validate-run`; the
  generated Slurm wrapper records the model configuration and output paths.
- Inputs, search databases, external model checkouts, and target run outputs
  are ignored by Git. Recreate them with the setup scripts and the documented
  YAML configuration rather than committing machine-local artifacts.

## Structural and Pose Analysis

| Question | Utility | Evidence produced |
|---|---|---|
| Do sampled co-folds contain distinct receptor states? | `scripts/cluster_cofolding_ensemble.py` | chain-aware assembly/protomer RMSD clusters |
| Does ligand-specific co-folding move the receptor or pocket? | `scripts/analyze_holo_receptor_variation.py` | global/pocket CA-RMSD and ligand-centroid spread CSV |
| Are all pose sources expressed in the docking receptor frame? | `scripts/check_frame_consistency.py`, `scripts/audit_frame_50.py` | run-wide checks and sampled alignment diagnostics |
| Does a selected pose preserve a valid receptor and ligand layout? | `scripts/lint_lg_submission.py`, `scripts/lint_ligand_series_stage2.py` | CASP LG grammar and series-specific validation |

## Ligand-Series and Pose-Scorer Analysis

`scripts/score_poses.py` evaluates pose ensembles with external research
scorers and writes a per-pose table. `scripts/analyze_stage1_signals.py` and
`scripts/analyze_stage1_bapred.py` join revealed ligand-series labels to the
available signals and report AUC, BEDROC, enrichment factors, and a permutation
baseline. The plotting scripts under `scripts/plot_*` render the associated
summary figures from these tables.

The binder-ranking analyses are retrospective research utilities. They are not
the same as Mosaic-PL's CASP pose-selection path, which uses RMSD-Pred LSCORE
on a receptor-aligned candidate pool.

## Template and Solvent Research

The core template bridge maps ligand-bound templates into the co-fold receptor
frame and clusters their ligand centroids. Supporting utilities make the
underlying structural evidence inspectable:

- `scripts/extract_template_pockets.py` and
  `scripts/cluster_template_pockets.py` generate consensus pocket evidence.
- `scripts/export_solvent_donor_bundle.py`,
  `scripts/cluster_solvent_donors.py`, and
  `scripts/make_solvent_donor_viewer.py` support consensus solvent-site
  analysis for ordered-solvent targets.
- `scripts/make_solvent_ts_submission.py` and
  `scripts/lint_ts_submission.py` construct and validate PFRMAT TS outputs.

Target-specific submission reruns, backups, and generated viewers remain local
operational artifacts and are intentionally excluded from the public history.
