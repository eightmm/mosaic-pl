# CASP17 Ligand-Series Stage-1 Postmortem (L01 / L02)

What our stage-1 binder ranking actually achieved, measured against the real
answers that CASP released with the stage-2 reveal on **2026-07-29**.

Reproduce with:

```bash
curl -s "https://predictioncenter.org/casp17/target.cgi?target=L01&view=smiles" \
  -o inputs/ligand_series/L01/ligands_truth.csv
uv run python scripts/analyze_stage1_signals.py --target L01 --legacy-id-shift
uv run python scripts/analyze_stage1_signals.py --target L02 --restrict-common
```

## The reveal

The truth arrived as a new `binding` column in each target's SMILES CSV — there
was no announcement page and the upload-gateway text did not change.

| target | protein | fragments | real binders | rate |
|---|---|---|---|---|
| L01 | BFT1 / Fragilysin, 364 res | 1210 | 80 | 6.6 % |
| L02 | MmaA1, 271 res | 647 | 29 | 4.5 % |

L01's truth file has **1210** rows; our submitted `L01LG129.bind.txt` had 1209.
The missing row is not a tail truncation — it is the symptom of an id shift that
mislabelled most of the L01 submission. See *Data defect* below.

Stage-2 deadlines are the *Human Expiration* dates: **L01 2026-08-28**,
**L02 2026-08-27**.

## Method

Every signal is normalised to *higher = more likely to bind*, then scored with
AUC, BEDROC (α = 20, weighting roughly the top 8 %) and enrichment factors at
1 / 5 / 10 %. Because BEDROC has no fixed random value, a 2000-draw permutation
baseline is computed per (N, n_actives) pair; `*` marks a signal above the 95th
percentile of that baseline.

Per-pose scores are aggregated per fragment as `best` / `p90` / `mean`. Score
directions follow `scripts/score_poses.py`: `ak_score`, `ak_ens`, `ak_dock`
lower-is-better; `gen_score`, `equi_pred` higher-is-better.

Coverage caveats: L01 has rescored poses for 1207/1210 fragments and Boltz
affinity for 1197. **L02 rescoring never ran** and its docking was still in
flight — only 269/647 fragments had Boltz output at measurement time, so L02's
Boltz rows are computed on that subset and a second table restricts every signal
to the same 269 for an apples-to-apples read. EquiScore is absent from both
targets by design (skipped over an unfixed GPU bug — see
`ligand_series_submission.md`), so stage 1 was a **two**-scorer pipeline.

## Data defect: the L01 id ↔ SMILES shift

Found 2026-07-29 while building the phase-2 fragment list. **The official L01 list
contains one duplicated compound** — `L010356` and `L010357` carry the identical
SMILES `CN1C(=O)CC(CO)C1c1cccs1`. Our scrape pipeline deduplicated it, so
`inputs/ligand_series/L01/ligands.csv` had 1209 rows and **every id from L010357
on was shifted down by one**: our `L010357` actually held official L010358's
molecule. Verified: `local[i] == truth[i+1]` for all 853 positions after the
duplicate, and L02 matches 647/647 (unaffected).

Consequences:

- No molecule was lost — L010357 is chemically identical to L010356, which we did
  run. Only the labels moved.
- **853 of the 1209 submitted L01 lines carried another molecule's score and
  pocket**, and L011210 got no line at all. That is what CASP scored.
- The first version of this postmortem read signals out of run dirs named by
  official id, so its L01 numbers were corrupted the same way. They are corrected
  below; `scripts/analyze_stage1_signals.py --legacy-id-shift` reproduces the
  mapping (official N → dir `L01{N-1}` for N ≥ 358, `L010356` for N = 357).
- `ligands.csv` has been rebuilt from the truth CSV (1210 rows, duplicate kept);
  the shifted file is preserved as `ligands.csv.shifted_bak`.

Phase-2 runs are generated straight from the truth CSV, so they carry official
ids by construction.

## L01 — N = 1210, 80 binders

Random baseline: AUC 0.500 | BEDROC 0.090 (95th pct 0.142) | EF 1.00

Signals are read with the shift corrected. The `submitted` row is **not**
remapped — it is the file as CASP received and scored it.

| group | signal | AUC | BEDROC₂₀ | EF1% | EF5% | EF10% | hits@5% |
|---|---|---:|---:|---:|---:|---:|---|
| submitted | `L01LG129.bind.txt` (as sent) | 0.557 | 0.116 | 0.00 | 1.76 | 1.62 | 7/80 |
| boltz | prob_mean | 0.561 | 0.099 | 1.25 | 1.00 | 1.00 | 4/80 |
| boltz | prob_max | 0.564 | 0.094 | 1.25 | 1.00 | 1.00 | 4/80 |
| boltz | aff_min | 0.598 | 0.124 | 1.25 | 1.25 | 1.12 | 5/80 |
| boltz | aff_mean | 0.606 | 0.121 | 2.50 | 1.25 | 1.12 | 5/80 |
| boltz | **aff / heavy (LE)** | **0.628** | 0.138 | 1.25 | 1.25 | 1.50 | 5/80 |
| akscore2 | ak_score_best | 0.547 | 0.087 | 0.00 | 0.75 | 1.12 | 3/80 |
| akscore2 | ak_score_mean | 0.542 | 0.081 | 1.24 | 0.25 | 1.37 | 1/80 |
| akscore2 | ak_ens_best | 0.498 | 0.069 | 0.00 | 0.25 | 1.00 | 1/80 |
| akscore2 | ak_ens_mean | 0.526 | 0.070 | 1.24 | 0.50 | 1.00 | 2/80 |
| akscore2 | ak_dock_best | 0.561 | 0.137 | **2.49** | 1.50 | 1.37 | 6/80 |
| akscore2 | ak_dock_mean | 0.544 | 0.086 | 1.25 | 0.50 | 1.37 | 2/80 |
| genscore | **gen_score_mean** | **0.636** | 0.133 | 0.00 | 1.76 | 1.62 | 7/80 |
| genscore | gen_score_p90 | 0.625 | 0.141 | 0.00 | 1.51 | 1.62 | 6/80 |
| genscore | gen_score_best | 0.620 | **0.154** \* | 0.00 | 2.01 | 1.75 | 8/80 |
| molprop | **QED** | 0.626 | **0.182** \* | **2.52** | 2.27 | 2.12 | 9/80 |
| molprop | **heavy_atoms** | 0.624 | 0.128 | 1.26 | 1.51 | 1.50 | 6/80 |
| molprop | cLogP | 0.622 | 0.057 | 0.00 | 0.25 | 0.62 | 1/80 |
| molprop | MW | 0.609 | 0.124 | 1.26 | 1.26 | 1.62 | 5/80 |
| molprop | rings | 0.600 | 0.062 | 0.00 | 0.76 | 1.25 | 3/80 |
| molprop | HBA | 0.564 | 0.130 | 1.26 | 1.76 | 1.62 | 7/80 |
| molprop | arom_rings | 0.562 | 0.071 | 1.26 | 1.76 | 1.25 | 7/80 |
| molprop | rot_bonds | 0.545 | 0.043 | 0.00 | 1.26 | 1.00 | 5/80 |
| molprop | TPSA | 0.514 | 0.131 | 2.52 | 2.02 | 1.38 | 8/80 |
| molprop | HBD | 0.484 | 0.067 | 1.26 | 0.50 | 0.88 | 2/80 |
| molprop | fracCsp3 | 0.457 | 0.014 | 0.00 | 0.00 | 0.25 | 0/80 |
| molprop | formal_chg | 0.449 | 0.031 | 1.26 | 0.76 | 0.38 | 3/80 |

## L02 — N = 647, 29 binders

Random baseline: AUC 0.500 | BEDROC 0.075 (95th pct 0.151) | EF 1.00

| group | signal | n | AUC | BEDROC₂₀ | EF1% | EF5% | EF10% |
|---|---|---:|---:|---:|---:|---:|---:|
| submitted | `L02LG129.bind.txt` | 647 | **0.626** | **0.180** \* | **7.44** | 2.09 | 1.72 |
| boltz | prob_mean | 269 | 0.621 | 0.079 | 0.00 | 0.00 | 1.42 |
| boltz | prob_max | 269 | 0.617 | 0.074 | 0.00 | 0.00 | 1.42 |
| boltz | aff_min | 269 | 0.536 | 0.039 | 0.00 | 0.00 | 0.71 |
| boltz | aff_mean | 269 | 0.538 | 0.041 | 0.00 | 0.00 | 0.71 |
| boltz | aff / heavy (LE) | 269 | 0.498 | 0.053 | 0.00 | 0.00 | 0.71 |
| molprop | arom_rings | 647 | 0.597 | 0.129 | 3.72 | 2.09 | 1.03 |
| molprop | **HBA** | 647 | 0.573 | **0.178** \* | **7.44** | 2.09 | 2.06 |
| molprop | rings | 647 | 0.544 | 0.112 | 3.72 | 1.39 | 1.03 |
| molprop | formal_chg | 647 | 0.538 | 0.090 | 0.00 | 2.09 | 1.72 |
| molprop | TPSA | 647 | 0.522 | 0.060 | 0.00 | 0.70 | 0.69 |
| molprop | heavy_atoms | 647 | 0.512 | 0.060 | 0.00 | 0.70 | 1.03 |
| molprop | MW | 647 | 0.509 | 0.036 | 0.00 | 0.00 | 0.69 |
| molprop | cLogP | 647 | 0.484 | 0.074 | 0.00 | 1.39 | 1.37 |
| molprop | HBD | 647 | 0.482 | 0.093 | 3.72 | 1.39 | 1.03 |
| molprop | QED | 647 | 0.464 | 0.042 | 0.00 | 0.70 | 0.69 |
| molprop | fracCsp3 | 647 | 0.448 | 0.030 | 0.00 | 0.70 | 0.34 |
| molprop | rot_bonds | 647 | 0.410 | 0.005 | 0.00 | 0.00 | 0.34 |

Restricted to the 269 Boltz-covered fragments (14 binders), the submitted score
reads AUC 0.636 but BEDROC 0.045 — *below* that subset's random mean of 0.081.
L02's apparent early enrichment comes entirely from the 378 fragments that had no
Boltz signal at all, i.e. from the rank-fusion fallback rather than from a model.

## Findings

1. **AUC and BEDROC disagree, and the disagreement is the story.** cLogP ranks
   third on AUC (0.622) with a BEDROC of 0.057 — *below* random: greasier
   fragments sit slightly higher throughout the list without ever reaching the top
   of it. Conversely `ak_dock_best` is only AUC 0.561 but has the best top-1 % hit
   rate of any docking signal (EF1% 2.49). Global rank quality and early
   enrichment are different properties; report both.

2. **GenScore is the strongest single signal once the ids are right** — AUC
   0.620-0.636 across aggregates, top of the whole table. Its BEDROC (0.133-0.154)
   barely clears random, so it orders the library without surfacing the binders.
   AKScore2's `ak_ens` stays near-random in its correct (lower-is-better)
   direction: best 0.498, mean 0.526.

3. **Descriptors are competitive but not ahead.** QED 0.626 and heavy_atoms 0.624
   sit just under GenScore 0.636 and Boltz ligand efficiency 0.628. QED does keep
   the best early enrichment of anything measured (BEDROC 0.182, EF1% 2.52), and
   it inverts on L02 (0.464) — library chemotype bias, not a transferable prior.
   The first version of this document claimed descriptors beat every GPU-computed
   score; that was an artifact of the id shift.

4. **The two targets behave oppositely.** L01 is strongly size- and
   lipophilicity-biased (heavy_atoms 0.624). L02 is not (heavy_atoms 0.512,
   MW 0.509); its weak signals are aromatic-ring count (0.597) and HBA (0.573),
   consistent with MmaA1 being a SAM-dependent methyltransferase with a polar,
   aromatic-friendly site.

5. **Rank fusion diluted rather than combined.** Measured on the (shifted)
   stage-1 mapping: heavy alone 0.624 > heavy+gen 0.618 > heavy+gen+boltz 0.611 >
   heavy+gen+boltz+ak_dock 0.598. Equal-weight RRF over signals that are
   individually near-random drags the best one down. Not recomputed on the
   corrected mapping — the ordering of the inputs changed, so treat the exact
   figures as indicative and re-measure before reusing any fusion.

6. **Much of the apparent skill is molecule size.** On the shifted mapping,
   regressing heavy-atom count out dropped gen_score_p90 0.578 → 0.560 and
   boltz_prob 0.540 → 0.524. The size-independent exception is **ligand
   efficiency**: `boltz_aff / heavy_atoms` scores 0.628 on the corrected mapping
   versus 0.598 for raw `boltz_aff_min`. Re-run the residualisation on the
   corrected mapping before quoting the other two numbers.

7. **EF1% is ~0 almost everywhere.** On L01 most signals put zero binders in the
   top 12. As a practical screen, stage 1 was unusable.

## Consequences for stage 2

Binder identity is now given, so ranking quality no longer matters — the same
scores are only used for **pose selection** among the ~870 poses per fragment.
Carry forward: prefer GenScore, keep `ak_dock`/`ak_score` as a secondary vote,
leave `ak_ens` out until its behaviour is understood, and do not re-use
equal-weight rank fusion without validating it on a labelled set first.

Related: `docs/ligand_series_submission.md` (format + pipeline),
`docs/pose_ranker_design.md` (pose-level ranking work).
