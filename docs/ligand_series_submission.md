# CASP17 Ligand-Series (Lxx) Submission Method

Understanding of the CASP17 **ligand-series** experiment submission for targets
`L01` (BFT1 / Fragilysin, 364 res) and `L02` (MmaA1, 271 res). Source: official
Ligand Series Prediction Upload Form + target pages
(`predictioncenter.org/casp17/target.cgi?id=221` L01, `id=222` L02).

**This is NOT the standard single-LG-file flow.** Dedicated gateway, two stages.

**Timeline** (per the upload form): stage 1 gets **3 weeks** (submit before the
target's *Soft Deadline*), stage 2 gets **5 weeks** (submit before *Human
Expiration*). Both dates come from the Target List page.

## Scale (critical)
- **L01 = 1209 fragments** to screen (L010001 … L011209).
- **L02 = 647 fragments** (L020001 …).
- Approach taken (2026-07): a **per-fragment holo run** of the full pipeline
  (receptor + that fragment + affinity → cofold → docking → post-analysis). This
  is only affordable because the receptor is identical across fragments, so the
  AF3 MSA is computed once and reused by every run — see the pipeline notes in
  `experiments/ligand_series/`. An earlier plan (fixed-receptor dock-only screen,
  `scripts/screen_ligand_series.py`) was dropped: apo docking misses induced fit.

## Stage 1 — binding + pocket  (`LXXLGYYY.bind.txt`)
One text file, one line per fragment:
```
L010001 0.95    A75,A77,A155,B10,B20
L010002 0
L011234 0.75    A5,A7,A355,A357
```
- col1 = ligand id `Lxx` + 4-digit number (our ranked.json `name` "010001" →
  prefix "L" → `L010001`).
- col2 = **binding probability** in [0,1]; `0` = predicted non-binder.
- col3 = **pocket residues** = every receptor residue with ≥1 heavy atom within
  **5 Å** of any ligand heavy atom, `ChainResnum` (e.g. `A75`), comma-separated.
  Omitted when prob 0.
- Evaluation: (a) ranking task — discriminate binders from non-binders by score;
  (b) pocket-prediction task on the residues.
- `YYY` = numeric **group number** = **129** (confirmed by the user 2026-07-28).
  Distinct from the registration code `6095-5696-9732` (group LCDD).
  → stage-1 files are **`L01LG129.bind.txt`** and **`L02LG129.bind.txt`**.

## Stage 2 — poses for actual binders (`L01LG129.tgz`)
Revealed after stage 1: the true binding answers are given (but NOT the pocket
identities), and only the compounds that actually bind must be modeled. Per
complex CASP provides a FASTA of the receptor sequence + a text file with the
compound SMILES **and how many copies of the ligand bind**.

**The reveal landed 2026-07-29** as a new `binding` column in the target SMILES
CSV (no announcement page, gateway text unchanged):

```bash
curl -s "https://predictioncenter.org/casp17/target.cgi?target=L01&view=smiles" \
  -o inputs/ligand_series/L01/ligands_truth.csv
```

- **L01: 80 binders of 1210** (the truth file has 1210 rows; our stage-1 file had
  1209 — the extra `L011210` is a non-binder).
- **L02: 29 binders of 647.**
- Deadlines (*Human Expiration*): **L01 2026-08-28, L02 2026-08-27**.
- The promised per-complex FASTA + SMILES + **copy count** was still not
  published as of 2026-07-29 — only the binding column. Re-check before building
  poses for any multi-copy ligand.

### Stage-2 spec, superseded twice (message board 2026-07-30 and 2026-08-07)

The binder list and the per-complex composition both changed after the first
reveal. **The 2026-08-07 files are authoritative**; everything modeled before
that date is built against a wrong ligand set.

1. **2026-07-30** (news 581): the 2026-07-29 answer file for L01 contained a
   duplicated SMILES record, so its ids were shifted. Replaced; binder count
   restated as 79.
2. **2026-08-07** (news 582): after further checks, one segment marked
   non-binder turns out to be a binder — **L01 is back to 80 binders**, L02
   stays at 29. Two things that were never in any earlier file also arrive:
   - **Every L01 receptor carries a Zn²⁺; every L02 receptor carries an SFG
     cofactor** (sinefungin, the SAM analogue). Both must be in the model.
   - **Per-fragment stoichiometry.** 43 of the 80 L01 complexes and 5 of the 29
     L02 complexes contain **more than one copy of the same fragment**.

Authoritative files (vendored as `inputs/ligand_series/{L01,L02}/ligands_stage2.csv`):

```bash
curl -s "https://predictioncenter.org/download_area/CASP17/extra_experiments/ligands/L01.smiles.stage2.csv" \
  -o inputs/ligand_series/L01/ligands_stage2.csv
curl -s "https://predictioncenter.org/download_area/CASP17/extra_experiments/ligands/L02.smiles.stage2.csv" \
  -o inputs/ligand_series/L02/ligands_stage2.csv
```

Columns repeat per component — `CASP target, ligand code, canonical_smiles,
stoichiometry, ligand code, canonical_smiles, stoichiometry` — so parse them
**positionally**, not with `csv.DictReader` (duplicate header names collapse).
CASP16-style per-target SMILES files are also published as
`L0{1,2}.smiles.tgz` in the same directory.

**Ligand naming in the LG file** (news 581): the MDL section header takes the
binder id from the answers file, e.g. `LIGAND 1 L010016`.

**Deadlines extended to compensate**: **L01 2026-09-11, L02 2026-09-10**.
Resubmission is explicitly encouraged and *only the newest submission is
assessed*, so a rebuilt tarball fully replaces anything sent earlier.

How our stage-1 ranking actually scored (L01 AUC 0.557, L02 0.626, EF1% ≈ 0) and
which signals carry what: `docs/ligand_series_stage1_postmortem.md`, reproducible
via `scripts/analyze_stage1_signals.py`.

One model per complex: receptor coordinates (PDB) + the binder(s) (MDL), in
CASP17 LG format (`docs/casp17_lg_format.md`), named `L010001LG129_1`. Put the
models in `./L01`, then:
```bash
tar -czf L01LG129.tgz ./L01
```
Upload the single tarball before the stage-2 expiration date.

## What we already have (as of 2026-07-28)
Per-fragment holo runs under `experiments/ligand_series/{L01,L02}/holo/<fragID>/`:
- **L01: 1207 / 1209 complete** (poses under `outputs/analysis/poses/*.sdf`,
  ≈870 poses per fragment across vina/adg × binding-site source × seed).
  The 2 missing are `L010202` (`[2H]O[2H]`, deuterated water — not dockable) and
  `L011170` (fused cage polycycle, RDKit/meeko embed failed).
- **L02: docking still running.**
- Per fragment also: Boltz-2 `affinity_*.json` (one per boltz2/boltz2x × seed)
  with `affinity_pred_value` (lower = stronger) and
  `affinity_probability_binary` (0-1 binder probability) — available even when
  docking produced no poses.
- Rescoring of every pose with AKScore2 + GenScore →
  `outputs/scoring/pose_scores.csv` (`scripts/score_poses.py`, fed by
  `scripts/feed_rescore_queue.py`). EquiScore was skipped for this batch (unfixed
  `SequentialDistributedSampler` GPU bug), so it is a **two**-scorer pipeline.
  Score directions: `ak_score` / `ak_ens` / `ak_dock` lower-better, `gen_score`
  higher-better. (`ak_ens` was mislabelled higher-better in `score_poses.py`'s
  `score_directions` metadata until 2026-07-29; that dict is written to the
  summary JSON only and was never read by code.)

## Stage 1 — what shipped (2026-07-29)
Both files were submitted and accepted. `scripts/make_bind_txt.py` writes the
Boltz-probability variant, `scripts/make_bind_fused.py` the rank-fused variant;
pocket residues come from the cofold complex at 5 Å either way. Fragments with no
signal at all get a bare `0`.

Measured against the released answers, the ranking task scored close to nothing
(L01 AUC 0.557 / EF1% 0.00, L02 AUC 0.626). Full per-signal breakdown, the
permutation baselines, and why rank fusion hurt:
`docs/ligand_series_stage1_postmortem.md`.

## Stage 2 — what exists locally (as of 2026-08-10)

Per-binder runs under `experiments/ligand_series/{L01,L02}/phase2/<id>/`, built
from the 2026-07-29 answer file, i.e. **before** the 08-07 correction:

| | dirs | analysis done | matches the 08-07 spec |
|---|---:|---:|---|
| L01 | 80 | 72 | no — 58 dirs mislabelled, no Zn, single copy only |
| L02 | 29 | 29 | no — no SFG, single copy only |

No `submission_scores.json` and no LG file has been produced for either.

The mislabelling is *only* in the names: matching each run's modeled SMILES
against the 08-07 CSV gives a clean 1:1 map (saved as
`inputs/ligand_series/{L01,L02}/phase2_dir_to_official.json`, generated by
comparing `inputs/alphafold3_input.json` ligand SMILES to the CSV). So the
docking work is reusable — but the receptor composition is not, because none of
these runs contains the Zn/SFG cofactor or the extra fragment copies.

Two gaps beyond relabelling:
- official **`L010123` was never modeled** (it is not the shifted image of any
  local dir), and one local dir maps onto an id another dir already claims;
- 8 L01 dirs never finished post-analysis: `L011160 L011161 L011167 L011168
  L011178 L011180 L011200 L011208`.

## Stage 2 — the `phase2v2` runs (2026-08-14)

`phase2` above is superseded. `experiments/ligand_series/{L01,L02}/phase2v2/`
holds 80 + 29 runs rebuilt against the 2026-08-07 answer file, so each carries
its cofactor (Zn / SFG) and the right number of fragment copies in the input.
All 109 have Boltz + Protenix + AF3 structures, Track-1 docking and
post-analysis.

### Build

```bash
# 1. one LG per run, from the poses already on disk (CPU, ~14 s/run)
uv run python scripts/rebuild_ligand_series_lg.py --series L01 L02
sbatch experiments/ligand_series/rebuild_lg.sbatch.sh
#    → experiments/ligand_series/{L01,L02}/submissions_stage2/<id>_LCDD.lg

# 2. rewrite them into the stage-2 form and tar
uv run python scripts/make_ligand_series_stage2.py --series L01 L02
#    → experiments/ligand_series/stage2/{L01,L02}/<id>LG129_1
#      experiments/ligand_series/stage2/{L01,L02}LG129.tgz

# 3. gate
uv run python scripts/lint_lg_submission.py experiments/ligand_series/stage2/L0*/*
uv run python scripts/lint_ligand_series_stage2.py experiments/ligand_series/stage2/L0*/*
```

Step 1 writes to `submissions_stage2/`, **not** the shared `submissions/`
directory. That directory holds 1206 stage-1 files; on the first redo four runs'
builds failed, nothing overwrote their stage-1 file, and the packager could not
tell an eleven-day-old submission from a fresh one. A per-run output that either
exists because this build wrote it or does not exist at all removes that.

### What step 2 changes, and why the generic builder cannot

`make_casp_submission.py` gets the receptor, the pose per MODEL and the AFFNTY
right; it does not know the three rules specific to this experiment (format page
Example 6.1 as rewritten by message 582 — `docs/casp17_lg_format.md` §8b):

- **The cofactor is a `LIGAND` block**, last index, own `LSCORE`, `M  CHG` for a
  monoatomic ion. `cif_to_pdb_with_plddt` calls `remove_ligands_and_waters()`,
  so the generic path submitted the complex without its catalytic metal.
- **One `LIGAND` block per copy**, numbered 1..N. Docking prep collapses copies
  to one entry (one SMILES, one SDF), so the generic path emitted one block for
  the 43 L01 + 5 L02 multi-copy complexes.
- **The name is the ligand code** (`HTX00050570`, `ZN`) in both the `LIGAND`
  record and the MDL title line — not `LIG`, and not the 0-indexed number the
  generic path derives from the docking summary.

Extra copies come from the cofolding models, not from docking: docking strips
every copy from the receptor, so it never saw an N-copy complex and its poses
are not mutually consistent. `collect_arrangements` gathers up to 6 distinct,
internally clash-free N-copy placements across the run's aligned CIFs; a MODEL
keeps its docked pose and fills the rest from those, or — when the docked pose
overlaps every free placement — falls back wholesale to one arrangement, a
different one per MODEL so the five stay distinct.

### Defects this flow fixed

| defect | effect | fix |
|---|---|---|
| `remove_ligands_and_waters()` in the receptor writer | no Zn / SFG in any model | cofactor emitted as a `LIGAND` block |
| copies collapsed in docking prep | 1 block where the answers file asks 2-3 | copies from cofold arrangements |
| `LIGAND 0 LIG` | wrong id and name | code + 1..N from `ligands_stage2.csv` |
| pdb2pqr drops HETATMs its AMBER FF cannot parameterise | **SFG absent from every L02 `receptor.pdbqt`** — docking scored an occupied site as empty pocket, 26 % of L020255's poses sat inside the cofactor | `_reattach_dropped_hetatms` re-types them through obabel; `_cofactor_heavy_coords` + `_pose_overlaps_cofactor` drop such poses at selection time |
| `_receptor_heavy_coords` strips non-polymer residues | the existing clash filter could not see a cofactor either | dedicated cofactor filter, 2.5 Å floor, one pair is enough |
| chiral centres compared by zip position | mirror-image false alarms on anything read from a CIF/PDB (all 6 SFG centres) and a pose filter that compared unrelated atoms | `HasSubstructMatch(ref, useChirality=True)` — order-independent, and a centre the answers file leaves unspecified matches either hand |

### Template evidence: why L01 has almost none and L02 has a lot

Measured on the `phase2v2` runs, so this is what the submitted MODELs actually
had to work with:

| | filtered hits | USalign attempted | aligned (TM ≥ 0.5) | pockets | clusters | unique PDB surviving |
|---|---:|---:|---:|---:|---:|---:|
| **L01** | 1128 | 753 | **12** | 20 | 4 | **4** (7pol / 7poo / 7poq / 7pou) |
| **L02** | 1974 | 1626 | **1473** | 2035 | 10 (of 34) | hundreds |

L01 is not short of hits — it has 1128. It loses **98 % of them at the
`extract_template_pockets --min-tmscore 0.5` gate** (`n_low_quality_align`
741/753, `n_failed_align` 0, so every one of them aligned and simply scored
below the cutoff). The surviving TM scores are 0.908-0.948 and belong to four
BFT1 entries; there is nothing between 0.5 and 0.9. The distribution is bimodal
because the only close structural neighbours of the fragilysin zymogen are
fragilysin itself — the rest of the metzincin clan is too remote to superpose.
Lowering the cutoff would not recover usable templates, it would admit
alignments that are not the same fold.

L02 (MmaA1) sits in the SAM-methyltransferase family, so 1473 templates clear
the same gate at TM 0.97-0.99.

L02's abundance is also narrower than the raw count suggests. One cluster holds
almost all of it:

```
tcons_1: d(SFG)= 2.69  n=1904  uniq PDB 624  score 3222   ← the SAM/SFG site itself
tcons_2: d(SFG)=23.42  n=20    uniq PDB 10   score 33.7
tcons_3: d(SFG)=25.10  n=15    uniq PDB 10   score 23.8
   (4-10: n ≤ 11, 15-37 Å away)
```

So the 624-PDB consensus reports where the *cofactor* binds — which the answers
file already gave us as SFG. It says little about the fragment site.

Track 3 (lig-align) produced **0 SDFs in both series**: no template ligand
reaches `mcs_threshold = 0.5` against an 8-22 heavy-atom fragment.

### How much of the selection the templates actually drove

`select_diverse_top_k_anchored` folds anchors in at three points, and only one
of them touches which poses ship:

- **Set membership is pure `lscore`** (`ordered = sorted(cand, key=-lscore)`).
  Templates contribute nothing here.
- **Hard coverage** force-promotes one pose per uncovered anchor
  (`coverage_dist` 8 Å, `promote_floor` 0.05).
- **Final order** sorts by `eff = lscore + bonus`.

The bonus is `0.15 · size_mult · weight · exp(-d/6)` with
`size_mult = 1 + 1.5·clamp((heavy − 15)/20, 0, 1)`. Fragment heavy-atom counts
are L01 median 15 (9-22) and L02 median 14 (8-21) — **at or below `size_lo`, so
`size_mult = 1.0` and the size term is off by construction.** The mechanism was
built for large ligands whose template binding mode outweighs their LSCORE.

Result, measured as heavy-atom distance from every shipped copy to its nearest
anchor (830 copies, 109 files):

| | MODEL 1 | 2 | 3 | 4 | 5 | files with no MODEL ≤ 8 Å |
|---|---:|---:|---:|---:|---:|---:|
| **L01** median | 13.9 | 15.5 | 5.0 | 10.7 | 6.0 | 4 / 80 |
| L01 ≤ 8 Å | 43 % | 45 % | 54 % | 42 % | 52 % | MODEL 1 outside: 27 / 80 |
| **L02** median | 2.5 | 3.0 | 3.9 | 18.5 | 19.0 | 0 / 29 |
| L02 ≤ 8 Å | 85 % | 71 % | 62 % | 21 % | 15 % | MODEL 1 outside: 0 / 29 |

L02's MODELs 1-3 sit on the top template anchor, but not because the ranker put
them there — LSCORE and the templates happen to agree, because the best-scoring
site *is* the SAM pocket. L01's disagree, and LSCORE wins; coverage promotion is
what pulls MODELs 4-5 back toward the anchors, which is why their LSCORE
collapses (median 0.589 / 0.465, and 26 % / 40 % below 0.3, against 0.985 for
MODEL 1).

That ratio is the right weight for L01's evidence: four PDBs should not evict a
0.99-LSCORE pose. It is recorded here so the next person does not read
`pose=…template_consensus_3…` in a METHOD line as evidence that a template
chose the pose. It names the docking box, nothing more.

### L01's T2 / T4 pockets are the same site on the template's other chain

The viewer draws four template-pocket clusters for every L01 run and only two of
them ever receive a MODEL, which looks like a sampling gap and is not one.

7POL and 7POO both carry **two protein chains** in the asymmetric unit with one
copy of the fragment bound to each, and the contact counts are symmetric --
7X9 in 7POL chain A sees 52 chain-A atoms and 4 chain-B atoms, the chain-B copy
sees 4 and 52; 7WK in 7POO splits 60/1 and 1/58. It is one site on two
protomers, not two sites.

Our receptor is a monomer and USalign matches it to chain A, so the chain-B copy
lands where chain B would have been -- in solvent. Measured against the
submitted MODEL 1 receptor over all 80 runs:

| cluster | what it is | rec heavy atoms within 8 Aa (median, range) |
|---|---|---|
| T1 | 7WK, chain A | 59 (52-66) |
| T2 | 7WK, **chain B** | **1 (1-3)** |
| T3 | 7X9, chain A | 36 (31-64) |
| T4 | 7X9, **chain B** | 21 (0-25) |

So T2 is open solvent and T4 is the same pocket as T3 displaced onto the missing
protomer. Neither is worth a MODEL slot, and re-docking a corrected T2 box would
dock into water -- one of L01's three `template_consensus` boxes was aimed at a
site that does not exist on our receptor.

`cluster_template_pockets.py --surface-margin` (default 10 Aa) is documented as
the guard against exactly this multi-chain USalign artifact, and it does not
catch T2: the centroid sits 7.4 Aa off the surface, inside the margin. A
distance-to-surface test cannot separate "off the monomer" from "in a shallow
groove".

**Root cause and fix (2026-09-09).** `extract_template_pockets.py` ran one
USalign per hit row and moved every ligand in the entry with that one
transform. Its host-chain filter keyed on the row's `chain_id`, which cannot
identify a host: mmseqs reports the **entity id** (`1`) where foldseek reports
**auth chains** (`A`/`B`), so the union filter emits three rows per entry and
the entity row matches no chain, hits the "keep all ligands" fallback, and
places the other protomer's copies with the wrong transform. Measured share of
mis-transformed pockets (`template_chain != ligand_chain`): **L01 50 %**
(120/240, all from the `1` row), **L02 15 %** (3693/24380).

The fix attributes each ligand to the protomer it actually touches (heavy-atom
contacts within 6 Aa), aligns **that** chain, caches alignments per
`(pdb_id, host chain)` so the duplicate rows cost one USalign each, and emits
one pocket per physical ligand instance (`(pdb_id, host chain, ccd, seqid)`) so
repeated rows stop inflating evidence counts. On L010016 every chain-A/chain-B
pair then superposes within 0.3 Aa:

| | before | after |
|---|---|---|
| pockets | 20 | 10 |
| T1 | 12 / 4 PDB / 24.5 -- real main site | 8 / 4 / 16.4 -- real main site |
| T2 | 4 / 4 / 8.2 -- **solvent phantom** | 2 / 1 / 4.0 -- real second site |
| T3 | 3 / 1 / 6.0 -- real second site | -- |
| T4 | 1 / 1 / 2.0 -- **solvent phantom** | -- |

Member counts drop because the old ones double-counted one physical ligand per
hit row; 7POL turns out to carry **two** 7X9 copies per chain, 8.9 Aa apart, so
the second site is genuine and now clears `TEMPLATE_CONSENSUS_MIN_MEMBERS = 2`
on its own.

**Existing runs are fixed without re-running anything.** A pocket placed by a
chain-specific alignment satisfies `template_chain == ligand_chain`, so the
mis-transformed points are identifiable in the files already on disk. On
L010016 filtering the shipped `template_pockets.json` on that test reproduces
the re-extraction's 10 pockets exactly, in the submission's own frame -- which
matters, because the two frames differ: the shipped pockets sit 0.3 Aa from the
`tcons_1` docking box and 1.1 Aa from the pose cloud, while a fresh extraction
against the `alignment_summary` reference lands 23.7 Aa away. **Re-extracting
would have imported the wrong frame.** `research_prior._load_pockets` therefore
applies the filter at read time and `_cluster_supported` rejects a consensus
cluster with no surviving pocket within 5 Aa; both are no-ops on runs built by
the fixed extractor. On L010016 the real clusters keep support at 0.3 / 0.0 Aa
while the phantoms sit 37.5 / 35.4 Aa from anything that survived.

**L01 moved a lot; L02 did not move at all.** Turning the hedge cap off (above)
disables only stage 3. Stage 2 -- hard coverage -- still forces a MODEL within
8 Aa of every anchor, and the phantom anchor and the wasted `tcons_2` docking
box name the *same* solvent point, so poses really had been docked there for
coverage to promote. Removing the anchor removes the forced promotion:

| L01 fragment LSCORE | M1 | M2 | M3 | M4 | M5 | mean | `<0.3` |
|---|---|---|---|---|---|---|---|
| 2026-08-31 build | 0.982 | 0.972 | 0.935 | 0.596 | 0.465 | 0.764 | 21 % |
| + hedge cap off | 0.982 | 0.972 | 0.952 | 0.721 | 0.764 | 0.829 | 8 % |
| + phantom anchor rejected | 0.983 | 0.974 | 0.955 | **0.954** | **0.921** | **0.933** | **0 %** |

59 of 80 L01 files changed (117/400 slots). L02 is byte-identical to the
previous build (0/29 files, 0/145 slots) -- its two anchors were genuine all
along -- and its SFG coverage is unchanged at 0/29 uncovered. L01 cofactor
coverage stays where it was (76/80 -> 77/80 uncovered; the zymogen's Zn site is
latched, so there was never anything to cover). Lint: 0 ERROR, WARN files
11 -> 13, all of the same multi-copy class (one copy of a 2-copy fragment
repeats between MODELs while the arrangement differs, because the 2 Aa
diversity gate scores whole arrangements).

Separately, 29 of 80 L01 runs carry a **frame drift of >= 10 Aa** (max 65.6;
L02: 0/29) between the stored cluster centres and the re-aligned frame --
`template_pockets.json` was written 2026-08-11 05:06 and the cofold outputs were
re-aligned 2026-08-13 16:44 (`rmsd_before` 18.6 Aa), with docking prep reading
the pre-realignment centres in between. The frame-consistency fix (`018b78f`,
2026-05-04) does not cover this path.

### Hedge-cap calibration: `--max-anchorless` is per-series, not global

`select_diverse_top_k_anchored`'s third stage (`max_anchorless`, default **2**)
caps how many MODELs may sit at no anchor and swaps the excess for the best
anchor-covering pose clearing `promote_floor = 0.05`. Measured on the 2026-08-31
build, that swap was the single largest quality loss in both series — and the
right setting is opposite per series, because the two anchor sets differ in
whether they mark a site a ligand can actually occupy.

**L01 — turn it off (`-1`).** The BFT1 zymogen's catalytic Zn is latched by
prodomain Asp161, which is why `COFACTOR_ANCHOR` has no L01 entry. 75 of 80
shipped files already had no MODEL within 8 Aa of the Zn, so the cap was
defending a site nothing reaches while still evicting good poses:

| L01 fragment LSCORE | M1 | M2 | M3 | M4 | M5 | mean | files with no MODEL <= 8 Aa |
|---|---|---|---|---|---|---|---|
| cap = 2 (shipped) | 0.982 | 0.972 | 0.935 | 0.596 | 0.465 | 0.764 | 75 / 80 |
| cap off | 0.982 | 0.972 | 0.952 | 0.721 | 0.764 | **0.829** | 76 / 80 |

54 of 400 slots swapped across 24 files; the poses dropped had median LSCORE
0.430 and the ones that replaced them 0.927. `<0.3` slots fell from 30 %/40 %
(M4/M5) to 20 %/21 %.

**L02 — tighten to 3, do not remove.** Here the anchor is real: `tcons_1` is 624
unique PDB entries 2.69 Aa from SFG. Removing the cap buys LSCORE by walking
off that site in **9 of 29 complexes**.

| L02 | M1 | M2 | M3 | M4 | M5 | mean | `<0.3` | no-cover files |
|---|---|---|---|---|---|---|---|---|
| cap = 2 (shipped) | 0.821 | 0.805 | **0.619** | 0.813 | 0.699 | 0.655 | 18 % | 0 / 29 |
| cap = 3 | 0.821 | 0.805 | **0.818** | 0.774 | 0.706 | 0.706 | 11 % | 0 / 29 |
| cap off | 0.877 | 0.835 | 0.804 | 0.738 | 0.695 | 0.766 | 2 % | **9 / 29** |

`cap = 3` leaves M1/M2 byte-identical to the shipped build (coverage 100 %/93 %)
and fills only the M3 trough, while every complex keeps a MODEL at the SFG site.

Generalisation: the cap is worth its cost only where the anchors mark a site a
ligand can occupy. Check coverage of the *shipped* build first — if it is
already near zero, the cap is paying LSCORE for nothing and should be off.

Rebuild path (`--max-anchorless` passthrough added to the driver 2026-09-08):

```bash
uv run python scripts/rebuild_ligand_series_lg.py --series L01 \
  --out-name submissions_nohedge --max-anchorless -1
uv run python scripts/rebuild_ligand_series_lg.py --series L02 \
  --out-name submissions_hedge3  --max-anchorless 3
sbatch experiments/ligand_series/rebuild_lg.sbatch.sh   # regenerated per call
uv run python scripts/make_ligand_series_stage2.py --series L01 \
  --submissions submissions_nohedge --out-dir experiments/ligand_series/stage2_v2
```

### Pre-submission audit of the v3 build (2026-09-09)

Re-derived from the raw scores rather than by re-running the builder: the pool
is rebuilt from `rmsd_pred_*.tsv` (`lscore = 1 - Is_Above_2A`) plus the staged
pose SDFs, and each shipped LIGAND block is matched back by exact coordinates.

Clean:

| check | result |
|---|---|
| format vs Example 6.1 + `ligands_stage2.csv` | 109/109 |
| fragment inside the receptor (floor 2.2 Aa) | 0 / 830 (closest 2.21) |
| fragment clashing the cofactor (floor 1.6 Aa) | 0 / 830 (closest 2.50) |
| copies of one fragment overlapping | 0 / 335 (closest 2.51) |
| pose adrift in solvent | 0 (farthest 3.75 Aa from the receptor) |
| reported LSCORE exists in the run's pool | 545/545 |
| percentile of each MODEL's LSCORE in its pool | median top 0.71 % |

Two behaviours the audit surfaced are deliberate, not defects:

- **L02 MODEL 1 is often not the pool's top pose.** `anchor_reorder` promotes
  the pose at the cofactor site: L020601 ships LSCORE 0.107 at 3.1 Aa from SFG
  as MODEL 1 and demotes 0.894 at 19.9 Aa to MODEL 2. Order only -- membership
  is untouched, and MODEL 1 sits at the SFG site in 29/29 complexes.
- **Every extra copy of a multi-copy fragment carries the docked pose's
  LSCORE**, not its own (`make_ligand_series_stage2.py`, `lscores =
  [m.ligands[0].lscore] * len(chosen_mdls)`). Copy #1 carries its own score in
  532/540 blocks; copies #2..N do not in 106/107. The gap overstates by a
  median of **+0.542**, up to +0.983 (L010167 M2 copy 2 reports 0.976 where
  that placement's own estimate is 0.000). 106 of 830 blocks, i.e. the 43 L01
  and 5 L02 multi-copy complexes. The poses are unaffected -- this is the
  self-estimate only -- and LSCORE is optional per the format page, so it does
  not block submission; fix it in the builder before any resubmission.

Three findings during this audit were defects in the *checker*, not the
submission, and are worth remembering because each one produced a convincing
false alarm: a staged `.sdf` holds every pose of one (source, seed) rather than
one pose; the LG builder reorders atoms to the reference SMILES so an
order-sensitive RMSD reports a difference where the pose is identical; and
cofold poses are staged as `<source>.sdf` while the score table names them
`<source>_<a>_<b>`, which made 20 % of poses look unmatched until the mapping
was fixed (they then matched to 0.0000 Aa).

### What the CASP validator rejected (2026-09-09/10)

Two rounds, both packaging/receptor faults rather than pose faults. The poses,
scores and selection are unchanged throughout -- verified by reassembling the
split members and diffing against the pre-split build (109/109 byte-identical).

**Round 1 — one model per file.** Every member of the L02 tarball came back
with `Only one model per file is allowed for the Ensemble target category.
Please submit each model in a separate file.` The ligand series is an
**Ensemble** target category; the `_1` in `<target>LG129_1` is the *model*
number, not a file counter. `make_ligand_series_stage2.split_models` now writes
one file per MODEL (`_1`..`_5`, MODEL number matching the suffix): L01 80 -> 400
files, L02 29 -> 145.

The rule was already in `docs/casp17_lg_format.md` §0.1 from the 2026-05-06
validator run, and `lint_lg_submission.py` enforces it -- but its filename test
is `_([1-5])\.lg$` and these files carry no `.lg` suffix, so the check never
fired, and the ligand-series lint does not carry the rule. The scope in that
doc was also too broad: our own `T2451_LCDD.lg` (**10** MODELs) and
`T2455_LCDD.lg` (5) were accepted on 2026-08-10/11, so multi-MODEL is fine for
regular targets and only the Ensemble category forbids it.

**Round 2 — modified residue vs the target sequence.**
`# ERROR! Check atom number 2057 residue: # 253 chain 'A' (In TARGET: C 253)`.
AF3/Protenix model MmaA1's redox-gated Cys253 as **CSO** (S-hydroxycysteine),
which the validator rejects because the target sequence has `C`. CSO also
carries `het_flag='H'`, so an ATOM-only copy dropped it outright -- L020426 and
L020501 shipped with **no residue 253 at all** (270 of 271). One fix covers
both: `make_casp_submission.normalize_modified_residues` rewrites CSO -> CYS,
drops the extra `OD`, and sets `het_flag='A'`, running **before**
`remove_ligands_and_waters` in `cif_to_pdb_with_plddt`, which every receptor
path goes through. Only entries verified against a real target are in
`MODIFIED_PARENT`; anything else non-standard is reported, not guessed.

Post-fix, checked the way the validator checks -- every modelled residue against
`inputs/ligand_series/<series>/receptor.fasta`:

| | files | residues/file | sequence mismatches | non-standard atoms |
|---|---|---|---|---|
| L01 | 400 | 364 / 364 | 0 | 0 |
| L02 | 145 | 271 / 271 | 0 | 0 |

`lint_lg_submission.py` (with `.lg` appended so the single-model rule fires):
L01 400/400, L02 145/145, **0 ERROR / 0 WARN**. Physical checks are unchanged
from the pre-split build, as expected for a receptor-residue fix.

## Deadlines
Stage 1 = 3 weeks (Soft Deadline), Stage 2 = 5 weeks (Human Expiration). See
target list page for the dated cutoffs. Stage 2 was extended on 2026-08-07 to
**L01 2026-09-11 / L02 2026-09-10**.
