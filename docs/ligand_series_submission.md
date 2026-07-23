# CASP17 Ligand-Series (Lxx) Submission Method

Understanding of the CASP17 **ligand-series** experiment submission for targets
`L01` (BFT1 / Fragilysin, 364 res) and `L02` (MmaA1, 271 res). Source: official
Ligand Series Prediction Upload Form + target pages
(`predictioncenter.org/casp17/target.cgi?id=221` L01, `id=222` L02).

**This is NOT the standard single-LG-file flow.** Dedicated gateway, two stages.

## Scale (critical)
- **L01 = 1209 fragments** to screen (L010001 … L011209).
- **L02 = 647 fragments** (L020001 …).
- Our CASP17_own runs so far modeled only **3 fragments each** (test subset) via
  per-fragment cofold — infeasible to repeat ×1209/×647. Stage 1 needs a
  **fixed-receptor docking screen** (fold receptor once, dock+score every
  fragment), not per-fragment cofolding.

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
- `YYY` = numeric **group number** (e.g. 987) — distinct from the registration
  code `0887-0325-2808`; MUST be confirmed before upload.

## Stage 2 — poses for actual binders (`LXXLGYYY.tgz`)
Revealed after stage 1 (binding answers given, pockets hidden). For each binding
complex: receptor (PDB) + binder(s) (MDL), CASP17 LG format, one model per
complex, named e.g. `L010001LGYYY_1`. Put models in `./L01`, then
`tar -czf L01LGYYY.tgz ./L01`. Upload the tarball.

## What we already have (per fragment, in `runs/Lxx/score/ranked.json`)
- ranked poses with `lscore`, `consensus_score`, `ranker_score`, `plddt_pocket`,
  `pocket_residue_plddt` (chain+resi), `pose_id`, aligned SDF + receptor PDB in
  `runs/Lxx/dock/_aligned/`.
- → for the 3 modeled fragments a bind.txt line is directly derivable.

## Gaps to actually submit
1. **Full fragment library** — pull all 1209 (L01) / 647 (L02) SMILES from the
   target page input files; current runs cover 3.
2. **Scalable screen** — fixed-receptor dock (Vina/GNINA) + score across the full
   library; per-fragment cofold does not scale.
3. **Binding probability** — define/calibrate a 0-1 binder score. Current
   `lscore`/`ranker_score` are pose quality, not a binding yes/no; needs a
   binder classifier or calibrated cutoff (GNINA CNNaffinity / Boltz binder_prob
   candidates).
4. **Pocket residues @5 Å** — compute from the chosen pose (aligned SDF vs
   receptor PDB); do not reuse pocket-def residues verbatim.
5. **bind.txt writer** — not yet in the repo (`series-preflight` only checks
   readiness). Need a generator: ranked.json/poses → `LXXLGYYY.bind.txt`.
6. **Group number YYY** — confirm the numeric CASP group id.

## Deadlines
Stage 1 = 3 weeks (Soft Deadline), Stage 2 = 5 weeks (Human Expiration). See
target list page for the dated cutoffs.
