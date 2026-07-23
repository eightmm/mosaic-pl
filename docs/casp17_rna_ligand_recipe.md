# CASP17 RNA-Ligand Targets — Working Recipe

Source: <https://predictioncenter.org/casp17/targetlist.cgi?view=ligand>
Format spec (parent): `docs/casp17_lg_format.md`
First written: 2026-05-06

This file is the **fast-lookup** when building / submitting RNA-ligand
predictions for CASP17. It supplements `docs/casp17_lg_format.md` with
RNA-specific notes and stores the per-target metadata so we don't have
to re-derive it from the website each time.

---

## 1. RNA-ligand targets

| ID    | Length | RNA sequence (5'→3')                                                                                                                                                                                                                                            | Ligand               | CCD | SMILES                                            | Expires (PDT 09:00) |
| ----- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | --- | ------------------------------------------------- | ------------------- |
| R2314 | 25 nt  | `CGAGGACCGGUACGGCCGCCACUCG`                                                                                                                                                                                                                                       | L-Tryptophan         | TRP | `N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O`                | 2026-05-06          |
| R2317 | 129 nt | `GCGGGUGAAUGUAAGCAGAGAGACUGCGAAAAGCGGCGCCGACGGGGAAAGCAUGUAUUAUGUGAAACUCUCAGGCAAAAGGAUGUUUACGGGACGCAACUCUGGAGUCAUUUUUGUGUUACGACAGGG`                                                                                                                              | Glycine (Lmo riboswitch) | GLY | `NCC(=O)O`                                        | 2026-05-07          |
| R2318 | 140 nt | `CACUGGAUGAGGUUUUCAGGAGAACAGGGUAAGCUAACCAUGAUGAACUGAAAACGGACAGAACUCUGGAGAGUUCCGCAAGGACGCCGAAGGGGCAAGACAGCAAAGCUGUUCAAUCUCUCAGGCAAAAGGACAGAGCG`                                                                                                                  | Glycine (Dha riboswitch) | GLY | `NCC(=O)O`                                        | 2026-05-07          |
| R2329 | 60 nt  | `CGGAGUCAUGGCUCAGGGCUGUUCGCAGCCGCUGCAGUCAGUCGAAAGACUGAGACUCCG`                                                                                                                                                                                                    | S-adenosylmethionine (SAMURI ribozyme holo) | SAM | `C[S@@+](CC[C@H](N)C(=O)[O-])C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O` | 2026-05-14          |
| R2330 | 75 nt  | `ACUCGGGGUGCCCUUCAAAAGAAGGCUGAGAAAUACCCGUAUCACCUGAUCUGGAUAAUGCCAGCGUAGGGAAGU`                                                                                                                                                                                    | Thiamine pyrophosphate (E. coli TPP riboswitch) | TPP | `Cc1ncc(C[n+]2csc(CCOP(=O)([O-])OP(=O)([O-])[O-])c2C)c(N)n1` | 2026-05-14          |
| R2387 | 43 nt  | `GGUGCGUUGCUUCCGAUGACGGCACCUUAAAAACAAUAGGAGA`                                                                                                                                                                                                                    | NAD+ (RNA-NAD+ complex; id=153) | NAD | `NC(=O)c1ccc[n+](c1)[C@@H]2O[C@H](CO[P]([O-])(=O)O[P@](O)(=O)OC[C@H]3O[C@H]([C@H](O)[C@@H]3O)n4cnc5c(N)ncnc45)[C@@H](O)[C@H]2O` | 2026-06-17 (human 2026-06-29) |
| R2390 | 34 nt  | `GGUGGGUUCUUCCUCGCCACGGUAAAAACAAGGA`                                                                                                                                                                                                                             | NMN (RNA-NMN complex; id=154) | NMN | `NC(=O)c1ccc[n+](c1)[C@@H]2O[C@H](CO[P](O)(O)=O)[C@@H](O)[C@H]2O` | 2026-06-19 (human 2026-07-01) |

Sequences also stored as fasta in `inputs/casp17_rna_lig/{R2314,R2317,R2318}.fasta`
(downloaded via `target.cgi?id={34,37,38}&view=sequence` — note **numeric id**,
not target name).

Authoritative ligand candidates:
- **R2314** — Trp aptamer; canonical PDB analog: in vitro selected
  L-tryptophan-binding aptamer family (CCD `TRP`, neutral
  zwitterionic L-form is correct for an aptamer that recognises the
  carboxylate + α-amine + indole).
- **R2317 / R2318** — Glycine riboswitches (Rfam family **RF00504**).
  Bound species in deposited Glycine riboswitch structures is **glycine
  (CCD `GLY`)**, neutral form OK for cofolding (charge protonation is
  inside the cofolding model).
- **R2329** — SAMURI is a SAM-utilizing ribozyme; target description
  says holo but does not publish a separate ligand file on the target
  page, so the working ligand is cognate CCD `SAM`.
- **R2330** — TPP riboswitch cognate ligand is thiamine pyrophosphate,
  CCD `TPP`. Use the deprotonated pyrophosphate SMILES in the LG MDL
  topology; the neutral/protonated phosphate form is rejected by the
  live CASP ligand validator.

---

## 2. Pipeline reality check (this repo)

| Stage                       | Works for RNA-only? | Why                                                                                                          |
| --------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `template-search-sequence`  | skip                | MMseqs2 DB is RCSB **protein** seqDB.                                                                        |
| `template-search-structure` | skip                | Foldseek RCSB DB indexes protein structures.                                                                 |
| AF3 unified MSA pipeline    | **yes**             | `/home/jaemin/DB/AlphaFold3` already has `nt_rna_*.fasta`, `rfam_14_9_*.fasta`, `rnacentral_*.fasta`. AF3's data pipeline runs nhmmer against these for RNA chains. |
| Boltz colabfold MSA         | no                  | `use_msa_server=true` returns protein MSA only; RNA chains stay single-seq.                                  |
| Boltz-2 / Boltz-2x cofolding | yes (single-seq RNA) | Adapter has explicit RNA branch (`adapters.py:263`); ligand block accepts SMILES.                          |
| Protenix cofolding          | yes (single-seq RNA) | RNA branch exists (`adapters.py:283`); the existing bridge only redistributes protein MSAs to Protenix's `rnaSequence`, so RNA chain runs without external MSA. |
| AF3 cofolding               | yes (with RNA MSA)  | RNA block builds (`adapters.py:490`); when `msa_pipeline.enabled=true`, the data pipeline JSON is fed to inference and AF3 consumes the RNA MSA it produced itself. |
| Track-1 docking (Vina/ADG/PxDock) | skip          | Receptor PDBQT pipeline is protein-only; PxDock tleap params don't ship RNA force field.                     |
| Track-2 (template box)      | skip                | Depends on RCSB structural template hits.                                                                    |
| Track-3 (lig-align)         | skip                | Same dependency as Track-2.                                                                                  |
| Post-analysis (BA-Pred / RMSD-Pred) | skip          | Models trained on protein-ligand; scores on RNA-ligand are out of distribution.                              |
| LG submission                | yes                 | Format is receptor-agnostic; PDB ATOM lines use RNA residue / atom names.                                    |

**Effective scope:** cofolding only (Boltz-2 + Boltz-2x + Protenix + AF3),
output cif → submission. AF3 receives a real RNA MSA via the unified
MSA pipeline; Boltz and Protenix RNA chains are single-seq.

---

## 3. MSA sharing — what is and isn't redistributed today

`scripts/bridge_distribute_msa_templates.py` consumes AF3's data
pipeline output (`{name}_data.json`) and:

- patches `proteinChain` blocks in Protenix JSON with `unpairedMsaPath`
  / `pairedMsaPath` / `templatesPath`,
- patches `protein` blocks in Boltz YAML with the same a3m + templates,
- replaces the AF3 input JSON with the data-pipeline output so AF3
  inference reads the same MSA + templates.

The bridge currently has **no `rnaSequence` / `rna` branches**, so for
RNA-only targets:

- **AF3** gets the RNA MSA correctly (it generated it; the inference JSON
  is the same JSON).
- **Boltz / Protenix** get nothing extra on the RNA chain — single-seq
  fallback inside each tool.

If we want to feed the AF3-built RNA MSA to Boltz and Protenix as well,
the bridge needs a small RNA-aware patch (~30 lines: dump RNA chain
`unpairedMsa` to `shared_msa/{chain}_rna_unpaired.a3m`, set
`rnaSequence.msa.precomputed_msa_dir` for Protenix and Boltz's
`rnaSequence.msa: <path>`). This is a follow-up; not done before the
R2314 deadline.

---

## 4. LG submission — RNA-ligand specifics

(The full grammar is in `docs/casp17_lg_format.md`; this section only
flags the parts that change for RNA-ligand targets.)

### 4.1 Header

```
PFRMAT LG
TARGET R2314             # use the RNA TARGET ID; no rename to L-prefix
AUTHOR <12-digit code>   # CASP17 registration code, supplied at submit time
METHOD Boltz-2 + Boltz-2x cofolding (single-seq RNA MSA, 5 seeds × 5 samples)
```

The TARGET ID is whatever the organizers issued (`R2314`, `R2317`,
`R2318`); LG format does not require an `L`-prefix when the target was
released as an RNA target with a ligand.

### 4.2 Receptor block (RNA)

`PARENT N/A` for de novo cofolding.
PDB `ATOM` records use RNA residue names `A`, `U`, `G`, `C`. Atom
naming follows the standard nucleic acid set:

```
P  OP1  OP2  O5'  C5'  C4'  O4'  C3'  O3'  C2'  O2'  C1'   <backbone + 2'OH>
+ base atoms:
  A: N9  C8  N7  C5  C6  N6  N1  C2  N3  C4
  G: N9  C8  N7  C5  C6  O6  N1  C2  N2  N3  C4
  C: N1  C2  O2  N3  C4  N4  C5  C6
  U: N1  C2  O2  N3  C4  O4  C5  C6
```

Chain id: the builder rewrites cofolding's native chain id (`A`) to **`0`**
in the LG output — the CASP-issued RNA receptor template uses `0` (PDB
column 22, single zero character) for RNA monomers, and the assessor
matches by chain id. Override via `--receptor-chain-id` (pass empty
string to keep cofolding's native id). End with a single `TER`.

The B-factor column should carry the per-residue confidence (pLDDT)
on the 0–100 scale. For Boltz mmcif output we read `_atom_site.B_iso_or_equiv`,
which is already a 0–100 pLDDT-derived value (Boltz convention).

### 4.3 Ligand block

```
LIGAND 001 LIG          # 3-digit ID = 001 (single ligand per target);
                        # name "LIG" is safe across all three targets.
LSCORE 0.55             # Boltz interface confidence proxy (we use 1 - average
                        # ipTM-style score) clipped to [0, 1]; document the
                        # exact derivation in METHOD.
<title line>            # e.g. "boltz2x_seed_42_sample_3"
     RDKit          3D
                        # blank line
 <NN> <MM>  0  0  0  0  0  0  0  0999 V2000
<atom block>
<bond block>
M  END
```

Hydrogens optional; we omit them.

### 4.4 AFFNTY (optional, RNA caveat)

For R2314/R2317/R2318, **do NOT emit AFFNTY by default**. BA-Pred is a
protein-ligand model; reporting its number on RNA targets would be
misleading. If we want to submit any AFFNTY at all, the only defensible
source is **Boltz-2 affinity head** with the binder probability filter
(`binder_prob ≥ 0.5`). If `binder_prob < 0.5` for all seeds, drop the
record entirely.

### 4.5 MODEL ordering / diversity

Five MODELs per file. We pick:

1. MODEL 1 = best (max Boltz `confidence_score` after sample averaging)
2. MODELs 2–5 = next four samples that satisfy
   *ligand heavy-atom RMSD ≥ 2.0 Å* against MODEL 1 (per-ligand basis,
   per `docs/casp17_lg_format.md`).

If diversity cannot be satisfied across 25 Boltz-2 + 25 Boltz-2x = 50
samples, fall back to lowest-RMSD acceptable poses; never duplicate.

---

## 5. Build order for one target (e.g. R2314)

```bash
# 0. inputs already present:
#    inputs/casp17_rna_lig/R2314.fasta        (sequence, reference)
#    inputs/casp17_rna_lig/R2314.yaml         (CommonInput; RNA + ligand, ccd only)
#    inputs/casp17_rna_lig/runner.yaml        (Boltz + Protenix + AF3, msa_pipeline on)

# 1. prepare wrapper (master node, no GPU):
uv run casp17-pl prepare-wrapper \
  --input inputs/casp17_rna_lig/R2314.yaml \
  --config inputs/casp17_rna_lig/runner.yaml \
  --output-root experiments/CASP17 \
  --backend slurm

# 2. submit:
sbatch --partition=heavy --time=06:00:00 \
  experiments/CASP17/R2314/scripts/run_wrapper.sbatch.sh

# Wrapper executes (RNA-only path):
#   1) AF3 data pipeline (jackhmmer + nhmmer + Rfam + RNAcentral + nt_rna)
#   2) bridge_distribute_msa_templates.py (RNA-aware: rna.msa for Boltz,
#      rnaSequence.unpairedMsaPath for Protenix, AF3 input replaced wholesale)
#   3) cofolding (Boltz + Protenix + AF3, 5 seeds × 5 samples each)
#   4) align_cofolding_outputs.py (USalign whole-system superposition;
#      writes <name>_aligned.cif next to every original cif)

# 3. build LG submission:
uv run python scripts/build_rna_ligand_lg_submission.py \
  --run-dir experiments/CASP17/R2314 \
  --target-id R2314 \
  --ligand-name TRP \
  --author <12-DIGIT-REG-CODE> \
  --method "Boltz-2 + Protenix + AF3 cofolding (5 seeds × 5 samples; AF3 unified RNA MSA pipeline; whole-system USalign superposition + ligand pose clustering at 3 Å)"
  # default --output is now experiments/CASP17/submissions/R2314_LCDD.lg
  # (pass --output only to override the directory)

# Builder consumes only *_aligned.cif so every MODEL is in the same frame.
# Ligand poses are heavy-atom RMSD clustered (--cluster-rmsd, default 3 Å);
# the highest-confidence member of each top cluster becomes a MODEL.

# 4. routine validation:
make lint-lg LG_FILES=experiments/CASP17/submissions/R2314_LCDD.lg
# (or `make test` to exercise the linter unit tests under tests/test_lg_lint.py)
```

---

## 6. Known unknowns / risks

- **R2314 expiration is today** (2026-05-06 09:00 PDT ≈ 2026-05-07 01:00
  KST). Plan: submit Boltz-2 + Boltz-2x cofolding immediately, build LG
  from whichever finishes first. Skipping Protenix/AF3 is acceptable for
  R2314 alone.
- **Ligand name in `LIGAND 001 XXX`**: the format spec says the name
  comes from the organizer's SMILES file. Without confirmed access we
  default to `LIG`. If the organizer page later exposes a per-target
  smiles file with a name, swap it in before submit.
- **Glycine zwitterion**: SMILES `NCC(=O)O` is the neutral form. Boltz
  / AF3 internally protonate the model. If the assessment uses the
  zwitterionic form `[NH3+]CC(=O)[O-]` with the same heavy-atom set the
  RMSD scoring is unchanged.
- **No RNA template input**: setting `templates=[]` is mandatory for
  AF3 RNA blocks (otherwise AF3 schema rejects). Already handled by
  `adapters.py`.
