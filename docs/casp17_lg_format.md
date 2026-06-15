# CASP17 LG (Ligand) Submission Format — Authoritative Reference

Source: <https://predictioncenter.org/casp17/index.cgi?page=format>
Format text is identical to CASP16 LG format
(<https://predictioncenter.org/casp16/index.cgi?page=format#LG>).
Last verified against the live page: 2026-05-06 (KST).

> **Read this file before generating or modifying any LG-format
> submission.** It is the canonical mirror of the CASP17 format page,
> and it is the contract that every LG builder + linter in this repo
> respects:
>
> - `scripts/make_casp_submission.py` (protein-ligand)
> - `scripts/build_rna_ligand_lg_submission.py` (RNA-ligand)
> - `scripts/lint_lg_submission.py` (format validator, also wired to
>   `make lint-lg` and `tests/test_lg_lint.py`)
>
> If the upstream CASP page changes, update this file first and then
> propagate behaviour into the builders / linter so the mirror stays
> trusted.

RNA-ligand specifics (chain id `0` convention, OP3 5'-end residue,
ENDMDL handling vs. TS-style templates) live in
`docs/casp17_rna_ligand_recipe.md`.

## 0. Server-verified rules (override the spec page when they conflict)

These items were confirmed against the live CASP17 LG validator on
2026-05-06 by submitting R2314 / R2317 / R2318 and observing what the
server actually accepts. The format page text and the validator
disagree on a handful of points; **the validator wins**. Source:
`memory/casp17_lg_format_spec.md`.

1. **Single MODEL per file.** "Submission files in LG and QA categories
   should contain only one model." To submit five alternates, upload
   five separate files (`<TID>_1.lg` … `<TID>_5.lg`). Multi-MODEL files
   trigger `validation script crashed, get in touch with the
   Prediction Center`.
2. **`LIGAND <id>` uses the SMILES file integer ID verbatim — no
   zero-padding.** The format page's `LIGAND 001 LIG` example is
   misleading. Fetch the per-target SMILES file
   (`target.cgi?target=<TID>&view=smiles`) and copy the `ID` column
   straight in (e.g. `LIGAND 0 TRP`).
3. **MDL atom block omits hydrogens for current RNA-ligand targets.**
   The CASP17 format page says hydrogens are optional, and the live
   validator rejected explicit-H SAM/TPP blocks as invalid ligand
   topology. Emit heavy atoms only, with bond orders copied from the
   CASP SMILES reference.
4. **MDL bond stereo column must be `0` for every bond.** Reset chirality
   tags + bond directions before emit:
   `atom.SetChiralTag(CHI_UNSPECIFIED)` and
   `bond.SetBondDir(BondDir.NONE)`. 3D coordinates already encode
   stereochemistry; a `1` (wedge up) in the stereo column crashes the
   validator on chiral atoms.
5. **`TER` is the bare four-letter line, not the full PDB-style record
   with serial/resName.** Match `spec example 6.1`.
6. **No `ENDMDL`.** The LG MODEL terminator is `END` only.
7. **AUTHOR / METHOD trailing line.** The `casp17_own` group code is
   `0887-0325-2808`. Append a fixed METHOD continuation line:
   `METHOD Pipeline configured and executed via Claude Code agentic decision-making`.
8. **Chain id by polymer type.** Protein chains use alphabetic ids
   (`A`/`B`/...); RNA and DNA chains use digit ids (`0`/`1`/...). For
   RNA-only targets the receptor lives on chain `0`; cofolding emits
   chain `A` natively, so the LG writer rewrites column 22 to `0`
   before output. Mixed protein-RNA targets (`M*`) need per-chain
   remapping (still TODO).
9. **B-factors must vary** across the receptor (CASP rejects flat
   B-factors). Boltz/Protenix/AF3 all populate per-residue pLDDT, so
   this falls out naturally for cofolded models.

The repo enforces all of the above via:
- `scripts/build_rna_ligand_lg_submission.py` — emits one target-level
  multi-MODEL `<TID>.lg` submission file.
- `scripts/lint_lg_submission.py` — validates target-level multi-MODEL
  submission files.
- `scripts/make_casp_submission.py::build_lg_submission` — the
  shared LG assembler used by both protein-ligand and RNA-ligand
  pipelines.

---

## 1. The big picture

**One LG file per target.** Inside the file:

```
<header records: PFRMAT / TARGET / AUTHOR / METHOD>
MODEL 1
  <one complete snapshot of the prediction>
END
MODEL 2
  <alternate complete snapshot>
END
...
MODEL 5
  <alternate complete snapshot>
END
```

- **Up to 5 MODEL blocks per file** (top-5 alternate predictions for the
  same target). The server keeps up to 5.
- **Each MODEL is one *complete* snapshot**: receptor coordinates + one
  LIGAND block per ligand + optional AFFNTY. All ligands of the target
  must appear inside every MODEL — the MODEL is the "whole complex at
  rank k" unit, not "one ligand".
- MODEL 1 is the group's primary prediction and is used as the main
  ranking target during assessment; MODELs 2-5 are alternatives.

### Inside a MODEL

```
MODEL k
REMARK <anything>                 # optional
PARENT <template>                 # required when receptor coords present
ATOM   ...                        # receptor PDB coords
...
TER
LIGAND 1 LIG                      # first ligand, ID verbatim (no zero-pad)
LSCORE 0.82                       # optional, per-ligand, in [0, 1]
<MDL V2000 body for ligand 1>
M  END
LIGAND 2 LIG                      # second ligand (multi-ligand target)
LSCORE 0.65
<MDL V2000 body for ligand 2>
M  END
AFFNTY 0.050 aa                   # optional, per MODEL, after last LIGAND
END
```

Protein-ligand-ligand-(affinity) repeats as one unit, and that unit
appears once per MODEL.

---

## 2. Critical rules (violating these → invalid submission)

1. **Each MODEL contains all ligands of the target.** A target with 2
   ligands → every MODEL has 2 LIGAND blocks. Do NOT split ligands across
   MODELs.
2. Up to **5 MODEL blocks** per file. MODEL 6+ are dropped by the server.
3. `PARENT` is **mandatory** inside every MODEL that carries receptor
   coordinates. Use `PARENT N/A` for de novo.
4. `LSCORE` lives **immediately after** its `LIGAND` record and before
   the MDL body. It is **per-ligand**, not per-MODEL. Range `[0, 1]`.
5. `AFFNTY` appears **after the last LIGAND block** in the MODEL and
   before the MODEL's terminating `END`. It is one-per-MODEL (or absent).
6. Each MDL body terminates with `M  END` (two spaces between `M` and
   `END` — MDL convention).
7. Each MODEL terminates with a single `END` line. No `ENDMDL` anywhere.
8. Counts line in the MDL body must exactly match the atom/bond counts.
   Incorrect connectivity → submission marked invalid.

---

## 3. Record catalog

| Record | Example | Notes |
|---|---|---|
| `PFRMAT LG` | `PFRMAT LG` | Required, first line. |
| `TARGET` | `TARGET T1214` | L-prefix for ligand-specific targets; T/H/M-prefix for structure targets with incidental ligands. |
| `AUTHOR` | `AUTHOR 0123-4567-8901` | 12-digit CASP registration code. |
| `METHOD` | `METHOD <text>` | Multiple allowed. |
| `REMARK` | `REMARK <text>` | Optional, anywhere after header; NOT inside MDL body. |
| `MODEL k` | `MODEL 1` | `k` ∈ {1, 2, 3, 4, 5}. MODEL 1 is primary. |
| `PARENT` | `PARENT N/A` or `PARENT 1abc_A` | Required when receptor coords present. |
| `ATOM` / `HETATM` / `TER` | Standard PDB 80-col format | Receptor coords. `TER` closes the protein chain. |
| `LIGAND n XXX` | `LIGAND 001 LIG` | `n` = zero-padded 3-digit ligand number from SMILES file; `XXX` = ligand name from SMILES file. |
| `LSCORE` | `LSCORE 0.82` | Per-LIGAND confidence in `[0, 1]`. Optional but strongly recommended. |
| *(MDL V2000 body)* | header + counts + atoms + bonds | Ends with `M  END`. |
| `M  END` | `M  END` | Terminates each MDL body. Two spaces between M and END. |
| `AFFNTY x type` | `AFFNTY 0.050 aa` | After last LIGAND, before MODEL's END. |
| `END` | `END` | Terminates the MODEL. One per MODEL. |

### MDL V2000 body structure

```
<title line>                      # pose identifier, e.g. "vina_seed_42_3"
     RDKit          3D            # program/timestamp line
                                  # blank line
 17 18  0  0  0  0  0  0  0  0999 V2000
   0.0000   0.0000   0.0000 C   0  0 ...
   ...
  1  2  1  0
  ...
M  END
```

Hydrogens optional. Counts must be exact.

---

## 4. AFFNTY types (priority order)

| Type | Format | Meaning |
|---|---|---|
| `aa` | `AFFNTY 0.050 aa` | Absolute Kd in **nM** |
| `ra` | `AFFNTY 0.900 ra` | Relative: Kd / Kd_reference |
| `lr` | `AFFNTY 1 lr`     | Rank: integer, 1 = strongest binder |

Use `aa` when absolute Kd in nM is available (our ensemble-from-BA-Pred
path). Use `ra` for relative ordering only. `lr` is fallback.

---

## 5. Task types

| Task | Required blocks |
|---|---|
| **P (Pose)** | receptor + ligand(s), no AFFNTY |
| **A (Affinity)** | AFFNTY only, no coordinates |
| **PA (Pose + Affinity)** | receptor + ligand(s) + AFFNTY (any valid combination) |

---

## 6. Complete reference example — 2 ligands, 2 alternate MODELs

```
PFRMAT LG
TARGET T1999
AUTHOR 0123-4567-8901
METHOD Boltz-2x + multi-track ensemble + lig-align (5 seeds x 5 samples)
METHOD Cofolding(25x4) + docking(Vina/ADG/PxDock x 3 sites x 5 seeds)
MODEL 1
REMARK Primary prediction
PARENT 1abc
ATOM      1  N   GLU A   1      10.982  -9.774   1.377  1.00 90.00           N
<... full receptor ATOM records ...>
TER
LIGAND 001 LIG
LSCORE 0.82
vina_cofolding_seed_42_3
     RDKit          3D

 16 17  0  0  0  0  0  0  0  0999 V2000
<... ligand 001 pose A atoms/bonds ...>
M  END
LIGAND 002 LIG
LSCORE 0.65
cofold_protenix_0
     RDKit          3D

 28 30  0  0  0  0  0  0  0  0999 V2000
<... ligand 002 pose A atoms/bonds ...>
M  END
AFFNTY 0.045 aa
END
MODEL 2
REMARK Alternate: template-based docking
PARENT 2def
ATOM      1  N   GLU A   1      10.880  -9.812   1.391  1.00 88.30           N
<... receptor atoms (same protein; may differ slightly) ...>
TER
LIGAND 001 LIG
LSCORE 0.71
template_2def_vina_2
     RDKit          3D

 16 17  0  0  0  0  0  0  0  0999 V2000
<... ligand 001 pose B ...>
M  END
LIGAND 002 LIG
LSCORE 0.58
cofold_boltz2x_1
     RDKit          3D

 28 30  0  0  0  0  0  0  0  0999 V2000
<... ligand 002 pose B ...>
M  END
AFFNTY 0.050 aa
END
```

---

## 7. Implementation checklist for `make_casp_submission.py`

Current implementation (pre-refactor) embeds one LIGAND per MODEL and
reaches 5 MODELs by using 5 alternate poses of a single ligand. That
worked for single-ligand targets but is wrong for multi-ligand targets.

- [ ] Read **all** ligands from `inputs/docking/docking_prep_summary.json`
      (field `ligands[*].id` gives `L`, `L2`, `L3`; ligand_number is the
      index+1 zero-padded to 3 digits).
- [ ] For each ligand, aggregate scored poses from the full pool
      (cofolding × 4 + Vina/ADG/PxDock × 3 sites + template × K + lig-align)
      and rank them.
- [ ] Produce **5 alternate snapshots**: snapshot k picks one pose per
      ligand (typically the k-th best for the primary ligand, with
      per-ligand diversity constraints).
- [ ] Emit **5 MODEL blocks**. Each MODEL:
      - Receptor ATOM + TER (same protein structure across MODELs OK,
        or best cofolded structure per snapshot).
      - N × (LIGAND nnn XXX + LSCORE + MDL + `M  END`).
      - AFFNTY at MODEL level (ensemble across ligands, or primary
        ligand's AFFNTY).
      - Single `END`.
- [ ] Per-ligand cofolded-ligand fallback when a ligand has zero docked
      poses — extract that ligand from the cofolding CIF, emit with
      LSCORE ≈ 0.10.
- [ ] Diversity: within one ligand's 5 picks, ≥ 2 Å pairwise RMSD still
      recommended. Across MODELs, the pose for ligand 001 in MODEL k
      should differ meaningfully from ligand 001 in MODEL k-1.

---

## 8. Target L-prefix vs T-prefix

- **L-prefix** (e.g. `L1214`): ligand target. Primary evaluation is
  ligand pose + affinity.
- **T/H/M-prefix** (e.g. `T1214`): structure target where ligand is
  secondary. LG carries the bound-pose alongside the primary TS submission.

---

## 9. Gotchas

1. `LIGAND nnn` is **zero-padded to 3 digits** (`001`, `002`, ...).
2. Ligand name `XXX` matches the SMILES file's declared name — often
   `LIG` for pharma targets, or a CCD code (`HEM`, `ATP`). Not a free label.
3. LSCORE is **per-ligand**, placed right after `LIGAND`, before MDL.
4. `M  END` = two spaces. Not a tab, not one space.
5. B-factors in receptor ATOM must vary (CASP rejects uniform B-factors);
   populate with pLDDT or a synthetic gradient.
6. MODEL 1 is the primary prediction — put the best-scored snapshot
   there. MODELs 2-5 are alternatives.
7. No `ENDMDL` record exists in LG. Only `M  END` (per MDL) and `END`
   (per MODEL).
