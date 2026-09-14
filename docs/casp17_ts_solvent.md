# CASP17 `PFRMAT TS` — ordered-solvent targets

Covers R2386. This is a different deliverable from the LG path: no docking, no
BA-Pred, no `LIGAND` blocks. The target asks for **water and ion positions**
around an RNA, in PDB-style TS.

## 1. What the target asks for

From the R2386 page, verbatim:

> The challenge is to generate up to 5 discrete solvent models for the
> previously released target W2386. […] This target, R2386, is the group IIC
> intron structure determined under the following solvent conditions: 10 mM
> MgCl2, 5 mM Na-cacodylate (pH 6.5), 100 mM KCl. […] Predicted water and ions
> of each model (up to five models) will be assessed, and the best prediction
> will be taken as the group score. Teams should aim to predict the more
> ordered water+ion positions.
>
> Models should be submitted in the PFRMAT TS format for PDBs and include
> predictions of the group IIC intron RNA structure in addition to the
> following ligands: Mg2+, H2O, Na+, K+. Ligands should be included in the PDB
> file as HETATM records. Predictors may include AltLoc for ligand positions,
> but total occupancy must be equal to the number of ligands. B-factors must be
> included for every ligand.

Metadata: all-group target, ligand target, 417 nt, *Oceanobacillus iheyensis*
group IIC intron, method EM, entry 2026-07-29, server expiration 2026-08-05,
**human expiration 2026-08-26**.

### 1b. Amended instructions (message board 2026-08-07, news item 582)

The experimentalists added two rules after the target was released. Both change
what a good submission looks like:

> The experimentalists/assessors have provided an additional instruction that
> R2386 models should include **500 ligands**.
>
> Only ligands located in the core, well-resolved regions will be assessed.
> Ligands closest to the following non-core residues will be ignored: 6–7,
> 56–60, 86–106, 167–173, 206–220, 276–287, 309–320, 335–357, 394–417. These
> residue numbers are in **CASP numbering, which is +5 relative to the numbering
> in PDB entry 9C6I**.

The target page states it more bluntly: *"Modelers should submit 500 ligands."*

The +5 offset is stated against **9C6I's** numbering — and 9C6I is precisely the
entry whose own numbering skips a value, so taking "+5" literally against its
deposited residue numbers shifts the 5' end by one (§4d).

The count is a **budget, not a cap on quality**: a site nearest a non-core
residue is discarded before scoring, so it costs a slot and returns nothing.
The number that matters is *assessable sites* = 500 minus whatever lands in the
excluded regions. Measured on the 2026-08-06 build:

| MODEL | solvent | near non-core | assessable | short of 500 |
|---|---:|---:|---:|---:|
| 1 | 479 | 104 | 375 | −125 |
| 2 | 486 | 103 | 383 | −117 |
| 3 |  68 |   5 |  63 | −437 |
| 4 | 353 |  68 | 285 | −215 |
| 5 | 133 |  19 | 114 | −386 |

So every MODEL is under budget, and the two richest waste ~20 % of what they do
carry on regions that will not be scored. A rebuild should fill each MODEL to
500 *core* sites — the donor pool is large enough (41 usable experimental
structures, §3) — and the high-precision bets (MODELs 3 and 5) stop making sense
as "few sites": with a fixed budget, precision has to be expressed as *which*
500, not *how many*.

Two consequences that shape everything else:

- **Best of five, not model 1.** The general TS rule ("assessment will focus on
  the model labeled 1") does not apply here — the target page overrides it. So
  the five models should be five *different bets*, spread from high-precision
  to high-recall, not five variants of one.
- **The RNA is context, the solvent is the answer.** A wrong RNA frame moves
  every solvent site with it, so the frame matters — but only through the
  solvent.

## 2. TS format rules

| Record | Where | Note |
|---|---|---|
| `PFRMAT TS` | first line | |
| `TARGET <id>` | header | |
| `AUTHOR <code>` | header | LCDD: `6095-5696-9732` |
| `METHOD <text>` | header | may repeat for multiple lines |
| `MODEL n` | opens a model | `n` = 1..5, in order |
| `PARENT` | after MODEL | `N/A` when not a single-template copy |
| `ATOM` / `HETATM` | body | fixed-column PDB |
| `END` | closes a model | **not** `ENDMDL` |

- The header block appears **once**, above the first `MODEL`.
- A model must contain **no target residue repetition**.
- **Every atom of the target template must be present.** The general TS rule
  allows omitting residues; R2386's verification server does not — it walks the
  zero-coordinate PDB the target page publishes (417 residues, chain 0, one
  canonical atom set per base) and errors on anything absent. This is what
  rejected the 2026-08-26 rev 2 submission (§4e).
- **Solvent must not reuse the RNA's residue numbers.** The server keys residues
  on `(chain, resseq)`, so a water numbered 394 stands where nucleotide 394
  should be. Number the ligands past the last target residue.
- The B-factor column carries a **0-100 confidence** (CASP's general TS rule
  says pLDDT-scaled), and R2386 additionally requires one on every ligand atom.
  It must **vary**: a model that repeats one value draws a verification warning
  — "the assessors will not be able to perform some of their analyses and may
  penalize your model" — and the format page asks for a *per-atom* estimate,
  not a per-residue one (§4f).
- Occupancies of the ligands must sum to the ligand count. Writing every
  ligand at occupancy 1.00 with no AltLoc satisfies this by construction.

`scripts/lint_ts_submission.py` checks all of the above, including the
residue-repetition rule (both duplicate records and a residue split across two
blocks) and the occupancy sum.

## 3. Method — template-based solvent transfer

The intron is one of the best-characterised RNAs in the PDB. An RCSB sequence
search on the target returns **63 entries**, several at 2.6-2.9 Å with hundreds
of modelled waters and a conserved Mg2+ core:

| PDB | res (Å) | nt | HOH | Mg | K | Na |
|---|---:|---:|---:|---:|---:|---:|
| **3G78** | 2.80 | 389 | **435** | **51** | 26 | 0 |
| 3IGI | 3.12 | 389 | 89 | 44 | 22 | 0 |
| 4E8Q | 2.84 | 393 | 56 | 30 | 17\* | 0 |
| 4FAU | 2.87 | 393 | 55 | 24 | 0 | 0 |
| 4FAR | 2.86 | 390 | 49 | 30 | 19 | 0 |
| 4FAQ | 3.11 | 396 | 45 | 0 | 20 | 0 |
| 4E8K | 3.03 | 388 | 35 | 0 | 18 | 0 |
| 4FAX | 3.10 | 392 | 32 | 21 | 0 | **8** |
| 4E8N | 2.96 | 393 | 31 | 31 | 0 | 0 |
| 4FAW | **2.70** | 390 | 22 | 32 | 19 | 0 |
| 8OLS | 3.00 | 390 | 11 | 24 | 18 | 0 |
| 9C6I | **2.56** | 393 | 0 | 21 | 0 | 0 |

`*` 4E8Q's potassium sites are deposited as **Tl+** — thallium is the standard
heavy-atom mimic the Marcia & Pyle series used to *locate* K+. Those sites are
read as K+ (`SOLVENT_ALIASES`); everything else non-requested (HEPES, spermine,
Ca2+, NH4+) is dropped.

No predicted fold competes with a 2.7 Å crystal of the identical molecule as a
frame for solvent placement, so each model takes an experimental structure as
its RNA frame.

### Numbering

Most entries map onto the target with a **constant offset of +5**
(`target_resnum = deposited_seqid + 5`). That offset is checked, never assumed:
an entry whose own numbering skips a value comes out shifted, which is exactly
what happened to the cryo-EM entry (§4d). `renumber_to_target` scores the author
numbering against the target sequence, scores every offset of the chain read in
order, and adopts whichever agrees best — 388/389 to 393/393 per entry. Crystal
chains cover target residues ~7-395; the disordered termini (1-6, 396-417) are
left out.

### Sequence reconciliation

The crystal constructs are not byte-identical to the target. Every X-ray entry
carries a **G at target position 364 where the target has A**; the cryo-EM entry
9C6I differs at its 5' end instead. CASP verifies the model sequence, so a
residue whose name disagrees has its **base** replaced from a donor that has the
right one — fitted on the shared base ring so the frame's own backbone and base
plane stay put (§4d) — or, if no donor has it, the residue is **dropped**.
Renaming a residue is never done: that would leave the wrong atoms under the
right label.

```
3G78: grafted 1 ['364<-4DS6'], dropped 0
9C6I: grafted 0 [], dropped 0
4FAW: grafted 1 ['364<-3G78'], dropped 1 [396]
4E8Q: grafted 4 ['3<-4FAU', '4<-4FAU', '5<-4FAU', '364<-4FAW'], dropped 1 [395]
```

9C6I needs no graft at all once its numbering is read correctly; the three it
used to "need" were an artefact of the shift.

### Solvent pooling

Donor structures are superposed onto the frame by **P atoms of shared target
residue numbers** (Kabsch). Observed spread against 3G78:

```
3IGI 0.33   4FAW 0.66   4FAU 0.82   4FAR 1.29   8OLS 1.41
4E8Q 1.44   4FAX 1.63   9C6I 2.14        (P-atom RMSD, ~389 shared)
```

Sites of the same kind within **1.0 Å** collapse to one entry at the running
mean; the number of *distinct donors* hitting it becomes its **support**, which
also becomes its B-factor confidence (`100 × support / max_support`). Sites
landing within **1.8 Å** of an RNA atom are dropped — those are artefacts of
the transfer, not predictions. The count of dropped sites is printed, since it
measures how lossy a given frame/donor pair is (75 for the 9C6I frame at
2.14 Å; 0-5 for the crystal frames).

That pooling buckets **by kind**, so it cannot see a water sitting on top of an
Mg²⁺, and because each absorb moves the running mean two same-kind sites can
drift back under the cutoff. The first build shipped both: **24 % of the union
models' solvent atoms had another site within 1 Å** (mostly HOH/MG and HOH/K,
down to 0.16 Å). A final global pass now walks the sites best-supported first
and keeps one only when it clears everything already kept, regardless of kind.
Real crystal solvent never comes closer than ~1.8 Å — MODEL 1, which ships
3G78's own refined waters untouched, has a 1.85 Å minimum — so at a 1 Å cutoff
this removes transfer artefacts only. It drops 6-110 sites per model and leaves
**zero sub-1 Å pairs** in all five.

RNA B-factors are a per-residue confidence from cross-structure P-atom spread
(`100 − 20 × mean deviation`, clamped).

## 4. The five models

| MODEL | frame | solvent | Mg | K | Na | HOH | bet |
|---|---|---|---:|---:|---:|---:|---|
| 1 | 3G78 (2.80 Å) | its own | 51 | 26 | 0 | 435 | richest single experimental set, zero transfer error |
| 2 | 3G78 | union of all 12 | 89 | 48 | 1 | 602 | max recall |
| 3 | 3G78 | support ≥ 3 | 31 | 15 | 0 | 38 | max precision |
| 4 | 9C6I (2.56 Å) | union | 81 | 42 | 1 | 549 | cryo-EM frame — same modality as the target |
| 5 | 4FAW (2.70 Å) | support ≥ 2 | 40 | 25 | 0 | 128 | independent, highest-resolution X-ray frame |

MODEL 1 is the safest single bet rather than the most aggressive, because it
is internally consistent: 3G78's waters were refined against 3G78's RNA.

## 4b. The 500-site build (2026-08-10)

Two scripts implement the amendment.

`scripts/cluster_solvent_donors.py` pools every usable donor and merges the
observations. Two changes over the old inline pooling:

- **41 donors, not 12.** A quality gate replaces the hand-picked list: coverage
  80-105 % of the target and a P-atom superposition onto the frame under 3 A.
  That gate matters — seven entries (8K0S, 7UIN, 8K0P, 8K0Q, 8K0R, 8T2T, 7UIM)
  share the P numbering but are longer constructs that superpose at 24-27 A, and
  their solvent was being poured into the pool at effectively random positions.
- **Merging on physical contact, not 1.0 A.** Two observations are one site when
  they are closer than the shortest real separation for that species pair
  (`MIN_SOLVENT_CONTACT`). At the old 1.0 A two waters interpenetrate — an O-O
  hydrogen bond is 2.6-3.0 A — so sites that could never coexist survived as
  separate predictions. Merging is now the exact complement of the clash filter.
- **Repairing the drift the greedy pass leaves behind** (added 2026-08-11). The
  greedy loop compares each observation against a *running mean*, so a cluster's
  centre moves every time it accretes; two clusters seeded more than a floor
  apart can be pulled together by later observations, and nothing looked back.
  That left **107 pairs — 155 of 829 sites — closer than their species pair
  physically allows**, contradicting the property the merge exists to guarantee.
  `repair_overlaps()` now merges the closest offending pair and re-checks until
  a round finds none; the script refuses to write a list that still overlaps.
  Found by eye in the viewer, not by any test: two K sites 1.74 A apart against
  a 3.00 A floor.

Result: 3,430 observations from 41 donors collapse to **746 consensus sites**
(829 from the greedy pass, 83 merged away by the repair). 587 clear the RNA
floors. Support is thin in the tail — 277 sites are seen by two or more donors —
so a 500-site budget necessarily reaches down to single-donor observations for
roughly a third of its slots. 127 sites have donors that disagree on the species
(one is called K by 19 crystals, water by 19, Na by 3 and Mg by 2): the position
is solid, the identity is not.

The repair changes what gets built. Before it, the budget selector was rejecting
**325 candidates per model** as too close to something already chosen; after it,
2. The selector was silently absorbing the clustering defect — which is why the
submitted-candidate file was physically clean both before and after (0
solvent-solvent and 0 solvent-RNA floor violations in all five models either
way). The defect cost site *quality*, not validity: slots went to duplicates of
sites already taken instead of to distinct positions.

`make_solvent_ts_submission.py --consensus-json` then spends the budget. Core
regions are exhausted before any non-core site is taken, because a non-core site
is worth exactly what an empty slot is worth. The five models are five bets on
*which* 500:

| MODEL | frame | bet | sites | scored (core) |
|---|---|---|---:|---:|
| 1 | 3G78 | consensus support | 500 | 468 |
| 2 | 3G78 | the frame's own refined solvent first | 500 | 500 |
| 3 | 3G78 | ions before water | 500 | 468 |
| 4 | 3G78 | support-ranked but spatially spread | 500 | 468 |
| 5 | 4FAW | independent high-resolution X-ray frame | 500 | 447 |

All five carry zero RNA-floor and zero solvent-solvent violations.

## 4c. Four of those "five bets" were one bet (2026-08-26)

The build above ships **MODEL 1 and MODEL 4 byte-identical** — the same 500
positions — with MODEL 3 sharing 495 of 500 with them. Three of the five slots
held one prediction. Superposed onto MODEL 1 and compared at 1 A, the shipped
set agreed 100 / 57 / 99 / 100 / — per cent.

The cause is arithmetic, not a coding slip. Every frame yields **fewer core
candidates than the budget**:

| frame | P-RMSD to 3G78 | pool after RNA floors | core | own consensus fills |
|---|---:|---:|---:|---:|
| 3G78 | 0.00 | 587 | 469 | 500 |
| 4FAW | 0.63 | 570 | 454 | 500 |
| 4FAR | 0.68 | 571 | 459 | 500 |
| 4E8Q | 0.86 | 493 | 394 | 492 |
| 9C6I | 1.71 | 481 | 376 | 481 |

With 469 core candidates and 500 slots, a selector that only *reorders* takes
the entire core whatever the order — `support`, `ions` and `spread` cannot
disagree about a set they all exhaust. Ordering is a lever only when there is
something to leave out.

Two axes do change the answer:

- **Frame.** It moves every site at once. The X-ray forms sit 0.39-0.86 A from
  each other; the cryo-EM entry is 1.7-2.1 A from all of them, the one frame
  that is a genuinely different position bet — and it matches the target's own
  modality.
- **Candidate pool.** `native` adds the frame's refined solvent to the
  consensus, the one way to get more candidates than slots. In 3G78 it lifts
  core coverage 468 → 500 and shares only 283/500 sites with the
  consensus-only pick.

Two changes make the frame axis usable. **Each frame reads a consensus
clustered in itself** (`cluster_solvent_donors.py --frame <ID>`) — transferring
3G78's list into 9C6I buries 253 of 754 sites in its RNA, which is why the
earlier note concluded the cryo-EM frame "cannot fill the budget"; clustered in
its own frame it holds 481. And `select_budget` takes a **`tail`**: when a
frame's own consensus is smaller than the budget, the base consensus
transferred in tops it up, admitted only where the species floors allow, so it
adds positions rather than duplicates. That lifted MODEL 4 from 492 sites / 393
core to 500 / 425.

The rebuilt five:

| MODEL | frame | bet | sites | scored (core) |
|---|---|---|---:|---:|
| 1 | 3G78 (2.80 A) | frame's own refined solvent first, consensus fills the rest | 500 | **500** |
| 2 | 3G78 | same frame, cross-structure consensus only | 500 | 468 |
| 3 | 4FAW (2.70 A) | highest-resolution X-ray frame | 500 | 454 |
| 4 | 4E8Q (2.84 A) | independent form, furthest from the anchor that still fills | 500 | 425 |
| 5 | 9C6I (2.56 A) | cryo-EM frame — the target's modality, the only distant one | 486 | 382 |

Pairwise agreement (same position within 1 A *and* same species, after
superposing each model's RNA onto MODEL 1):

```
        M1    M2    M3    M4    M5
M1     100%   74%   67%   60%   55%
M2      74%  100%   86%   79%   74%
M3      67%   86%  100%   81%   71%
M4      60%   79%   81%  100%   68%
M5      57%   76%   73%   70%  100%
```

No pair is a duplicate; the closest is 86 %. All five keep zero RNA-floor and
zero solvent-solvent violations (minimum contact 1.96-2.04 A).

MODEL 5 stops at **486 of 500**. The 9C6I frame holds no more solvent that
clears the floors, and the base consensus transferred in adds only five sites
before every remaining candidate duplicates one already chosen. 486 sites with
valid geometry is worth more than 14 atoms that no structure could show — and
the 14 would have been non-core, which is discarded before scoring anyway.

A third defect fixed here: `main()` reconciled only the frames named in
`DEFAULT_PLAN`, so a frame used exclusively by `CONSENSUS_PLAN` shipped with its
own residue names — the G-for-A at target position 364 that CASP's sequence
check would reject. The loop now covers every frame either plan names.

## 4d. The RNA chain was broken in every model (2026-08-26)

The solvent was checked against physical floors from the first build; the RNA it
sits in was not. It had two defects, both invisible to the format linter and to
every check listed above, and both shipped in the 13:37 submission.

### The graft moved whole residues by a global fit

The crystal constructs disagree with the target at a few positions — every
X-ray entry carries G where the target has A at 364 — so `reconcile_sequence`
replaced the residue with the same position taken from a donor that has the
right base. It copied the **entire donor residue**, positioned by the *global*
P-atom superposition, whose RMSD is 0.3-2 A. Locally that is the difference
between a bond and a clash:

```
O3'(363) - P(364)   0.25 - 0.92 A     (a bond is 1.60; these atoms overlap)
O3'(364) - P(365)   3.75 - 4.21 A     (backbone break)
```

Two more in MODEL 5 at the 5' end (8.01 A and 6.08 A), and MODEL 4 shipped
residue 395 as a three-atom phosphate stub inherited from a partly-modelled
crystal residue.

Only the base differs between two nucleotides, so `graft_base` now changes only
the base and leaves the frame's own sugar-phosphate untouched. The donor is
fitted on the **shared base-ring atoms** — the nine-atom purine ring for G↔A,
the six-atom pyrimidine ring for C↔U — not on the ribose. Anchoring on the
ribose instead preserves the donor's glycosidic torsion, and at 364 that swung
the new adenine into its Watson-Crick partner G293 (N1···N1 at 1.02 A where the
pair wants 2.9). A residue with no ribose left to anchor against is dropped
rather than written: a phosphate-only stub declares a nucleotide the model does
not place.

### One entry's numbering skips a value

`load_template` renumbered by the deposited author numbering plus a constant +5.
That is right for the crystal forms, whose author numbering already encodes
their disorder gaps. **9C6I runs -5, -4, -3, -2, -1, 1 — there is no 0**, so a
constant +5 puts its first five residues one position low and opens a phantom
gap at target 5. Read in chain order it aligns to the target at offset 0 with
**393/393** residues agreeing; the author numbering plus 5 gives 389/393.

The four displaced residues then looked like sequence mismatches, and the graft
"repaired" them by transplanting bases that were never wrong — which is where
MODEL 5's 8 A backbone breaks came from. 4DS6, the donor 3G78 draws its
position-364 adenine from, had the same numbering and so was donating the
residue next door.

`renumber_to_target` now scores the author numbering against the target, scores
every offset of the chain read in order, and adopts whichever agrees best. It
runs before the identity gate, so a mis-numbered entry is judged on its real
sequence agreement rather than being rejected or silently mis-mapped.

### After both fixes

| | before | after |
|---|---|---|
| broken backbone bonds (O3'-P outside 1.4-1.8 A) | 2 / 2 / 2 / 3 / 2 | 0 in all five |
| O3'-P range | 0.25 - 4.21 A | 1.52 - 1.68 A |
| RNA non-neighbour pairs < 2 A | 0 / 0 / 0 / 10 / 7 | 0 in all five |
| residues without a C1' | 1 (MODEL 4) | 0 |
| MODEL 5 residues | 392, gap at 5 | 393, no gap |
| grafts needed on 9C6I | 3 | 0 |

Solvent counts and the model-to-model agreement are unchanged — these defects
were in the frame, not in the sites.

**What to check on a rebuild**, since none of it is covered by the linter: every
O3'(i)-P(i+1) within 1.4-1.8 A, no non-neighbour RNA pair under 2 A, no residue
without a C1', and each template's numbering re-scored against the target rather
than assumed.

## 4e. Rejected: the check wants the whole molecule (2026-08-26)

The rev 2 submission came back rejected. Transport said HTTP 200; the format
verification server, whose answer arrives by email, did not:

```
# ERROR! Chain 0 Residue U resseq=394 atom P present in the template
          not found in the model 1 counting from the top of the prediction file
# Number of Errors:     509
```

Two causes, and every local check passed both.

### The template is an atom list, not a suggestion

The target page publishes a PDB with all coordinates zeroed: 417 residues,
chain 0, and exactly one atom set per base (A 22, C 20, G 23, U 20 — every
residue carrying a 5'-phosphate). The check requires all 8958 of those atoms.
Our frames modelled 389-393 residues, leaving the disordered termini out on the
strength of "omitting residues is allowed" — true of TS in general, not of this
check. The arithmetic confirms it: MODEL 5, which reaches furthest, is missing
508 template atoms against the reported 509, and target residue 394 is the first
residue it leaves out.

### The solvent was standing on the RNA's residue numbers

Waters, Mg and K were written into chain 0 numbered from 1 per species — HOH
1..424, MG 1..64, K 1..12 — colliding with all 389 RNA residues. Nothing in the
file was lost, but a server that keys residues on `(chain, resseq)` finds a
one-atom water where nucleotide 394 should be.

### Completing the chain

`scripts/complete_ts_chain.py` takes a built submission and the target template
and returns models that carry the full atom list. Residues 397-417 are the
ligated 3' exon and **no entry in the 49-template set places them** — the
secondary chains that carry any of it cover 406-414 (3IGI) and 413-417 (4FAW,
8OLZ) and nothing between. So the missing runs are built, in this order:

1. **Donor, junction-fitted.** A model that covers the whole target donates the
   run, seated on the anchor residue by an exact fit on the three atoms around
   the bond about to form — three non-collinear points fix a rigid transform
   exactly, so the donor's own O3'-P distance carries over. The donor used here
   is the CASP17_own R2386 build: a 9C6I frame with a Boltz-2 3' tail.
2. **Donor, flank-fitted.** The same copy fitted on the residues either side of
   the gap, for interior runs.
3. **Extension.** Failing both, a real stretch of the frame itself becomes the
   geometric unit — its bonds are real because the stretch is.

A candidate is finished before it is judged — bases swapped to the target
sequence, missing atoms filled from another residue of the same base — because
both steps move atoms, and a run that was clash-free as bare backbone can stop
being so once its base is on. If nothing seats cleanly, the run is widened into
the residues already modelled, which hands the donor the exit vector as well as
the tail; that is what the 3G78 frame needed, where every tail built from
residue 395 alone ran into the body.

One ordering detail decides whether the 5' end is buildable at all: a crystal's
5'-terminal residue is deposited without its phosphate, and that phosphate is
the atom a 5' extension anchors on. Fill inside existing residues *before*
building runs, not after.

### Result

| | rev 2 | rev 3 |
|---|---|---|
| residues per model | 389-393 | **417** in all five |
| ATOM records | 8343-8450 | **8958**, the template's own count |
| template atoms missing | 508-596 | **0** |
| atoms not in the template | 0 | 0 |
| solvent residue numbers | 1..424, colliding with the RNA | 418.. , past the last residue |
| backbone bonds | 1.52-1.68 A | 1.52-1.68 A, none broken |
| scored core sites | 500 / 468 / 454 / 425 / 382 | 499 / 466 / 453 / 424 / 382 |

That last row is the check the built termini made necessary. The assessors
assign each predicted ligand to its nearest residue and discard the ones nearest
a non-core residue, so a tail wandering past the solvent could quietly move
sites out of the scored set. It costs 0-2 sites per model.

Residues 1-6 and 394-417 are now predicted context rather than transferred
crystal coordinates, and all of 394-417 is inside the declared non-core region.

## 4f. A flat B-factor column earns a warning (2026-08-26)

Rev 3 was accepted, with one warning per model:

> The same error estimate (i.e., value in the temperature factor field in the
> PDB format) was used for all residues in model 1 for target R2386. The
> prediction will be accepted, but the assessors will not be able to perform
> some of their analyses and may penalize your model.

Self-inflicted: `complete_ts_chain.py` wrote a constant 50.00 into the B-factor
column, discarding the per-residue confidence the builder had computed. Rev 2
carried 359 distinct values; rev 3 carried one.

The fix restores the column and upgrades it to what the format page asks for.
`atom_confidence` replaces `residue_confidence`: instead of scoring a residue by
how far its phosphate moves across the donor structures, it measures every atom
separately. That is real information, not noise — in MODEL 1's residue 394 the
phosphate scores 75.45 and C3' 52.26, because a backbone phosphate is pinned
between crystal forms and the sugar is not. Residues we built carry no such
evidence, so they take a value that decays with each step into the terminus,
which is honestly how much less is known about them.

| | rev 3 | rev 4 |
|---|---:|---:|
| distinct RNA B-factors per model | 1 | 3047 - 3961 |
| range | 50.00 | 0 - 94.28 |

The lesson generalises past this target: a post-processing step that rewrites
coordinate lines has to carry every column it did not compute, not just the ones
it did.

## 5. Running it

Cluster the donor solvent once per frame the plan uses. The base list carries
no suffix; a frame with its own `consensus_sites_<ID>.json` beside it reads that
instead of having the base list transferred in.

```bash
D=data/solvent_templates/R2386
for F in 3G78 4FAW 4E8Q 9C6I; do
  out=$D/consensus_sites_$F.json
  [ "$F" = 3G78 ] && out=$D/consensus_sites.json
  uv run python scripts/cluster_solvent_donors.py \
    --templates "$D/*.cif" --sequence inputs/R2386.fasta \
    --frame "$F" --out "$out"
done
```

```bash
uv run python scripts/make_solvent_ts_submission.py \
  --target-id R2386 \
  --sequence inputs/R2386.fasta \
  --templates data/solvent_templates/R2386/*.cif \
  --author 6095-5696-9732 \
  --consensus-json data/solvent_templates/R2386/consensus_sites.json \
  --output experiments/CASP17/R2386/submissions/R2386_LCDD.ts

uv run python scripts/complete_ts_chain.py \
  --ts experiments/CASP17/R2386/submissions/R2386_LCDD.ts \
  --template data/solvent_templates/R2386/R2386_casp_template.pdb \
  --donor-ts submissions/R2386.ts \
  --out experiments/CASP17/R2386/submissions/R2386_LCDD.ts

uv run python scripts/lint_ts_submission.py \
  experiments/CASP17/R2386/submissions/R2386_LCDD.ts
```

The completion step is not optional (§4e): the builder produces only what the
crystal frames model, and the verification server wants the whole molecule.
`R2386_casp_template.pdb` is the zero-coordinate PDB from the target page, kept
under `data/solvent_templates/R2386/` because it is the authority on which atoms
must exist.

Templates are fetched once from `https://files.rcsb.org/download/<ID>.cif` and
kept under `data/solvent_templates/<target>/`. CPU only, a few minutes.

Dropping `--consensus-json` falls back to the pre-amendment `DEFAULT_PLAN`,
which builds whatever solvent each frame supports rather than a fixed 500.

The linter checks format and occupancy. It does not check whether the five
models are different predictions (§4c), whether the RNA backbone is connected
(§4d), or whether every template atom is present (§4e) — three failures it
passed clean. Verify a rebuild by superposing the models onto
each other and comparing solvent within 1 Å; any pair above ~90 % is a wasted
slot.

## 5b. Physical contact floors (2026-08-06)

Pooling solvent from a dozen crystal forms through one superposition produces
pairs that no structure actually shows. The original build used two blunt
numbers — merge at 1.0 Å, drop anything within 1.8 Å of RNA — and shipped
geometry that is chemically impossible: waters 1.80 Å from RNA, K⁺ at 1.81 Å,
and 458 solvent-solvent pairs closer than a hydrogen bond in MODEL 2 alone
(minimum 1.00 Å).

The filters are now per species, set just below the shortest real example of
each interaction so only impossible geometry is removed:

| contact | floor (Å) | why |
|---|---|---|
| Mg²⁺ – RNA O/N | 1.95 | inner-sphere coordination is 2.07 |
| Na⁺ – RNA O/N | 2.20 | Na–O ≈ 2.4 |
| K⁺ – RNA O/N | 2.55 | K–O 2.7–3.0 |
| H₂O – RNA O/N | 2.40 | hydrogen bond 2.6–3.0 |
| any – RNA C/P | 2.80–3.00 | van der Waals only; no H-bond or coordination |
| H₂O – H₂O | 2.40 | hydrogen bond |
| cation – H₂O | 1.95 / 2.20 / 2.55 | same coordination distances |
| cation – cation | 3.00 | never closer than a shared bridging ligand allows |

Constants live in `MIN_RNA_CONTACT` / `MIN_SOLVENT_CONTACT`. `ROUNDING_MARGIN`
(0.005 Å) is added when enforcing them, because PDB coordinates are written to
0.001 Å and a pair that clears a floor in float can fall ~0.004 Å below it once
written — the first pass left one Mg···HOH at 1.9499 Å against a 1.95 floor.

Effect on R2386: violations went from 9/40/10/155/22 (RNA) and 30/458/8/403/57
(solvent-solvent) across the five MODELs to **zero in all ten counts**, at a
cost of 6–48 % of the sites per MODEL. Verify any rebuild the same way before
submitting.

## 6. Known limits

- **28 of 417 residues are unmodelled** (1-6, 396-417) in the crystal frames —
  disordered termini that carry no ordered solvent. MODEL 4's cryo-EM frame
  reaches 1-393. Filling the rest would need a folded model, which would move
  the frame and therefore the solvent.
- **Na+ is thin**: only 4FAX deposits sodium (8 sites), so only the union
  models carry any — and after the overlap pass just **1** survives, because 7
  of those 8 sit on a site another structure calls K+ or water and lose the
  support comparison. The target's 5 mM Na-cacodylate is a minor component next
  to 100 mM KCl, so a near-absence of Na+ is probably the right shape, but it is
  a gap, and the contested sites say the assignment itself is not settled.
- Solvent transferred between crystal forms inherits the superposition error
  (0.3-2.1 Å). The consensus models exist to hedge against exactly that.
