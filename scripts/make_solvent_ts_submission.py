#!/usr/bin/env python3
"""Build a CASP17 ``PFRMAT TS`` ordered-solvent submission.

R2386 asks for something the LG pipeline does not produce: up to five discrete
*solvent* models for a 417-nt group IIC intron, submitted as PDB-style TS with
the RNA plus ``Mg2+``, ``K+``, ``Na+`` and ``H2O`` as HETATM records.  From the
target page:

    Models should be submitted in the PFRMAT TS format for PDBs and include
    predictions of the group IIC intron RNA structure in addition to the
    following ligands: Mg2+, H2O, Na+, K+.  Ligands should be included in the
    PDB file as HETATM records.  Predictors may include AltLoc for ligand
    positions, but total occupancy must be equal to the number of ligands.
    B-factors must be included for every ligand.  [...] Predicted water and
    ions of each model (up to five models) will be assessed, and the best
    prediction will be taken as the group score.  Teams should aim to predict
    the more ordered water+ion positions.

Method.  The molecule is the *Oceanobacillus iheyensis* group II intron, one of
the best-characterised RNAs in the PDB — 63 entries match the target sequence,
several at 2.6-2.9 Å with hundreds of modelled waters and a conserved Mg2+
core.  A predicted fold cannot compete with that as a frame for solvent
placement, so each model takes an experimental structure as its RNA frame and
carries either that structure's own solvent (zero transfer error) or solvent
pooled across structures (wider coverage, superposition error added).  Since
the assessment takes the *best* of five models, the five are deliberately
spread from high-precision to high-recall rather than being five variants of
one bet.

Numbering.  Most matching PDB entries map onto the target with a constant
offset of +5 (``target_resnum = deposited_seqid + 5``), but that cannot be
assumed: an entry whose own numbering skips a value — 9C6I runs -5..-1 and then
1 — comes out shifted, so ``renumber_to_target`` scores the author numbering
against the target sequence, scores every offset of the chain read in order,
and keeps whichever agrees best.  Crystal chains cover roughly target residues
7-395; the disordered termini are left out rather than invented — TS forbids
residue *repetition*, not omission, and unmodelled termini carry no ordered
solvent anyway.

B-factor column.  CASP asks for a 0-100 confidence there.  RNA atoms get a
per-residue value from cross-template coordinate spread; solvent sites get one
from how many independent structures support the site.  Occupancy is 1.00 with
no AltLoc, which satisfies the "total occupancy equals the number of ligands"
rule by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: target_resnum = template_seqid + NUMBERING_OFFSET, checked per template.
NUMBERING_OFFSET = 5

#: Only these four are requested by the target page. Everything else in the
#: crystals (HEPES, spermine, Ca2+, NH4+, ...) is crystallisation additive and
#: is dropped.
SOLVENT_NAMES = ("MG", "K", "NA", "HOH")

#: Tl+ is the standard heavy-atom mimic for K+ and is what the Marcia & Pyle
#: series used to *locate* the potassium sites (4E8Q and friends). A Tl site in
#: those entries is a potassium site, so it is remapped rather than discarded.
SOLVENT_ALIASES = {"TL": "K"}

#: Two solvent sites closer than this (after superposition onto the frame) are
#: the same site seen twice. This only controls *merging*; the physical floors
#: below decide what may coexist.
DEFAULT_MERGE_CUTOFF = 1.0

#: A solvent site closer than this to an RNA atom is a clash, not a site.
#: Kept as the coarse fallback for kinds not in ``MIN_RNA_CONTACT``.
MIN_RNA_DISTANCE = 1.8

#: Shortest contact each solvent species can actually make with an RNA atom.
#:
#: ``polar`` = RNA O/N: a cation sits at its inner-sphere coordination distance
#: (Mg-O 2.07, Na-O 2.4, K-O 2.7-3.0 Å) and a water makes a hydrogen bond
#: (2.6-3.0 Å; 2.4 Å is already strained). ``other`` = C/P, where there is
#: neither coordination nor an H-bond, so the floor is the van der Waals
#: approach. Values are set just below the observed minimum of each interaction
#: so that only impossible geometry is removed, never a tight-but-real site.
#:
#: The single 1.8 Å cutoff this replaces let through contacts no crystal shows:
#: the first R2386 build had waters at 1.80 Å and K+ at 1.81 Å from RNA.
MIN_RNA_CONTACT = {
    "MG": {"polar": 1.95, "other": 2.80},
    "NA": {"polar": 2.20, "other": 2.80},
    "K": {"polar": 2.55, "other": 3.00},
    "HOH": {"polar": 2.40, "other": 2.90},
}

#: Shortest contact between two solvent sites. Cation-water pairs are again
#: inner-sphere coordination; water-water is a hydrogen bond; two cations never
#: approach each other closer than a shared bridging ligand allows (~3.5 Å),
#: so anything under 3.0 Å is a pooling artefact.
MIN_SOLVENT_CONTACT = {
    ("HOH", "HOH"): 2.40,
    ("MG", "HOH"): 1.95,
    ("NA", "HOH"): 2.20,
    ("K", "HOH"): 2.55,
    ("MG", "MG"): 3.00,
    ("MG", "K"): 3.00,
    ("MG", "NA"): 3.00,
    ("K", "K"): 3.00,
    ("K", "NA"): 3.00,
    ("NA", "NA"): 3.00,
}


#: PDB coordinates are written to 0.001 Å, so a distance that clears a floor in
#: float can fall ~0.004 Å below it once written. Enforce the floors with this
#: margin so the *file* satisfies them, not just the in-memory values.
ROUNDING_MARGIN = 0.005


def _solvent_floor(a: str, b: str) -> float:
    """Minimum allowed separation between two solvent sites of kinds ``a``/``b``."""
    return MIN_SOLVENT_CONTACT.get((a, b)) or MIN_SOLVENT_CONTACT.get((b, a)) or 3.00


@dataclass
class Template:
    """One experimental structure, renumbered onto the target."""

    pdb_id: str
    path: Path
    resolution: float
    #: target_resnum → {atom_name: (x, y, z)}
    residues: dict[int, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    #: target_resnum → one-letter residue name
    resnames: dict[int, str] = field(default_factory=dict)
    #: (kind, x, y, z, b_iso)
    solvent: list[tuple[str, float, float, float, float]] = field(default_factory=list)

    def p_map(self) -> dict[int, tuple[float, float, float]]:
        return {n: a["P"] for n, a in self.residues.items() if "P" in a}


def load_template(path: Path, *, offset: int = NUMBERING_OFFSET) -> Template:
    """Read an mmCIF entry and renumber its RNA onto the target."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    tpl = Template(pdb_id=path.stem.upper(), path=path,
                   resolution=float(st.resolution or 0.0))

    # The intron is the longest nucleic-acid chain; short chains are the
    # ligated exon, which the target sequence does not carry as a separate
    # strand.
    best_chain, best_n = None, 0
    for chain in st[0]:
        n = sum(1 for r in chain
                if (info := gemmi.find_tabulated_residue(r.name))
                and info.is_nucleic_acid())
        if n > best_n:
            best_chain, best_n = chain, n

    for chain in st[0]:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            name = res.name.strip().upper()
            if info and info.is_nucleic_acid():
                if chain is not best_chain:
                    continue
                num = res.seqid.num + offset
                atoms = {a.name.strip(): (a.pos.x, a.pos.y, a.pos.z) for a in res}
                # A duplicated residue number would violate TS's "no target
                # residue repetitions"; keep the first occurrence.
                tpl.residues.setdefault(num, atoms)
                tpl.resnames.setdefault(num, name[-1])
                continue
            kind = SOLVENT_ALIASES.get(name, name)
            if kind not in SOLVENT_NAMES:
                continue
            for atom in res:
                tpl.solvent.append(
                    (kind, atom.pos.x, atom.pos.y, atom.pos.z, float(atom.b_iso)))
    return tpl


def check_offset(tpl: Template, sequence: str) -> tuple[int, int]:
    """Return ``(matches, total)`` of renumbered residues against the target."""
    ok = 0
    for num, one in tpl.resnames.items():
        if 1 <= num <= len(sequence) and sequence[num - 1] == one:
            ok += 1
    return ok, len(tpl.resnames)


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Row-vector superposition: ``target ≈ mobile @ R + t``."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    m, t = mobile - mc, target - tc
    v, _, w = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1.0, 1.0, d]) @ w
    rmsd = float(np.sqrt(((m @ r - t) ** 2).sum(axis=1).mean()))
    return r, tc - mc @ r, rmsd


def superpose(donor: Template, frame: Template) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Transform taking ``donor`` coordinates into ``frame``'s frame."""
    dp, fp = donor.p_map(), frame.p_map()
    shared = sorted(set(dp) & set(fp))
    if len(shared) < 20:
        raise ValueError(f"{donor.pdb_id} shares only {len(shared)} P atoms with "
                         f"{frame.pdb_id} — cannot superpose")
    r, t, rmsd = kabsch(np.array([dp[k] for k in shared]),
                        np.array([fp[k] for k in shared]))
    return r, t, rmsd, len(shared)


def pool_solvent(
    frame: Template,
    donors: list[Template],
    *,
    min_support: int,
    merge_cutoff: float = DEFAULT_MERGE_CUTOFF,
) -> list[tuple[str, float, float, float, int, float]]:
    """Merge solvent sites from ``donors`` into ``frame``'s frame.

    Sites of the same kind within ``merge_cutoff`` collapse to one entry at the
    running mean, and the number of *distinct donors* that hit it becomes its
    support. A final pass then resolves overlaps *across* kinds and any residue
    the running mean left behind. Returns ``(kind, x, y, z, support, mean_b)``
    sorted by support.
    """
    buckets: dict[str, list[dict]] = {k: [] for k in SOLVENT_NAMES}
    for donor in donors:
        if donor.pdb_id == frame.pdb_id:
            r, t = np.eye(3), np.zeros(3)
        else:
            r, t, _, _ = superpose(donor, frame)
        for kind, x, y, z, b in donor.solvent:
            p = np.asarray([x, y, z], dtype=float) @ r + t
            group = buckets[kind]
            for site in group:
                if float(np.linalg.norm(site["xyz"] - p)) <= merge_cutoff:
                    site["donors"].add(donor.pdb_id)
                    site["b"].append(b)
                    n = len(site["b"])
                    site["xyz"] = site["xyz"] + (p - site["xyz"]) / n
                    break
            else:
                group.append({"xyz": p, "donors": {donor.pdb_id}, "b": [b]})

    out = []
    for kind, group in buckets.items():
        for site in group:
            support = len(site["donors"])
            if support < min_support:
                continue
            x, y, z = site["xyz"]
            out.append((kind, float(x), float(y), float(z), support,
                        float(np.mean(site["b"]))))
    out.sort(key=lambda s: (-s[4], s[0]))

    # The pooling above buckets by kind, so it cannot see a water sitting on top
    # of an Mg²⁺; and because each absorb moves the running mean, two same-kind
    # sites that started apart can drift under the cutoff. Both leave physically
    # impossible pairs in the output (measured on the first build: 24 % of the
    # union models' solvent atoms had another site within 1 Å, mostly HOH/MG and
    # HOH/K). Resolve globally — best-supported first, keep a site only when it
    # clears everything already kept.
    #
    # The threshold is the *pair's* physical floor, not `merge_cutoff`: at 1 Å
    # the first build still shipped water pairs 1.00 Å apart, which no crystal
    # shows. Two waters need a hydrogen bond (>=2.4 Å) and an ion needs its
    # coordination distance, so those are the distances that decide coexistence.
    kept: list[tuple[str, float, float, float, int, float]] = []
    dropped = 0
    for site in out:
        p = np.asarray(site[1:4], dtype=float)
        if any(float(np.linalg.norm(p - np.asarray(q[1:4], dtype=float)))
               < _solvent_floor(site[0], q[0]) + ROUNDING_MARGIN for q in kept):
            dropped += 1
            continue
        kept.append(site)
    if dropped:
        print(f"    deduped {dropped} site(s) below the solvent-solvent floor")
    return kept


#: Residues the 2026-08-07 amendment declares non-core: a site whose nearest
#: residue is one of these is discarded before scoring, so a slot spent there
#: returns nothing. CASP numbering, i.e. already the numbering this script uses.
NONCORE_RESIDUES = frozenset(
    r for a, b in [(6, 7), (56, 60), (86, 106), (167, 173), (206, 220),
                   (276, 287), (309, 320), (335, 357), (394, 417)]
    for r in range(a, b + 1)
)

#: The same amendment fixes the size of a model: "R2386 models should include
#: 500 ligands". It is a budget, not a cap on quality — an unfilled slot returns
#: nothing and neither does a site in a non-core region, so the question is
#: which 500, not how many.
TARGET_SITE_COUNT = 500


def load_consensus(path: Path) -> list[dict]:
    """Sites from ``cluster_solvent_donors.py`` (frame of that run)."""
    return json.loads(path.read_text())


def consensus_in_frame(sites: list[dict], source: Template,
                       frame: Template) -> list[tuple]:
    """Move consensus sites into ``frame`` and return writer tuples."""
    if source.pdb_id == frame.pdb_id:
        rot, trans = np.eye(3), np.zeros(3)
    else:
        rot, trans, _, _ = superpose(source, frame)
    out = []
    for s in sites:
        x, y, z = np.asarray(s["xyz"], dtype=float) @ rot + trans
        out.append((s["kind"], float(x), float(y), float(z), int(s["n_donors"]), 0.0))
    return out


def nearest_residue(sites: list[tuple], frame: Template) -> list[int]:
    """Residue number nearest each site — what decides core vs non-core."""
    nums, atoms = [], []
    for num, named in frame.residues.items():
        for xyz in named.values():
            nums.append(num)
            atoms.append(xyz)
    a = np.asarray(atoms, dtype=float)
    p = np.asarray([s[1:4] for s in sites], dtype=float)
    out = []
    for i in range(0, len(p), 2000):
        d = np.linalg.norm(p[i:i + 2000, None, :] - a[None, :, :], axis=-1)
        out.extend(nums[j] for j in d.argmin(axis=1))
    return out


def select_budget(sites: list[tuple], frame: Template, *, count: int,
                  strategy: str, native: list[tuple] | None = None,
                  tail: list[tuple] | None = None,
                  spacing: float = 3.5) -> tuple[list[tuple], dict]:
    """Pick exactly ``count`` sites under one bet, physical floors respected.

    Core sites are always exhausted before a non-core one is taken: a non-core
    site is ignored by the assessors, so it is worth only what an empty slot is
    worth, and the budget is fixed either way.

    ``native`` leads the order, ``tail`` follows it. A distant frame's own
    consensus can be smaller than the budget (481 sites for the cryo-EM entry),
    and ``tail`` is the lower-confidence pool that tops it up: sites the base
    consensus places there, taken only once the frame's own list runs out. The
    floor check rejects any that duplicate something already chosen, so a tail
    can only add positions the pick did not already hold.
    """
    resnum = nearest_residue(sites, frame)
    core = [r not in NONCORE_RESIDUES for r in resnum]
    is_ion = {"MG": 0, "K": 0, "NA": 0, "HOH": 1}

    def key(i):
        s = sites[i]
        if strategy == "ions":
            return (is_ion.get(s[0], 1), -s[4])
        return (-s[4],)

    order = sorted(range(len(sites)), key=key)
    if native:
        # The frame's own refined solvent carries no transfer error at all, so
        # it leads; consensus fills the rest.
        sites = list(native) + sites
        order = list(range(len(native))) + [i + len(native) for i in order]
    if tail:
        base = len(sites)
        sites = list(sites) + list(tail)
        order = order + [base + i for i in
                         sorted(range(len(tail)), key=lambda j: -tail[j][4])]
    if native or tail:
        resnum = nearest_residue(sites, frame)
        core = [r not in NONCORE_RESIDUES for r in resnum]

    kept: list[tuple] = []
    stats = {"core": 0, "noncore": 0, "floor_blocked": 0, "spacing_blocked": 0}

    def try_add(i, min_gap):
        s = sites[i]
        p = np.asarray(s[1:4], dtype=float)
        for q in kept:
            d = float(np.linalg.norm(p - np.asarray(q[1:4], dtype=float)))
            if d < _solvent_floor(s[0], q[0]) + ROUNDING_MARGIN:
                stats["floor_blocked"] += 1
                return False
            if min_gap and d < min_gap:
                stats["spacing_blocked"] += 1
                return False
        kept.append(s)
        stats["core" if core[i] else "noncore"] += 1
        return True

    # pass 1: core, with the strategy's spacing; pass 2: core again without it;
    # pass 3: non-core to top the budget up.
    passes = [(order, True, spacing if strategy == "spread" else 0.0),
              (order, True, 0.0),
              (order, False, 0.0)]
    seen: set[int] = set()
    for idx_list, want_core, gap in passes:
        for i in idx_list:
            if len(kept) >= count:
                break
            if i in seen or core[i] != want_core:
                continue
            if try_add(i, gap):
                seen.add(i)
        if len(kept) >= count:
            break
    return kept, stats


def drop_clashing_sites(sites, frame: Template, cutoff: float = MIN_RNA_DISTANCE):
    """Remove solvent sites that overlap an RNA atom of the frame.

    Sites pooled from another crystal form land in this frame through a global
    superposition, so a fraction of them fall inside the RNA. Those are
    artefacts of the transfer, not predictions.

    The floor is per solvent species *and* per RNA atom type (``MIN_RNA_CONTACT``):
    an Mg²⁺ legitimately touches a phosphate oxygen at 2.07 Å but must stay ~2.8 Å
    from carbon, and a water needs a full hydrogen bond either way. A single
    distance for all four species cannot express that — with one 1.8 Å cutoff the
    first R2386 build shipped 117 waters and 7 K⁺ closer to RNA than any crystal
    has ever shown. ``cutoff`` remains the fallback for unknown species.
    """
    atoms = [(name, xyz)
             for res in frame.residues.values() for name, xyz in res.items()]
    if not atoms or not sites:
        return sites, 0
    rna = np.array([xyz for _n, xyz in atoms])
    # RNA atom names are PDB-style, so the leading character is the element:
    # O/N accept coordination or a hydrogen bond, C/P offer neither.
    polar = np.array([n[:1] in ("O", "N") for n, _x in atoms])

    pts = np.array([[s[1], s[2], s[3]] for s in sites])
    keep, dropped = [], 0
    # Chunked to keep the distance matrix small on 8k RNA atoms × ~1k sites.
    for i in range(0, len(pts), 256):
        chunk = pts[i:i + 256]
        d = np.linalg.norm(rna[None, :, :] - chunk[:, None, :], axis=-1)
        d_polar = d[:, polar].min(axis=1) if polar.any() else np.full(len(chunk), 9e9)
        d_other = d[:, ~polar].min(axis=1) if (~polar).any() else np.full(len(chunk), 9e9)
        for j in range(len(chunk)):
            site = sites[i + j]
            floors = MIN_RNA_CONTACT.get(site[0])
            if floors is None:
                bad = min(d_polar[j], d_other[j]) < cutoff + ROUNDING_MARGIN
            else:
                bad = (d_polar[j] < floors["polar"] + ROUNDING_MARGIN
                       or d_other[j] < floors["other"] + ROUNDING_MARGIN)
            if bad:
                dropped += 1
            else:
                keep.append(site)
    return keep, dropped


#: Sugar-phosphate atoms. Identical for all four bases, so a base swap can keep
#: the frame's own backbone and leave every bond length untouched.
BACKBONE_ATOMS = frozenset(
    {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
     "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"})

#: Ribose atoms used to seat a donor's base in the frame's own sugar.
GRAFT_ANCHOR = ("C1'", "C2'", "C3'", "C4'", "O4'")


def graft_base(frame_res: dict, donor_res: dict) -> dict | None:
    """Replace a residue's base while keeping the frame's sugar-phosphate.

    The whole-residue graft this replaced copied the donor residue positioned by
    the *global* superposition, whose RMSD is 0.3-2 A. Locally that is the
    difference between a bond and a clash: at target position 364 it left
    O3'(363)-P(364) at 0.25-0.92 A — atoms on top of each other — and
    O3'(364)-P(365) at 3.8-4.2 A, a backbone break, in every model built before
    2026-08-26.

    Only the base actually differs between two nucleotides, so the fix is to
    change only the base, leaving every backbone atom exactly where the frame
    put it. Returns ``None`` when there is nothing to seat the base against, and
    the caller drops the residue.

    The donor is fitted on the **shared base-ring atoms** where they exist — the
    nine-atom purine ring for G↔A, the six-atom pyrimidine ring for C↔U. That
    keeps the frame's own base plane, which matters because a base is usually
    paired: seating the donor on the ribose instead preserves the *donor's*
    glycosidic torsion, and at target position 364 that swung the new adenine
    into its partner G293 — N1···N1 at 1.02 A where the pair wants 2.9. Only a
    purine↔pyrimidine swap has no shared ring, and there the ribose is the only
    anchor available.
    """
    ring = [a for a in frame_res
            if a in donor_res and a not in BACKBONE_ATOMS]
    common = ring if len(ring) >= 3 else [
        a for a in GRAFT_ANCHOR if a in frame_res and a in donor_res]
    if len(common) < 3:
        return None
    r, t, _ = kabsch(np.array([donor_res[a] for a in common], dtype=float),
                     np.array([frame_res[a] for a in common], dtype=float))
    out = {a: xyz for a, xyz in frame_res.items() if a in BACKBONE_ATOMS}
    for aname, xyz in donor_res.items():
        if aname in BACKBONE_ATOMS:
            continue
        p = np.asarray(xyz, dtype=float) @ r + t
        out[aname] = (float(p[0]), float(p[1]), float(p[2]))
    return out


def renumber_to_target(tpl: Template, sequence: str) -> dict:
    """Adopt whichever numbering agrees best with the target sequence.

    ``load_template`` renumbers by the deposited author numbering plus a
    constant. That is right for the crystal forms, whose author numbering
    already encodes their disorder gaps, but wrong for any entry whose
    numbering itself skips a value: 9C6I runs -5..-1 and then 1, with no 0, so
    a constant +5 puts its first five residues one position low and opens a
    phantom gap at target 5. Everything downstream of that jump lands correctly,
    which is why the error survived a spot check — and the four residues it
    displaced then looked like sequence mismatches and were "repaired" by
    grafting the wrong bases onto them.

    So the numbering is chosen, not assumed: score the author-based numbering
    against the target, score every offset of the chain read in order, and keep
    the best. 9C6I goes from 389/393 residues agreeing to 393/393; the crystal
    forms keep the numbering they already had.
    """
    order = sorted(tpl.residues)
    if not order:
        return {"changed": False, "match": 0, "n": 0}
    names = [tpl.resnames.get(n) for n in order]
    current = sum(1 for n in order
                  if 1 <= n <= len(sequence) and tpl.resnames.get(n) == sequence[n - 1])

    best_off, best = None, current
    for off in range(-15, 16):
        score = sum(1 for i, nm in enumerate(names)
                    if 0 <= i + off < len(sequence) and sequence[i + off] == nm)
        if score > best:
            best, best_off = score, off
    if best_off is None:
        return {"changed": False, "match": current, "n": len(order)}

    mapping = {old: i + best_off + 1 for i, old in enumerate(order)}
    tpl.residues = {mapping[o]: tpl.residues[o] for o in order}
    tpl.resnames = {mapping[o]: tpl.resnames[o] for o in order if o in tpl.resnames}
    return {"changed": True, "match": best, "was": current, "n": len(order)}


def reconcile_sequence(frame: Template, donors: list[Template], sequence: str) -> dict:
    """Make ``frame``'s residues agree with the target sequence.

    The crystal constructs are not byte-identical to the target: every X-ray
    entry carries a G where the target has A at position 364, and the cryo-EM
    entry differs at its 5' end. CASP verifies the model sequence against the
    target, so a residue whose name disagrees has its **base** replaced from a
    donor that does have the right one (see ``graft_base`` — the frame's own
    backbone stays put), or is dropped. Guessing a base by renaming would put
    the wrong atoms in the file.

    A residue with no ribose left to anchor a base is dropped rather than
    written: a phosphate-only stub declares a nucleotide the model does not
    actually place.
    """
    grafted, dropped = [], []
    ranked: list[tuple[float, Template]] = []
    for donor in donors:
        if donor.pdb_id == frame.pdb_id:
            continue
        try:
            r, t, rmsd, _ = superpose(donor, frame)
        except ValueError:
            continue
        ranked.append((rmsd, donor))
    ranked.sort(key=lambda x: x[0])

    for num in sorted(frame.residues):
        if not (1 <= num <= len(sequence)):
            frame.residues.pop(num), frame.resnames.pop(num, None)
            dropped.append(num)
            continue
        want = sequence[num - 1]
        if frame.resnames.get(num) == want:
            if "C1'" not in frame.residues[num]:
                frame.residues.pop(num)
                frame.resnames.pop(num, None)
                dropped.append(num)
            continue
        for _rmsd, donor in ranked:
            if donor.resnames.get(num) != want:
                continue
            swapped = graft_base(frame.residues[num], donor.residues[num])
            if swapped is None:
                continue
            frame.residues[num] = swapped
            frame.resnames[num] = want
            grafted.append((num, donor.pdb_id))
            break
        else:
            frame.residues.pop(num)
            frame.resnames.pop(num, None)
            dropped.append(num)
    return {"grafted": grafted, "dropped": dropped}


def residue_confidence(frame: Template, donors: list[Template]) -> dict[int, float]:
    """Per-residue 0-100 confidence from cross-template P-atom spread.

    A residue whose P atom lands in the same place in every independent
    structure is one we are confident about; a residue that moves between
    crystal forms is not.
    """
    fp = frame.p_map()
    spread: dict[int, list[float]] = {n: [] for n in fp}
    for donor in donors:
        if donor.pdb_id == frame.pdb_id:
            continue
        try:
            r, t, _, _ = superpose(donor, frame)
        except ValueError:
            continue
        dp = donor.p_map()
        for n, xyz in dp.items():
            if n not in fp:
                continue
            p = np.asarray(xyz, dtype=float) @ r + t
            spread[n].append(float(np.linalg.norm(p - np.asarray(fp[n]))))
    conf = {}
    for n in fp:
        vals = spread.get(n) or []
        # 0 Å spread → 100; 5 Å or worse → 0.
        d = float(np.mean(vals)) if vals else 0.0
        conf[n] = round(max(0.0, min(100.0, 100.0 - 20.0 * d)), 2)
    return conf


def atom_confidence(frame: Template, donors: list[Template]) -> dict[int, dict[str, float]]:
    """Per-**atom** 0-100 confidence from cross-template coordinate spread.

    CASP asks for an error estimate per atom, and one value repeated over a whole
    model draws a warning from the verification server — the assessors use the
    column and cannot when it is flat. The information is real and already here:
    superpose every donor on the frame and measure how far each atom moves.
    A base that stacks identically in twelve crystal forms is not the same
    prediction as the 2'-OH that rotates between them, and a residue's phosphate
    is usually pinned while its base edge is not.

    Falls back to the residue's mean for an atom no donor shares.
    """
    spread: dict[int, dict[str, list[float]]] = {
        n: {a: [] for a in res} for n, res in frame.residues.items()}
    for donor in donors:
        if donor.pdb_id == frame.pdb_id:
            continue
        try:
            r, t, _, _ = superpose(donor, frame)
        except ValueError:
            continue
        for n, res in frame.residues.items():
            dres = donor.residues.get(n)
            if not dres:
                continue
            for aname, xyz in res.items():
                if aname not in dres:
                    continue
                p = np.asarray(dres[aname], dtype=float) @ r + t
                spread[n][aname].append(
                    float(np.linalg.norm(p - np.asarray(xyz, dtype=float))))

    out: dict[int, dict[str, float]] = {}
    for n, atoms in spread.items():
        seen = [v for vals in atoms.values() for v in vals]
        fallback = float(np.mean(seen)) if seen else 0.0
        out[n] = {}
        for aname, vals in atoms.items():
            d = float(np.mean(vals)) if vals else fallback
            out[n][aname] = round(max(0.0, min(100.0, 100.0 - 20.0 * d)), 2)
    return out


def _atom_line(record: str, serial: int, name: str, resname: str, chain: str,
               resnum: int, x: float, y: float, z: float, occ: float,
               bfac: float, element: str) -> str:
    """One fixed-column PDB ATOM/HETATM line.

    Atom names follow the PDB rule that 1-3 character names start in column 14
    (leaving column 13 for the element of 4-character names like ``C5'``).
    """
    name = name.strip()
    aname = f"{name:<4s}" if len(name) >= 4 else f" {name:<3s}"
    return (f"{record:<6s}{serial:>5d} {aname}{' '}{resname:>3s} {chain:1s}"
            f"{resnum:>4d}{'':4s}{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{bfac:6.2f}{'':10s}{element:>2s}")


def build_model_lines(
    frame: Template,
    sites,
    conf: dict[int, dict[str, float]],
    chain: str,
) -> tuple[list[str], dict]:
    """Render one model's ATOM + HETATM records."""
    lines: list[str] = []
    serial = 1
    for num in sorted(frame.residues):
        resname = frame.resnames.get(num, "N")
        per_atom = conf.get(num) or {}
        for aname, (x, y, z) in frame.residues[num].items():
            element = aname[0] if aname[0].isalpha() else aname[1]
            lines.append(_atom_line("ATOM", serial, aname, resname, chain, num,
                                    x, y, z, 1.00, per_atom.get(aname, 50.0),
                                    element))
            serial += 1
    lines.append(f"TER   {serial:>5d} {'':5s}"
                 f"{frame.resnames.get(max(frame.residues), 'N'):>3s} {chain:1s}"
                 f"{max(frame.residues):>4d}")
    serial += 1

    counts = {k: 0 for k in SOLVENT_NAMES}
    n_donor_max = max((s[4] for s in sites), default=1)
    for kind, x, y, z, support, _b in sites:
        counts[kind] += 1
        element = "O" if kind == "HOH" else kind
        bfac = round(100.0 * support / max(1, n_donor_max), 2)
        lines.append(_atom_line("HETATM", serial, "O" if kind == "HOH" else kind,
                                kind, chain, counts[kind], x, y, z, 1.00, bfac,
                                element))
        serial += 1
    return lines, counts


def build_ts_submission(
    target_id: str,
    author: str,
    method: str,
    models: list[tuple[list[str], str]],
) -> str:
    """Assemble the TS file: one header, then ``MODEL n`` … ``END`` per model."""
    if not models:
        raise ValueError("no models to write")
    if len(models) > 5:
        raise ValueError(f"TS allows at most 5 models; got {len(models)}")
    out = ["PFRMAT TS", f"TARGET {target_id}", f"AUTHOR {author}"]
    for line in method.splitlines():
        out.append(f"METHOD {line}")
    for idx, (lines, remark) in enumerate(models, start=1):
        out.append(f"MODEL {idx}")
        if remark:
            out.append(f"REMARK {remark}")
        out.append("PARENT N/A")
        out.extend(lines)
        out.append("END")
    return "\n".join(out) + "\n"


#: (frame pdb id, minimum donor support, label). ``min_support=0`` means the
#: frame's own solvent only. The five are spread precision → recall on purpose:
#: the assessment scores the best model, so five variants of one bet waste four
#: slots.
DEFAULT_PLAN = [
    ("3G78", 0, "native solvent of the richest single structure"),
    ("3G78", 1, "union of all donor structures (max recall)"),
    ("3G78", 3, "sites seen in >=3 independent structures (max precision)"),
    ("9C6I", 1, "cryo-EM frame, pooled solvent"),
    ("4FAW", 2, "highest-resolution X-ray frame, sites seen in >=2 structures"),
]


#: (frame, strategy, label) — five different bets on *which* 500, since the
#: assessment takes the best model and the count is fixed for all of them.
#:
#: Ordering strategies alone cannot produce five different models here. Every
#: frame yields fewer core candidates (376-469) than the 500-site budget, so a
#: selector that only reorders ends up taking the whole core whatever the order:
#: the first build shipped ``support`` and ``spread`` as byte-identical models
#: and ``ions`` at 495/500 shared with them. Two axes do change the answer:
#:
#: - **frame** — it moves every site. The X-ray forms sit 0.39-0.86 A apart
#:   (measured P-atom RMSD), the cryo-EM entry 1.7-2.1 A from all of them.
#: - **candidate pool** — ``native`` adds the frame's own refined solvent to
#:   the consensus, which is the one way to get more candidates than slots.
#:   In 3G78 that lifts core coverage 468 -> 500 and shares only 283/500 sites
#:   with the consensus-only pick.
CONSENSUS_PLAN = [
    ("3G78", "native", "the richest frame's own refined solvent first, consensus fills the rest"),
    ("3G78", "support", "same frame, cross-structure consensus only, ranked by supporting structures"),
    ("4FAW", "support", "highest-resolution X-ray frame (2.70 A), consensus re-clustered in it"),
    ("4E8Q", "support", "independent crystal form, the furthest from the anchor that still fills"),
    # The cryo-EM frame matches the target's own modality and is the only frame
    # that is genuinely far from the rest (1.7-2.1 A). Its consensus has to be
    # clustered in it — transferring 3G78's buries 253 of 754 sites in its RNA.
    # Even then the record only supports 481 sites there, 19 short of the
    # budget; 481 with valid geometry beats padding with impossible atoms.
    ("9C6I", "support", "cryo-EM frame - the target's own modality, and the only distant frame"),
]


def build_from_consensus(args, templates, donors, sequence) -> int:
    """Fixed-budget models against the 2026-08-07 amendment."""
    base_id = args.consensus_frame.upper()
    if base_id not in templates:
        print(f"consensus frame {base_id} not among the templates", file=sys.stderr)
        return 1
    cons_cache: dict[str, tuple[list[dict], str]] = {}

    def consensus_for(frame_id: str) -> tuple[list[dict], str]:
        """Prefer a site list clustered in this frame over one transferred in.

        Transferring the 3G78 consensus into a distant frame buries a large
        share of it inside that frame's RNA — 253 of 754 sites for the cryo-EM
        entry — so a frame that has its own ``consensus_sites_<ID>.json`` next
        to the base list uses it.
        """
        if frame_id not in cons_cache:
            own = args.consensus_json.with_name(f"consensus_sites_{frame_id}.json")
            path, src = ((own, frame_id) if own.exists()
                         else (args.consensus_json, base_id))
            cons_cache[frame_id] = (load_consensus(path), src)
            print(f"consensus for {frame_id}: {len(cons_cache[frame_id][0])} sites "
                  f"clustered in {src}")
        return cons_cache[frame_id]

    print()
    plan = [p for p in CONSENSUS_PLAN if p[0] in templates]
    models, summary = [], []
    conf_cache: dict[str, dict[int, float]] = {}
    for frame_id, strategy, label in plan:
        frame = templates[frame_id]
        if frame_id not in conf_cache:
            conf_cache[frame_id] = atom_confidence(frame, donors)
        raw, src_id = consensus_for(frame_id)
        sites = consensus_in_frame(raw, templates[src_id], frame)
        sites, dropped = drop_clashing_sites(sites, frame)
        native = None
        if strategy == "native":
            native, _ = drop_clashing_sites(
                pool_solvent(frame, [frame], min_support=1,
                             merge_cutoff=args.merge_cutoff), frame)
        # A frame reading its own consensus can hold fewer sites than the
        # budget; the base consensus transferred in tops it up rather than
        # leaving slots empty.
        tail = None
        if src_id != base_id and len(sites) < args.target_count:
            tail, _ = drop_clashing_sites(
                consensus_in_frame(consensus_for(base_id)[0],
                                   templates[base_id], frame), frame)
        kept, stats = select_budget(sites, frame, count=args.target_count,
                                    strategy=strategy, native=native, tail=tail)
        lines, counts = build_model_lines(frame, kept, conf_cache[frame_id], args.chain)
        remark = (f"frame {frame_id} ({frame.resolution:.2f} A); {len(kept)} sites, "
                  f"{stats['core']} in core regions; {label}")
        models.append((lines, remark))
        summary.append(f"{frame_id}/{strategy}")
        print(f"  MODEL {len(models)}: {frame_id}/{strategy:7s} "
              f"{len(kept):3d} sites (core {stats['core']:3d}, non-core "
              f"{stats['noncore']:3d})  MG={counts['MG']} K={counts['K']} "
              f"NA={counts['NA']} HOH={counts['HOH']}  "
              f"[{dropped} RNA-clashing dropped, {stats['floor_blocked']} "
              f"blocked by the solvent floor]")
        if len(models) == 5:
            break

    method = args.method or (
        "Template-based ordered-solvent prediction, built to the 2026-08-07 "
        "instruction of 500 sites per model. RNA frame from an experimental "
        "structure of the identical sequence (constant numbering offset +5, "
        "verified per entry). Mg2+/K+/Na+/H2O pooled from 41 donor structures "
        "after P-atom superposition, rejecting any donor whose superposition "
        "exceeds 3 A; Tl+ read as K+. Observations merge into one site when "
        "they are closer than the shortest real separation for that species "
        "pair, so a site is what could physically be occupied once rather than "
        "what falls inside an arbitrary radius. Sites are then filtered against "
        "per-species floors to RNA (Mg 1.95, Na 2.20, K 2.55, water 2.40 A to "
        "O/N; 2.80-3.00 A to C/P) and between each other. The budget is spent "
        "on core regions first - a site nearest one of the declared non-core "
        "residues is not scored, so it is worth no more than an empty slot. The "
        "experimental record supports fewer core sites than the budget, so the "
        "five models differ where that can still change the answer: the RNA "
        "frame, and whether the frame's own refined solvent leads the "
        "consensus. Frames span 0.39-2.06 A of P-atom RMSD from each other and "
        "each reads a consensus clustered in its own frame. B-factors carry a "
        "0-100 confidence: RNA from cross-structure coordinate spread, solvent "
        "from the number of supporting structures.\n"
        f"Frames/modes: {', '.join(summary)}."
    )
    text = build_ts_submission(args.target_id, args.author, method, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"\nwrote {args.output} ({text.count(chr(10))} lines, {len(models)} models)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--sequence", required=True,
                        help="target sequence, or a path to a file holding it")
    parser.add_argument("--templates", nargs="+", type=Path, required=True,
                        help="experimental mmCIF entries; frames are picked from "
                             "these by PDB id, and all of them act as solvent donors")
    parser.add_argument("--author", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain", default="0",
                        help="chain id (default 0, matching the CASP template)")
    parser.add_argument("--offset", type=int, default=NUMBERING_OFFSET,
                        help=f"target_resnum - template_seqid (default {NUMBERING_OFFSET})")
    parser.add_argument("--merge-cutoff", type=float, default=DEFAULT_MERGE_CUTOFF,
                        metavar="A",
                        help="two same-kind sites within this distance are one site "
                             f"(default {DEFAULT_MERGE_CUTOFF} Å)")
    parser.add_argument("--consensus-json", type=Path, default=None,
                        help="site list from scripts/cluster_solvent_donors.py. "
                             "Switches to the 2026-08-07 rules: a fixed budget of "
                             "--target-count sites per model, spent on core "
                             "regions first.")
    parser.add_argument("--consensus-frame", default="3G78",
                        help="frame the consensus JSON was clustered in")
    parser.add_argument("--target-count", type=int, default=TARGET_SITE_COUNT,
                        help=f"sites per model (default {TARGET_SITE_COUNT})")
    parser.add_argument("--min-identity", type=float, default=0.95,
                        help="reject a template whose renumbered residues agree with "
                             "the target below this fraction (default 0.95)")
    args = parser.parse_args()

    seq_arg = Path(args.sequence)
    sequence = (seq_arg.read_text().split(">")[-1].split("\n", 1)[-1]
                if seq_arg.exists() else args.sequence)
    sequence = "".join(sequence.split()).upper().replace("T", "U")
    print(f"target {args.target_id}: {len(sequence)} nt")

    print("\nLoading templates...")
    templates: dict[str, Template] = {}
    for path in args.templates:
        tpl = load_template(path, offset=args.offset)
        fix = renumber_to_target(tpl, sequence)
        ok, total = check_offset(tpl, sequence)
        frac = ok / total if total else 0.0
        counts = {k: sum(1 for s in tpl.solvent if s[0] == k) for k in SOLVENT_NAMES}
        flag = "" if frac >= args.min_identity else "  REJECTED (identity)"
        print(f"  {tpl.pdb_id}  res={tpl.resolution:4.2f}  nt={total:3d}  "
              f"seq-match={ok}/{total} ({frac:.1%})  "
              f"MG={counts['MG']:3d} K={counts['K']:3d} NA={counts['NA']:3d} "
              f"HOH={counts['HOH']:4d}{flag}"
              + (f"  RENUMBERED {fix['was']}->{fix['match']}" if fix["changed"] else ""))
        if frac >= args.min_identity:
            templates[tpl.pdb_id] = tpl
    if not templates:
        print("no usable template", file=sys.stderr)
        return 1

    donors = list(templates.values())
    plan = [p for p in DEFAULT_PLAN if p[0] in templates]
    if not plan:
        # Fall back to the highest-resolution frame available.
        best = min(donors, key=lambda t: t.resolution or 99.0)
        plan = [(best.pdb_id, 0, "native solvent"), (best.pdb_id, 1, "pooled solvent")]
        print(f"\nNOTE: none of the planned frames is present; using {best.pdb_id}")

    print("\nReconciling frames with the target sequence...")
    # Every frame that will actually be written has to be reconciled, not just
    # the ones in DEFAULT_PLAN: a frame whose residue names disagree with the
    # target fails CASP's sequence check.
    used_frames = [p[0] for p in DEFAULT_PLAN] + [p[0] for p in CONSENSUS_PLAN]
    for frame_id in dict.fromkeys(f for f in used_frames if f in templates):
        report = reconcile_sequence(templates[frame_id], donors, sequence)
        g, d = report["grafted"], report["dropped"]
        print(f"  {frame_id}: grafted {len(g)} {[f'{n}<-{p}' for n, p in g]}, "
              f"dropped {len(d)} {d}")

    if args.consensus_json:
        return build_from_consensus(args, templates, donors, sequence)

    print("\nBuilding models...")
    models: list[tuple[list[str], str]] = []
    summary: list[str] = []
    conf_cache: dict[str, dict[int, float]] = {}
    for frame_id, min_support, label in plan:
        frame = templates[frame_id]
        if frame_id not in conf_cache:
            conf_cache[frame_id] = atom_confidence(frame, donors)
        if min_support == 0:
            sites = pool_solvent(frame, [frame], min_support=1,
                                 merge_cutoff=args.merge_cutoff)
            source = f"{frame_id} only"
        else:
            sites = pool_solvent(frame, donors, min_support=min_support,
                                 merge_cutoff=args.merge_cutoff)
            source = f"{len(donors)} structures, support>={min_support}"
        sites, dropped = drop_clashing_sites(sites, frame)
        lines, counts = build_model_lines(frame, sites, conf_cache[frame_id], args.chain)
        remark = (f"frame {frame_id} ({frame.resolution:.2f} A); solvent from "
                  f"{source}; {label}")
        models.append((lines, remark))
        n_res = len(frame.residues)
        summary.append(f"{frame_id}/{'native' if not min_support else f'>={min_support}'}")
        print(f"  MODEL {len(models)}: frame {frame_id} ({n_res} nt) "
              f"MG={counts['MG']} K={counts['K']} NA={counts['NA']} "
              f"HOH={counts['HOH']}  (dropped {dropped} RNA-clashing sites)")
        if len(models) == 5:
            break

    method = args.method or (
        "Template-based ordered-solvent prediction. RNA frame from an "
        "experimental structure of the identical sequence (RCSB sequence "
        "search, constant numbering offset +5, verified per entry); Mg2+/K+/"
        "Na+/H2O pooled across independent crystal and cryo-EM structures after "
        "P-atom superposition, merged at 1.0 A, Tl+ sites read as K+. Sites are "
        "then filtered against per-species physical floors: inner-sphere "
        "coordination to RNA O/N (Mg 1.95, Na 2.20, K 2.55 A), a hydrogen bond "
        "for water (2.40 A), van der Waals approach to C/P (2.80-3.00 A), and "
        "the same floors between solvent sites (water-water 2.40 A, cation-"
        "cation 3.00 A). Models span precision (sites seen in >=3 "
        "structures) to recall (union). B-factor column carries a 0-100 "
        "confidence: RNA from cross-structure coordinate spread, solvent from "
        "the number of supporting structures.\n"
        f"Frames/modes: {', '.join(summary)}."
    )
    text = build_ts_submission(args.target_id, args.author, method, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)

    n_lines = text.count("\n")
    print(f"\n{'=' * 60}")
    print(f"  TS submission written: {args.output}")
    print(f"  Size: {len(text):,} bytes   Lines: {n_lines:,}   MODELs: {len(models)}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
