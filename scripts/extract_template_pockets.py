#!/usr/bin/env python3
"""Extract pocket centers from every candidate template hit.

After ``run_template_filter.py`` writes ``filtered_hits.tsv`` (union of
mmseqs + foldseek), this script walks every hit, aligns the template
structure onto the best cofolding model, and records the heavy-atom
centroid of every bound candidate ligand instance — projected into the
cofolding reference frame.

The output is a *flat list* of pocket points; clustering is the next step
and runs separately so it can be ablated independently.

Alignment uses **USalign** (structure-based TM-align algorithm), not
gemmi's sequence-anchored superposition. Foldseek finds templates by
3Di + structural similarity, so the reciprocal alignment must also be
structure-based — sequence-anchored gemmi collapses on distant homologs
(matched-residue count drops to <50, transform RMSD blows up to 15+ Å,
ligand centroids end up tens of Å off-pocket). USalign reproduces the
same "fold view" foldseek used to find the hit.

Usage::

    python extract_template_pockets.py \
        --run-dir experiments/runs/<target> \
        --rcsb-dir ~/DB/RCSB/raw/mmCIF_data \
        --max-templates 100

Output::

    <run-dir>/outputs/template_pockets/template_pockets.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# CIF lookup / extraction helpers come from the ion-placement script so we
# don't duplicate that path. Alignment moved to the shared usalign module
# (structure-based, not the gemmi sequence-anchored superposition).
from collect_template_ions import (  # noqa: E402
    extract_cif,
    find_best_cofolding_structure,
    find_template_cif,
)
from casp17.usalign import run_usalign, transform_point  # noqa: E402


@dataclass
class PocketPoint:
    """One pocket observation: centroid of a candidate ligand bound to a
    template, expressed in the cofolding reference frame."""
    template_pdb_id: str
    template_chain: str
    ligand_ccd: str
    ligand_chain: str
    ligand_n_heavy: int
    centroid_x: float
    centroid_y: float
    centroid_z: float
    # USalign-reported quality of the template→reference superposition.
    # ``alignment_tmscore`` (reference-normalized) is the canonical
    # structure-similarity score; ``alignment_rmsd`` is on the matched
    # subset only (not structure-wide).
    alignment_tmscore: float
    alignment_rmsd: float
    in_mmseqs: bool
    in_foldseek: bool
    pident: float
    qtmscore: float
    best_tanimoto: float
    best_mcs_coverage: float


def _parse_filtered_hits(tsv_path: Path) -> list[dict]:
    if not tsv_path.exists():
        return []
    rows: list[dict] = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                if int(row.get("num_ligands", 0)) <= 0:
                    continue
            except ValueError:
                continue
            rows.append(row)
    return rows


def _candidate_ccds(row: dict) -> set[str]:
    """Return the set of candidate-ligand CCDs reported in the filter row.

    The filter writes ``ligand_codes = "AAA;BBB;..."`` where each entry is one
    instance of an ``is_candidate=1`` ligand attached to the PDB. We only
    need the *unique CCDs*; per-residue enumeration happens by walking the
    CIF directly so we also pick up multiple binding sites for the same CCD.
    """
    raw = row.get("ligand_codes") or ""
    return {x.strip().upper() for x in raw.split(";") if x.strip()}


#: Heavy-atom contact radius used to attribute a bound ligand to its host
#: protomer. 6 Aa is past a hydrogen bond and well inside a packing contact,
#: so the chain that actually cradles the ligand wins by a wide margin.
_HOST_CONTACT_A = 6.0


def _ligand_centroids(cif_path: Path, candidate_ccds: set[str]) -> list[dict]:
    """For every residue in the CIF whose name is a candidate CCD, return its
    heavy-atom centroid + element count. Multiple binding sites for the same
    CCD therefore yield separate pocket observations.
    """
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    pockets: list[dict] = []
    for model in structure:
        # Polymer atoms per chain, so each ligand can be attributed to the
        # protomer it actually touches. A ligand's own asym id is not a
        # reliable host: mmCIF gives HETATM residues their own label_asym_id,
        # and the chain a hit row reports may be an entity id ("1") that names
        # no chain at all. Contact count is the physical answer.
        poly: dict[str, list] = {}
        for chain in model:
            pts = [a.pos for res in chain
                   if res.entity_type == gemmi.EntityType.Polymer
                   for a in res if a.element.atomic_number > 1]
            if pts:
                poly[chain.name] = pts
        for chain in model:
            for residue in chain:
                if residue.name.upper() not in candidate_ccds:
                    continue
                # Heavy atoms only — ignore hydrogens so the centroid matches
                # the ligand-shape expectation downstream consumers (vina box
                # center, swinsite-equivalent pocket marker) make.
                heavy = [a for a in residue if a.element.atomic_number > 1]
                if not heavy:
                    continue
                cx = sum(a.pos.x for a in heavy) / len(heavy)
                cy = sum(a.pos.y for a in heavy) / len(heavy)
                cz = sum(a.pos.z for a in heavy) / len(heavy)
                best_host, best_n = None, 0
                for cname, pts in poly.items():
                    n = sum(1 for a in heavy for q in pts
                            if a.pos.dist(q) <= _HOST_CONTACT_A)
                    if n > best_n:
                        best_host, best_n = cname, n
                pockets.append({
                    "ccd": residue.name.upper(),
                    "chain": chain.name,
                    "host_chain": best_host or chain.name,
                    "host_contacts": best_n,
                    "seqid": residue.seqid.num,
                    "n_heavy": len(heavy),
                    "x": cx, "y": cy, "z": cz,
                })
        break  # first model only
    return pockets


def _extract_chain_pdb(cif_path: Path, chain_id: str, output_pdb: Path) -> Path | None:
    """Write polymer atoms of a single chain to PDB, for chain-specific USalign.

    USalign on a multi-chain template returns one transform tied to whichever
    chain it best-fit to the cofold monomer. That transform only places the
    ligands of *that* host chain correctly; other protomers' ligands fly into
    deep space (validated 2026-05-02 on 7hqq: 7/10 cluster centroids landed
    50-200 Å away). Aligning a single-chain extraction to the reference makes
    the host chain explicit and removes USalign's chain-mapping ambiguity, so
    the returned R/t is guaranteed to correspond to ``chain_id``.

    Returns ``None`` when the requested chain isn't present (the caller falls
    back to whole-CIF alignment in that case so single-chain templates with
    synthetic asym ids still work).
    """
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    new_struct = gemmi.Structure(); new_struct.name = structure.name
    new_model = gemmi.Model("1")
    matched = False
    target = chain_id.upper()
    for model in structure:
        for chain in model:
            if chain.name.upper() != target:
                continue
            # PDB format only allows single-character chain ids — multi-char
            # asym ids (foldseek often hits ``8qrt_CCC`` etc.) crash
            # ``write_pdb`` with "chain name too long for the PDB format".
            # We're writing a single-chain extraction anyway; rename to "A".
            new_chain = gemmi.Chain("A")
            for res in chain:
                if res.entity_type == gemmi.EntityType.Polymer:
                    new_chain.add_residue(res)
            if len(new_chain) > 0:
                new_model.add_chain(new_chain)
                matched = True
        break  # first model only
    if not matched:
        return None
    new_struct.add_model(new_model)
    new_struct.write_pdb(str(output_pdb))
    return output_pdb


def _safe_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--filtered-hits-tsv",
        type=Path,
        default=None,
        help="Path to filtered_hits.tsv. Defaults to "
             "<run-dir>/outputs/template_search_sequence/filtered_hits.tsv.",
    )
    parser.add_argument(
        "--rcsb-dir",
        type=Path,
        default=Path.home() / "DB/RCSB/raw/mmCIF_data",
    )
    parser.add_argument(
        "--reference-cif",
        type=Path,
        default=None,
        help="Cofolding CIF used as the alignment reference. Defaults to "
             "auto-pick via find_best_cofolding_structure().",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=2000,
        help="Cap how many top-evidence templates USalign aligns. USalign "
             "is fast — measured at ~0.5 s/template on typical 300-aa "
             "structures — so 2000 templates cost ~17 min/target, well "
             "inside the SLURM 12h budget. Default chosen to match "
             "``template_search_structure.max_hits=2000`` so we don't "
             "silently drop the foldseek tail; tail-rank dual-source hits "
             "(mmseqs hit also at foldseek rank 1800) need the full pool "
             "to surface. Beyond 2000 is diminishing returns since most "
             "queries don't have that many credible candidates.",
    )
    # The single authoritative quality gate on the union template pool.
    # Since USalign aligns every selected template, its TM-score is the
    # ground truth — foldseek's qtmscore_min is now disabled by default
    # so this is the only filter that decides which templates contribute
    # pockets. 0.5 is the canonical Zhang/Skolnick "same fold" cutoff;
    # below it the binding-site-equivalence premise of consensus
    # extraction breaks down.
    parser.add_argument(
        "--min-tmscore",
        type=float,
        default=0.5,
        help="Drop a template if USalign's reference-normalized TM-score "
             "falls below this. Default 0.5 — Zhang/Skolnick canonical "
             "same-fold cutoff. With foldseek pre-filter disabled, this "
             "is the only TM gate.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    hits_tsv = args.filtered_hits_tsv or (
        run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
    )
    output_dir = args.output_dir or (run_dir / "outputs" / "template_pockets")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_extract_work"
    work_dir.mkdir(exist_ok=True)

    rows = _parse_filtered_hits(hits_tsv)
    if not rows:
        print(f"[pockets] no filtered hits with ligands at {hits_tsv}; nothing to extract.")
        (output_dir / "template_pockets.json").write_text(json.dumps({
            "n_templates_attempted": 0,
            "n_templates_aligned": 0,
            "n_pockets": 0,
            "pockets": [],
        }, indent=2))
        return 0

    rows = rows[: args.max_templates]
    reference_cif = args.reference_cif or find_best_cofolding_structure(run_dir)
    if reference_cif is None:
        print("[pockets] no cofolding reference structure found; cannot align.")
        return 1
    print(f"[pockets] reference cif: {reference_cif}")
    print(f"[pockets] processing {len(rows)} template hits (max_templates={args.max_templates})")

    pockets: list[PocketPoint] = []
    #: (pdb_id, host chain) -> USalign result, so the duplicate hit rows the
    #: union filter emits for one entry (mmseqs entity id + foldseek auth
    #: chains) share one alignment each instead of repeating it.
    align_cache: dict[tuple[str, str], tuple | None] = {}
    #: (pdb_id, host chain, ccd, seqid) already emitted — one pocket per
    #: physical ligand instance, however many rows point at it.
    seen_instances: set[tuple[str, str, str, int]] = set()
    n_aligned = 0
    n_failed_cif = 0
    n_failed_align = 0
    n_low_quality_align = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        pdb_id = (row.get("pdb_id") or "").lower()
        candidate_ccds = _candidate_ccds(row)
        if not pdb_id or not candidate_ccds:
            continue

        cif_packed = find_template_cif(pdb_id, args.rcsb_dir)
        if cif_packed is None:
            n_failed_cif += 1
            continue
        try:
            cif = extract_cif(cif_packed, work_dir)
        except Exception as e:
            print(f"  [{pdb_id}] extract_cif failed: {e}")
            n_failed_cif += 1
            continue

        # Per-protomer alignment. Every bound ligand is placed by the transform
        # of the chain that actually cradles it, not by one transform for the
        # whole entry. Aligning once and reusing that R/t threw the other
        # protomers' copies into solvent: on L01 that produced two phantom
        # clusters (7WK/7X9 chain-B copies, 1 and 21 receptor heavy atoms
        # within 8 Aa against 59 and 36 for the chain-A copies of the same
        # site) and half of every run's pockets. It reached selection as a
        # weight-1.20 anchor sitting in open solvent in 7 of 10 L01 complexes.
        #
        # A hit row's ``chain_id`` cannot drive this: mmseqs reports the entity
        # id ("1") where foldseek reports auth chains ("A"/"B"), so the same
        # entry arrives three times and the entity row matches no chain at all.
        # Alignments are cached per (pdb_id, host chain) so those duplicate
        # rows cost one USalign each rather than one per row.
        ligand_centroids = _ligand_centroids(cif, candidate_ccds)
        if not ligand_centroids:
            continue

        by_host: dict[str, list[dict]] = {}
        for lig in ligand_centroids:
            by_host.setdefault(lig["host_chain"], []).append(lig)

        row_aligned = False
        for host_chain, ligs in by_host.items():
            ck = (pdb_id, host_chain.upper())
            if ck in align_cache:
                align = align_cache[ck]
            else:
                chain_pdb = _extract_chain_pdb(
                    cif, host_chain, work_dir / f"{pdb_id}_{host_chain}.pdb"
                )
                # Whole-CIF fallback only for genuinely single-chain entries,
                # where there is no other protomer to confuse the transform.
                align_input = chain_pdb if chain_pdb is not None else cif
                align = run_usalign(align_input, reference_cif)
                align_cache[ck] = align
            if align is None:
                n_failed_align += 1
                continue
            R, t_vec, tm_score, rmsd_aligned = align
            if tm_score < args.min_tmscore:
                n_low_quality_align += 1
                continue
            row_aligned = True

            for lig in ligs:
                # One pocket per physical ligand instance. Without this the
                # A-row, B-row and entity-row of the same entry each emit the
                # same copies and inflate every cluster's evidence count.
                ik = (pdb_id, host_chain.upper(), lig["ccd"], lig["seqid"])
                if ik in seen_instances:
                    continue
                seen_instances.add(ik)
                tx, ty, tz = transform_point(R, t_vec, lig["x"], lig["y"], lig["z"])
                pockets.append(PocketPoint(
                    template_pdb_id=pdb_id,
                    template_chain=host_chain,
                    ligand_ccd=lig["ccd"],
                    ligand_chain=lig["chain"],
                    ligand_n_heavy=lig["n_heavy"],
                    centroid_x=round(tx, 3),
                    centroid_y=round(ty, 3),
                    centroid_z=round(tz, 3),
                    alignment_tmscore=round(float(tm_score), 4),
                    alignment_rmsd=round(float(rmsd_aligned), 3),
                    in_mmseqs=row.get("in_mmseqs", "0") == "1",
                    in_foldseek=row.get("in_foldseek", "0") == "1",
                    pident=_safe_float(row.get("pident")),
                    qtmscore=_safe_float(row.get("qtmscore")),
                    best_tanimoto=_safe_float(row.get("best_tanimoto")),
                    best_mcs_coverage=_safe_float(row.get("best_mcs_coverage")),
                ))
        if row_aligned:
            n_aligned += 1

        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(rows) - i)
            print(f"  [{i}/{len(rows)}] aligned={n_aligned} pockets={len(pockets)} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    summary = {
        "reference_cif": str(reference_cif),
        "alignment_tool": "USalign",
        "n_templates_attempted": len(rows),
        "n_templates_aligned": n_aligned,
        "n_failed_cif": n_failed_cif,
        "n_failed_align": n_failed_align,
        "n_low_quality_align": n_low_quality_align,
        "min_tmscore": args.min_tmscore,
        "n_pockets": len(pockets),
        "pockets": [asdict(p) for p in pockets],
    }
    out_path = output_dir / "template_pockets.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[pockets] wrote {len(pockets)} pockets from {n_aligned}/{len(rows)} aligned templates "
        f"(low-quality dropped: {n_low_quality_align}, tm_score < {args.min_tmscore}) → {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
