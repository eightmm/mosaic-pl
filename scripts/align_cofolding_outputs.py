#!/usr/bin/env python3
"""Align all cofolding outputs to a common reference frame.

After cofolding, each model (Boltz2/2x, Protenix, AF3) produces structures in
its own coordinate frame. This bridge aligns every CIF to a single reference:

  1. If a template PDB/CIF is available (from template search), align to that.
  2. Otherwise, pick the highest-pLDDT cofolding model as the reference and
     align all others to it.

The aligned CIFs are written alongside the originals as ``*_aligned.cif``.
Downstream scripts (docking prep, post-analysis, submission) should prefer
``_aligned.cif`` when it exists.

Usage:
    python align_cofolding_outputs.py --run-dir experiments/runs/L2001_input
    python align_cofolding_outputs.py --run-dir ... --reference path/to/template.cif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_confidence(run_dir: Path) -> dict[str, float]:
    """Read pLDDT scores for each cofolding model (reuse prepare_docking_inputs logic)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_docking_inputs import read_confidence_score

    scores: dict[str, float] = {}
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = run_dir / "outputs" / model
        if model_dir.exists():
            s = read_confidence_score(model_dir, model)
            if s > 0:
                scores[model] = s
    return scores


def _pick_best_template(run_dir: Path, rcsb_dir: Path | None) -> Path | None:
    """Find the top template hit CIF from template search results."""
    hits_tsv = run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
    if not hits_tsv.exists() or rcsb_dir is None:
        return None

    import csv
    with open(hits_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pdb_id = row.get("pdb_id", row.get("target", "")).strip()[:4].lower()
            if not pdb_id or len(pdb_id) != 4:
                continue
            # Try common mmCIF paths
            for candidate in [
                rcsb_dir / f"{pdb_id}.cif",
                rcsb_dir / f"{pdb_id}.cif.gz",
                rcsb_dir / pdb_id[1:3] / f"{pdb_id}.cif",
                rcsb_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz",
            ]:
                if candidate.exists():
                    print(f"  Template reference: {pdb_id} → {candidate}")
                    return candidate
            break  # only try the top hit
    return None


def _pick_best_cofolding_as_ref(run_dir: Path, scores: dict[str, float]) -> Path | None:
    """Pick the first CIF from the highest-pLDDT cofolding model."""
    if not scores:
        return None
    best_model = max(scores, key=scores.get)
    model_dir = run_dir / "outputs" / best_model
    for cif in sorted(model_dir.rglob("*.cif")):
        if "_aligned" in cif.name:
            continue
        print(f"  Cofolding reference: {best_model} (pLDDT {scores[best_model]:.1f}) → {cif.name}")
        return cif
    return None


def _collect_ca(structure):
    """Extract (chain, resi, resname, Position) for all CA atoms."""
    cas = []
    for model in structure:
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", "*")
                if ca:
                    cas.append((chain.name, res.seqid.num, res.name, ca.pos))
        break  # first model only
    return cas


def _sequence_match_cas(ref_cas, query_cas):
    """Match CA atoms by 1-letter residue sequence with sliding offset.
    Returns list of (ref_Position, query_Position) pairs."""

    _AA3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
        "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V", "MSE": "M",
    }

    # Group CAs by chain, build per-chain 1-letter sequences
    def _by_chain(cas):
        chains: dict[str, list] = {}
        for ch, resi, resname, pos in cas:
            one = _AA3TO1.get(resname)
            if one:
                chains.setdefault(ch, []).append((resi, one, pos))
        return chains

    ref_chains = _by_chain(ref_cas)
    query_chains = _by_chain(query_cas)

    # For each query chain (sorted by size desc), find best matching ref chain
    pairs_ref = []
    pairs_query = []
    used_ref_chains = set()

    for q_ch, q_list in sorted(query_chains.items(), key=lambda kv: -len(kv[1])):
        q_seq = "".join(tri for _, tri, _ in q_list)
        best_chain = None
        best_offset = 0
        best_matches = 0

        for r_ch, r_list in ref_chains.items():
            if r_ch in used_ref_chains:
                continue
            r_seq = "".join(tri for _, tri, _ in r_list)
            # Sliding offset
            for offset in range(-len(r_seq) + 1, len(q_seq)):
                matches = sum(
                    1 for i in range(len(r_seq))
                    if 0 <= i + offset < len(q_seq) and r_seq[i] == q_seq[i + offset]
                )
                if matches > best_matches:
                    best_matches = matches
                    best_offset = offset
                    best_chain = r_ch

        if best_chain is None or best_matches < 10:
            continue

        r_list = ref_chains[best_chain]
        for i, (_, r_one, r_pos) in enumerate(r_list):
            j = i + best_offset
            if 0 <= j < len(q_list) and q_list[j][1] == r_one:
                pairs_ref.append(r_pos)
                pairs_query.append(q_list[j][2])
        used_ref_chains.add(best_chain)
        break  # one chain is enough for alignment

    return pairs_ref, pairs_query


def align_structure(query_path: Path, ref_structure, ref_cas, output_path: Path) -> dict:
    """Align a query CIF to the reference structure using CA superposition.

    Writes the aligned structure to ``output_path`` and returns a summary dict.
    """
    import gemmi

    query_st = gemmi.read_structure(str(query_path))
    query_cas = _collect_ca(query_st)

    pairs_ref, pairs_query = _sequence_match_cas(ref_cas, query_cas)
    n_matched = len(pairs_ref)

    if n_matched < 10:
        return {
            "query": str(query_path),
            "aligned": None,
            "n_matched_ca": n_matched,
            "rmsd_before": None,
            "rmsd_after": None,
            "status": "skipped_insufficient_matches",
        }

    # Compute superposition: finds transform that maps query onto ref
    sup = gemmi.superpose_positions(pairs_ref, pairs_query)
    rmsd_after = sup.rmsd

    # Compute RMSD before alignment
    import math
    rmsd_before = math.sqrt(
        sum(
            (r.x - q.x) ** 2 + (r.y - q.y) ** 2 + (r.z - q.z) ** 2
            for r, q in zip(pairs_ref, pairs_query)
        ) / n_matched
    )

    # Apply transform to ALL atoms in the query structure
    transform = sup.transform
    for model in query_st:
        for chain in model:
            for res in chain:
                for atom in res:
                    v = transform.apply(atom.pos)
                    atom.pos = gemmi.Position(v.x, v.y, v.z)

    doc = query_st.make_mmcif_document()
    doc.write_file(str(output_path))

    return {
        "query": str(query_path),
        "aligned": str(output_path),
        "n_matched_ca": n_matched,
        "rmsd_before": round(rmsd_before, 3),
        "rmsd_after": round(rmsd_after, 3),
        "status": "aligned",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Align cofolding outputs to a common reference frame.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None,
                        help="Explicit reference CIF/PDB. If omitted, auto-selects from template search or best cofolding model.")
    parser.add_argument("--rcsb-dir", type=Path, default=None,
                        help="RCSB mmCIF directory for template lookup.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    print("Aligning cofolding outputs to common reference frame...")

    # 1. Determine reference
    import gemmi

    ref_path = args.reference
    if ref_path is None:
        ref_path = _pick_best_template(run_dir, args.rcsb_dir)
    if ref_path is None:
        scores = _read_confidence(run_dir)
        ref_path = _pick_best_cofolding_as_ref(run_dir, scores)
    if ref_path is None:
        print("ERROR: no reference structure found (no template, no cofolding outputs)")
        return 1

    print(f"  Reference: {ref_path}")
    ref_st = gemmi.read_structure(str(ref_path))
    ref_cas = _collect_ca(ref_st)
    print(f"  Reference CA atoms: {len(ref_cas)}")

    # 2. Align each cofolding model's CIFs
    results = []
    n_aligned = 0
    n_skipped = 0

    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue

        cifs = sorted(
            c for c in model_dir.rglob("*.cif")
            if "_aligned" not in c.name
        )

        for cif in cifs:
            # Skip if this IS the reference
            if cif.resolve() == ref_path.resolve():
                # Still write an _aligned copy (identity transform)
                aligned_path = cif.with_name(cif.stem + "_aligned.cif")
                if not aligned_path.exists():
                    import shutil
                    shutil.copy2(cif, aligned_path)
                results.append({
                    "query": str(cif),
                    "aligned": str(aligned_path),
                    "n_matched_ca": len(ref_cas),
                    "rmsd_before": 0.0,
                    "rmsd_after": 0.0,
                    "status": "reference_identity",
                })
                n_aligned += 1
                continue

            aligned_path = cif.with_name(cif.stem + "_aligned.cif")
            result = align_structure(cif, ref_st, ref_cas, aligned_path)
            results.append(result)

            if result["status"] == "aligned":
                n_aligned += 1
                print(f"  {model}/{cif.name}: {result['n_matched_ca']} CA, "
                      f"RMSD {result['rmsd_before']:.1f} → {result['rmsd_after']:.2f} Å")
            else:
                n_skipped += 1
                print(f"  {model}/{cif.name}: skipped ({result['status']})")

    # 3. Write alignment summary
    summary = {
        "reference": str(ref_path),
        "n_aligned": n_aligned,
        "n_skipped": n_skipped,
        "details": results,
    }
    summary_path = run_dir / "outputs" / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nAlignment complete: {n_aligned} aligned, {n_skipped} skipped")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
