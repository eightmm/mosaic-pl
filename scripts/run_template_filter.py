#!/usr/bin/env python3
"""Bridge script: filter template search hits by ligand type and MCS similarity.

Reads mmseqs_hits.tsv, queries rcsb_index.db for ligand info, computes
Tanimoto/MCS against the target SMILES, writes filtered_hits.tsv.

Usage:
    python run_template_filter.py \
        --hits-tsv outputs/template_search_sequence/mmseqs_hits.tsv \
        --rcsb-db ~/DB/RCSB/processed/rcsb_index.db \
        --input-yaml inputs/boltz_input.yaml \
        --output-tsv outputs/template_search_sequence/filtered_hits.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_first_smiles(yaml_path: Path) -> str | None:
    """Extract first ligand SMILES from unified input YAML.

    Uses yaml.safe_load so block-style nested mappings (the default Boltz
    adapter output where ``id`` and ``smiles`` sit under a ``- ligand:``
    list entry) parse correctly. Returns the first ligand SMILES found or
    None if there is no ligand block.
    """
    try:
        import yaml  # pyyaml is already a hub dep
    except Exception:
        return None
    data = yaml.safe_load(yaml_path.read_text()) or {}
    for entry in data.get("sequences", []) or []:
        if isinstance(entry, dict) and "ligand" in entry:
            smi = (entry["ligand"] or {}).get("smiles")
            if smi:
                return str(smi).strip().strip("'\"")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter template hits with ligand + MCS scoring.")
    parser.add_argument(
        "--hits-tsv",
        type=Path,
        default=None,
        help="MMseqs2 easy-search TSV. Optional — pass --foldseek-tsv alone if "
             "you only have structural hits.",
    )
    parser.add_argument(
        "--foldseek-tsv",
        type=Path,
        default=None,
        help="Foldseek easy-search TSV (output of template_search_structure stage). "
             "When provided, hits are union-merged with --hits-tsv on (pdb_id, chain_id).",
    )
    parser.add_argument("--rcsb-db", type=Path, required=True)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument(
        "--max-deposition-date",
        type=str,
        default=None,
        help="Drop template hits whose RCSB deposition_date is on or after this "
             "ISO date (YYYY-MM-DD). Used for time-split benchmarks.",
    )
    args = parser.parse_args()

    if args.hits_tsv is None and args.foldseek_tsv is None:
        print("Need at least one of --hits-tsv or --foldseek-tsv; nothing to filter.")
        return 0
    if args.hits_tsv is not None and not args.hits_tsv.exists():
        print(f"  warn: --hits-tsv {args.hits_tsv} not found, falling back to foldseek-only.")
        args.hits_tsv = None
    if args.foldseek_tsv is not None and not args.foldseek_tsv.exists():
        print(f"  warn: --foldseek-tsv {args.foldseek_tsv} not found, falling back to mmseqs-only.")
        args.foldseek_tsv = None
    if args.hits_tsv is None and args.foldseek_tsv is None:
        print("No template hit TSVs available, skipping filter.")
        return 0

    target_smiles = extract_first_smiles(args.input_yaml) if args.input_yaml.exists() else None
    print(f"Target SMILES: {target_smiles[:60]}..." if target_smiles and len(target_smiles) > 60 else f"Target SMILES: {target_smiles}")
    print(f"  mmseqs hits TSV: {args.hits_tsv}")
    print(f"  foldseek hits TSV: {args.foldseek_tsv}")

    from casp17.template_filter import filter_hits_with_ligands

    results = filter_hits_with_ligands(
        hits_tsv=args.hits_tsv,
        foldseek_tsv=args.foldseek_tsv,
        db_path=args.rcsb_db,
        target_smiles=target_smiles,
        output_path=args.output_tsv,
        max_deposition_date=args.max_deposition_date,
    )
    n_both = sum(1 for h in results if h.in_mmseqs and h.in_foldseek)
    n_seq = sum(1 for h in results if h.in_mmseqs and not h.in_foldseek)
    n_struct = sum(1 for h in results if h.in_foldseek and not h.in_mmseqs)
    print(
        f"Filtered: {len(results)} unique hits with ligands → {args.output_tsv} "
        f"(both={n_both}, mmseqs-only={n_seq}, foldseek-only={n_struct})"
    )
    for h in results[:5]:
        prov = "+".join(s for s, on in [("mmseqs", h.in_mmseqs), ("foldseek", h.in_foldseek)] if on)
        print(
            f"  {h.pdb_id}_{h.chain_id} [{prov}] "
            f"pident={h.pident:.1f}% qtm={h.qtmscore:.3f} "
            f"tanimoto={h.best_tanimoto:.3f} mcs={h.best_mcs_coverage:.3f} ligs={len(h.ligands)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
