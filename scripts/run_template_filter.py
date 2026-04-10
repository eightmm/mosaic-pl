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
    """Extract first ligand SMILES from unified input YAML."""
    text = yaml_path.read_text()
    in_ligand = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ligand:") or stripped == "ligand:":
            in_ligand = True
            continue
        if in_ligand:
            if stripped.startswith("smiles:"):
                return stripped.split(":", 1)[1].strip().strip("'\"")
            if stripped.startswith("- ") or (stripped and not stripped.startswith((" ", "#"))):
                in_ligand = False
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter template hits with ligand + MCS scoring.")
    parser.add_argument("--hits-tsv", type=Path, required=True)
    parser.add_argument("--rcsb-db", type=Path, required=True)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    if not args.hits_tsv.exists():
        print(f"No hits TSV found at {args.hits_tsv}, skipping filter.")
        return 0

    target_smiles = extract_first_smiles(args.input_yaml) if args.input_yaml.exists() else None
    print(f"Target SMILES: {target_smiles[:60]}..." if target_smiles and len(target_smiles) > 60 else f"Target SMILES: {target_smiles}")

    from casp17.template_filter import filter_hits_with_ligands

    results = filter_hits_with_ligands(
        hits_tsv=args.hits_tsv,
        db_path=args.rcsb_db,
        target_smiles=target_smiles,
        output_path=args.output_tsv,
    )
    print(f"Filtered: {len(results)} hits with ligands → {args.output_tsv}")
    for h in results[:5]:
        print(f"  {h.pdb_id} pident={h.pident:.1f}% tanimoto={h.best_tanimoto:.3f} mcs={h.best_mcs_coverage:.3f} ligs={len(h.ligands)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
