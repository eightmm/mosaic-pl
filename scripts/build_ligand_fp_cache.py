#!/usr/bin/env python3
"""One-shot precompute of Morgan ECFP4 fingerprints for every RCSB CCD.

Reads ``ccd_code, smiles`` from ``rcsb_index.db.ccd_components`` (the
authoritative SMILES list — 49k entries as of 2026-05) and writes
``data/processed/ligand_fp_cache.db`` with one Morgan FP per CCD.

Idempotent: ``INSERT OR REPLACE`` so re-running just refreshes any rows
whose SMILES changed upstream.

Usage::

    .venv/bin/python scripts/build_ligand_fp_cache.py [--rcsb-db PATH] [--cache-db PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# Make casp17 module importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from casp17.ligand_fp_cache import (  # noqa: E402
    DEFAULT_CACHE_DB,
    DEFAULT_NBITS,
    DEFAULT_RADIUS,
    _connect,
)

DEFAULT_RCSB_DB = Path("/home/jaemin/DB/RCSB/processed/rcsb_index.db")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rcsb-db", type=Path, default=DEFAULT_RCSB_DB,
                   help="Source sqlite with ``ccd_components`` table.")
    p.add_argument("--cache-db", type=Path, default=DEFAULT_CACHE_DB,
                   help="Output cache sqlite path.")
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    p.add_argument("--nbits", type=int, default=DEFAULT_NBITS)
    args = p.parse_args()

    if not args.rcsb_db.exists():
        print(f"error: rcsb db not found at {args.rcsb_db}", file=sys.stderr)
        return 1

    from rdkit import Chem
    from rdkit.Chem import AllChem

    src = sqlite3.connect(str(args.rcsb_db))
    rows = src.execute(
        "SELECT ccd_code, smiles FROM ccd_components "
        "WHERE smiles IS NOT NULL AND smiles != ''"
    ).fetchall()
    src.close()
    print(f"[build_fp] {len(rows)} CCDs with SMILES from {args.rcsb_db}")

    dst = _connect(args.cache_db)
    n_ok = 0
    n_skip = 0
    n_existing = 0
    t0 = time.time()
    for i, (ccd, smi) in enumerate(rows, 1):
        ccd_u = ccd.upper()
        existing = dst.execute(
            "SELECT 1 FROM ligand_fps WHERE ccd_code=? AND fp_radius=? AND fp_nbits=?",
            (ccd_u, args.radius, args.nbits),
        ).fetchone()
        if existing:
            n_existing += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skip += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.nbits)
        dst.execute(
            "INSERT OR REPLACE INTO ligand_fps (ccd_code, fp_radius, fp_nbits, smiles, fp_blob) "
            "VALUES (?, ?, ?, ?, ?)",
            (ccd_u, args.radius, args.nbits, smi, fp.ToBinary()),
        )
        n_ok += 1
        if i % 5000 == 0:
            dst.commit()
            elapsed = time.time() - t0
            print(f"  [{i}/{len(rows)}] cached={n_ok} skipped={n_skip} existing={n_existing} "
                  f"elapsed={elapsed:.0f}s")
    dst.commit()
    elapsed = time.time() - t0
    size_mb = args.cache_db.stat().st_size / 1024 / 1024 if args.cache_db.exists() else 0
    print(f"[build_fp] done in {elapsed:.0f}s — wrote {n_ok}, skipped (bad SMILES) {n_skip}, "
          f"already-cached {n_existing}.  cache file: {args.cache_db} ({size_mb:.1f} MB)")
    dst.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
