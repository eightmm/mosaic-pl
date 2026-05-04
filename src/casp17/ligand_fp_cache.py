"""Cached Morgan ECFP4 fingerprints keyed by CCD code.

Computing 50k Morgan FPs from SMILES on every target run wastes 25-150 s
inside ``template_filter`` because the same RCSB ligands (NAD, FAD, COA,
common cofactors) reappear across thousands of hits. Pre-baking
``ccd_code → FP`` into a local sqlite (~12 MB for ECFP4 r=2 / 2048 bits)
lets ``filter_hits_with_ligands`` look up by primary key instead.

Cache miss path: compute on demand from SMILES and ``INSERT OR REPLACE``
so the next run hits the cache. Schema is forward-compatible — radius
and nBits live in their own columns so multiple FP variants can coexist.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Default location matches data/search_dbs sibling layout.
DEFAULT_CACHE_DB = Path("data/processed/ligand_fp_cache.db")
DEFAULT_RADIUS = 2
DEFAULT_NBITS = 2048


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ligand_fps (
            ccd_code TEXT NOT NULL,
            fp_radius INTEGER NOT NULL,
            fp_nbits INTEGER NOT NULL,
            smiles TEXT,
            fp_blob BLOB,
            PRIMARY KEY (ccd_code, fp_radius, fp_nbits)
        )
        """
    )
    return conn


class LigandFPCache:
    """Read-mostly cache that returns RDKit ``ExplicitBitVect`` for a CCD.

    Pass ``allow_compute=False`` for a strict read-only mode that returns
    ``None`` on miss instead of falling back to RDKit. Useful when the
    caller wants to surface "ligand has no fingerprint" rather than pay
    the on-demand SMILES → FP cost.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_CACHE_DB,
        radius: int = DEFAULT_RADIUS,
        nbits: int = DEFAULT_NBITS,
        allow_compute: bool = True,
    ):
        self._conn = _connect(Path(db_path))
        self._radius = radius
        self._nbits = nbits
        self._allow_compute = allow_compute
        self._mem: dict[str, object] = {}  # in-process cache for hot CCDs

    def get_fp(self, ccd_code: Optional[str], smiles: Optional[str] = None):
        """Return ``ExplicitBitVect`` for the ligand, or ``None`` on failure.

        Lookup precedence: in-process dict → sqlite cache → on-demand
        compute (if ``smiles`` given and ``allow_compute=True``). The
        in-process layer matters because a single ``filter_hits_with_ligands``
        call can ask for the same CCD across hundreds of hits.
        """
        if not ccd_code:
            return self._compute_from_smiles(smiles) if smiles else None
        ccd = ccd_code.upper()
        cached = self._mem.get(ccd)
        if cached is not None:
            return cached
        row = self._conn.execute(
            "SELECT fp_blob FROM ligand_fps WHERE ccd_code = ? AND fp_radius = ? AND fp_nbits = ?",
            (ccd, self._radius, self._nbits),
        ).fetchone()
        if row and row[0]:
            # Round-trip RDKit's binary serialization. Note: do NOT use
            # ``CreateFromBinaryText`` — that's a different format and
            # silently truncates to a wrong nBits (192 instead of 2048).
            from rdkit.DataStructs import ExplicitBitVect
            fp = ExplicitBitVect(bytes(row[0]))
            self._mem[ccd] = fp
            return fp
        if not self._allow_compute or not smiles:
            return None
        fp = self._compute_from_smiles(smiles)
        if fp is not None:
            self._store(ccd, smiles, fp)
            self._mem[ccd] = fp
        return fp

    def _compute_from_smiles(self, smiles: Optional[str]):
        if not smiles:
            return None
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, self._radius, nBits=self._nbits)
        except Exception:
            return None

    def _store(self, ccd_code: str, smiles: str, fp) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO ligand_fps (ccd_code, fp_radius, fp_nbits, smiles, fp_blob) "
                "VALUES (?, ?, ?, ?, ?)",
                (ccd_code, self._radius, self._nbits, smiles, fp.ToBinary()),
            )
            self._conn.commit()
        except sqlite3.Error:
            pass  # cache update failure must never break the caller

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def tanimoto(fp_a, fp_b) -> float:
    """Tanimoto similarity between two ExplicitBitVects, ``0.0`` on bad input."""
    if fp_a is None or fp_b is None:
        return 0.0
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
