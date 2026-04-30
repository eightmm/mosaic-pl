"""Shared CCD / residue sets for receptor preparation steps.

Centralised so the docking-prep stage and the Track 2 template prep stay
in lock-step. Adding a residue here propagates to both call sites; without
this single source of truth the two locations were drifting (the Track 2
copy already gained ``T`` / ``DI`` / ``I`` ahead of the Stage 3 copy).
"""

from __future__ import annotations

# Standard RNA / DNA residue names that appear in mmCIF / PDB ``label_comp_id``.
# When a receptor PDB contains any of these we route through obabel for
# protonation + PDBQT instead of pdb2pqr — pdb2pqr's AMBER FF doesn't
# parameterize nucleotides reliably and either fails outright or strips
# the entire nucleic chain.
NUCLEIC_RESIDUES: frozenset[str] = frozenset({
    # RNA standard
    "A", "U", "G", "C",
    # RNA explicit prefix (rare in cofold cifs but seen in some PDBs)
    "RA", "RU", "RG", "RC",
    # DNA standard
    "DA", "DT", "DG", "DC",
    # DNA legacy single-letter (T = thymine)
    "T",
    # Modified backbones
    "DI", "I",
})


__all__ = ["NUCLEIC_RESIDUES"]
