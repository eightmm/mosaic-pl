"""Shared geometry utilities for pose evaluation scripts.

Consolidates the Kabsch alignment, PDB CA parsing, ligand coordinate
transformation, bond-order reassignment, MDL block parsing, and
symmetry-aware pose RMSD implementations that previously lived as
near-duplicates across five separate evaluation scripts:

    experiments/casp16_test/L1000/evaluate.py
    experiments/casp16_test/L1000/best_pose.py
    experiments/casp16_test/L1000/compare_selection.py
    experiments/casp16_test/L1000/compare_cofold_metrics.py
    experiments/novel2025_test/evaluate.py

Import site:
    from casp17.geometry import (
        parse_ca,
        kabsch,
        transform_mol,
        reassign_bonds,
        mol_from_mdl_body,
        pose_rmsd,
    )
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

# Three-letter → one-letter amino-acid mapping used when extracting
# crystal sequences from mmCIF/PDB residue lists. MSE (selenomethionine)
# is treated as methionine because most crystals use it as a Met
# isomorph and downstream tooling doesn't distinguish the two.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


# ---------- PDB / CA parsing ----------

def parse_ca(
    source: str | Iterable[str] | Path,
    chain: str | None = "A",
) -> dict[int, np.ndarray]:
    """Return ``{resid: xyz}`` for the CA atoms of a given chain.

    ``source`` may be a PDB file path, a multi-line string, or an iterable
    of lines (as produced by parsing an LG MODEL block). Only ``ATOM``
    records with atom name ``CA`` are considered. ``chain=None`` disables
    the chain filter.
    """
    if isinstance(source, Path):
        lines: Iterable[str] = source.read_text().splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = source

    out: dict[int, np.ndarray] = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        if chain is not None and line[21] != chain:
            continue
        try:
            resid = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        out[resid] = np.array([x, y, z], dtype=np.float64)
    return out


# ---------- Kabsch rigid-body alignment ----------

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the rigid-body transform that aligns ``P`` onto ``Q``.

    Returns ``(R, t, rmsd)`` where ``R`` is a 3×3 rotation matrix, ``t``
    is a 3-vector translation, and ``rmsd`` is the residual RMSD after
    applying ``q = R @ p + t``. ``P`` and ``Q`` must be Nx3 arrays of the
    same length (typically corresponding CA coordinates).
    """
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc
    aligned = P @ R.T + t
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum() / len(P)))
    return R, t, rmsd


# ---------- Ligand coordinate transform ----------

def transform_mol(mol: Chem.Mol, R: np.ndarray, t: np.ndarray, *, inplace: bool = True) -> Chem.Mol:
    """Apply ``R @ xyz + t`` to every atom of an RDKit Mol.

    When ``inplace=True`` the input Mol is mutated and returned; otherwise
    a deep copy is modified. Returns the transformed molecule.
    """
    out = mol if inplace else Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = np.array(conf.GetAtomPosition(i))
        q = R @ p + t
        conf.SetAtomPosition(i, q.tolist())
    return out


# ---------- Bond-order reassignment via SMILES template ----------

def reassign_bonds(mol: Chem.Mol | None, template: Chem.Mol) -> Chem.Mol | None:
    """Use ``template`` (built from SMILES) to assign correct bond orders
    onto a ligand parsed from PDB/MDL (where bond orders are unreliable).

    Returns a new Mol with template bond orders, or ``None`` if atom
    counts disagree or RDKit's subgraph match fails. The caller is
    responsible for stripping hydrogens before passing mol in.
    """
    if mol is None:
        return None
    if mol.GetNumAtoms() != template.GetNumAtoms():
        return None
    try:
        return AssignBondOrdersFromTemplate(template, mol)
    except Exception:
        return None


# ---------- MDL block parsing (for LG MODEL ligand extraction) ----------

def mol_from_mdl_body(body: str) -> Chem.Mol | None:
    """Parse an MDL/SDF body into an RDKit Mol.

    The LG format stores the ligand MDL block *without* the 3-line header
    (the ligand name sits in a separate ``LIGAND`` line). This helper
    prepends a synthetic ``lig\\n`` name line and removes explicit
    hydrogens so that downstream bond reassignment gets a heavy-atom-only
    graph that matches the SMILES template's atom count.

    ``sanitize=False`` is required because several writers (pxdock in
    particular) emit atoms with out-of-range valence that only become
    valid after ``reassign_bonds`` runs.
    """
    if not body.strip():
        return None
    mol = Chem.MolFromMolBlock("lig\n" + body, removeHs=False, sanitize=False)
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        return mol


# ---------- Symmetry-aware pose RMSD (no re-superposition) ----------

def pose_rmsd(pred: Chem.Mol, ref: Chem.Mol, *, max_matches: int = 1000) -> float:
    """Symmetry-aware heavy-atom RMSD at the current coordinates — **no**
    rigid-body re-alignment of the ligand.

    Uses ``rdkit.Chem.rdMolAlign.CalcRMS`` (symmetry-aware, no alignment).

    Why CalcRMS and not GetBestRMS:
      * ``GetBestRMS`` performs an internal Kabsch alignment of the ligand
        before computing RMSD, returning a conformational distance rather
        than a pocket-placement error — useless for CASP-style evaluation
        where the ligand must already be in the correct pocket frame.
      * The previous hand-rolled enumeration via
        ``GetSubstructMatches(uniquify=False)`` was correct but combinatorial
        in symmetric ligands; on the 499-target batch one such target hung
        score_per_metric.py for >1 h because the SIGALRM 180-s timeout
        cannot fire inside RDKit's C-extension loop. ``CalcRMS`` caps the
        permutation search internally.

    NEVER substitute ``rdMolAlign.GetBestRMS`` here. See
    ``docs/per_pose_rmsd_method.md`` for the rule.

    Returns ``nan`` when atom mapping fails (typically atom-count mismatch
    or sanitization failure — the caller should fall back to MCS RMSD).
    """
    from rdkit.Chem import rdMolAlign  # type: ignore
    pred_h = Chem.RemoveHs(pred)
    ref_h = Chem.RemoveHs(ref)
    try:
        return float(rdMolAlign.CalcRMS(pred_h, ref_h))
    except Exception:
        # Try the reverse direction (probe/ref swap) for edge cases where
        # only one ordering admits a substructure match.
        try:
            return float(rdMolAlign.CalcRMS(ref_h, pred_h))
        except Exception:
            return math.nan
