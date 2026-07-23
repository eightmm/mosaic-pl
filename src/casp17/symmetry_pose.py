"""Pseudo-symmetry pose expansion.

Some ligands are (pseudo-)symmetric — e.g. a di-carbamate (two identical arms on
a ring) where either arm can face the catalytic residue, giving two near-
degenerate binding modes related by a ~180° flip. Docking / pose selection often
captures only one. This module derives the ligand's internal rigid symmetry
operation(s) from a *docked pose's own conformer* (so they are already in the
receptor frame) and applies them to generate the flipped pose in place — a
cheap, deterministic hedge that keeps both orientations available for the 5
submitted MODELs.

Deterministic (RDKit graph automorphism + Kabsch), fail-open: a non-symmetric
ligand yields no ops and nothing changes.
"""
from __future__ import annotations


def _kabsch(P, Q):
    """Rigid transform (R, t) mapping P onto Q (both N×3). ``Q ≈ R·P + t``."""
    import numpy as np
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def pose_symmetry_ops(mol, rmsd_thresh: float = 1.2, max_ops: int = 2):
    """Rigid ops (R, t, rmsd) that map the pose's conformer onto itself via a
    non-trivial graph automorphism within ``rmsd_thresh`` Å — the ligand's
    pseudo-symmetry flips, expressed in the pose's (receptor) frame.

    Returns [] for a rigid/asymmetric ligand or missing conformer. Ops are
    de-duplicated and sorted by fit RMSD (tightest symmetry first)."""
    try:
        import numpy as np
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return []
    if mol is None or mol.GetNumConformers() == 0:
        return []
    n = mol.GetNumAtoms()
    if n < 4:
        return []
    conf = mol.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                   conf.GetAtomPosition(i).z] for i in range(n)])
    try:
        matches = mol.GetSubstructMatches(mol, uniquify=False, maxMatches=2000)
    except Exception:
        return []
    ident = tuple(range(n))
    ops = []
    for perm in matches:
        if perm == ident or len(perm) != n:
            continue
        Q = P[list(perm)]
        R, t = _kabsch(P, Q)
        # skip near-identity rotations (trace≈3 → no real flip)
        if (np.trace(R) - 3.0) > -0.05:
            continue
        rmsd = float(np.sqrt(((R @ P.T).T + t - Q) ** 2).sum(1).mean())
        if rmsd <= rmsd_thresh:
            ops.append((R, t, rmsd))
    ops.sort(key=lambda o: o[2])
    # de-dup ops whose rotation axes/angles nearly coincide
    kept = []
    for R, t, rmsd in ops:
        if any(np.allclose(R, R2, atol=0.1) for R2, _, _ in kept):
            continue
        kept.append((R, t, rmsd))
        if len(kept) >= max_ops:
            break
    return kept


# hydrolase scissile groups — atoms ordered so the first 4 define orientation.
_SCISSILE_SMARTS = (
    ("carbamate", "[NX3][CX3](=O)[OX2H0]"),
    ("ester",     "[#6][CX3](=O)[OX2H0][#6]"),
    ("amide",     "[NX3][CX3](=O)[#6]"),
)


def functional_flip_ops(mol, anchor_pts, core_rmsd_max: float = 1.0, max_ops: int = 2):
    """Rigid ops (R, t, rmsd) that swap *which equivalent scissile group* faces
    the catalytic anchor, for ligands with ≥2 chemically-equivalent hydrolase
    groups (e.g. a di-carbamate) that a graph automorphism misses because a
    distal substituent breaks strict symmetry.

    The group whose carbonyl-C is nearest an ``anchor`` is the "engaged" one;
    each other group of the same type is superimposed (Kabsch on the 4 core
    atoms, in the pose frame) onto the engaged group → a transform that presents
    the alternate group at the catalytic site. Fail-open: [] when <2 equivalent
    groups or no conformer/anchor."""
    try:
        import numpy as np
        from rdkit import Chem
    except ImportError:
        return []
    if mol is None or mol.GetNumConformers() == 0 or not anchor_pts:
        return []
    conf = mol.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                   conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    apts = [np.asarray(a, dtype=float) for a in anchor_pts]
    ops = []
    for _name, smarts in _SCISSILE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if len(matches) < 2:
            continue
        core = [m[:4] for m in matches if len(m) >= 4]
        if len(core) < 2:
            continue
        # engaged group = carbonyl C (match[1]) nearest any anchor
        def cdist(m):
            c = P[m[1]]
            return min(float(np.linalg.norm(c - ap)) for ap in apts)
        eng = min(core, key=cdist)
        for g in core:
            if g == eng:
                continue
            R, t = _kabsch(P[list(g)], P[list(eng)])
            rmsd = float(np.sqrt(((R @ P[list(g)].T).T + t - P[list(eng)]) ** 2).sum(1).mean())
            if rmsd <= core_rmsd_max:
                ops.append((R, t, rmsd))
        if ops:
            break  # one scissile type is enough
    ops.sort(key=lambda o: o[2])
    return ops[:max_ops]


def engaged_scissile_group(mol, anchor_pts):
    """Which equivalent hydrolase group faces the catalytic anchor.

    Returns ``(group_key, n_groups)`` where ``group_key`` identifies the
    scissile group whose carbonyl-C is nearest an anchor (a stable tuple of the
    group's core atom indices), and ``n_groups`` is how many equivalent groups
    the ligand has. ``(None, 0)`` when no scissile group / conformer / anchor.

    Two docked poses of the same ligand with *different* ``group_key`` present
    opposite binding orientations at the site — one can be submitted as the
    alternate-orientation hedge without any synthetic transform."""
    try:
        import numpy as np
        from rdkit import Chem
    except ImportError:
        return None, 0
    if mol is None or mol.GetNumConformers() == 0 or not anchor_pts:
        return None, 0
    conf = mol.GetConformer()
    apts = [np.asarray(a, dtype=float) for a in anchor_pts]
    for _name, smarts in _SCISSILE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = [m for m in mol.GetSubstructMatches(patt) if len(m) >= 4]
        if len(matches) < 2:
            continue

        def cdist(m):
            p = conf.GetAtomPosition(m[1])  # carbonyl C
            c = np.array([p.x, p.y, p.z])
            return min(float(np.linalg.norm(c - ap)) for ap in apts)

        eng = min(matches, key=cdist)
        return tuple(sorted(eng[:4])), len(matches)
    return None, 0


def write_flipped_sdf(mol, R, t, out_path, title: str) -> bool:
    """Apply (R, t) to ``mol``'s conformer and write the flipped pose to
    ``out_path`` as SDF (connectivity preserved). Returns True on success."""
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except ImportError:
        return False
    try:
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            v = R @ np.array([p.x, p.y, p.z]) + t
            conf.SetAtomPosition(i, Point3D(float(v[0]), float(v[1]), float(v[2])))
        m.SetProp("_Name", title)
        w = Chem.SDWriter(str(out_path))
        w.write(m)
        w.close()
        return True
    except Exception:
        return False
