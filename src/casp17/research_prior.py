"""Research- and template-informed pocket *anchors* for pose selection.

Deterministic, fail-open signals tell the submission ranker where the ligand
almost certainly belongs, so those sites are never absent from the 5 submitted
MODELs:

1. **Research catalytic pocket** — ``<run_dir>/research.json`` (vendored from the
   CASP17_own science-skills briefing) names the catalytic residues in target
   numbering. Grounding them on the submission receptor (same frame as the docked
   poses) gives a chemically anchored centre. Absent json / residues → skipped.

2. **Dominant template-consensus cluster** — when the union-template pockets pile
   up on one site (e.g. the kinase ATP pocket), ``template_pocket_clusters.json``
   has a top cluster that dwarfs the rest. Its centroid is a strong anchor even
   with no research briefing.

3. **Top / ligand template COM** — the bound-ligand centroid of the best-TM
   template ("top template") and of the most chemically-similar template ("ligand
   template"). The CASP meeting wants a pose reproducing the top template's
   binding mode guaranteed a MODEL even when its LSCORE is mediocre.

All return centres in the receptor/pose coordinate frame. Everything is
best-effort: any missing/broken input yields fewer (or no) anchors and never
raises, leaving pose selection unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Preferred functional atom per residue (falls back to CA). Nucleophile / charge-
# relay atoms first so the centre sits on the chemically relevant point.
_FUNCTIONAL_ATOM = {
    "SER": "OG", "THR": "OG1", "TYR": "OH", "CYS": "SG", "LYS": "NZ",
    "HIS": "NE2", "ASP": "OD1", "GLU": "OE1", "ASN": "ND2", "GLN": "NE2",
    "ARG": "NH1", "TRP": "NE1",
}
_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
_THREE_SET = set(_THREE.values())
_RES_RE = re.compile(r"^\s*([A-Za-z]{0,3})\s*(\d+)")

#: Free standard residues deposited as ligands are crystallisation / covalent-
#: modification remnants, not binding-site markers. They pass the CCD candidate
#: filter (a free CYS is ``peptide_like``, 7 heavy atoms), so a template whose
#: only "bound ligand" is one of these would otherwise anchor the ranker on a
#: surface residue. Observed on T2451, where 4LYK's CYS sat 15 Å from the real
#: cofactor site.
_MONOMER_LIGANDS = _THREE_SET | {
    "MSE", "SEC", "PYL", "UNK",
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "N",
}


@dataclass(frozen=True)
class Anchor:
    """A pocket centre the ranker should keep represented among the MODELs."""
    xyz: tuple[float, float, float]
    kind: str          # "research" | "template_consensus" | "top_template" | "ligand_template"
    weight: float      # relative strength (research > weak consensus)
    label: str


def _parse_residue(token: str) -> tuple[str | None, int] | None:
    m = _RES_RE.match(str(token))
    if not m:
        return None
    name_raw, num = m.group(1), int(m.group(2))
    up = name_raw.upper()
    if not up:
        return None, num
    if len(up) == 3 and up in _THREE_SET:
        return up, num
    if len(up) == 1 and up in _THREE:
        return _THREE[up], num
    if up[:3] in _THREE_SET:
        return up[:3], num
    return None, num


def _catalytic_residues(research_json: Path) -> list[tuple[str | None, int]]:
    try:
        data = json.loads(research_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    protein = data.get("protein") if isinstance(data, dict) else None
    toks = protein.get("catalytic_residues_target_numbering") if isinstance(protein, dict) else None
    if not isinstance(toks, list):
        return []
    out = []
    for t in toks:
        parsed = _parse_residue(t)
        if parsed is not None:
            out.append(parsed)
    return out


def research_centers(run_dir: Path, receptor_cif: Path) -> list[Anchor]:
    """Ground research catalytic residues on the receptor → one Anchor each."""
    rj = Path(run_dir) / "research.json"
    if not rj.is_file() or not Path(receptor_cif).is_file():
        return []
    residues = _catalytic_residues(rj)
    if not residues:
        return []
    try:
        import gemmi
    except ImportError:
        return []
    try:
        st = gemmi.read_structure(str(receptor_cif))
    except Exception:
        return []
    if not len(st):
        return []
    model = st[0]
    # One anchor per (chain, residue), NOT per residue. A homodimer carries the
    # catalytic set once per protomer, so stopping at the first chain leaves the
    # other protomer's site unanchored — a pose bound there then reads as
    # "covers nothing" and the ranker treats it as off-site (T2451: chains A and
    # B are both residues 1-255; 3 of 5 submitted MODELs bound chain B).
    polymers = [ch for ch in model
                # polymer chains only — a ligand/ion het residue can share a
                # seqid with a catalytic residue number and would otherwise be
                # grounded as one.
                if any(res.find_atom("CA", "*") is not None for res in ch)]
    multi = len(polymers) > 1
    out: list[Anchor] = []
    for chain in polymers:
        for name, num in residues:
            for res in chain:
                if res.seqid.num != num:
                    continue
                if name and res.name.upper() != name:
                    continue
                want = _FUNCTIONAL_ATOM.get(res.name.upper())
                atom = (res.find_atom(want, "*") if want else None) or res.find_atom("CA", "*")
                if atom is None and len(res):
                    atom = res[0]
                if atom is not None:
                    p = atom.pos
                    label = f"{name or ''}{num}" + (chain.name if multi else "")
                    out.append(Anchor((float(p.x), float(p.y), float(p.z)),
                                      "research", 1.0, label))
                    break
    return out


def dominant_cluster_center(
    run_dir: Path,
    *,
    min_share: float = 0.30,
    min_unique_pdb: int = 3,
) -> list[Anchor]:
    """Top template-consensus cluster centroid, if it dominates the pool.

    "Dominant" = its ``evidence_score`` is ≥ ``min_share`` of the total AND it
    is backed by ≥ ``min_unique_pdb`` distinct template PDBs (guards against a
    single-structure fluke). Weighted by its evidence share.
    """
    for rel in ("outputs/template_pockets/template_pocket_clusters.json",
                "template_pocket_clusters.json"):
        p = Path(run_dir) / rel
        if p.is_file():
            path = p
            break
    else:
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    clusters = data if isinstance(data, list) else data.get("clusters", [])
    if not clusters:
        return []
    total = sum(float(c.get("evidence_score", 0) or 0) for c in clusters)
    top = clusters[0]
    ev = float(top.get("evidence_score", 0) or 0)
    share = (ev / total) if total else 0.0
    centroid = top.get("centroid") or top.get("center")
    if (not centroid or len(centroid) != 3
            or share < min_share
            or int(top.get("n_unique_pdb", 0) or 0) < min_unique_pdb
            or not _cluster_supported(centroid, _pocket_points(run_dir))):
        return []
    return [Anchor(tuple(float(v) for v in centroid), "template_consensus",
                   round(share, 3), f"consensus/{share:.0%}")]


def dominant_site_mode(
    run_dir: Path,
    *,
    radius: float = 5.0,
    min_neighbours: int = 20,
    weight: float = 2.0,
) -> list[Anchor]:
    """Densest point of the raw template-pocket cloud — the *mode*, not the mean.

    ``dominant_cluster_center`` / ``template_site_centers`` report a cluster's
    running weighted **centroid**, which the greedy first-match clusterer lets
    drift: a 392-pocket cluster chains outward and its mean can land several Å
    off the actual pile-up, in shallower ground (T2451: centroid 4 Å from the
    mode, with 7 receptor heavy atoms within 5 Å against the mode's pile of 377
    pockets). Recomputing the mode straight from ``template_pockets.json``
    removes that drift and gives a point that is, by construction, where the
    most template ligands actually sit.

    Emitted only when the mode is backed by ``min_neighbours`` pockets, so a
    thin template set yields nothing rather than a noisy anchor. Fail-open.
    """
    for rel in ("outputs/template_pockets/template_pockets.json",
                "template_pockets.json"):
        p = Path(run_dir) / rel
        if p.is_file():
            path = p
            break
    else:
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    pockets = data if isinstance(data, list) else data.get("pockets")
    if not isinstance(pockets, list) or not pockets:
        return []
    pts = []
    for pk in pockets:
        c = (pk.get("centroid_x"), pk.get("centroid_y"), pk.get("centroid_z"))
        if not any(v is None for v in c):
            pts.append([float(v) for v in c])
    if len(pts) < min_neighbours:
        return []
    try:
        import numpy as np
    except ImportError:
        return []
    X = np.asarray(pts, dtype=float)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    near = d2 <= radius * radius
    counts = near.sum(1)
    i = int(np.argmax(counts))
    n = int(counts[i])
    if n < min_neighbours:
        return []
    # local mean of the mode's own neighbourhood: same pile-up, less quantised
    centre = X[near[i]].mean(axis=0)
    return [Anchor(tuple(float(v) for v in centre), "template_mode",
                   float(weight), f"mode/{n}pk")]


#: Radius (Å) within which a surviving pocket must sit for a consensus cluster
#: to count as real. Matches ``cluster_template_pockets.py --cutoff``, so a
#: cluster keeps its anchor as long as one of its own members survived.
_CLUSTER_SUPPORT_A = 5.0


def _load_pockets(run_dir: Path) -> list[dict]:
    """``template_pockets.json`` with mis-transformed protomer copies dropped.

    Until 2026-09-09 ``extract_template_pockets.py`` aligned one chain per hit
    row and moved *every* ligand in the entry with that transform, so copies
    bound to the other protomers landed wherever that chain was not. Those
    points are identifiable after the fact: a pocket placed by a chain-specific
    alignment has ``template_chain == ligand_chain``. On L01 half of every run's
    pockets fail that test and they formed two whole phantom clusters, one of
    which reached MODEL selection as a weight-1.20 anchor sitting in open
    solvent (1 receptor heavy atom within 8 Å against 59 for the real site).

    Filtering here rather than rewriting the run files keeps existing runs
    usable without re-running USalign, and is a no-op on runs produced by the
    fixed extractor. Fail-open: older files that carry neither chain field are
    returned unchanged.
    """
    for rel in ("outputs/template_pockets/template_pockets.json",
                "template_pockets.json"):
        path = Path(run_dir) / rel
        if path.is_file():
            break
    else:
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    pockets = data.get("pockets") if isinstance(data, dict) else data
    if not isinstance(pockets, list):
        return []

    keep, seen = [], set()
    for pk in pockets:
        tc = str(pk.get("template_chain") or "").strip().upper()
        lc = str(pk.get("ligand_chain") or "").strip().upper()
        if tc and lc and tc != lc:
            continue
        key = (pk.get("template_pdb_id"), tc, pk.get("ligand_ccd"),
               round(float(pk.get("centroid_x") or 0), 1),
               round(float(pk.get("centroid_y") or 0), 1),
               round(float(pk.get("centroid_z") or 0), 1))
        if key in seen:
            continue
        seen.add(key)
        keep.append(pk)
    # A file with no chain provenance at all would filter to nothing; keep it.
    return keep or pockets


def _pocket_points(run_dir: Path) -> list[tuple[float, float, float]]:
    out = []
    for pk in _load_pockets(run_dir):
        c = (pk.get("centroid_x"), pk.get("centroid_y"), pk.get("centroid_z"))
        if not any(v is None for v in c):
            out.append(tuple(float(v) for v in c))
    return out


def _cluster_supported(centroid, pts, radius: float = _CLUSTER_SUPPORT_A) -> bool:
    """True when a surviving pocket sits within ``radius`` of the centroid.

    A cluster built entirely from mis-transformed copies has none, which is what
    separates a phantom from a real site without re-clustering anything.
    """
    if not pts:
        return True  # no provenance to judge by — fail open
    r2 = radius * radius
    cx, cy, cz = (float(v) for v in centroid)
    return any((q[0] - cx) ** 2 + (q[1] - cy) ** 2 + (q[2] - cz) ** 2 <= r2
               for q in pts)


def template_com_centers(
    run_dir: Path,
    *,
    top_weight: float = 1.5,
    lig_weight: float = 1.0,
    min_lig_tanimoto: float = 0.3,
    min_corroboration: int = 3,
    corroboration_radius: float = 5.0,
) -> list[Anchor]:
    """Bound-ligand COM of the **top template** (best TM-score) and the **ligand
    template** (best Tanimoto), in the pose coordinate frame.

    These are the anchors the CASP meeting asked MODEL selection to follow so a
    pose that reproduces the top template's binding mode is guaranteed a MODEL
    even when its LSCORE is mediocre. Read from ``template_pockets.json`` (each
    pocket already carries a bound-ligand centroid in the aligned frame). The
    ligand-template anchor is emitted only when it sits at a *different* site
    than the top template and its ligand is genuinely similar
    (``best_tanimoto >= min_lig_tanimoto``).

    Two guards keep a single mis-annotated template from anchoring the ranker on
    a non-site, since these COMs come from ONE structure and have no averaging to
    fall back on:

    * pockets whose ligand is a free standard residue are ignored outright
      (``_MONOMER_LIGANDS``);
    * an emitted COM must be *corroborated* — at least ``min_corroboration``
      other template pockets within ``corroboration_radius`` Å. A COM that sits
      alone in the pocket cloud is an outlier, not a binding site.

    Fail-open: missing/broken file → []."""
    pockets = _load_pockets(run_dir)
    if not pockets:
        return []

    all_pts: list[tuple[float, float, float]] = []
    by_pdb: dict[str, dict] = {}
    for pk in pockets:
        c = (pk.get("centroid_x"), pk.get("centroid_y"), pk.get("centroid_z"))
        if any(v is None for v in c):
            continue
        all_pts.append(tuple(float(v) for v in c))
        if str(pk.get("ligand_ccd") or "").strip().upper() in _MONOMER_LIGANDS:
            continue
        pdb = pk.get("template_pdb_id") or ""
        g = by_pdb.setdefault(pdb, {"pts": [], "tm": None, "tan": None})
        g["pts"].append([float(v) for v in c])
        tm = pk.get("alignment_tmscore")
        if tm is None:
            tm = pk.get("qtmscore")
        if tm is not None and (g["tm"] is None or tm > g["tm"]):
            g["tm"] = float(tm)
        tan = pk.get("best_tanimoto")
        if tan is not None and (g["tan"] is None or tan > g["tan"]):
            g["tan"] = float(tan)
    by_pdb = {k: v for k, v in by_pdb.items() if v["pts"]}
    if not by_pdb:
        return []

    def com(pts):
        return tuple(sum(v[k] for v in pts) / len(pts) for k in range(3))

    r2 = corroboration_radius * corroboration_radius

    def corroborated(pt) -> bool:
        """≥ ``min_corroboration`` *other* template pockets within the radius."""
        if min_corroboration <= 0:
            return True
        n = 0
        for q in all_pts:
            d2 = sum((q[k] - pt[k]) ** 2 for k in range(3))
            if d2 <= r2:
                n += 1
                if n > min_corroboration:  # >: the anchor's own pocket counts once
                    return True
        return False

    out: list[Anchor] = []
    top_pdb, top_g = max(by_pdb.items(), key=lambda kv: (kv[1]["tm"] or 0.0))
    top_com = com(top_g["pts"])
    if corroborated(top_com):
        out.append(Anchor(top_com, "top_template", top_weight,
                          f"top/{top_pdb} TM{top_g['tm'] or 0:.2f}"))
    lig_pdb, lig_g = max(by_pdb.items(), key=lambda kv: (kv[1]["tan"] or 0.0))
    if lig_pdb != top_pdb and (lig_g["tan"] or 0.0) >= min_lig_tanimoto:
        lig_com = com(lig_g["pts"])
        if corroborated(lig_com):
            out.append(Anchor(lig_com, "ligand_template", lig_weight,
                              f"lig/{lig_pdb} sim{lig_g['tan']:.2f}"))
    return out


def template_site_centers(
    run_dir: Path,
    *,
    top_k: int = 2,
    min_share: float = 0.20,
    min_unique_pdb: int = 2,
) -> list[Anchor]:
    """Up to ``top_k`` *major* template-consensus binding sites.

    Where ``dominant_cluster_center`` returns only the single top cluster when it
    dominates, this returns every cluster whose evidence share ≥ ``min_share``
    and that is backed by ≥ ``min_unique_pdb`` distinct PDBs, up to ``top_k``. So
    a target whose templates pile up on TWO real sites (e.g. T2412: a catalytic
    site + a lower pocket) yields two anchors, and the ranker then splits the
    submitted MODELs across both sites (CASP meeting request). Single-dominant
    targets return one anchor exactly like before."""
    for rel in ("outputs/template_pockets/template_pocket_clusters.json",
                "template_pocket_clusters.json"):
        p = Path(run_dir) / rel
        if p.is_file():
            path = p
            break
    else:
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    clusters = data if isinstance(data, list) else data.get("clusters", [])
    if not clusters:
        return []
    total = sum(float(c.get("evidence_score", 0) or 0) for c in clusters) or 1.0
    pts = _pocket_points(run_dir)
    out: list[Anchor] = []
    for c in clusters:
        if len(out) >= top_k:
            break
        ev = float(c.get("evidence_score", 0) or 0)
        share = ev / total
        centroid = c.get("centroid") or c.get("center")
        if (not centroid or len(centroid) != 3
                or share < min_share
                or int(c.get("n_unique_pdb", 0) or 0) < min_unique_pdb
                or not _cluster_supported(centroid, pts)):
            continue
        # weight > research (1.0) so a major template site is covered/protected
        # before individual catalytic residues in the hard-coverage pass.
        out.append(Anchor(tuple(float(v) for v in centroid), "template_site",
                          round(1.0 + share, 3), f"site{len(out)+1}/{share:.0%}"))
    return out


def anchor_centers(
    run_dir: Path,
    receptor_cif: Path,
    *,
    multi_site: bool = True,
) -> list[Anchor]:
    """All anchors (research + top/ligand template + template sites), deduped by
    proximity. Template-COM anchors come before the generic site clusters so,
    when they land on the same site, the template-labelled anchor wins the dedup.

    ``multi_site`` (default on) uses ``template_site_centers`` (top-2 major
    sites) so 2-site targets split their MODELs; set False for the legacy
    single-dominant-cluster behaviour."""
    sites = (template_site_centers(run_dir) if multi_site
             else dominant_cluster_center(run_dir))
    anchors = (research_centers(run_dir, receptor_cif)
               + template_com_centers(run_dir)
               + sites)
    merged: list[Anchor] = []
    for a in anchors:
        if any((a.xyz[0]-b.xyz[0])**2 + (a.xyz[1]-b.xyz[1])**2
               + (a.xyz[2]-b.xyz[2])**2 < 9.0 for b in merged):  # <3 Å → same site
            continue
        merged.append(a)
    return merged
