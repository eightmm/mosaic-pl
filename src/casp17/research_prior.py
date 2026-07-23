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
    out: list[Anchor] = []
    for name, num in residues:
        pos = None
        for chain in model:
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
                    pos = (float(p.x), float(p.y), float(p.z))
                    break
            if pos is not None:
                break
        if pos is not None:
            out.append(Anchor(pos, "research", 1.0, f"{name or ''}{num}"))
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
            or int(top.get("n_unique_pdb", 0) or 0) < min_unique_pdb):
        return []
    return [Anchor(tuple(float(v) for v in centroid), "template_consensus",
                   round(share, 3), f"consensus/{share:.0%}")]


def template_com_centers(
    run_dir: Path,
    *,
    top_weight: float = 1.5,
    lig_weight: float = 1.0,
    min_lig_tanimoto: float = 0.3,
) -> list[Anchor]:
    """Bound-ligand COM of the **top template** (best TM-score) and the **ligand
    template** (best Tanimoto), in the pose coordinate frame.

    These are the anchors the CASP meeting asked MODEL selection to follow so a
    pose that reproduces the top template's binding mode is guaranteed a MODEL
    even when its LSCORE is mediocre. Read from ``template_pockets.json`` (each
    pocket already carries a bound-ligand centroid in the aligned frame). The
    ligand-template anchor is emitted only when it sits at a *different* site
    than the top template and its ligand is genuinely similar
    (``best_tanimoto >= min_lig_tanimoto``). Fail-open: missing/broken file → []."""
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
    pockets = data.get("pockets") if isinstance(data, dict) else None
    if not isinstance(pockets, list) or not pockets:
        return []

    by_pdb: dict[str, dict] = {}
    for pk in pockets:
        c = (pk.get("centroid_x"), pk.get("centroid_y"), pk.get("centroid_z"))
        if any(v is None for v in c):
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

    out: list[Anchor] = []
    top_pdb, top_g = max(by_pdb.items(), key=lambda kv: (kv[1]["tm"] or 0.0))
    out.append(Anchor(com(top_g["pts"]), "top_template", top_weight,
                      f"top/{top_pdb} TM{top_g['tm'] or 0:.2f}"))
    lig_pdb, lig_g = max(by_pdb.items(), key=lambda kv: (kv[1]["tan"] or 0.0))
    if lig_pdb != top_pdb and (lig_g["tan"] or 0.0) >= min_lig_tanimoto:
        out.append(Anchor(com(lig_g["pts"]), "ligand_template", lig_weight,
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
    out: list[Anchor] = []
    for c in clusters:
        if len(out) >= top_k:
            break
        ev = float(c.get("evidence_score", 0) or 0)
        share = ev / total
        centroid = c.get("centroid") or c.get("center")
        if (not centroid or len(centroid) != 3
                or share < min_share
                or int(c.get("n_unique_pdb", 0) or 0) < min_unique_pdb):
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
