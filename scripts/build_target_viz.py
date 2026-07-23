#!/usr/bin/env python3
"""Build self-contained HTML 3D viewers for CASP17 LG submissions.

Scans a directory tree for finished submissions (``*_LCDD.lg``) and emits,
per target, an offline HTML page that shows — in an interactive 3Dmol.js
viewer — every submitted MODEL as an independent snapshot (its own receptor +
ligand pose), every binding-site prediction (translucent spheres, toggleable
by source), and the per-MODEL scores (LSCORE, AFFNTY, predicted RMSD / pose
reliability / pKd when post-analysis exists).

Why per-MODEL receptors: RNA cofolding produces a *different* fold per MODEL,
so MODEL 1's receptor is not representative. Protein-ligand MODELs share one
receptor (docking only moves the ligand) — identical receptor blocks are
de-duplicated before embedding so the HTML stays small.

Sources (all already produced by the pipeline, nothing re-computed):
  * receptor + ligand poses : parsed straight out of the ``*_LCDD.lg`` file.
  * binding sites           : ``docking_prep_summary.json`` under the target's
    run tree. RNA / docking-off targets simply have none.
  * per-MODEL metrics       : ``outputs/analysis/{ba,rmsd}_pred_*.tsv`` keyed by
    the pose ``Name`` (== the MDL title line). Absent for RNA targets.

3Dmol.js is vendored at ``<output-dir>/assets/3Dmol-min.js`` so the pages work
fully offline (no CDN, no network).

Usage:
    uv run python scripts/build_target_viz.py --output-dir viz
    # default scan = experiments/CASP17 (lg files live in CASP17/submissions
    # and CASP17/{target}/submissions). Pass --scan-root experiments for all.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

ASSET_REL = "assets/3Dmol-min.js"
ASSET_URL = "https://3Dmol.org/build/3Dmol-min.js"

# Per-MODEL colours (MODEL 1..5).
MODEL_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
# Binding-site source family -> colour.
BSITE_FAMILY_COLORS = {
    "cofolding": "#2ca02c",
    "swinsite": "#ff7f0e",
    "p2rank": "#1f77b4",
    "template_consensus": "#9467bd",
}


def _family(source: str) -> str:
    """'swinsite_2' -> 'swinsite', 'template_consensus_10' -> 'template_consensus'."""
    return re.sub(r"_\d+$", "", source)


# --------------------------------------------------------------------------- #
# LG parsing — per-MODEL receptor + ligand poses
# --------------------------------------------------------------------------- #
def parse_lg(lg_path: Path) -> dict:
    lines = lg_path.read_text().splitlines()
    target = ""
    models: list[dict] = []
    cur_model: dict | None = None
    cur_receptor: list[str] = []
    in_ligand = False
    lig_name = ""
    lig_lscore: float | None = None
    mdl_lines: list[str] = []
    capturing_mdl = False

    def flush_ligand():
        nonlocal in_ligand, mdl_lines, capturing_mdl, lig_lscore, lig_name
        if cur_model is not None and mdl_lines:
            title = mdl_lines[0].strip() if mdl_lines else ""
            cur_model["ligands"].append({
                "name": lig_name or "LIG",
                "lscore": lig_lscore,
                "title": title,
                "molblock": "\n".join(mdl_lines),
            })
        in_ligand = False
        capturing_mdl = False
        mdl_lines = []
        lig_lscore = None
        lig_name = ""

    def stash_receptor():
        if cur_model is not None and cur_model.get("receptor_pdb") is None and cur_receptor:
            cur_model["receptor_pdb"] = "\n".join(cur_receptor) + "\nEND\n"

    for ln in lines:
        if ln.startswith("TARGET"):
            parts = ln.split(None, 1)
            target = parts[1].strip() if len(parts) > 1 else ""
            continue
        if ln.startswith("MODEL"):
            flush_ligand()
            cur_model = {"model": int(ln.split()[1]) if len(ln.split()) > 1 else len(models) + 1,
                         "affinity": None, "receptor_pdb": None, "ligands": []}
            models.append(cur_model)
            cur_receptor = []
            continue
        if cur_model is None:
            continue
        if ln.startswith(("ATOM", "HETATM", "TER")) and not in_ligand:
            cur_receptor.append(ln)
            continue
        if ln.startswith("LIGAND"):
            flush_ligand()
            in_ligand = True
            parts = ln.split()
            lig_name = parts[2] if len(parts) > 2 else "LIG"
            stash_receptor()
            # Begin MDL capture immediately: LSCORE is OPTIONAL per the LG spec,
            # so a ligand block may go straight from `LIGAND` to the MDL body
            # (no LSCORE line). Gating capture on LSCORE alone silently dropped
            # such ligands (e.g. M2415 MODEL 1 duplex pose). An LSCORE line, if
            # present, is intercepted below and resets mdl_lines before the body.
            capturing_mdl = True
            mdl_lines = []
            continue
        if in_ligand:
            if ln.startswith("LSCORE"):
                try:
                    lig_lscore = float(ln.split()[1])
                except (IndexError, ValueError):
                    lig_lscore = None
                capturing_mdl = True
                mdl_lines = []
                continue
            if ln.startswith("AFFNTY"):
                flush_ligand()
                parts = ln.split(None, 1)
                cur_model["affinity"] = parts[1].strip() if len(parts) > 1 else None
                continue
            if capturing_mdl:
                mdl_lines.append(ln)
                if ln.strip() == "M  END":
                    capturing_mdl = False
                continue
            if ln.startswith("END"):
                flush_ligand()
    flush_ligand()
    return {"target": target or lg_path.stem.replace("_LCDD", ""), "models": models}


# --------------------------------------------------------------------------- #
# Locate target run roots (bounded — never rglob a benchmark mega-tree)
# --------------------------------------------------------------------------- #
def _target_run_roots(lg_path: Path, target: str, scan_roots: list[Path]) -> list[Path]:
    roots: list[Path] = []
    if lg_path.parent.parent.name == target:
        roots.append(lg_path.parent.parent)
    bases = set()
    for sr in scan_roots:
        bases.add(sr)
        bases.add(sr / "CASP17")
        # tolerate LG/run living in sibling trees (e.g. the central
        # CASP17/submissions/ lg vs the actual run under CASP17/{target})
        bases.add(sr.parent)
        bases.add(sr.parent / "CASP17")
    for base in bases:
        for cand in (base / target, base / target / "run" / target):
            if cand.exists() and cand.is_dir():
                roots.append(cand)
    seen, uniq = set(), []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(r)
    return uniq


# --------------------------------------------------------------------------- #
# Binding sites
# --------------------------------------------------------------------------- #
def find_binding_sites(roots: list[Path], target: str) -> dict:
    candidates: list[Path] = []
    for root in roots:
        candidates += list(root.rglob("docking/docking_prep_summary.json"))
        candidates += list(root.rglob("docking_prep_summary.json"))

    def score(p: Path) -> tuple:
        s = str(p)
        return (f"/{target}/" in s, "inputs/docking" in s)

    for prep in sorted(set(candidates), key=score, reverse=True):
        try:
            d = json.loads(prep.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        bs = d.get("binding_site_predictions")
        if not isinstance(bs, dict) or not bs:
            continue
        sites = []
        for src, v in bs.items():
            c = v.get("center") if isinstance(v, dict) else None
            if isinstance(c, (list, tuple)) and len(c) == 3:
                sites.append({"source": src, "family": _family(src),
                              "center": [float(x) for x in c]})
        box = d.get("box_size") or [22.5, 22.5, 22.5]
        if isinstance(box, (int, float)):
            box = [float(box)] * 3
        return {"sites": sites, "box_size": [float(x) for x in box[:3]],
                "prep_path": str(prep)}
    return {"sites": [], "box_size": [22.5, 22.5, 22.5], "prep_path": None}


# --------------------------------------------------------------------------- #
# Template pockets — every aligned template's bound-ligand centroid, already in
# the cofold frame the viewer uses (from extract_template_pockets.py).
# --------------------------------------------------------------------------- #
def find_template_pockets(roots: list[Path], target: str) -> dict:
    candidates: list[Path] = []
    for root in roots:
        candidates += list(root.rglob("template_pockets/template_pockets.json"))
        candidates += list(root.rglob("template_pockets.json"))

    def score(p: Path) -> tuple:
        return (f"/{target}/" in str(p),)

    for tp in sorted(set(candidates), key=score, reverse=True):
        try:
            d = json.loads(tp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        raw = d.get("pockets")
        if not isinstance(raw, list) or not raw:
            continue
        pockets = []
        cl0 = []  # consensus (cluster 0) centroids → crop focus
        for p in raw:
            c = (p.get("centroid_x"), p.get("centroid_y"), p.get("centroid_z"))
            if any(v is None for v in c):
                continue
            xyz = [float(v) for v in c]
            if p.get("cluster_index") == 0:
                cl0.append(xyz)
            pockets.append({
                "pdb": p.get("template_pdb_id", ""),
                "ccd": p.get("ligand_ccd", ""),
                "xyz": xyz,
                "tm": p.get("alignment_tmscore"),
                "qtm": p.get("qtmscore"),
                "tan": p.get("best_tanimoto"),
                "mcs": p.get("best_mcs_coverage"),
                "pid": p.get("pident"),
                "ms": bool(p.get("in_mmseqs")),
                "fs": bool(p.get("in_foldseek")),
            })
        focus = ([sum(v) / len(cl0) for v in zip(*cl0)] if cl0
                 else ([sum(v) / len(pockets) for v in zip(*[q["xyz"] for q in pockets])]
                       if pockets else None))
        # per-PDB summary rows (sorted by best TM desc), each keeps its pocket idxs
        by_pdb: dict[str, dict] = {}
        for i, p in enumerate(pockets):
            g = by_pdb.setdefault(p["pdb"], {
                "pdb": p["pdb"], "idxs": [], "ccds": set(),
                "best_tm": None, "best_tan": None, "best_mcs": None, "best_pid": None,
                "ms": False, "fs": False})
            g["idxs"].append(i)
            if p["ccd"]:
                g["ccds"].add(p["ccd"])
            for k, src in (("best_tm", "tm"), ("best_tan", "tan"), ("best_mcs", "mcs"), ("best_pid", "pid")):
                v = p[src]
                if v is not None and (g[k] is None or v > g[k]):
                    g[k] = v
            g["ms"] = g["ms"] or p["ms"]
            g["fs"] = g["fs"] or p["fs"]
        rows = sorted(by_pdb.values(),
                      key=lambda g: (g["best_tm"] is not None, g["best_tm"] or 0),
                      reverse=True)
        for g in rows:
            g["ccds"] = sorted(g["ccds"])
            g["n"] = len(g["idxs"])
        return {"pockets": pockets, "by_pdb": rows,
                "n_pdb": len(rows), "path": str(tp), "focus": focus,
                "reference_cif": d.get("reference_cif"),
                "work_dir": str(tp.parent / "_extract_work")}
    return {"pockets": [], "by_pdb": [], "n_pdb": 0, "path": None,
            "focus": None, "reference_cif": None, "work_dir": None}


def _dist3(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def template_anchors(templates: dict) -> list[dict]:
    """Center-of-mass anchors the meeting asked pose selection to respect.

    Poses that land near a template's bound-ligand COM should surface as
    candidates even when their LSCORE is mediocre, so the board ranks/marks
    proximity to these anchors alongside LSCORE. Anchors (all already in the
    cofold frame the viewer uses):

      * ``top_template``    — bound-ligand COM of the #1 hit by TM-score (the
        "top template" fold the meeting wants poses to follow).
      * ``ligand_template`` — bound-ligand COM of the hit whose ligand is most
        chemically similar (max Tanimoto) — the "ligand template". Omitted when
        it coincides with the top template.
      * ``consensus``       — the cluster-0 consensus centroid (``focus``).
    """
    pockets = templates.get("pockets") or []
    by_pdb = templates.get("by_pdb") or []

    def com(idxs):
        pts = [pockets[i]["xyz"] for i in idxs if 0 <= i < len(pockets)]
        return [sum(v) / len(pts) for v in zip(*pts)] if pts else None

    anchors: list[dict] = []
    top_pdb = None
    if by_pdb:
        top = by_pdb[0]  # rows are TM-sorted desc
        c = com(top["idxs"])
        if c:
            top_pdb = top["pdb"]
            anchors.append({"kind": "top_template", "pdb": top["pdb"], "xyz": c,
                            "tm": top.get("best_tm"), "tan": top.get("best_tan")})
    lig = max(by_pdb, key=lambda g: (g.get("best_tan") or 0), default=None)
    if lig and (lig.get("best_tan") or 0) > 0 and lig["pdb"] != top_pdb:
        c = com(lig["idxs"])
        if c:
            anchors.append({"kind": "ligand_template", "pdb": lig["pdb"], "xyz": c,
                            "tm": lig.get("best_tm"), "tan": lig.get("best_tan")})
    focus = templates.get("focus")
    if focus:
        anchors.append({"kind": "consensus", "pdb": None,
                        "xyz": [float(v) for v in focus], "tm": None, "tan": None})
    return anchors


def _nearest_anchor(centroid, anchors: list[dict]) -> dict | None:
    """Closest template anchor to a pose/ligand centroid: {kind, pdb, dist}."""
    if not centroid or not anchors:
        return None
    best, bd = None, 1e18
    for a in anchors:
        d = _dist3(centroid, a["xyz"])
        if d < bd:
            bd, best = d, a
    return {"kind": best["kind"], "pdb": best.get("pdb"),
            "dist": round(bd, 2)} if best else None


# colours for embedded template structures (cartoon), by rank
TPL_STRUCT_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                     "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]

_BACKBONE = {"N", "CA", "C", "O"}
_WATER = {"HOH", "WAT", "DOD"}


def _slim_pdb(pdb_str: str, center=None, radius: float = 22.0,
              chain_radius: float = 12.0, ca_only: bool = False) -> str:
    """Shrink an embedded template PDB so the offline HTML stays small while the
    cartoon stays a *whole* fold (no mid-chain truncation):
      * polymer → backbone atoms only (enough for a 3Dmol cartoon),
      * drop water,
      * keep only the chains that touch the pocket — a chain is kept in FULL if
        any of its atoms is within ``chain_radius`` Å of ``center`` — so ribbons
        don't get chopped, unlike a raw distance crop,
      * hetero (crystal ligands / ions) within ``radius`` Å of the pocket kept.
    Falls back to keeping everything when ``center`` is None.
    """
    def dist2(ln: str):
        try:
            dx = float(ln[30:38]) - center[0]
            dy = float(ln[38:46]) - center[1]
            dz = float(ln[46:54]) - center[2]
        except (ValueError, TypeError):
            return None
        return dx*dx + dy*dy + dz*dz

    lines = pdb_str.splitlines()
    keep_chains = None
    if center is not None:
        cr2 = chain_radius * chain_radius
        keep_chains = set()
        for ln in lines:
            if ln[:6].strip() == "ATOM":
                d = dist2(ln)
                if d is not None and d <= cr2:
                    keep_chains.add(ln[21])
    r2 = radius * radius

    keep = []
    for ln in lines:
        rec = ln[:6].strip()
        if rec == "ATOM":
            if ln[17:20].strip() in _WATER:
                continue
            allowed = {"CA"} if ca_only else _BACKBONE
            if ln[12:16].strip() not in allowed:
                continue
            if keep_chains is not None and ln[21] not in keep_chains:
                continue
        elif rec == "HETATM":
            if ln[17:20].strip() in _WATER:
                continue
            if center is not None:
                d = dist2(ln)
                if d is None or d > r2:
                    continue
        keep.append(ln)
    return "\n".join(keep) + "\n"


def load_template_structures(templates: dict, top_n: int, seq_n: int = 3,
                             lig_cap: int = 60, cartoon_cap: int = 12,
                             timeout: int = 90) -> dict:
    """Align template host chains onto the cofold reference (same frame as the
    pocket centroids) and return per-template overlays:

      * ``cartoon`` — CA-only trace of the host chain (compact; 3Dmol still draws
        a cartoon). Only the *chosen* subset gets one: top_n by TM ∪ top by
        Tanimoto ∪ seq_n mmseqs-only.
      * ``lig`` — the transformed crystal ligand (HETATM, the copy nearest the
        consensus pocket). Computed for EVERY aligned template up to ``lig_cap``,
        so pressing "all" shows the bound ligand structure even for templates
        that only render as a centroid dot (no cartoon).

    Aligning the host chain (not the whole oligomeric CIF) guarantees the
    overlay lands on the cofold receptor and reproduces the pipeline transform.
    """
    if top_n <= 0 or not templates.get("by_pdb"):
        return {}
    ref = templates.get("reference_cif")
    work = templates.get("work_dir")
    if not ref or not work or not Path(ref).exists():
        return {}
    try:
        import gemmi
        import numpy as np
        from casp17.usalign import run_usalign
    except ImportError:
        return {}
    refp = Path(ref)
    work = Path(work)
    focus = np.asarray(templates["focus"], dtype=float) if templates.get("focus") else None
    rows = templates["by_pdb"]  # already TM-sorted desc
    n = len(rows)
    by_tm = list(range(min(top_n, n)))
    sim_n = max(10, top_n // 2)
    by_sim = sorted(range(n), key=lambda i: (rows[i]["best_tan"] or 0), reverse=True)[:sim_n]
    seq_extra = [i for i, g in enumerate(rows) if g["ms"] and not g["fs"]][:seq_n]
    # cartoons are full-backbone (heavy on size) → cap to the strongest few by
    # TM; ligand overlays (small) still cover up to lig_cap templates.
    cartoon_ranked = list(dict.fromkeys([*by_tm, *by_sim, *seq_extra]))
    cartoon_set = set(sorted(cartoon_ranked,
                             key=lambda i: (rows[i]["best_tm"] or 0),
                             reverse=True)[:cartoon_cap])
    # align this many (for ligand overlays); always include the cartoon set
    lig_idx = set(range(min(lig_cap, n))) | cartoon_set
    out: dict[str, dict] = {}
    for gi in sorted(lig_idx):  # ascending == TM-desc (rows pre-sorted)
        g = rows[gi]
        pdb = g["pdb"]
        ccds = set(g["ccds"])
        # Align EVERY pre-extracted host chain ("{pdb}_{CHAIN}.pdb"). Each gives a
        # valid transform for the ligand bound to THAT protomer. The cartoon uses
        # the best-TM chain; the ligand uses whichever aligned chain places its
        # own bound copy nearest the pocket — so a ligand on a non-best hit chain
        # still lands right (was hidden by the single-transform approach).
        chain_files = sorted(work.glob(f"{pdb}_*.pdb"))
        aligns = []  # (cf, letter, R, t, tm, rmsd)
        for cf in chain_files:
            r = run_usalign(cf, refp, timeout=timeout)
            if not r or r[2] < 0.3:
                continue
            letter = cf.stem.split("_", 1)[1] if "_" in cf.stem else ""
            aligns.append((cf, letter, r[0], r[1], r[2], r[3]))
        if not aligns:  # no host-chain extraction → whole-CIF fallback
            cif = work / f"{pdb}.cif"
            if not cif.exists():
                continue
            r = run_usalign(cif, refp, timeout=timeout)
            if not r or r[2] < 0.3:
                continue
            aligns.append((cif, "", r[0], r[1], r[2], r[3]))
        best = max(aligns, key=lambda a: a[4])  # cartoon = best TM chain
        align_input, _bl, R, t, tm, rmsd = best
        Rm, tv = np.asarray(R), np.asarray(t)
        try:
            cif_path = work / f"{pdb}.cif"
            # ligand: best-placed copy across all aligned chains (strict host copy
            # per chain); fall back to any copy under the best chain's transform.
            lig, lig_d = "", None
            for (cf, letter, cR, cT, ctm, crm) in aligns:
                lines, d = _template_ligand_hetatm(
                    cif_path, ccds, np.asarray(cR), np.asarray(cT), focus,
                    host_chain=letter, strict=True)
                if lines and (lig_d is None or (d is not None and d < lig_d)):
                    lig, lig_d = lines, d
            # no hit-chain copy near the pocket → align the ligand's OWN chain
            if not lig or (lig_d is not None and lig_d > 15.0):
                l2, d2 = _ligand_on_own_chain(cif_path, ccds, refp, focus, timeout=timeout)
                if l2 and (lig_d is None or (d2 is not None and d2 < lig_d)):
                    lig, lig_d = l2, d2
            if not lig:
                lig, _ = _template_ligand_hetatm(cif_path, ccds, Rm, tv, focus,
                                                 host_chain="", strict=False)
            cartoon = ""
            if gi in cartoon_set:
                st = gemmi.read_structure(str(align_input))
                st.remove_alternative_conformations()
                st.remove_empty_chains()
                for model in st:
                    for chain in model:
                        for resi in chain:
                            for atom in resi:
                                p = atom.pos
                                v = Rm @ np.array([p.x, p.y, p.z]) + tv
                                atom.pos = gemmi.Position(float(v[0]), float(v[1]), float(v[2]))
                    break  # first model only
                # full backbone (N,CA,C,O) — 3Dmol cartoon needs it (CA-only draws
                # nothing). Single host chain, so size stays bounded.
                cartoon = _slim_pdb(st.make_pdb_string())
        except Exception:
            continue
        if not (cartoon.strip() or lig.strip()):
            continue
        out[pdb] = {"cartoon": cartoon, "lig": lig,
                    "tm": round(float(tm), 3), "rmsd": round(float(rmsd), 2),
                    "color": TPL_STRUCT_COLORS[len(out) % len(TPL_STRUCT_COLORS)]}
    return out


_LIG_SKIP = _WATER | {"HOH"}  # never render water as a "ligand"


def _ligand_on_own_chain(cif_path, ccds, refp, focus, timeout=90, max_chains=4):
    """Place the crystal ligand by aligning ITS OWN chain to the reference.

    For templates with no pre-extracted host-chain file (or whose ligand sits on
    a chain that wasn't a search hit), extract each candidate-ligand-bearing
    chain on the fly, USalign it to the reference, and transform that chain's
    ligand. Returns ``(hetatm_lines, dist_to_focus)`` for the copy nearest
    ``focus`` — or ``("", None)``. Bounds work at ``max_chains`` USalign calls."""
    if not cif_path.exists() or not ccds:
        return "", None
    try:
        import gemmi
        import numpy as np
        import tempfile
        from casp17.usalign import run_usalign
    except ImportError:
        return "", None
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return "", None
    if not len(st):
        return "", None
    model = st[0]
    lig_chains = []
    for ch in model:
        if any(r.name in ccds and r.name.upper() not in _LIG_SKIP for r in ch):
            lig_chains.append(ch.name)
    best, best_d = "", None
    for cn in lig_chains[:max_chains]:
        ns = gemmi.Structure(); ns.name = st.name
        nm = gemmi.Model("1"); nc = gemmi.Chain("A")
        for ch in model:
            if ch.name != cn:
                continue
            for r in ch:
                if r.entity_type == gemmi.EntityType.Polymer:
                    nc.add_residue(r)
        if len(nc) < 30:  # too short to align meaningfully
            continue
        nm.add_chain(nc); ns.add_model(nm)
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            tmp = Path(f.name)
        try:
            ns.write_pdb(str(tmp))
            r = run_usalign(tmp, refp, timeout=timeout)
        finally:
            try: tmp.unlink()
            except OSError: pass
        if not r or r[2] < 0.3:
            continue
        lines, d = _template_ligand_hetatm(cif_path, ccds, np.asarray(r[0]),
                                           np.asarray(r[1]), focus,
                                           host_chain=cn, strict=True)
        if lines and (best_d is None or (d is not None and d < best_d)):
            best, best_d = lines, d
    return best, best_d


def _template_ligand_hetatm(cif_path, ccds, Rm, tv, focus, host_chain: str = "",
                            strict: bool = False):
    """Return ``(hetatm_lines, dist_to_focus)`` for the crystal ligand of a
    template, transformed by ``Rm/tv``.

    ``strict=True``: only a candidate-CCD copy physically on ``host_chain`` (the
    protomer this transform aligns) — returns ``("", None)`` if none. Used to
    place each aligned chain's own bound ligand. ``strict=False``: host copy if
    present else the copy nearest ``focus`` (fallback). Water is never emitted."""
    if not cif_path.exists() or not ccds:
        return "", None
    try:
        import gemmi
        import numpy as np
    except ImportError:
        return "", None
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return "", None
    host = (host_chain or "").upper()
    best, best_d = None, None            # any-copy nearest focus
    host_best, host_best_d = None, None  # copy on the host chain
    for model in st:
        for chain in model:
            on_host = chain.name.upper() == host
            for res in chain:
                if res.name not in ccds or res.name.upper() in _LIG_SKIP:
                    continue
                atoms = []
                for a in res:
                    p = a.pos
                    v = Rm @ np.array([p.x, p.y, p.z]) + tv
                    atoms.append((a.name, (a.element.name or "").upper(), v))
                if not atoms:
                    continue
                cen = np.mean([v for _, _, v in atoms], axis=0)
                d = float(np.linalg.norm(cen - focus)) if focus is not None else 0.0
                if best_d is None or d < best_d:
                    best_d, best = d, (res.name, atoms)
                if on_host and (host_best_d is None or d < host_best_d):
                    host_best_d, host_best = d, (res.name, atoms)
        break  # first model only
    if strict:
        chosen, chosen_d = host_best, host_best_d
    else:
        chosen, chosen_d = (host_best, host_best_d) if host_best else (best, best_d)
    if not chosen:
        return "", None
    resn, atoms = chosen
    lines = []
    for i, (name, elem, v) in enumerate(atoms, 1):
        an = name if len(name) >= 4 else f" {name:<3}"
        lines.append("HETATM%5d %-4.4s%3.3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2.2s"
                     % (i % 100000, an, resn, 1, v[0], v[1], v[2], elem))
    return "\n".join(lines) + "\n", chosen_d


# --------------------------------------------------------------------------- #
# Per-pose metrics (BA-Pred + RMSD-Pred), keyed by pose Name == MDL title
# --------------------------------------------------------------------------- #
COFOLD_TOOLS = ["boltz2", "boltz2x", "protenix", "alphafold3"]
COFOLD_COLORS = {"boltz2": "#1f77b4", "boltz2x": "#17becf",
                 "protenix": "#2ca02c", "alphafold3": "#d62728"}


def load_cofold_receptors(roots: list[Path]) -> list[dict]:
    """One representative aligned receptor per cofolding tool (boltz2 / boltz2x
    / protenix / alphafold3), polymer-only, in the common aligned frame — for
    overlaying the four predicted folds. Ligand/water stripped via gemmi."""
    try:
        import gemmi
    except ImportError:
        return []
    out = []
    for tool in COFOLD_TOOLS:
        found = []
        for root in roots:
            for p in root.rglob("*_aligned.cif"):
                if f"/{tool}/" in str(p):
                    found.append(p)
        if not found:
            continue
        cif = sorted(set(found))[0]
        try:
            st = gemmi.read_structure(str(cif))
            st.remove_alternative_conformations()
            st.remove_ligands_and_waters()
            st.remove_empty_chains()
            pdb = st.make_pdb_string()
        except Exception:
            continue
        if pdb.strip():
            out.append({"tool": tool, "color": COFOLD_COLORS[tool], "pdb": pdb})
    return out


def _norm_pose(name: str) -> str:
    """Collapse a duplicated trailing index so BA-Pred / RMSD-Pred keys align.

    RMSD-Pred names cofolding poses as ``cofold_af3_L_10_10`` while BA-Pred and
    the LG MDL title use ``cofold_af3_L_10``. Only collapse a genuine repeat
    (``_(\\d+)_\\1$``); docked poses like ``..._seed_202_7`` are untouched.
    """
    return re.sub(r"_(\d+)_\1$", r"_\1", name.strip())


def load_pose_metrics(roots: list[Path]) -> dict:
    metrics: dict[str, dict] = {}
    analysis_dirs = []
    for root in roots:
        analysis_dirs += list(root.rglob("outputs/analysis"))
        analysis_dirs += [d for d in root.rglob("analysis") if d.is_dir()]
    for adir in dict.fromkeys(analysis_dirs):  # de-dupe, keep order
        for tsv in adir.glob("rmsd_pred_*.tsv"):
            _read_tsv(tsv, metrics, {"pRMSD": "prmsd", "Is_Above_2A": "p_above_2a"})
        for tsv in adir.glob("ba_pred_*.tsv"):
            _read_tsv(tsv, metrics, {"pKd": "pkd", "Kcal/mol": "kcal"})
    return metrics


def _pose_group(name: str) -> str:
    """Group key for a pose Name: drop seed + conformer index.

    'vina_cofolding_3_L_seed_202_7' -> 'vina_cofolding_3_L'
    'cofold_af3_L_10'               -> 'cofold_af3_L'
    'template_8bua_vina_L_5'        -> 'template_8bua_vina_L'
    """
    return re.sub(r"(_seed_\d+)?_\d+$", "", name)


# longer palette for per-source pose groups
GROUP_PALETTE = ["#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f",
                 "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]


def _track_of(group: str) -> str:
    """Classify a pose-group source into its pipeline track / receptor basis."""
    if group.startswith("cofold_"):
        return "cofold (AF3-frame)"
    if "template_consensus" in group:
        return "Track1 · consensus-box (AF3 receptor)"
    if group.startswith("template_") and ("_vina" in group or "_adg" in group):
        return "Track2 · template-receptor docking"
    return "Track1 · binding-site docking (AF3 receptor)"


# fixed track display order
_TRACK_ORDER = [
    "Track1 · binding-site docking (AF3 receptor)",
    "Track1 · consensus-box (AF3 receptor)",
    "cofold (AF3-frame)",
    "Track2 · template-receptor docking",
]


def build_pose_groups(roots: list[Path], metrics: dict, top_n: int = 3,
                      anchors: list[dict] | None = None) -> list[dict]:
    """Per docking/cofolding source, the top-N most reliable poses (lowest
    predicted RMSD; falls back to highest pKd). Molblocks come from the staged
    SDFs — same frame as the LG receptor (verified).

    When ``anchors`` (template COMs) are supplied, each source also surfaces its
    single pose nearest any anchor — even if its LSCORE is too low to make the
    top-N — so template-similar poses stay in the candidate board (per the CASP
    meeting: don't let LSCORE hide a pose that follows the top template)."""
    # locate the poses/ dir
    pose_dirs: list[Path] = []
    for root in roots:
        pose_dirs += [d for d in root.rglob("analysis/poses") if d.is_dir()]
        pose_dirs += [d for d in root.rglob("poses") if d.is_dir()]
    pose_dirs = list(dict.fromkeys(pose_dirs))
    if not pose_dirs:
        return []

    # group pose names by source
    groups: dict[str, list[str]] = {}
    for name in metrics:
        groups.setdefault(_pose_group(name), []).append(name)

    sdf_cache: dict[Path, list[str]] = {}

    def conformer(name: str) -> str | None:
        m = re.match(r"^(.*)_(\d+)$", name)
        if not m:
            return None
        stem, idx = m.group(1), int(m.group(2))
        for pd in pose_dirs:
            sdf = pd / f"{stem}.sdf"
            if not sdf.exists():
                continue
            if sdf not in sdf_cache:
                parts = [p.lstrip("\n") for p in sdf.read_text().split("$$$$")]
                sdf_cache[sdf] = [p for p in parts if p.strip()]
            confs = sdf_cache[sdf]
            if 0 <= idx < len(confs):
                return confs[idx]
        return None

    anchors = anchors or []

    def anchor_of(mb):
        """Nearest template anchor to a molblock's heavy-atom centroid."""
        if not anchors or not mb:
            return None
        cen = _ligand_centroid(mb)
        return _nearest_anchor(cen, anchors) if cen else None

    out = []
    for gi, (gkey, names) in enumerate(sorted(groups.items())):
        def rank(n):
            met = metrics.get(n, {})
            pr = met.get("prmsd")
            pk = met.get("pkd")
            # lower pRMSD best; missing pRMSD sinks to end; tie-break higher pKd
            return (pr if pr is not None else 1e9, -(pk if pk is not None else -1e9))
        picked = []
        chosen = set()
        for n in sorted(names, key=rank)[:top_n]:
            mb = conformer(n)
            if mb:
                a = anchor_of(mb)
                picked.append({"name": n, "molblock": mb, "metrics": metrics.get(n, {}),
                               "tpl": a, "anchored": False})
                chosen.add(n)
        # template-anchored surfacing: the group's single pose nearest any anchor,
        # regardless of LSCORE, so a template-following pose is never hidden.
        if anchors:
            best_n, best_mb, best_a = None, None, None
            for n in names:
                if n in chosen:
                    continue
                mb = conformer(n)
                a = anchor_of(mb)
                if a and (best_a is None or a["dist"] < best_a["dist"]):
                    best_n, best_mb, best_a = n, mb, a
            if best_n is not None:
                picked.append({"name": best_n, "molblock": best_mb,
                               "metrics": metrics.get(best_n, {}),
                               "tpl": best_a, "anchored": True})
        if picked:
            lscores = [1.0 - p["metrics"]["p_above_2a"] for p in picked
                       if "p_above_2a" in p["metrics"]]
            dists = [p["tpl"]["dist"] for p in picked if p.get("tpl")]
            out.append({"group": gkey, "color": GROUP_PALETTE[gi % len(GROUP_PALETTE)],
                        "track": _track_of(gkey),
                        "best_lscore": max(lscores) if lscores else -1.0,
                        "min_tpl_dist": min(dists) if dists else None,
                        "poses": picked})
    # order: by track (fixed order) then best_lscore desc within track
    out.sort(key=lambda g: (_TRACK_ORDER.index(g["track"]) if g["track"] in _TRACK_ORDER else 99,
                            -g.get("best_lscore", -1)))
    return out


def _read_tsv(tsv: Path, metrics: dict, cols: dict) -> None:
    try:
        with tsv.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                name = _norm_pose(row.get("Name") or "")
                if not name:
                    continue
                rec = metrics.setdefault(name, {})
                for src, dst in cols.items():
                    val = (row.get(src) or "").strip()
                    if val and val.lower() != "nan":
                        try:
                            rec[dst] = float(val)
                        except ValueError:
                            pass
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# HTML emit
# --------------------------------------------------------------------------- #
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
    "A": "A", "U": "U", "G": "G", "C": "C", "DA": "A", "DT": "T",
    "DG": "G", "DC": "C", "DU": "U",
}


def sequence_from_pdb(pdb: str) -> list[dict]:
    """One entry per residue (CA for protein / C1' for nucleic), in chain order:
    {chain, resi, resn, one}."""
    seq = []
    seen = set()
    for ln in pdb.splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        atom = ln[12:16].strip()
        if atom not in ("CA", "C1'", "P"):
            continue
        chain = ln[21].strip() or "?"
        try:
            resi = int(ln[22:26])
        except ValueError:
            continue
        key = (chain, resi)
        if key in seen:
            continue
        seen.add(key)
        resn = ln[17:20].strip()
        seq.append({"chain": chain, "resi": resi, "resn": resn,
                    "one": _THREE_TO_ONE.get(resn, "X")})
    return seq


def _ligand_centroid(molblock: str) -> tuple[float, float, float] | None:
    """Heavy-atom centroid of an MDL molblock (skips explicit H)."""
    lines = molblock.splitlines()
    counts_i = next((i for i, l in enumerate(lines) if l.rstrip().endswith("V2000")), None)
    if counts_i is None:
        return None
    try:
        na = int(lines[counts_i][:3])
    except ValueError:
        return None
    xs = ys = zs = 0.0
    n = 0
    for l in lines[counts_i + 1: counts_i + 1 + na]:
        try:
            x, y, z = float(l[0:10]), float(l[10:20]), float(l[20:30])
            elem = l[31:34].strip()
        except (ValueError, IndexError):
            continue
        if elem == "H":
            continue
        xs += x; ys += y; zs += z; n += 1
    return (xs / n, ys / n, zs / n) if n else None


def _molblock_heavy_coords(molblock: str) -> list[tuple[float, float, float]]:
    """Heavy-atom xyz from an MDL molblock (skips explicit H)."""
    lines = molblock.splitlines()
    ci = next((i for i, l in enumerate(lines) if l.rstrip().endswith("V2000")), None)
    if ci is None:
        return []
    try:
        na = int(lines[ci][:3])
    except ValueError:
        return []
    out = []
    for l in lines[ci + 1: ci + 1 + na]:
        try:
            x, y, z = float(l[0:10]), float(l[10:20]), float(l[20:30])
            elem = l[31:34].strip()
        except (ValueError, IndexError):
            continue
        if elem != "H":
            out.append((x, y, z))
    return out


def _pdb_heavy_coords(pdb: str) -> list[tuple[float, float, float]]:
    """Heavy-atom xyz from PDB ATOM/HETATM records (element col 77-78, skip H)."""
    out = []
    for l in pdb.splitlines():
        if not l.startswith(("ATOM", "HETATM")):
            continue
        elem = l[76:78].strip() if len(l) >= 78 else ""
        if elem == "H" or (not elem and l[12:16].strip()[:1] == "H"):
            continue
        try:
            out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        except (ValueError, IndexError):
            continue
    return out


def _ligand_receptor_clash(receptor_pdb: str, molblock: str,
                           cutoff: float = 2.2) -> dict:
    """Per-pose steric check: heavy-atom ligand↔receptor contacts below
    ``cutoff`` Å (a real covalent contact is ~2.6 Å+, so <2.2 Å = interpenetration)
    plus the single closest contact. Cofolding can jam a ligand into the
    receptor (notably single-seq RNA); docking output is normally clash-free.
    """
    rec = _pdb_heavy_coords(receptor_pdb)
    lig = _molblock_heavy_coords(molblock)
    if not rec or not lig:
        return {"n": None, "min": None}
    c2 = cutoff * cutoff
    n = 0
    best2 = 1e18
    for (lx, ly, lz) in lig:
        for (rx, ry, rz) in rec:
            d2 = (lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2
            if d2 < best2:
                best2 = d2
            if d2 < c2:
                n += 1
    return {"n": n, "min": round(best2 ** 0.5, 2)}


def _nearest_bsite(centroid, sites: list[dict]) -> dict | None:
    if not centroid or not sites:
        return None
    cx, cy, cz = centroid
    best, bd = None, 1e18
    for s in sites:
        x, y, z = s["center"]
        d = ((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2) ** 0.5
        if d < bd:
            bd, best = d, s
    return {"source": best["source"], "dist": round(bd, 2)} if best else None


def load_research(run_dir: Path, receptor_cif: Path | None = None) -> dict | None:
    """Compact summary of the vendored research briefing (`<run_dir>/research.json`)
    for display + grounded catalytic-residue markers. None when no briefing."""
    rj = Path(run_dir) / "research.json"
    if not rj.is_file():
        return None
    try:
        d = json.loads(rj.read_text())
    except (OSError, ValueError):
        return None
    prot = d.get("protein") or {}
    ligs = []
    for lg in (d.get("ligands") or []):
        ligs.append({"name": lg.get("name"), "smiles": lg.get("smiles"),
                     "pubchem": lg.get("pubchem_cid"), "chembl": lg.get("chembl_id"),
                     "known_targets": lg.get("known_protein_targets") or []})
    tpls = []  # rich template rows (resolution, seq-identity, bound ligand)
    for t in (d.get("template_pdbs") or [])[:8]:
        if isinstance(t, dict):
            tpls.append({"pdb": t.get("pdb"), "res": t.get("resolution_a"),
                         "seqid": t.get("seq_identity"), "ccd": t.get("bound_ligand")})
        elif t:
            tpls.append({"pdb": t})
    cov = d.get("covalent_suspicion") or {}
    covalent = None
    if cov.get("suspected"):
        pl = cov.get("proposed_link") or {}
        covalent = {"confidence": cov.get("confidence"),
                    "rationale": cov.get("rationale"),
                    "residue": f"{pl.get('residue_name','')}{pl.get('resi','')}".strip(),
                    "resi": pl.get("resi"), "res_name": pl.get("residue_name"),
                    "warhead": cov.get("warhead_class") or pl.get("warhead_class")}
    residues = []
    if receptor_cif is not None:
        try:
            from casp17.research_prior import research_centers  # type: ignore
            for a in research_centers(run_dir, receptor_cif):
                residues.append({"xyz": list(a.xyz), "label": a.label})
        except Exception:
            residues = []
    return {
        "name": prot.get("name"),
        "uniprot": prot.get("uniprot_homolog"),
        "ec": prot.get("ec"),
        "family": prot.get("family"),
        "catalytic": prot.get("catalytic_residues_target_numbering") or [],
        "cofactors": prot.get("cofactors") or [],
        "metals": prot.get("metals") or [],
        "ligands": ligs,
        "templates": [t for t in tpls if t.get("pdb")],
        "covalent": covalent,
        "caveats": d.get("caveats") or [],
        "residues": residues,
    }


def build_target_html(parsed: dict, bsites: dict, metrics: dict,
                      pose_groups: list[dict], cofold_receptors: list[dict],
                      asset_rel: str, templates: dict | None = None,
                      template_structs: dict | None = None,
                      research: dict | None = None,
                      anchors: list[dict] | None = None) -> str:
    target = parsed["target"]
    anchors = anchors or []
    # de-dupe identical receptor blocks; map each model to a receptor index.
    receptors: list[str] = []
    rec_index: dict[str, int] = {}
    model_payload = []
    for i, m in enumerate(parsed["models"]):
        rpdb = m.get("receptor_pdb") or ""
        if rpdb not in rec_index:
            rec_index[rpdb] = len(receptors)
            receptors.append(rpdb)
        ligs = []
        for lg in m["ligands"]:
            met = metrics.get(_norm_pose(lg["title"]), {})
            ligs.append({"molblock": lg["molblock"], "title": lg["title"],
                         "lscore": lg["lscore"], "metrics": met})
        centroid = _ligand_centroid(m["ligands"][0]["molblock"]) if m["ligands"] else None
        clash = (_ligand_receptor_clash(rpdb, m["ligands"][0]["molblock"])
                 if (rpdb and m["ligands"]) else {"n": None, "min": None})
        model_payload.append({
            "model": m["model"],
            "receptor_idx": rec_index[rpdb],
            "color": MODEL_COLORS[i % len(MODEL_COLORS)],
            "affinity": m.get("affinity"),
            "ligands": ligs,
            "pocket": _nearest_bsite(centroid, bsites["sites"]),
            "tpl": _nearest_anchor(centroid, anchors),
            "clash": clash,
        })
    payload = {
        "target": target,
        "receptors": receptors,
        "models": model_payload,
        "bsites": bsites["sites"],
        "box_size": bsites["box_size"],
        "family_colors": BSITE_FAMILY_COLORS,
        "shared_receptor": len(receptors) == 1 and len(parsed["models"]) > 1,
        "pose_groups": pose_groups,
        "cofold_receptors": cofold_receptors,
        "sequence": sequence_from_pdb(receptors[0] if receptors else ""),
        "templates": (templates or {}).get("pockets", []),
        "templates_by_pdb": (templates or {}).get("by_pdb", []),
        "template_structs": template_structs or {},
        "template_anchors": anchors,
        "research": research,
    }
    for g in payload["templates_by_pdb"]:
        s = payload["template_structs"].get(g["pdb"])
        g["has_struct"] = bool(s and s.get("cartoon"))   # full cartoon overlay
        g["has_lig"] = bool(s and s.get("lig"))           # crystal ligand overlay
    return _TARGET_TEMPLATE.format(
        target=html.escape(target),
        asset_rel=asset_rel,
        data_json=json.dumps(payload),
        n_models=len(parsed["models"]),
        n_sites=len(bsites["sites"]),
        n_templates=len(payload["templates"]),
        n_tpl_pdb=len(payload["templates_by_pdb"]),
        rec_note=("1 shared receptor" if payload["shared_receptor"] else f"{len(receptors)} per-MODEL receptors"),
        prep_note=html.escape(bsites["prep_path"] or "none (RNA / docking off)"),
    )


def build_index_html(entries: list[dict]) -> str:
    rows = []
    for e in sorted(entries, key=lambda x: x["target"]):
        rows.append(
            f"<tr><td><a href='{html.escape(e['target'])}.html'>{html.escape(e['target'])}</a></td>"
            f"<td>{html.escape(e.get('kind','-'))}</td>"
            f"<td>{e['n_models']}</td><td>{e['n_ligands']}</td><td>{e['n_sites']}</td>"
            f"<td>{html.escape(e['best_lscore'])}</td><td>{html.escape(e['affinity'])}</td></tr>"
        )
    return _INDEX_TEMPLATE.format(rows="\n".join(rows), n=len(entries))


_TARGET_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{target} — CASP17 viewer</title>
<script src="{asset_rel}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1115; color:#e7e9ee; }}
  header {{ padding:10px 16px; background:#161922; border-bottom:1px solid #2a2f3a; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
  header h1 {{ font-size:18px; margin:0; }} header .meta {{ font-size:12px; color:#9aa3b2; }}
  #seqbar {{ background:#fff; color:#222; border-bottom:1px solid #2a2f3a; overflow-x:auto; overflow-y:hidden;
             font:11px ui-monospace, monospace; padding:2px 6px; }}
  #seqinner {{ display:inline-block; min-width:100%; }}
  #seqticks, #seqletters {{ white-space:nowrap; height:15px; line-height:15px; }}
  #seqticks {{ color:#8a8f98; }}
  .tick {{ display:inline-block; width:12px; position:relative; }}
  .tick .tno {{ position:absolute; right:1px; bottom:0; white-space:nowrap; font-size:9px; }}
  #seqbar .res {{ display:inline-block; width:12px; text-align:center; cursor:pointer; border-radius:2px; }}
  #seqbar .res:hover {{ background:#cde3ff; }}
  #seqbar .res.sel {{ background:#d62728; color:#fff; }}
  #seqbar .chsep {{ display:inline-block; width:12px; color:#bbb; text-align:center; }}
  #wrap {{ display:flex; height:calc(100vh - 96px); }}
  #panel {{ width:330px; flex:0 0 auto; min-width:200px; max-width:85vw; overflow:auto; padding:12px; background:#12151c; font-size:13px; }}
  #divider {{ flex:0 0 6px; cursor:col-resize; background:#2a2f3a; }}
  #divider:hover, #divider.drag {{ background:#3a4150; }}
  #pgList {{ max-height:300px; overflow:auto; border:1px solid #2a2f3a; border-radius:6px; padding:4px 6px; margin-top:4px; }}
  .trackhdr {{ font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#7f8794; margin:8px 0 2px; padding-top:4px; border-top:1px solid #222836; white-space:nowrap; }}
  .trackhdr:first-child {{ border-top:none; padding-top:0; margin-top:2px; }}
  label.row .txt {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #viewport {{ flex:1; position:relative; }}
  .sec {{ margin-bottom:16px; }}
  .sec h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#9aa3b2; margin:0 0 6px; border-bottom:1px solid #2a2f3a; padding-bottom:4px; }}
  label.row {{ display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer; }}
  label.row .sw {{ width:12px; height:12px; border-radius:3px; flex:0 0 12px; }}
  label.row .txt {{ flex:1; min-width:0; }} label.row .num {{ color:#9aa3b2; font-variant-numeric:tabular-nums; flex:0 0 auto; }}
  .mdl {{ border:1px solid #2a2f3a; border-radius:6px; padding:6px 8px; margin-bottom:6px; }}
  .mdl .met {{ font-size:11px; color:#9aa3b2; margin-top:3px; line-height:1.5; }}
  .mdl .met b {{ color:#cdd3df; font-weight:600; }}
  .mdl .src {{ font-size:10px; color:#6b7280; word-break:break-all; margin-top:2px; }}
  .badge {{ font-size:10px; font-weight:600; padding:1px 6px; border-radius:8px; margin-left:6px; flex:0 0 auto; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .badge.ok {{ background:#10331c; color:#5fd38a; border:1px solid #1d5e34; }}
  .badge.bad {{ background:#3a1115; color:#ff7a86; border:1px solid #6e1f27; }}
  .btn {{ background:#222838; color:#e7e9ee; border:1px solid #333b4e; border-radius:6px; padding:5px 9px; cursor:pointer; font-size:12px; margin:2px 2px 2px 0; }}
  .btn:hover {{ background:#2c344a; }}
  .tab {{ background:transparent; color:#9aa3b2; border:1px solid #333b4e; border-bottom:2px solid transparent; border-radius:6px 6px 0 0; padding:4px 9px; cursor:pointer; font-size:12px; }}
  .tab:hover {{ background:#1c2130; color:#e7e9ee; }}
  .tab.active {{ color:#e7e9ee; background:#222838; border-bottom-color:#5b8cff; }}
  label.tplrow {{ align-items:flex-start; padding:5px 0; }}
  .tplbody {{ flex:1; min-width:0; }}
  .tplhdr {{ display:flex; align-items:baseline; gap:6px; }}
  .tplhdr .pdb {{ font-weight:700; letter-spacing:.02em; }}
  .tplhdr .ccd {{ font-size:11px; color:#9aa3b2; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .tplhdr .srcb {{ margin-left:auto; font-size:9px; color:#7f8794; flex:0 0 auto; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:3px; }}
  .chip {{ font-size:10px; font-variant-numeric:tabular-nums; padding:1px 6px; border-radius:8px;
           background:#222838; border:1px solid #333b4e; color:#cdd3df; white-space:nowrap; }}
  .chip .k {{ color:#7f8794; margin-right:3px; }}
  #tools {{ position:absolute; top:10px; right:10px; z-index:6; display:flex; flex-direction:column; gap:5px; }}
  .tool {{ width:34px; height:34px; display:flex; align-items:center; justify-content:center;
           background:rgba(24,28,38,.82); color:#e7e9ee; border:1px solid #333b4e; border-radius:8px;
           cursor:pointer; font-size:15px; backdrop-filter:blur(3px); user-select:none; }}
  .tool:hover {{ background:#2c344a; }}
  .tool.on {{ background:#2b3f6b; border-color:#5b8cff; color:#cfe0ff; }}
  #tools select {{ width:34px; }}
  select {{ background:#222838; color:#e7e9ee; border:1px solid #333b4e; border-radius:6px; padding:4px; font-size:12px; width:100%; }}
  .small {{ font-size:11px; color:#6b7280; word-break:break-all; }}
</style></head>
<body>
<header>
  <h1>{target}</h1>
  <span class="meta">{n_models} MODELs · {rec_note} · {n_sites} binding-site centroids · {n_templates} template pockets / {n_tpl_pdb} PDB</span>
  <span class="meta small">prep: {prep_note}</span>
</header>
<div id="seqbar" title="click a residue to highlight it in 3D"><div id="seqinner"><div id="seqticks"></div><div id="seqletters"></div></div></div>
<div id="wrap">
  <div id="panel">
    <div class="sec" id="researchSec" style="display:none">
      <h2>Research (functional prior)</h2>
      <label class="row"><input type="checkbox" id="resMark"><span class="txt">mark catalytic residues (magenta)</span></label>
      <div id="researchBody" class="small" style="line-height:1.5; word-break:break-word;"></div>
    </div>
    <div class="sec">
      <h2>Receptor style</h2>
      <select id="recStyle">
        <option value="cartoon">cartoon</option>
        <option value="cartoon+surface">cartoon + surface</option>
        <option value="stick">stick</option>
        <option value="line">line</option>
        <option value="none">hide receptor</option>
      </select>
      <label class="row"><input type="checkbox" id="plddt"><span class="txt">color by pLDDT (B-factor)</span></label>
      <label class="row"><input type="checkbox" id="pocketRes"><span class="txt">show pocket residues (≤4.5Å)</span></label>
    </div>
    <div class="sec">
      <h2>MODELs (receptor + pose)</h2>
      <div><button class="btn" id="mAll">all</button><button class="btn" id="mNone">none</button><button class="btn" id="m1only">MODEL 1</button></div>
      <div id="mdlList"></div>
    </div>
    <div class="sec">
      <h2>Cofolding models (receptor compare)</h2>
      <div><button class="btn" id="cfAll">all</button><button class="btn" id="cfNone">none</button></div>
      <div id="cofoldList"></div>
    </div>
    <div class="sec">
      <h2>All poses · top 3 + ⚓template / source</h2>
      <div><button class="btn" id="pgAll">all</button><button class="btn" id="pgNone">none</button><button class="btn" id="pgTpl" title="select every source whose nearest pose is within 8 Å of a template COM">⚓ near-tpl</button></div>
      <div id="pgList"></div>
    </div>
    <div class="sec">
      <h2>Binding sites</h2>
      <label class="row"><input type="checkbox" id="boxShow"><span class="txt">show docking boxes</span></label>
      <label class="row"><input type="checkbox" id="lblShow" checked><span class="txt">show labels</span></label>
      <div><button class="btn" id="bsAll">all</button><button class="btn" id="bsNone">none</button></div>
      <div id="bsList"></div>
    </div>
    <div class="sec">
      <h2>Template pockets</h2>
      <label class="row"><input type="checkbox" id="tplAnchor" checked><span class="txt">show template COM anchors (★)</span></label>
      <div id="tplAnchorLegend" class="small" style="margin:2px 0 6px;"></div>
      <label class="row"><input type="checkbox" id="tplLigOnly"><span class="txt">template ligands only (ghost receptor + cartoon)</span></label>
      <label class="row"><span class="txt">color by</span>
        <select id="tplColor" style="width:auto; flex:0 0 auto;">
          <option value="tan">Tanimoto</option>
          <option value="tm">TM-score</option>
          <option value="source">source</option>
        </select></label>
      <div id="tplLegend" class="small" style="margin:4px 0;"></div>
      <div class="small" style="display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin:4px 0;">
        <span>sort</span>
        <select id="tplSort" style="width:auto; flex:0 0 auto;">
          <option value="tm">TM</option>
          <option value="pid">id (seq)</option>
          <option value="sim">sim (Tanimoto)</option>
          <option value="n">pockets</option>
        </select>
        <span>TM≥</span><input id="tplTmMin" type="number" min="0" max="1" step="0.05" value="0"
          style="width:50px; background:#222838; color:#e7e9ee; border:1px solid #333b4e; border-radius:5px; padding:2px 4px;">
        <span>sim≥</span><input id="tplSimMin" type="number" min="0" max="1" step="0.05" value="0"
          style="width:50px; background:#222838; color:#e7e9ee; border:1px solid #333b4e; border-radius:5px; padding:2px 4px;">
      </div>
      <label class="row"><input type="checkbox" id="tplLbl"><span class="txt">show template labels</span></label>
      <div id="tplTabs" style="display:flex; gap:2px; flex-wrap:wrap; border-bottom:1px solid #2a2f3a; margin-top:4px;"></div>
      <div style="margin-top:4px;"><button class="btn" id="tplAll">all</button><button class="btn" id="tplNone">none</button><button class="btn" id="tplTop">✓ top 10</button></div>
      <div id="tplList" style="max-height:280px; overflow:auto; border:1px solid #2a2f3a; border-top:none; border-radius:0 0 6px 6px; padding:4px 6px;"></div>
    </div>
    <div class="sec">
      <h2>View</h2>
      <button class="btn" id="reset">reset camera</button>
      <button class="btn" id="bg">toggle bg</button>
    </div>
  </div>
  <div id="divider" title="drag to resize sidebar"></div>
  <div id="viewport">
    <div id="tools">
      <div class="tool" id="tSpin"  title="toggle spin">⟳</div>
      <div class="tool" id="tFit"   title="fit / zoom to all">⤢</div>
      <div class="tool" id="tLig"   title="zoom to ligand(s)">◎</div>
      <div class="tool" id="tSurf"  title="toggle receptor surface">◐</div>
      <div class="tool" id="tProj"  title="perspective / orthographic">⬒</div>
      <div class="tool" id="tPng"   title="save PNG snapshot">📷</div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const $ = (id) => document.getElementById(id);
let viewer, bgWhite = true;
let spinning = false, orthographic = false, surfOn = false;
let recModelIds = [], ligModelIds = [];
const selResidues = new Set();  // "chain:resi"

function fmt(v, d) {{ return (v===undefined||v===null) ? "–" : (+v).toFixed(d); }}

// template-anchor identity/colour (COM the meeting wants poses to follow).
const ANCHOR_LBL = {{ top_template:"top-tpl", ligand_template:"lig-tpl", consensus:"consensus" }};
const ANCHOR_COL = {{ top_template:"#ffd400", ligand_template:"#ff5bd0", consensus:"#00e0c0" }};
function anchName(t) {{ return t ? (t.pdb ? t.pdb : (ANCHOR_LBL[t.kind]||t.kind)) : ""; }}
function anchColor(t) {{ return t ? (ANCHOR_COL[t.kind]||"#ffd400") : "#ffd400"; }}
// distance→colour: green near a template COM, red far off it.
function distCol(d) {{ return d==null ? "#6b7280" : (d<=6 ? "#5fd38a" : d<=12 ? "#e6a23c" : "#e6194B"); }}

// element colours (CPK-ish); carbon kept at the caller's colour so each pose /
// template stays distinguishable while heteroatoms read at a glance.
const ELEM_COLORS = {{ N:"#3050f8", O:"#ff2d2d", S:"#e6c53c", P:"#ff8000",
  F:"#4caf50", CL:"#2ecc40", BR:"#a0522d", I:"#7b2fbe", B:"#ffb5b5",
  H:"#e0e0e0", FE:"#e06633", ZN:"#7d80b0", MG:"#3cb371", CA:"#3dcf3d",
  NA:"#ab5cf2", K:"#8f40d4", MN:"#9c7ac7" }};
function carbonScheme(carbonColor) {{
  return {{ prop: "elem", map: Object.assign({{}}, ELEM_COLORS, {{ C: carbonColor }}) }};
}}

// centroid of a V2000 SDF/MOL block (heavy + H); used to anchor the pocket.
function molCentroid(mb) {{
  const lines = mb.split("\\n");
  const na = parseInt(lines[3].slice(0, 3));
  if (!na) return null;
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < na; i++) {{
    const l = lines[4 + i];
    x += parseFloat(l.slice(0, 10)); y += parseFloat(l.slice(10, 20)); z += parseFloat(l.slice(20, 30));
  }}
  return [x / na, y / na, z / na];
}}
// centroid of HETATM lines in a PDB fragment.
function pdbHetCentroid(pdb) {{
  let x = 0, y = 0, z = 0, n = 0;
  pdb.split("\\n").forEach(l => {{
    if (l.slice(0, 6).trim() === "HETATM") {{
      x += parseFloat(l.slice(30, 38)); y += parseFloat(l.slice(38, 46)); z += parseFloat(l.slice(46, 54)); n++;
    }}
  }});
  return n ? [x / n, y / n, z / n] : null;
}}
function dist3(a, b) {{ return Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); }}

function init() {{
  viewer = $3Dmol.createViewer("viewport", {{ backgroundColor: "#ffffff" }});
  const ml = $("mdlList");
  DATA.models.forEach((m, i) => {{
    const lg = m.ligands[0] || {{}};
    const met = lg.metrics || {{}};
    const cl = m.clash || {{}};
    const clashBadge = (cl.n===null || cl.n===undefined) ? "" :
      (cl.n>0
        ? `<span class="badge bad" title="${{cl.n}} ligand–receptor heavy-atom contacts under 2.2 Å; closest ${{cl.min}} Å — steric clash">✖ clash ${{cl.n}} · min ${{cl.min}}Å</span>`
        : `<span class="badge ok" title="no ligand–receptor heavy-atom contact under 2.2 Å; closest ${{cl.min}} Å">✓ no clash · min ${{cl.min}}Å</span>`);
    const metLine =
      `<b>LSCORE</b> ${{fmt(lg.lscore,3)}}` +
      (met.prmsd!==undefined ? ` · <b>pRMSD</b> ${{fmt(met.prmsd,2)}}Å` : "") +
      (met.p_above_2a!==undefined ? ` · <b>P(&gt;2Å)</b> ${{fmt(met.p_above_2a,2)}}` : "") +
      (met.pkd!==undefined ? ` · <b>pKd</b> ${{fmt(met.pkd,2)}}` : "") +
      (met.kcal!==undefined ? ` · <b>ΔG</b> ${{fmt(met.kcal,2)}}` : "") +
      (m.affinity ? ` · <b>AFFNTY</b> ${{m.affinity}}` : "") +
      (m.pocket ? ` · <b>pocket</b> ${{m.pocket.source}} <span style="color:${{m.pocket.dist>8?'#e6194B':'#6b7280'}}">@${{fmt(m.pocket.dist,1)}}Å${{m.pocket.dist>8?' ⚠off':''}}</span>` : "") +
      (m.tpl ? ` · <b>tpl</b> ${{anchName(m.tpl)}} <span style="color:${{distCol(m.tpl.dist)}}">@${{fmt(m.tpl.dist,1)}}Å</span>` : "");
    ml.insertAdjacentHTML("beforeend",
      `<div class="mdl"><label class="row"><input type="checkbox" class="mChk" data-i="${{i}}" ${{i===0?"checked":""}}>`+
      `<span class="sw" style="background:${{m.color}}"></span>`+
      `<span class="txt">MODEL ${{m.model}}</span>${{clashBadge}}</label>`+
      `<div class="met">${{metLine}}</div>`+
      `<div class="src">${{lg.title||""}}</div></div>`);
  }});
  // pose-group checkboxes (top-N per source)
  const pgl = $("pgList");
  if (!DATA.pose_groups || DATA.pose_groups.length === 0)
    pgl.innerHTML = '<div class="small">no post-analysis poses (RNA / docking off)</div>';
  let curTrack = null;
  (DATA.pose_groups || []).forEach((g, i) => {{
    if (g.track !== curTrack) {{
      curTrack = g.track;
      pgl.insertAdjacentHTML("beforeend", `<div class="trackhdr">${{g.track}}</div>`);
    }}
    const ls = (g.best_lscore >= 0) ? `LS ${{fmt(g.best_lscore,2)}}` : "";
    const td = (g.min_tpl_dist!=null)
      ? ` · <span style="color:${{distCol(g.min_tpl_dist)}}">⚓${{fmt(g.min_tpl_dist,1)}}Å</span>` : "";
    pgl.insertAdjacentHTML("beforeend",
      `<label class="row"><input type="checkbox" class="pgChk" data-i="${{i}}" data-tpl="${{g.min_tpl_dist!=null?g.min_tpl_dist:1e9}}">`+
      `<span class="sw" style="background:${{g.color}}"></span>`+
      `<span class="txt">${{g.group}}</span><span class="num">${{ls}}${{td}} · ${{g.poses.length}}p</span></label>`);
  }});
  // cofolding receptor checkboxes
  const cl = $("cofoldList");
  if (!DATA.cofold_receptors || DATA.cofold_receptors.length === 0)
    cl.innerHTML = '<div class="small">no cofolding receptors found</div>';
  (DATA.cofold_receptors || []).forEach((c, i) => {{
    cl.insertAdjacentHTML("beforeend",
      `<label class="row"><input type="checkbox" class="cfChk" data-i="${{i}}">`+
      `<span class="sw" style="background:${{c.color}}"></span><span class="txt">${{c.tool}}</span></label>`);
  }});
  // sequence ribbon: ticks row (numbers every 10) + clickable letters row,
  // both built with identical cell/chsep structure so they stay aligned.
  const ticks = $("seqticks"), letters = $("seqletters");
  let lastChain = null;
  (DATA.sequence || []).forEach(r => {{
    if (lastChain !== null && r.chain !== lastChain) {{
      ticks.insertAdjacentHTML("beforeend", '<span class="chsep"></span>');
      letters.insertAdjacentHTML("beforeend", '<span class="chsep">·</span>');
    }}
    lastChain = r.chain;
    const no = (r.resi % 10 === 0) ? `<span class="tno">${{r.resi}}</span>` : "";
    ticks.insertAdjacentHTML("beforeend", `<span class="tick">${{no}}</span>`);
    letters.insertAdjacentHTML("beforeend",
      `<span class="res" data-c="${{r.chain}}" data-r="${{r.resi}}" title="${{r.resn}} ${{r.resi}} (chain ${{r.chain}})">${{r.one}}</span>`);
  }});
  const sb = $("seqbar");
  sb.addEventListener("click", e => {{
    const el = e.target.closest(".res"); if (!el) return;
    const key = el.dataset.c + ":" + el.dataset.r;
    if (selResidues.has(key)) {{ selResidues.delete(key); el.classList.remove("sel"); }}
    else {{ selResidues.add(key); el.classList.add("sel"); }}
    render();
    if (el.classList.contains("sel")) {{ viewer.zoomTo({{ chain: el.dataset.c, resi: +el.dataset.r }}); viewer.render(); }}
  }});
  const bl = $("bsList");
  if (DATA.bsites.length === 0)
    bl.innerHTML = '<div class="small">no binding-site predictions (RNA / docking off)</div>';
  DATA.bsites.forEach((s, i) => {{
    const col = DATA.family_colors[s.family] || "#888";
    bl.insertAdjacentHTML("beforeend",
      `<label class="row"><input type="checkbox" class="bsChk" data-i="${{i}}">`+
      `<span class="sw" style="background:${{col}}"></span><span class="txt">${{s.source}}</span></label>`);
  }});
  // template pockets — tabbed by source (All / Both / Seq-mmseqs / Struct-foldseek)
  const tl = $("tplList"), tpdb = DATA.templates_by_pdb || [];
  const tplSrc = (g) => (g.ms && g.fs) ? "both" : (g.ms ? "mmseqs" : "foldseek");
  // colour a 0..1 metric chip: low=blue .. high=red, dark bg so white text reads
  const chipHeat = (v) => {{
    if (v==null) return "";
    const h = ((1 - Math.max(0,Math.min(1,v))) * 240).toFixed(0);
    return `background:hsl(${{h}},55%,26%); border-color:hsl(${{h}},45%,38%); color:#eef1f7;`;
  }};
  const chip = (k, v, css) => `<span class="chip" style="${{css||''}}"><span class="k">${{k}}</span>${{v}}</span>`;
  const tplRow = (g, i) => {{
    const src = tplSrc(g);
    const srcTxt = src==="both" ? "seq+str" : src==="mmseqs" ? "seq" : "str";
    const st = (g.has_struct?'<span title="3D structure embedded — cartoon + crystal ligand" style="color:#5fd38a">◈</span> ':'');
    const chips =
      (g.best_tm !=null ? chip("TM",  fmt(g.best_tm,2),  chipHeat(g.best_tm))  : "") +
      (g.best_pid!=null && g.best_pid>0 ? chip("id", `${{fmt(g.best_pid,0)}}%`, chipHeat(g.best_pid/100)) : "") +
      (g.best_tan!=null ? chip("sim", fmt(g.best_tan,2), chipHeat(g.best_tan)) : "") +
      (g.best_mcs!=null && g.best_mcs>0 ? chip("mcs", fmt(g.best_mcs,2), chipHeat(g.best_mcs)) : "") +
      chip("", `${{g.n}} site${{g.n>1?'s':''}}`);
    return `<label class="row tplrow" data-src="${{src}}" data-tm="${{g.best_tm!=null?g.best_tm:0}}" data-sim="${{g.best_tan!=null?g.best_tan:0}}" data-pid="${{g.best_pid!=null?g.best_pid:0}}" data-n="${{g.n}}"><input type="checkbox" class="tplChk" data-i="${{i}}" data-src="${{src}}">`+
      `<div class="tplbody">`+
        `<div class="tplhdr">${{st}}<span class="pdb">${{g.pdb}}</span><span class="ccd">${{g.ccds.join(", ")}}</span><span class="srcb">${{srcTxt}}</span></div>`+
        `<div class="chips">${{chips}}</div>`+
      `</div></label>`;
  }};
  let tplTab = "all";
  if (tpdb.length === 0) {{
    tl.innerHTML = '<div class="small">no template pockets (no hits / RNA)</div>';
  }} else {{
    const cnt = {{ all: tpdb.length, both: 0, mmseqs: 0, foldseek: 0 }};
    tpdb.forEach(g => cnt[tplSrc(g)]++);
    $("tplTabs").innerHTML = [["all","All"],["both","Both"],["mmseqs","Seq"],["foldseek","Struct"]]
      .filter(([k]) => k === "all" || cnt[k])
      .map(([k,l]) => `<button class="tab${{k==="all"?" active":""}}" data-tab="${{k}}">${{l}} ${{cnt[k]}}</button>`).join("");
    tpdb.forEach((g, i) => tl.insertAdjacentHTML("beforeend", tplRow(g, i)));  // flat, TM order
  }}
  // filter (tab + TM≥ + sim≥) and sort (TM / sim / pockets) the template list
  const tplApply = () => {{
    const tmMin = +$("tplTmMin").value || 0, simMin = +$("tplSimMin").value || 0;
    const key = $("tplSort").value;
    const rowsEl = [...tl.querySelectorAll(".row")];
    rowsEl.forEach(r => {{
      const vis = (tplTab === "all" || r.dataset.src === tplTab)
        && (+r.dataset.tm) >= tmMin && (+r.dataset.sim) >= simMin;
      r.style.display = vis ? "" : "none";
      r.dataset.vis = vis ? "1" : "0";
    }});
    const kf = r => key === "sim" ? +r.dataset.sim : key === "pid" ? +r.dataset.pid
                  : key === "n" ? +r.dataset.n : +r.dataset.tm;
    rowsEl.sort((a, b) => (b.dataset.vis - a.dataset.vis) || (kf(b) - kf(a)));
    rowsEl.forEach(r => tl.appendChild(r));
  }};
  $("tplTabs").addEventListener("click", e => {{
    const b = e.target.closest(".tab"); if (!b) return;
    tplTab = b.dataset.tab;
    document.querySelectorAll("#tplTabs .tab").forEach(x => x.classList.toggle("active", x === b));
    tplApply();
  }});
  ["tplSort","tplTmMin","tplSimMin"].forEach(id => $(id).addEventListener("input", tplApply));
  // visible checkboxes (respect current tab + threshold filter), in sorted order
  const visTpl = () => [...tl.querySelectorAll(".row")]
    .filter(r => r.dataset.vis !== "0").map(r => r.querySelector(".tplChk"));
  // research (functional prior) panel
  if (DATA.research) {{
    const r = DATA.research;
    const li = (r.ligands||[]).map(l =>
      `${{l.name||"?"}}${{l.pubchem?` · PubChem ${{l.pubchem}}`:""}}${{l.chembl?` · ${{l.chembl}}`:""}}`+
      `${{(l.known_targets&&l.known_targets.length)?` · targets: ${{l.known_targets.slice(0,3).join(", ")}}`:""}}`).join("<br>");
    const tpl = (r.templates||[]).map(t =>
      `<b>${{t.pdb}}</b>${{t.res!=null?` ${{fmt(t.res,1)}}Å`:""}}${{t.seqid!=null?` · id ${{Math.round(t.seqid*100)}}%`:""}}${{t.ccd?` · ${{t.ccd}}`:""}}`).join("<br>");
    const cov = r.covalent ? (
      `<div style="margin-top:8px; padding:6px 8px; background:#3a2a10; border:1px solid #7a5a1e; border-radius:6px;">`+
      `<b style="color:#ffcf6b">⚠ Covalent suspected</b>${{r.covalent.confidence?` <span style="color:#9aa3b2">(${{r.covalent.confidence}})</span>`:""}}`+
      (r.covalent.residue?`<br>site: <b>${{r.covalent.residue}}</b>`:"")+
      (r.covalent.warhead?`<br>warhead: ${{r.covalent.warhead}}`:"")+
      (r.covalent.rationale?`<br><span style="color:#c9b98f">${{r.covalent.rationale}}</span>`:"")+
      `</div>`) : "";
    const cav = (r.caveats&&r.caveats.length) ?
      `<details style="margin-top:8px;"><summary style="cursor:pointer; color:#9aa3b2">caveats (${{r.caveats.length}})</summary>`+
      `<ul style="margin:4px 0 0; padding-left:16px;">${{r.caveats.map(c=>`<li>${{c}}</li>`).join("")}}</ul></details>` : "";
    $("researchBody").innerHTML =
      (r.name?`<b>${{r.name}}</b><br>`:"") +
      (r.uniprot?`<span style="color:#7f8794">UniProt ${{r.uniprot}}</span> · `:"") +
      (r.ec&&r.ec.length?`EC ${{r.ec.join(", ")}}`:"") +
      (r.family?`<br><span style="color:#9aa3b2">${{r.family}}</span>`:"") +
      (r.catalytic&&r.catalytic.length?`<br><br><b>Catalytic:</b> ${{r.catalytic.join(", ")}}`:"") +
      (r.cofactors&&r.cofactors.length?`<br><b>Cofactors:</b> ${{r.cofactors.join(", ")}}`:"") +
      (r.metals&&r.metals.length?`<br><b>Metals:</b> ${{r.metals.join(", ")}}`:"") +
      cov +
      (li?`<br><br><b>Ligand:</b><br>${{li}}`:"") +
      (tpl?`<br><br><b>Templates:</b><br>${{tpl}}`:"") +
      cav;
    $("researchSec").style.display = "";
  }}
  // template-anchor legend (top-tpl / lig-tpl / consensus present for this target)
  if ($("tplAnchorLegend")) {{
    const seen = new Set((DATA.template_anchors||[]).map(a => a.kind));
    const parts = [];
    (DATA.template_anchors||[]).forEach(a => {{
      const nm = a.pdb ? `${{ANCHOR_LBL[a.kind]||a.kind}} ${{a.pdb}}` : (ANCHOR_LBL[a.kind]||a.kind);
      parts.push(`<span style="color:${{anchColor(a)}}">★</span> ${{nm}}`);
    }});
    $("tplAnchorLegend").innerHTML = parts.length ? parts.join(" · ")
      : '<span style="color:#6b7280">no template anchors (no hits / RNA)</span>';
  }}
  document.querySelectorAll(".mChk,.bsChk,.pgChk,.cfChk,.tplChk,#boxShow,#lblShow,#tplLbl,#tplAnchor,#tplLigOnly,#resMark,#plddt,#pocketRes,#tplColor").forEach(el => el.addEventListener("change", render));
  $("pgAll").onclick = () => {{ setChk(".pgChk", true); render(); }};
  $("pgNone").onclick = () => {{ setChk(".pgChk", false); render(); }};
  $("pgTpl").onclick = () => {{  // sources with a pose within 8 Å of a template COM
    document.querySelectorAll(".pgChk").forEach(c => c.checked = (+c.dataset.tpl) <= 8);
    render();
  }};
  $("recStyle").addEventListener("change", render);
  $("mAll").onclick = () => {{ setChk(".mChk", true); render(); }};
  $("mNone").onclick = () => {{ setChk(".mChk", false); render(); }};
  $("m1only").onclick = () => {{ document.querySelectorAll(".mChk").forEach(c => c.checked=(c.dataset.i==="0")); render(); }};
  $("bsAll").onclick = () => {{ setChk(".bsChk", true); render(); }};
  $("bsNone").onclick = () => {{ setChk(".bsChk", false); render(); }};
  $("cfAll").onclick = () => {{ setChk(".cfChk", true); render(); }};
  $("cfNone").onclick = () => {{ setChk(".cfChk", false); render(); }};
  $("tplAll").onclick = () => {{ visTpl().forEach(c => c.checked = true); render(); }};
  $("tplNone").onclick = () => {{ visTpl().forEach(c => c.checked = false); render(); }};
  $("tplTop").onclick = () => {{  // top 10 of the current sort/filter view
    const v = visTpl();
    v.forEach(c => c.checked = false);
    v.slice(0, 10).forEach(c => c.checked = true);
    render();
  }};
  tplApply();  // set initial visibility/order
  $("reset").onclick = () => {{ viewer.zoomTo(); viewer.render(); }};
  $("bg").onclick = () => {{ bgWhite=!bgWhite; viewer.setBackgroundColor(bgWhite?"#ffffff":"#0f1115"); viewer.render(); }};
  // draggable sidebar splitter
  const divider = $("divider"), panel = $("panel");
  let dragging = false;
  divider.addEventListener("mousedown", e => {{ dragging = true; divider.classList.add("drag"); e.preventDefault(); }});
  window.addEventListener("mousemove", e => {{
    if (!dragging) return;
    const w = Math.max(200, Math.min(e.clientX, window.innerWidth - 200));
    panel.style.width = w + "px";
  }});
  window.addEventListener("mouseup", () => {{
    if (!dragging) return;
    dragging = false; divider.classList.remove("drag");
    if (viewer.resize) viewer.resize();
    viewer.render();
  }});
  // right-side 3Dmol toolbar
  $("tSpin").onclick = () => {{
    spinning = !spinning;
    $("tSpin").classList.toggle("on", spinning);
    viewer.spin(spinning ? "y" : false);
  }};
  $("tFit").onclick = () => {{ viewer.zoomTo(); viewer.render(); }};
  $("tLig").onclick = () => {{
    if (ligModelIds.length) viewer.zoomTo({{ model: ligModelIds }});
    else viewer.zoomTo();
    viewer.render();
  }};
  $("tSurf").onclick = () => {{ surfOn = !surfOn; $("tSurf").classList.toggle("on", surfOn); render(); }};
  $("tProj").onclick = () => {{
    orthographic = !orthographic;
    $("tProj").classList.toggle("on", orthographic);
    viewer.setProjection(orthographic ? "orthographic" : "perspective");
    viewer.render();
  }};
  $("tPng").onclick = () => {{
    const uri = viewer.pngURI();
    const a = document.createElement("a");
    a.href = uri; a.download = (DATA.target || "viewer") + ".png";
    document.body.appendChild(a); a.click(); a.remove();
  }};
  render(true);
}}

function setChk(sel, v) {{ document.querySelectorAll(sel).forEach(c => c.checked = v); }}

function render(initial) {{
  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.removeAllLabels();
  const recStyle = $("recStyle").value;
  const usePlddt = $("plddt").checked;
  const PLDDT = {{ prop: "b", gradient: "roygb", min: 50, max: 90 }};  // blue = high
  const checked = [...document.querySelectorAll(".mChk")].filter(c => c.checked).map(c => +c.dataset.i);
  const multi = checked.length > 1;
  // "template ligands only" ghosts everything except template ligands so a small
  // buried ligand (e.g. a single amino-acid crystal fragment) isn't hidden inside
  // the opaque receptor. It drops template cartoons AND fades the MODEL receptor.
  const tplLigOnly = (($("tplLigOnly")||{{}}).checked) || false;
  const recOpacity = tplLigOnly ? 0.15 : 1.0;
  // each MODEL is a complete CASP snapshot (receptor + its ligand). Render them
  // independently — toggling MODEL N draws MODEL N's own receptor + pose, as
  // written in the LG file. (For protein the receptor is byte-identical across
  // MODELs, so multiple-on overlap exactly; view one at a time for a clean snapshot.)
  recModelIds = []; ligModelIds = [];
  // Draw a receptor for EVERY distinct receptor among the checked MODELs. Protein
  // targets share ONE predicted structure across MODEL 1-5 (only the ligand pose
  // differs), so the distinct-set is a single receptor drawn once — that is the
  // correct, complete "protein + all poses" picture (there is no 2nd protein).
  // RNA targets differ per MODEL, so each distinct fold is drawn. Deduping by
  // receptor_idx avoids stacking 5 identical cartoons (which z-fight and can look
  // like nothing rendered).
  const drawnRec = new Set();
  checked.forEach(i => {{
    const m = DATA.models[i];
    if (recStyle !== "none" && !drawnRec.has(m.receptor_idx)) {{
      drawnRec.add(m.receptor_idx);
      const rpdb = DATA.receptors[m.receptor_idx];
      if (rpdb) {{
        const rec = viewer.addModel(rpdb, "pdb");
        recModelIds.push(rec.getID());
        const polyColor = usePlddt ? {{ colorscheme: PLDDT }}
          : ((multi && !DATA.shared_receptor) ? {{ color: m.color }} : {{ color: "spectrum" }});
        if (recStyle === "cartoon" || recStyle === "cartoon+surface")
          rec.setStyle({{}}, {{ cartoon: Object.assign({{ opacity: recOpacity }}, polyColor) }});
        else if (recStyle === "stick")
          rec.setStyle({{}}, {{ stick: Object.assign({{ radius: 0.1, opacity: recOpacity }}, polyColor) }});
        else
          rec.setStyle({{}}, {{ line: polyColor }});
        if (recStyle === "cartoon+surface")
          viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.45, color: "white" }}, {{ model: rec.getID() }});
      }}
    }}
    // ligand pose — trust the MDL connection table (no distance re-perception)
    m.ligands.forEach(lg => {{
      const lm = viewer.addModel(lg.molblock, "sdf", {{ keepH: true, assignBonds: false }});
      const lcs = carbonScheme(m.color);  // C = MODEL colour, heteroatoms by element
      lm.setStyle({{}}, {{ stick: {{ radius: 0.15, colorscheme: lcs }}, sphere: {{ scale: 0.22, colorscheme: lcs }} }});
      ligModelIds.push(lm.getID());
    }});
  }});
  // if MODELs are shown but recStyle hid nothing was drawn, still nothing — but
  // when at least one MODEL is on, guarantee the shared receptor is visible.
  if (recStyle !== "none" && checked.length && !recModelIds.length && DATA.receptors[0]) {{
    const rec = viewer.addModel(DATA.receptors[0], "pdb");
    recModelIds.push(rec.getID());
    rec.setStyle({{}}, {{ cartoon: {{ color: "spectrum", opacity: recOpacity }} }});
  }}
  // optional receptor surface (right-toolbar toggle), independent of style menu
  if (surfOn && recModelIds.length)
    recModelIds.forEach(rid =>
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.45, color: "white" }}, {{ model: rid }}));
  // pocket residues: receptor atoms within 4.5Å of any shown ligand -> add sticks
  if ($("pocketRes").checked && recModelIds.length && ligModelIds.length) {{
    recModelIds.forEach(rid => ligModelIds.forEach(lid =>
      viewer.addStyle({{ model: rid, byres: true, within: {{ distance: 4.5, sel: {{ model: lid }} }} }},
                      {{ stick: {{ radius: 0.18, colorscheme: "cyanCarbon" }} }})));
  }}
  // cofolding receptors (compare folds) — cartoon in tool colour
  document.querySelectorAll(".cfChk").forEach(chk => {{
    if (!chk.checked) return;
    const c = DATA.cofold_receptors[+chk.dataset.i];
    const rm = viewer.addModel(c.pdb, "pdb");
    rm.setStyle({{}}, {{ cartoon: {{ color: c.color }} }});
  }});
  // extra poses (top-N per source) — thin sticks in the group colour
  document.querySelectorAll(".pgChk").forEach(chk => {{
    if (!chk.checked) return;
    const g = DATA.pose_groups[+chk.dataset.i];
    g.poses.forEach(p => {{
      const pm = viewer.addModel(p.molblock, "sdf", {{ keepH: true, assignBonds: false }});
      // template-anchored poses (surfaced regardless of LSCORE) draw thicker with
      // faint anchor-colour dots so a template-following pose stands out.
      if (p.anchored) {{
        const ac = anchColor(p.tpl);
        pm.setStyle({{}}, {{ stick: {{ radius: 0.16, colorscheme: carbonScheme(g.color) }},
                            sphere: {{ scale: 0.18, color: ac }} }});
      }} else {{
        pm.setStyle({{}}, {{ stick: {{ radius: 0.1, colorscheme: carbonScheme(g.color) }} }});
      }}
    }});
  }});
  // binding-site centroids — bold spheres + thick wire boxes (12 edge cylinders)
  const showBox = $("boxShow").checked, showLbl = $("lblShow").checked;
  const [bx, by, bz] = DATA.box_size;
  const drawThickBox = (ctr, w, h, d, color) => {{
    const hx = w/2, hy = h/2, hz = d/2;
    const cs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]]
      .map(([sx,sy,sz]) => ({{ x: ctr.x+sx*hx, y: ctr.y+sy*hy, z: ctr.z+sz*hz }}));
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    edges.forEach(([a,b]) =>
      viewer.addCylinder({{ start: cs[a], end: cs[b], radius: 0.18, color: color, opacity: 0.95, fromCap: 2, toCap: 2 }}));
  }};
  document.querySelectorAll(".bsChk").forEach(chk => {{
    if (!chk.checked) return;
    const s = DATA.bsites[+chk.dataset.i];
    const col = DATA.family_colors[s.family] || "#888";
    const c = {{ x: s.center[0], y: s.center[1], z: s.center[2] }};
    viewer.addSphere({{ center: c, radius: 1.6, color: col, opacity: 0.9 }});
    if (showLbl)
      viewer.addLabel(s.source, {{ position: c, fontSize: 10, backgroundColor: col,
                                  backgroundOpacity: 0.6, fontColor: "white" }});
    if (showBox)
      drawThickBox(c, bx, by, bz, col);
  }});
  // research catalytic-residue markers (magenta spheres + labels)
  if (($("resMark")||{{}}).checked && DATA.research && DATA.research.residues) {{
    DATA.research.residues.forEach(rr => {{
      viewer.addSphere({{ center: {{ x:rr.xyz[0], y:rr.xyz[1], z:rr.xyz[2] }},
                          radius: 1.4, color: "magenta", opacity: 0.85 }});
      viewer.addLabel(rr.label, {{ position: {{ x:rr.xyz[0], y:rr.xyz[1], z:rr.xyz[2] }},
                      fontSize: 10, backgroundColor: "magenta", backgroundOpacity: 0.7, fontColor: "white" }});
    }});
  }}
  // template pockets — centroid cloud, colour by selected metric
  const tplColor = ($("tplColor")||{{}}).value || "tan";
  const tplShowLbl = (($("tplLbl")||{{}}).checked) || false;
  const heat = (v) => {{  // 0->blue, 0.5->yellow, 1->red; null->grey
    if (v==null) return "#888888";
    v = Math.max(0, Math.min(1, v));
    const h = (1 - v) * 240;  // 240=blue .. 0=red
    return "hsl(" + h.toFixed(0) + ",85%,55%)";
  }};
  const srcCol = (p) => (p.ms && p.fs) ? "#2ca02c" : (p.fs ? "#1f77b4" : "#ff7f0e");
  const tplCol = (p) => tplColor==="source" ? srcCol(p)
                      : tplColor==="tm" ? heat(p.tm) : heat(p.tan);
  if ($("tplLegend"))
    $("tplLegend").innerHTML = (tplColor==="source")
      ? '<span style="color:#ff7f0e">●</span> mmseqs · <span style="color:#1f77b4">●</span> foldseek · <span style="color:#2ca02c">●</span> both'
      : 'low <span style="color:hsl(240,85%,55%)">●</span><span style="color:hsl(120,85%,55%)">●</span><span style="color:hsl(0,85%,55%)">●</span> high';
  const tplChecked = [...document.querySelectorAll(".tplChk")].filter(c => c.checked).map(c => +c.dataset.i);
  const TPLS = DATA.template_structs || {{}};
  // Keep a template ligand if it sits ON the receptor body (any pocket); hide
  // only symmetry-mate copies flung far off the protein by a wrong-chain
  // transform. Anchor = receptor CA centroid + max CA radius (+margin), NOT the
  // MODEL-1 pose (which may be at a different site than a template's pocket).
  let recCen = null, recR = 0;
  {{
    const rp = DATA.models[0] ? DATA.receptors[DATA.models[0].receptor_idx] : null;
    if (rp) {{
      const cas = [];
      rp.split("\\n").forEach(l => {{
        if (l.slice(0,6).trim()==="ATOM" && l.slice(12,16).trim()==="CA")
          cas.push([parseFloat(l.slice(30,38)),parseFloat(l.slice(38,46)),parseFloat(l.slice(46,54))]);
      }});
      if (cas.length) {{
        recCen = [0,1,2].map(k => cas.reduce((s,c)=>s+c[k],0)/cas.length);
        recR = Math.max(...cas.map(c => dist3(c, recCen)));
      }}
    }}
  }}
  const LIG_MARGIN = 8;  // Å beyond the receptor radius still counts as "on body"
  tplChecked.forEach(gi => {{
    const g = DATA.templates_by_pdb[gi];
    const s = TPLS[g.pdb];
    const p0 = g.idxs && g.idxs.length ? DATA.templates[g.idxs[0]] : null;
    const col = s ? s.color : (p0 ? tplCol(p0) : "#888");
    // cartoon + ligand in ONE model, styled via hetflag (proven path).
    // The bound ligand (often a small crystal fragment / free amino acid) is
    // drawn fat + opaque; the cartoon is translucent so the ligand never hides
    // inside a same-coloured fold. "ligands only" drops the cartoon entirely.
    let ligOk = false;
    if (s && s.lig) {{
      const lc = pdbHetCentroid(s.lig);
      ligOk = !recCen || !lc || dist3(lc, recCen) <= recR + LIG_MARGIN;
    }}
    const showCartoon = s && s.cartoon && !tplLigOnly;
    const parts = [];
    if (showCartoon) parts.push(s.cartoon.trim());
    if (s && s.lig && ligOk) parts.push(s.lig.trim());
    if (parts.length) {{
      const mdl = viewer.addModel(parts.join("\\n") + "\\n", "pdb");
      if (showCartoon) mdl.setStyle({{ hetflag: false }}, {{ cartoon: {{ color: col, opacity: 0.5 }} }});
      if (s.lig && ligOk)
        mdl.setStyle({{ hetflag: true }}, {{ stick: {{ radius: 0.3, colorscheme: carbonScheme(col) }},
                                            sphere: {{ scale: 0.6, colorscheme: carbonScheme(col) }} }});
    }} else {{  // nothing aligned — centroid dot fallback
      (g.idxs || []).forEach(pi => {{
        const p = DATA.templates[pi];
        viewer.addSphere({{ center: {{ x:p.xyz[0], y:p.xyz[1], z:p.xyz[2] }},
                            radius: 0.6, color: tplCol(p), opacity: 0.75 }});
      }});
    }}
    if (tplShowLbl && p0) {{
      const chem = `${{g.ccds.join(",")}}` +
        (g.best_tan!=null?` · sim ${{fmt(g.best_tan,2)}}`:"") +
        (g.best_mcs!=null&&g.best_mcs>0?` · mcs ${{fmt(g.best_mcs,2)}}`:"");
      const tmt = s ? ` · TM ${{fmt(s.tm,2)}}` : "";
      viewer.addLabel(`${{g.pdb}}${{tmt}} · ${{chem}}`, {{ position: {{ x:p0.xyz[0], y:p0.xyz[1], z:p0.xyz[2] }},
                      fontSize: 10, backgroundColor: col, backgroundOpacity: 0.7, fontColor: "white" }});
    }}
  }});
  // template COM anchors — the COMs the meeting wants poses to follow (top
  // template / ligand template / consensus). Bright translucent spheres + label.
  if (($("tplAnchor")||{{}}).checked && DATA.template_anchors) {{
    DATA.template_anchors.forEach(a => {{
      const c = {{ x:a.xyz[0], y:a.xyz[1], z:a.xyz[2] }};
      const col = anchColor(a);
      viewer.addSphere({{ center: c, radius: 1.9, color: col, opacity: 0.5 }});
      const nm = a.pdb ? `★ ${{ANCHOR_LBL[a.kind]||a.kind}} ${{a.pdb}}` : `★ ${{ANCHOR_LBL[a.kind]||a.kind}}`;
      viewer.addLabel(nm, {{ position: c, fontSize: 11, backgroundColor: col,
                            backgroundOpacity: 0.75, fontColor: "#111" }});
    }});
  }}
  // highlight residues picked from the sequence ribbon (magenta sticks)
  if (selResidues.size) {{
    const byChain = {{}};
    selResidues.forEach(k => {{ const [c, r] = k.split(":"); (byChain[c] = byChain[c] || []).push(+r); }});
    Object.entries(byChain).forEach(([c, rs]) =>
      viewer.setStyle({{ chain: c, resi: rs }},
                      {{ stick: {{ radius: 0.3, color: "magenta" }}, sphere: {{ scale: 0.45, color: "magenta" }} }}));
  }}
  if (initial) viewer.zoomTo();
  viewer.render();
}}
init();
</script>
</body></html>
"""


_INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CASP17 submissions — viewer index</title>
<style>
  body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1115; color:#e7e9ee; padding:24px; }}
  h1 {{ font-size:20px; }} .meta {{ color:#9aa3b2; font-size:13px; margin-bottom:16px; }}
  table {{ border-collapse:collapse; width:100%; max-width:760px; font-size:14px; }}
  th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid #2a2f3a; }}
  th {{ color:#9aa3b2; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  a {{ color:#6ea8fe; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  tr:hover td {{ background:#161922; }}
  h2 {{ font-size:15px; margin:28px 0 8px; color:#cdd3df; }}
  .card {{ max-width:860px; background:#12151c; border:1px solid #2a2f3a; border-radius:8px; padding:14px 18px; font-size:13px; line-height:1.6; color:#c2c8d4; }}
  .card ul {{ margin:6px 0; padding-left:20px; }}
  code {{ background:#1c2230; padding:1px 5px; border-radius:4px; font-size:12px; color:#9fd0ff; }}
  .flow {{ font-family:ui-monospace, monospace; font-size:12px; white-space:pre-wrap; color:#9aa3b2; background:#0c0e13; border-radius:6px; padding:10px 12px; }}
</style></head>
<body>
<h1>CASP17 submission viewers</h1>
<div class="meta">{n} target(s) · click a target to open its 3D viewer · fully offline</div>
<table>
<tr><th>Target</th><th>Type</th><th>MODELs</th><th>Ligands</th><th>Binding sites</th><th>Best LSCORE</th><th>AFFNTY</th></tr>
{rows}
</table>

<h2>Scoring</h2>
<div class="card">
  <p><b>LSCORE</b> — per-ligand pose confidence in [0,1], higher = better. Derivation differs by entity:</p>
  <ul>
    <li><b>protein-ligand</b>: <code>LSCORE = 1 − P(RMSD &gt; 2Å)</code> from the RMSD-Pred GNN (pose-reliability predictor).</li>
    <li><b>RNA-ligand</b>: cofolding model confidence (RMSD-Pred is protein-trained, skipped). Boltz = <code>confidence_score</code>; Protenix = <code>0.8·ipTM + 0.2·(pLDDT/100)</code>; AF3 = <code>0.8·ipTM + 0.2·pTM</code>. Clipped to [0,1].</li>
  </ul>
  <p><b>AFFNTY</b> — log-space ensemble dissociation constant Kd (nM): <code>log10(Kd_nM) = 9 − pKd</code>, ensembled across BA-Pred pKd + Boltz affinity head, filtered to binder_prob ≥ 0.5. Emitted once per MODEL. Not reported for RNA (BA-Pred is protein-trained).</p>
</div>

<h2>Pipeline by entity</h2>
<div class="card">
  <p><b>protein-ligand</b> (T*/L* targets):</p>
  <div class="flow">cofold (Boltz2 · Boltz2x · Protenix · AF3, 5 seeds × 5 samples)
→ union templates (MMseqs2 ∪ Foldseek) → consensus pockets
→ multi-track docking (Vina · AutoDock-GPU × {{cofolding, swinsite, p2rank, template_consensus}} + PxDock; template-box + lig-align)
→ post-analysis (BA-Pred · RMSD-Pred per pose)
→ diversity ranking → LG submission (top-5 MODELs)</div>
  <p style="margin-top:12px"><b>RNA-ligand</b> (R* targets):</p>
  <div class="flow">cofold only (Boltz2 · Protenix · AF3, 5 seeds × 5 samples; AF3 unified RNA MSA via nhmmer + Rfam + RNAcentral)
→ USalign whole-system alignment
→ ligand-pose clustering → LG submission (top-5 MODELs)
— template search / docking / post-analysis OFF (RCSB DBs + Vina/ADG/PxDock are protein-only)</div>
  <p style="margin-top:14px"><b>Clustering &amp; MODEL export</b> — how the up-to-5 MODELs are picked:</p>
  <ul>
    <li><b>protein-ligand</b> — <code>select_diverse_top_k</code> (greedy diversity rank): pool every docked/cofold pose, sort by LSCORE desc, take the top pose as MODEL 1, then walk down accepting a pose only if its heavy-atom RMSD is ≥ 2.0 Å to <em>every</em> already-chosen MODEL; stop at 5. Direct numpy heavy-atom RMSD (no symmetry correction). Fallback: BA-Pred pKd desc if no LSCORE. AFFNTY = log-space ensemble Kd per MODEL.</li>
    <li><b>RNA-ligand</b> — greedy single-link clustering: pool cofold samples sorted by confidence desc; first sample seeds cluster 1; each next sample joins the <em>first</em> cluster whose seed-vs-sample heavy-atom RMSD &lt; 3.0 Å, else spawns a new cluster (seed = highest-conf member). Clusters sorted by best confidence; the top-5 cluster seeds become MODEL 1..5 (MODEL 1 = strongest cluster).</li>
  </ul>
</div>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-root", type=Path, nargs="*",
                    default=[Path("experiments/CASP17")],
                    help="Director(ies) to scan for *_LCDD.lg. Default: the CASP17 tree "
                         "(avoids the large novel2025 tree). Pass 'experiments' for all.")
    ap.add_argument("--output-dir", type=Path, default=Path("viz"))
    ap.add_argument("--template-structs", type=int, default=0, metavar="N",
                    help="Embed the top-N template structures (by TM) as real "
                         "cartoon+ligand overlays via USalign re-alignment. "
                         "0 (default) = centroid clouds only (fast, no USalign).")
    ap.add_argument("--lg", type=Path, nargs="*", default=None,
                    help="Explicit LG file(s) instead of scanning.")
    args = ap.parse_args()

    out = args.output_dir
    (out / "assets").mkdir(parents=True, exist_ok=True)
    asset = out / "assets" / "3Dmol-min.js"
    if not asset.exists():
        raise SystemExit(f"3Dmol asset missing: {asset}\n  download: curl -fsSL -o {asset} {ASSET_URL}")

    scan_roots = list(args.scan_root)
    if args.lg:
        lg_files = [p for p in args.lg if p.exists()]
    else:
        lg_files = []
        for sr in scan_roots:
            if sr.exists():
                lg_files += list(sr.rglob("*_LCDD.lg"))
        lg_files = sorted(set(lg_files))
    if not lg_files:
        raise SystemExit(f"No *_LCDD.lg submissions found under {scan_roots}")

    by_target: dict[str, Path] = {}
    for p in lg_files:
        t = p.stem.replace("_LCDD", "")
        if t not in by_target or p.stat().st_mtime > by_target[t].stat().st_mtime:
            by_target[t] = p

    entries = []
    for target, lg in sorted(by_target.items()):
        parsed = parse_lg(lg)
        roots = _target_run_roots(lg, target, scan_roots)
        bsites = find_binding_sites(roots, target)
        metrics = load_pose_metrics(roots)
        templates = find_template_pockets(roots, target)
        anchors = template_anchors(templates)
        pose_groups = build_pose_groups(roots, metrics, anchors=anchors)
        cofold_receptors = load_cofold_receptors(roots)
        tstructs = load_template_structures(templates, args.template_structs)
        research = None
        for r in roots:
            rc = templates.get("reference_cif")
            research = load_research(r, Path(rc) if rc else None)
            if research:
                break
        (out / f"{target}.html").write_text(
            build_target_html(parsed, bsites, metrics, pose_groups,
                              cofold_receptors, ASSET_REL, templates, tstructs,
                              research, anchors))
        if tstructs:
            print(f"    + {len(tstructs)} embedded template structures")
        ligs = parsed["models"][0]["ligands"] if parsed["models"] else []
        best = next((m["ligands"][0]["lscore"] for m in parsed["models"]
                     if m["ligands"] and m["ligands"][0]["lscore"] is not None), None)
        aff = next((m["affinity"] for m in parsed["models"] if m["affinity"]), None)
        n_metrics = sum(1 for m in parsed["models"] for lg2 in m["ligands"]
                        if metrics.get(_norm_pose(lg2["title"])))
        entries.append({"target": target, "n_models": len(parsed["models"]),
                        "n_ligands": len(ligs), "n_sites": len(bsites["sites"]),
                        "best_lscore": f"{best:.3f}" if best is not None else "-",
                        "affinity": aff or "-"})
        is_protein = (n_metrics > 0 or len(bsites["sites"]) > 0)
        entries[-1]["kind"] = "protein-lig" if is_protein else "RNA-lig"
        n_extra = sum(len(g["poses"]) for g in pose_groups)
        print(f"  {target}: {len(parsed['models'])} MODELs, {len(ligs)} ligand(s), "
              f"{len(bsites['sites'])} bsites, {n_metrics} pose-metrics, "
              f"{len(pose_groups)} pose-groups/{n_extra} extra-poses, "
              f"{len(cofold_receptors)} cofold-receptors -> {target}.html")

    (out / "index.html").write_text(build_index_html(entries))
    print(f"\nWrote {len(entries)} viewer(s) + index -> {out/'index.html'}")


if __name__ == "__main__":
    main()
