#!/usr/bin/env python3
"""Build a browsable 3D viewer for a CASP17 ligand-series stage-2 submission.

The stage-2 answer is 109 files, each holding five complete complexes, and the
only questions that matter about them are spatial: does the fragment sit in a
pocket or inside the protein, do the copies of a multi-copy fragment occupy
different sites or the same one twice, is the cofactor where a cofactor should
be. A lint answers those numerically -- it says 2.21 Aa and nothing else -- so
the numbers are what got fixed, and this page is how the fix gets *looked at*.

Every complex therefore ships with the number that describes it and the picture
that explains the number, side by side, plus the pose the previous build had in
that slot wherever the receptor-clearance fix replaced one. The receptor is
slimmed to what the eye needs at this scale: a backbone trace for the fold, and
every heavy atom of the residues that line the pocket, which are the ones a
clash would be against.

Usage:
    uv run python scripts/make_ligand_series_viewer.py \
        --out viz/ligand_series_stage2_audit.html --prev <dir-of-previous-builds>
    # --series L01 L02 : which series to include (default: both)
    # --prev DIR       : holds <DIR>/L01 and <DIR>/L02 from an earlier build, so
    #                    each replaced pose can be shown beside the one it replaced
    # --json-only      : write the payload instead of the page, for size checks
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from make_ligand_series_stage2 import (  # noqa: E402
    COFACTOR_ANCHOR,
    COFACTOR_FLOOR,
    DIVERSITY_RMSD,
    RECEPTOR_FLOOR,
    load_stage2,
    mdl_coords,
    parse_lg,
    pose_rmsd,
    same_arrangement,
)

ASSET = REPO / "viz" / "assets" / "3Dmol-min.js"
STAGE2 = REPO / "experiments" / "ligand_series" / "stage2"

#: Residues with a heavy atom this close to any pose in any MODEL are drawn as
#: sticks. 6.5 Aa is the first contact shell plus a little: near enough that
#: everything shown is arguably touching the ligand, wide enough that a pose
#: sitting in the *wrong* place still has visible walls around it.
POCKET_RADIUS = 6.5

#: Coordinates ship as tenths of an Angstrom. The judgement this page supports
#: is "pocket or protein interior", which 0.1 Aa answers with room to spare,
#: and the integers cost a third of what full-precision floats would.
SCALE = 10


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def receptor_atoms(model) -> list[tuple[str, str, int, float, float, float]]:
    """(atom name, residue name, residue seq, x, y, z) for the receptor heavy
    atoms of one MODEL, in file order."""
    out = []
    for ln in model.prelude:
        if not ln.startswith(("ATOM  ", "HETATM")):
            continue
        if ln[76:78].strip() == "H":
            continue
        out.append((ln[12:16].strip(), ln[17:20].strip(), int(ln[22:26]),
                    float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def q(v: float) -> int:
    return int(round(v * SCALE))


def slim_receptor(atoms, ligand_pts) -> dict:
    """Split the receptor into a backbone trace and the pocket lining.

    Both halves are emitted as flat integer arrays rather than PDB text, which
    the page reassembles into two 3Dmol models -- one drawn as cartoon, one as
    sticks. Keeping them separate is what lets the pocket residues carry their
    full sidechains without the cartoon trying to thread through them.
    """
    lig = np.asarray(ligand_pts, dtype=float) if len(ligand_pts) else None

    # backbone, one entry per residue that has a complete N/CA/C/O set
    byres: dict[int, dict] = {}
    for name, rname, ri, x, y, z in atoms:
        byres.setdefault(ri, {"name": rname, "at": {}})["at"][name] = (x, y, z)

    bb: list[int] = []
    rn: list[str] = []
    ri_list: list[int] = []
    for ri in sorted(byres):
        rec = byres[ri]
        if not all(k in rec["at"] for k in ("N", "CA", "C", "O")):
            continue
        for k in ("N", "CA", "C", "O"):
            bb += [q(c) for c in rec["at"][k]]
        rn.append(rec["name"])
        ri_list.append(ri)

    # pocket lining: every heavy atom of any residue that reaches the ligand
    pocket_res: set[int] = set()
    if lig is not None:
        pts = np.asarray([a[3:6] for a in atoms], dtype=float)
        d = np.sqrt(((pts[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(1)
        for (name, rname, rix, *_), dist in zip(atoms, d):
            if dist <= POCKET_RADIUS:
                pocket_res.add(rix)

    pa: list[str] = []
    prn: list[str] = []
    pri: list[int] = []
    pc: list[int] = []
    for name, rname, rix, x, y, z in atoms:
        if rix in pocket_res:
            pa.append(name)
            prn.append(rname)
            pri.append(rix)
            pc += [q(x), q(y), q(z)]

    return {"bb": bb, "rn": "".join(f"{r:<3}" for r in rn), "ri": ri_list,
            "pa": " ".join(pa), "prn": "".join(f"{r:<3}" for r in prn),
            "pri": pri, "pc": pc}


def min_dist(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return float("inf")
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, list[int]]:
    """Shortest distance between two atom sets, and the two atoms that make it.

    The page draws that contact as a labelled line. A clearance figure in a
    table is a claim; the same figure drawn between the two atoms it came from
    is checkable -- you can see whether the contact is a sensible pocket
    contact or a pose passing through a sidechain.
    """
    if not len(a) or not len(b):
        return float("inf"), []
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(d.argmin(), d.shape)
    return float(d[i, j]), [q(v) for v in a[i]] + [q(v) for v in b[j]]


def centroid(pts) -> tuple[float, float, float] | None:
    if not pts:
        return None
    a = np.asarray(pts, dtype=float)
    return tuple(a.mean(0))


def build_complex(path: Path, spec: dict, prev: Path | None,
                  rescan: bool = False) -> dict:
    header, models = parse_lg(path.read_text())
    target = path.name.split("LG")[0]
    comps = spec["components"]
    # the fragment is the component that is not the shared cofactor; the CSV
    # lists it first, but keying off "has more than one atom in the SMILES" is
    # what actually distinguishes it from ZN.
    binder = comps[0]
    cofactor_codes = {c["code"] for c in comps[1:]}

    method = next((h[7:].strip() for h in header if h.startswith("METHOD")), "")

    prev_models = None
    if prev is not None:
        p = prev / path.name
        if p.exists():
            _, prev_models = parse_lg(p.read_text())

    rec_atoms = receptor_atoms(models[0])
    rec_pts = np.asarray([a[3:6] for a in rec_atoms], dtype=float)

    # Do the five MODELs really ship the same receptor? The viewer draws one
    # of them next to all five poses, so if they ever diverge the picture
    # would be lying about which structure a pose was measured against.
    rec_drift = 0.0
    for m in models[1:]:
        other = np.asarray([a[3:6] for a in receptor_atoms(m)], dtype=float)
        if other.shape == rec_pts.shape:
            rec_drift = max(rec_drift, float(np.abs(other - rec_pts).max()))
        else:
            rec_drift = float("inf")

    pool: list[str] = []

    def pool_id(mdl: str) -> int:
        try:
            return pool.index(mdl)
        except ValueError:
            pool.append(mdl)
            return len(pool) - 1

    # inventory of the previous build's fragment poses, for the "what changed"
    # overlay: every centroid it shipped, and the ones the current build no
    # longer carries anywhere
    prev_pool = prev_centroids = None
    prev_dr: list[float] = []
    prev_dropped: list[tuple[np.ndarray, float, int]] = []
    if prev_models is not None:
        prev_pool = [lg for pm in prev_models for lg in pm.ligands
                     if lg.name not in cofactor_codes]
        prev_centroids = [np.array(centroid(mdl_coords(lg.mdl))) for lg in prev_pool]
        now_centroids = [np.array(centroid(mdl_coords(lg.mdl)))
                         for mm in models for lg in mm.ligands
                         if lg.name not in cofactor_codes]
        for lg, c in zip(prev_pool, prev_centroids):
            if all(float(np.linalg.norm(c - n)) > 0.5 for n in now_centroids):
                prev_dropped.append(
                    (c, round(min_dist(np.asarray(mdl_coords(lg.mdl), dtype=float),
                                       rec_pts), 2), pool_id(lg.mdl)))
        # The previous build's own per-MODEL clearances, read straight off its
        # MODELs. Reconstructing them by patching the current slots only works
        # while the two builds agree on order, and anchoring broke that.
        prev_dr = [round(min((min_dist(np.asarray(mdl_coords(lg.mdl), dtype=float),
                                       rec_pts)
                              for lg in pm.ligands
                              if lg.name not in cofactor_codes),
                             default=float("inf")), 2)
                   for pm in prev_models]

    all_lig_pts: list[tuple[float, float, float]] = []
    out_models = []
    for mi, m in enumerate(models):
        ligs = []
        pts_by_lig = []
        for lg in m.ligands:
            pts = mdl_coords(lg.mdl)
            all_lig_pts += pts
            pts_by_lig.append((lg, np.asarray(pts, dtype=float)))
            ligs.append({"n": lg.number, "c": lg.name, "ls": lg.lscore,
                         "m": pool_id(lg.mdl),
                         "cof": lg.name in cofactor_codes})

        # every distance the lint gates on, recomputed here from the shipped
        # file so the badge and the picture cannot disagree
        for slot, (lg, pts) in zip(ligs, pts_by_lig):
            d, pair = closest_pair(pts, rec_pts)
            slot["dr"] = round(d, 2)
            slot["pair"] = pair
        pairs = [(i, j) for i in range(len(pts_by_lig))
                 for j in range(i + 1, len(pts_by_lig))]
        dl = min((min_dist(pts_by_lig[i][1], pts_by_lig[j][1]) for i, j in pairs),
                 default=float("inf"))

        frag = [s for s in ligs if not s["cof"]]
        cof = [s for s in ligs if s["cof"]]
        # Distance from the fragment to the cofactor: the cofactor marks the
        # functional site (catalytic Zn for L01, the SAM pocket for L02), so
        # this is the one number that says whether a pose is anywhere near the
        # chemistry rather than merely clear of the protein.
        cof_pts = np.vstack([p for s, (lg, p) in zip(ligs, pts_by_lig)
                             if s["cof"] and len(p)]) if cof else None
        dfc = float("inf")
        if cof_pts is not None:
            dfc = min((min_dist(p, cof_pts)
                       for s, (lg, p) in zip(ligs, pts_by_lig)
                       if not s["cof"] and len(p)), default=float("inf"))
        rec = {
            "ligs": ligs,
            "aff": m.affnty,
            "dr": round(min((s["dr"] for s in frag), default=float("inf")), 2),
            "dcof": round(min((s["dr"] for s in cof), default=float("inf")), 2),
            "dfc": round(dfc, 2) if dfc != float("inf") else None,
            "dl": round(dl, 2) if dl != float("inf") else None,
            "label": m.label,
        }

        if prev_pool is not None:
            # A pose counts as replaced only if it is absent from the previous
            # build *anywhere* -- MODEL order is anchored on the cofactor site,
            # so a pose that merely changed slots was not replaced, and pairing
            # slot k against slot k would report a rebuild that never happened.
            nf = [lg for lg in m.ligands if lg.name not in cofactor_codes]
            fresh = [lg for lg in nf
                     if all(np.linalg.norm(np.array(centroid(mdl_coords(lg.mdl)))
                                           - c) > 0.5 for c in prev_centroids)]
            if fresh and prev_dropped:
                # pair each new pose with the dropped pose nearest to it
                cn = np.array(centroid(mdl_coords(fresh[0].mdl)))
                near = min(prev_dropped,
                           key=lambda d: float(np.linalg.norm(cn - d[0])))
                rec["prev"] = {"m": [near[2]], "dr": near[1]}
        out_models.append(rec)

    # Nearest neighbour among the other MODELs, per ligand copy. Five MODELs
    # are five chances, so a pose that repeats one already shown spends a slot
    # without adding an answer -- and unlike a clash, nothing about the file
    # itself reveals it.
    by_slot: dict[int, list[tuple[int, str]]] = {}
    for mi, m in enumerate(models):
        for lg in m.ligands:
            if lg.name not in cofactor_codes:
                by_slot.setdefault(lg.number, []).append((mi, lg.mdl))
    for num, entries in by_slot.items():
        for a, (mi, mdl) in enumerate(entries):
            near = min(((pose_rmsd(mdl, other), mj)
                        for b, (mj, other) in enumerate(entries) if b != a),
                       default=(float("inf"), -1))
            slot = next(s for s in out_models[mi]["ligs"] if s["n"] == num)
            slot["nn"] = round(near[0], 2) if near[0] != float("inf") else None
            slot["nnm"] = near[1] + 1 if near[1] >= 0 else None

    # Which earlier MODEL, if any, this MODEL merely repeats. Permutation-
    # invariant: two MODELs holding the same set of copy positions say the same
    # thing even when the copy labels differ, and that is the one duplication
    # with no defence, so the page names the MODEL rather than only counting it.
    frag_by_model = [[lg.mdl for lg in m.ligands if lg.name not in cofactor_codes]
                     for m in models]
    for mi in range(1, len(frag_by_model)):
        for mj in range(mi):
            if same_arrangement(frag_by_model[mi], frag_by_model[mj]):
                out_models[mi]["dupof"] = models[mj].label
                break

    slim = slim_receptor(rec_atoms, all_lig_pts)
    ev = run_evidence(target, target[:3], rec_pts, rescan=rescan)
    # How many distinct structures the pocket evidence rests on. Counted from
    # the pocket file rather than from the shipped clusters, which carry only
    # their top few PDB ids each: taking the union of those said 53 for L02
    # where the run actually aligned 766 entries, and the point of the number
    # is precisely to separate "four entries of one study" from "hundreds of
    # independent structures".
    pj = (RUN_ROOT / target[:3] / "phase2v2" / target /
          "outputs" / "template_pockets" / "template_pockets.json")
    if pj.is_file() and isinstance(ev.get("tmpl"), dict):
        try:
            pk = json.loads(pj.read_text()).get("pockets") or []
            ev["tmpl"]["npdb"] = len({x.get("template_pdb_id", "").upper() for x in pk})
            ev["tmpl"]["npk_total"] = len(pk)
        except Exception:
            pass
    return {
        "run": ev,
        "t": target,
        "code": binder["code"],
        "smiles": binder["smiles"],
        "n": binder["n"],
        "cof": sorted(cofactor_codes),
        "method": method,
        "nres": len(slim["ri"]),
        "natom": len(rec_atoms),
        "drift": round(rec_drift, 3) if rec_drift != float("inf") else None,
        "rec": slim,
        "pool": pool,
        "prev_dr": prev_dr,
        "models": out_models,
    }



# --------------------------------------------------------------------------- #
# Run evidence: where the pipeline looked, and what it looked at
# --------------------------------------------------------------------------- #
#: Pose centroids closer than this collapse into one entry of the sampling map.
#: The map answers "was this site ever sampled", not "what was the pose", so a
#: 2 Aa bin is finer than the question needs and keeps the payload small enough
#: to ship all ~1100 poses per complex instead of a chosen few.
CLOUD_BIN = 2.0

RUN_ROOT = REPO / "experiments" / "ligand_series"

#: Pose sources whose coordinates are *not* in the submitted receptor's frame.
#: Track 2 docks our ligand into the **template's** receptor, so those poses
#: belong to that structure's coordinate system: for a close template they land
#: a few Angstrom out, for a distant homolog 60+ Aa away, and either way they
#: are not describing a site on the protein this page draws. Overlaying them
#: put 1,906 cloud cells in open solvent, up to 77 Aa off, which reads as a
#: rendering fault rather than the frame mismatch it is.
POSE_SOURCE_SKIP = ("template_",)

from extract_template_pockets import _extract_chain_pdb  # noqa: E402


def sdf_centroids(path: Path) -> list[tuple[float, float, float]]:
    """Heavy-atom centroid of every record in a multi-record SDF.

    Deliberately a text scan rather than RDKit: there are ~13,700 of these
    files across the two series and nothing here needs chemistry, only the
    counts line and the coordinates under it.
    """
    out = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if i + 3 >= len(lines):
            break
        try:
            n = int(lines[i + 3][:3])
        except (ValueError, IndexError):
            break
        sx = sy = sz = 0.0
        k = 0
        for j in range(i + 4, min(i + 4 + n, len(lines))):
            ln = lines[j]
            if ln[31:34].strip() == "H":
                continue
            try:
                sx += float(ln[0:10])
                sy += float(ln[10:20])
                sz += float(ln[20:30])
            except ValueError:
                break
            k += 1
        if k:
            out.append((sx / k, sy / k, sz / k))
        nxt = next((j for j in range(i + 4 + n, len(lines)) if lines[j].startswith("$$$$")),
                   None)
        if nxt is None:
            break
        i = nxt + 1
    return out


#: Where extracted run evidence is cached. Scanning every pose SDF costs ~10 s
#: per complex -- half an hour for a rebuild -- and none of it depends on the
#: submission being rebuilt, only on the run directory, which does not change.
#: Cached by target so editing the page costs nothing.
EVIDENCE_CACHE = REPO / "viz" / ".evidence-cache"



#: How many template clusters get their representative structure aligned and
#: shipped. One per cluster, best alignment first: the point is to show a
#: *different* pocket's real ligand each time, not five views of the same one.
TEMPLATE_ALIGN_TOP = 6

#: How many of those also ship the template's own backbone trace. A ligand is
#: a few dozen atoms; a chain is a few hundred, so every trace costs an order
#: of magnitude more payload and they overlay each other anyway once
#: superposed. Three is enough to see whether the fold agrees.
TEMPLATE_TRACE_TOP = 3


def aligned_template_ligands(run: Path, workdir: Path) -> list[dict]:
    """The bound ligand of each top template cluster, moved into our frame.

    The pocket extraction step already superposed every template and kept only
    the ligand *centroid*; a centroid says where, and says nothing about what.
    Re-running USalign for a handful of representatives buys the actual atoms:
    an experimentally observed ligand sitting in the pocket, next to the pose
    we are predicting for it. That is the only ground truth this page can show.

    USalign is re-run rather than reusing a stored transform because none was
    stored -- ``extract_template_pockets.py`` applies R/t to the centroid and
    discards it. The alignment is per host chain (the same single-chain
    extraction the extractor uses), since a whole-CIF transform only places the
    chain USalign happened to fit and sends the other protomers' ligands into
    deep space.
    """
    import gemmi

    from casp17.usalign import run_usalign

    pockets_json = run / "outputs" / "template_pockets" / "template_pockets.json"
    if not pockets_json.is_file():
        return []
    doc = json.loads(pockets_json.read_text())
    ref = Path(doc.get("reference_cif", ""))
    if not ref.is_file():
        return []
    work = run / "outputs" / "template_pockets" / "_extract_work"

    # best-aligned representative of each cluster, distinct pockets first
    best: dict[int, dict] = {}
    for pk in doc.get("pockets", []):
        ci = pk.get("cluster_index")
        if ci is None:
            continue
        if ci not in best or (pk.get("alignment_tmscore") or 0) > (
                best[ci].get("alignment_tmscore") or 0):
            best[ci] = pk
    reps = sorted(best.values(),
                  key=lambda k: (k.get("cluster_index", 99),
                                 -(k.get("alignment_tmscore") or 0)
                                 ))[:TEMPLATE_ALIGN_TOP]

    out: list[dict] = []
    for pk in reps:
        pdb_id = (pk.get("template_pdb_id") or "").lower()
        cif = work / f"{pdb_id}.cif"
        if not cif.is_file():
            continue
        chain_pdb = workdir / f"{pdb_id}_{pk.get('template_chain', 'A')}.pdb"
        try:
            moved = _extract_chain_pdb(cif, pk.get("template_chain", "A"), chain_pdb)
            align = run_usalign(moved or cif, ref)
        except Exception:
            align = None
        finally:
            chain_pdb.unlink(missing_ok=True)
        if align is None:
            continue
        R, tv, tm, rmsd = align
        try:
            st = gemmi.read_structure(str(cif))
        except Exception:
            continue
        want_ccd = (pk.get("ligand_ccd") or "").upper()
        want_chain = (pk.get("ligand_chain") or "").upper()
        stored = np.array([pk.get("centroid_x", 0.0), pk.get("centroid_y", 0.0),
                           pk.get("centroid_z", 0.0)])
        # A chain can hold the same CCD several times -- 7POL chain B carries two
        # copies of 7X9 at different sites -- and the pocket record names the
        # residue only by chain and code, not by which copy. Taking the first
        # match reports the *separation between copies* as if it were frame
        # drift, which is how a run whose frame never moved came out 8.9 Aa
        # stale. So transform every instance and keep the one the record was
        # actually describing: the one nearest its own stored centroid.
        instances: list[tuple[float, list[int], list[str]]] = []
        for model in st:
            for chain in model:
                if want_chain and chain.name.upper() != want_chain:
                    continue
                for res in chain:
                    if res.name.upper() != want_ccd:
                        continue
                    cc: list[int] = []
                    ee: list[str] = []
                    pts = []
                    for a in res:
                        if a.element.atomic_number <= 1:
                            continue
                        v = R @ np.array([a.pos.x, a.pos.y, a.pos.z]) + tv
                        pts.append(v)
                        cc += [q(v[0]), q(v[1]), q(v[2])]
                        ee.append(a.element.name)
                    if pts:
                        instances.append(
                            (float(np.linalg.norm(np.mean(pts, 0) - stored)), cc, ee))
            break
        if not instances:
            continue
        _best, coords, elems = min(instances, key=lambda x: x[0])
        # The delta between where this pocket was recorded and where the same
        # template lands now. Measured against *this pocket's own* stored
        # centroid, not against its cluster's mean -- a cluster centroid is an
        # average over members, so comparing a single representative to it
        # would report the cluster's spread as if it were frame drift.
        fresh = np.asarray(coords, dtype=float).reshape(-1, 3).mean(0) / SCALE
        delta = fresh - stored
        # The template's own backbone, moved by the same transform. Only the
        # CA trace: the question it answers is "does this fold sit on ours",
        # which a trace answers at a tenth of the cost of a full backbone.
        ca: list[int] = []
        if len(out) < TEMPLATE_TRACE_TOP:
            # ``template_chain`` is foldseek's label and is not always a chain
            # this CIF has -- 7POO's records carry '1' against a file whose
            # chains are A and B. The host of the ligand is the chain that
            # matters for the trace, so fall back to it, then to whatever the
            # file offers first.
            names = [c.name.upper() for c in st[0]]
            want_t = (pk.get("template_chain") or "").upper()
            if want_t not in names:
                want_t = (pk.get("ligand_chain") or "").upper()
            if want_t not in names:
                want_t = names[0] if names else ""
            for chain in st[0]:
                if chain.name.upper() != want_t:
                    continue
                for res in chain:
                    a = res.find_atom("CA", "*")
                    if a is None:
                        continue
                    v = R @ np.array([a.pos.x, a.pos.y, a.pos.z]) + tv
                    ca += [q(v[0]), q(v[1]), q(v[2])]

        out.append({"ca": ca, "pdb": pdb_id.upper(), "ccd": want_ccd,
                    "ch": pk.get("ligand_chain"),
                    "tm": round(float(tm), 3), "rmsd": round(float(rmsd), 2),
                    "cl": pk.get("cluster_index"),
                    "off": round(float(np.linalg.norm(delta)), 1),
                    "d": [q(v) for v in delta],
                    "e": " ".join(elems), "c": coords})
    return out

def refresh_poses(target: str, series: str) -> dict | None:
    """Recompute only the pose sampling map of a cached evidence record.

    The mirror of ``refresh_templates``: the template block costs a USalign run
    per cluster and the pose map costs a scan of every SDF the run produced, so
    a fix to one should not pay for the other.
    """
    hit = EVIDENCE_CACHE / f"{target}.json"
    if not hit.is_file():
        return None
    ev = json.loads(hit.read_text())
    poses = RUN_ROOT / series / "phase2v2" / target / "outputs" / "analysis" / "poses"
    if not poses.is_dir():
        return ev
    cells: dict[tuple, int] = {}
    bysrc: dict[str, int] = {}
    total = skipped = 0
    for f in sorted(poses.glob("*.sdf")):
        src = re.sub(r"_seed_\d+$", "", f.stem)
        if src.startswith(POSE_SOURCE_SKIP):
            skipped += 1
            continue
        cs = sdf_centroids(f)
        bysrc[src] = bysrc.get(src, 0) + len(cs)
        total += len(cs)
        for c in cs:
            key = tuple(int(round(v / CLOUD_BIN)) for v in c)
            cells[key] = cells.get(key, 0) + 1
    ev["cloud"] = [[k[0] * int(CLOUD_BIN * SCALE), k[1] * int(CLOUD_BIN * SCALE),
                    k[2] * int(CLOUD_BIN * SCALE), n]
                   for k, n in sorted(cells.items(), key=lambda kv: -kv[1])]
    ev["nposes"] = total
    ev["skipped_files"] = skipped
    ev["bysrc"] = dict(sorted(bysrc.items(), key=lambda kv: -kv[1]))
    hit.write_text(json.dumps(ev, separators=(",", ":")))
    return ev


def refresh_templates(target: str, series: str) -> dict | None:
    """Recompute only the template block of a cached evidence record.

    The pose sampling map costs ~60 s per complex to rebuild and depends on
    nothing that changes when the template code does, so a template fix should
    not have to pay for it. Returns the updated record, or None when there is
    no cache entry to update.
    """
    hit = EVIDENCE_CACHE / f"{target}.json"
    if not hit.is_file():
        return None
    ev = json.loads(hit.read_text())
    run = RUN_ROOT / series / "phase2v2" / target
    clu = run / "outputs" / "template_pockets" / "template_pocket_clusters.json"
    if not clu.is_file():
        return ev
    j = json.loads(clu.read_text())
    ev["tmpl"] = _template_block(j, run)
    _attach_template_ligands(ev, run)
    hit.write_text(json.dumps(ev, separators=(",", ":")))
    return ev


def run_evidence(target: str, series: str, rec_pts: np.ndarray,
                 rescan: bool = False) -> dict:
    """Binding sites, template pockets and the pose sampling map for one run.

    All three live in the frame ``align_cofolding_outputs.py`` puts every cofold
    model into, so they overlay on the receptor this file ships -- but the
    *structure* can still be a different model's prediction, because docking
    prep picks its receptor by mean pLDDT while the submission picks by score.
    ``recdev`` measures that, so the page can say so rather than implying the
    boxes were drawn against the protein on screen.
    """
    run = RUN_ROOT / series / "phase2v2" / target
    hit = EVIDENCE_CACHE / f"{target}.json"
    if not rescan and hit.is_file():
        return json.loads(hit.read_text())
    ev: dict = {}
    prep_path = run / "inputs" / "docking" / "docking_prep_summary.json"
    if prep_path.is_file():
        prep = json.loads(prep_path.read_text())
        ev["sites"] = [
            {"n": name, "c": [q(v) for v in s["center"]],
             "m": s.get("n_members"), "u": s.get("n_unique_models")
             or s.get("n_unique_pdb")}
            for name, s in (prep.get("binding_site_predictions") or {}).items()
            if s.get("center")
        ]
        ev["model"] = prep.get("cofolding_model")
        pdb = Path(prep.get("receptor_pdb", ""))
        if not pdb.is_absolute():
            pdb = run / pdb
        if pdb.is_file():
            dock = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                             for ln in pdb.read_text().splitlines()
                             if ln.startswith(("ATOM  ", "HETATM"))
                             and ln[76:78].strip() != "H"])
            if len(dock) and len(rec_pts):
                d = np.sqrt(((rec_pts[:, None, :] - dock[None, :, :]) ** 2)
                            .sum(-1)).min(1)
                ev["recdev"] = round(float(np.median(d)), 2)

    clu = run / "outputs" / "template_pockets" / "template_pocket_clusters.json"
    if clu.is_file():
        ev["tmpl"] = _template_block(json.loads(clu.read_text()), run)
        _attach_template_ligands(ev, run)

    poses = run / "outputs" / "analysis" / "poses"
    if poses.is_dir():
        cells: dict[tuple, int] = {}
        bysrc: dict[str, int] = {}
        total = 0
        skipped = 0
        for f in sorted(poses.glob("*.sdf")):
            src = re.sub(r"_seed_\d+$", "", f.stem)
            if src.startswith(POSE_SOURCE_SKIP):
                skipped += 1
                continue
            cs = sdf_centroids(f)
            bysrc[src] = bysrc.get(src, 0) + len(cs)
            total += len(cs)
            for c in cs:
                key = tuple(int(round(v / CLOUD_BIN)) for v in c)
                cells[key] = cells.get(key, 0) + 1
        ev["cloud"] = [[k[0] * int(CLOUD_BIN * SCALE), k[1] * int(CLOUD_BIN * SCALE),
                        k[2] * int(CLOUD_BIN * SCALE), n]
                       for k, n in sorted(cells.items(), key=lambda kv: -kv[1])]
        ev["nposes"] = total
        ev["skipped_files"] = skipped
        ev["bysrc"] = dict(sorted(bysrc.items(), key=lambda kv: -kv[1]))
    EVIDENCE_CACHE.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps(ev, separators=(",", ":")))
    return ev


def _template_block(j: dict, run: Path | None = None) -> dict:
    """The cluster summary as the page consumes it.

    ``sup`` marks whether a cluster survives the protomer-copy test the ranker
    applies (``research_prior._cluster_supported``). A cluster built entirely
    from ligands that were moved by another chain's transform has no surviving
    pocket near it and is not a site — the page has to say so, or a reader sees
    a pocket with no MODEL in it and reads that as a sampling gap.
    """
    pts: list = []
    if run is not None:
        try:
            from casp17.research_prior import _pocket_points
            pts = _pocket_points(run)
        except Exception:
            pts = []

    def _sup(k: dict):
        if not pts:
            return None
        try:
            from casp17.research_prior import _cluster_supported
            return bool(_cluster_supported(k["centroid"], pts))
        except Exception:
            return None

    return {
            "n_pockets": j.get("n_pockets"), "n_clusters": j.get("n_clusters"),
            "n_pockets_kept": len(pts) or None,
            "clusters": [
                {"c": [q(v) for v in k["centroid"]], "cl_index": i,
                 "sup": _sup(k),
                 "m": k.get("n_members"),
                 "u": k.get("n_unique_pdb"), "ev": round(k.get("evidence_score", 0), 1),
                 "tm": k.get("best_alignment_tmscore"), "pid": k.get("best_pident"),
                 "tan": k.get("best_tanimoto"),
                 "ccd": (k.get("unique_ccds") or [])[:4],
                 "pdb": sorted({mm.get("template_pdb_id", "").upper()
                                for mm in (k.get("members") or [])})[:6]}
                for i, k in enumerate((j.get("clusters") or [])[:10])],
        }


def _attach_template_ligands(ev: dict, run: Path) -> None:
    """Superpose each cluster's representative and correct the frame.

    Mutates ``ev["tmpl"]`` in place: adds the aligned ligands, shifts the
    cluster centroids onto the current frame, and records the drift.
    """
    wd = EVIDENCE_CACHE / "_work"
    wd.mkdir(parents=True, exist_ok=True)
    try:
        ligs = aligned_template_ligands(run, wd)
    except Exception as exc:                      # never fail the build on it
        ligs = []
        ev["tmpl"]["ligand_error"] = f"{type(exc).__name__}: {exc}"
    ev["tmpl"]["ligands"] = ligs

    # The stored cluster centroids were computed against the reference cif as
    # it stood at extraction time. On 29 of the 80 L01 runs the cofold outputs
    # were re-aligned afterwards, which moved the frame and left those
    # centroids pointing at nothing.
    #
    # The marker therefore goes at the representative's freshly aligned ligand
    # centroid. Shifting the stored centroid by that representative's delta was
    # tried first, to keep the cluster's own average -- but the frame change
    # carries a rotation, so a single translation leaves 2-3 Aa of residual and
    # pushed 89 markers off the protein surface, floating in solvent. An exact
    # position for one member beats an approximately corrected average of all
    # of them, and the members' spread is reported beside it anyway.
    #
    # ``off`` records how far the stored value had drifted, because that is a
    # fact about the run worth seeing rather than a bug to paper over. A
    # cluster whose representative would not re-align keeps no position at all:
    # a marker nobody can vouch for should not be drawn.
    by_cluster = {L["cl"]: L for L in ligs if L.get("cl") is not None}
    drift = []
    for k in ev["tmpl"]["clusters"]:
        rep = by_cluster.get(k.get("cl_index"))
        if rep is None:
            k["c"] = None                 # nothing vouches for this position
            continue
        pts = np.asarray(rep["c"], dtype=float).reshape(-1, 3)
        k["c"] = [int(round(v)) for v in pts.mean(0)]
        k["off"] = rep["off"]
        drift.append(rep["off"])
    ev["tmpl"]["drift"] = round(max(drift), 1) if drift else None

# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
#: What the submission is and when it is due, per series. Held here rather than
#: read from the tarball because these are facts about the *submission* -- the
#: group, the filename CASP expects, the deadline -- that no model file records,
#: and the page is where someone checks them before uploading.
SUBMISSION = {
    "L01": {"deadline": "2026-09-11", "archive": "L01LG129.tgz",
            "receptor": "BFT1 (fragilysin)", "cofactor": "ZN"},
    "L02": {"deadline": "2026-09-10", "archive": "L02LG129.tgz",
            "receptor": "MmaA1", "cofactor": "SFG (시네푼진)"},
}
GROUP = "129"
AUTHOR = "6095-5696-9732"


#: Residues worth marking on the sequence, per series: the chemistry the target
#: is *for*. Everything else the page highlights is derived from the run, so
#: these are the only positions asserted from outside it -- both confirmed by a
#: motif scan of the released sequence and by the CASP17_own research briefing.
SEQ_MARKS = {
    "L01": [(161, "Asp161 — prodomain, 촉매 아연 배위 (aspartate switch)", "warn"),
            (315, "His315 — Zn 배위", "site"), (316, "Glu316 — 촉매 염기", "site"),
            (319, "His319 — Zn 배위", "site"), (325, "His325 — Zn 배위", "site"),
            (333, "Met333 — Met-turn (metzincin 서명)", "site")],
    "L02": [(53, "SAM 모티프 I 시작 (LDVGCGWGG)", "site"),
            (54, "SAM 모티프 I", "site"), (55, "SAM 모티프 I", "site"),
            (56, "SAM 모티프 I", "site"), (57, "SAM 모티프 I", "site"),
            (58, "SAM 모티프 I", "site"), (59, "SAM 모티프 I", "site"),
            (60, "SAM 모티프 I", "site"), (61, "SAM 모티프 I 끝", "site"),
            (253, "Cys253 — redox 게이팅 잔기 (제출물에서는 산화형 CSO)", "warn")],
}


def _receptor_sequence(series: str) -> dict:
    """The released receptor sequence, with the positions worth marking on it.

    Shipped because every other thing this page shows is a coordinate, and a
    coordinate cannot say *which residue* -- the sequence is where a pocket
    lining becomes a list of names you can look up against the literature.
    """
    fa = REPO / "inputs" / "ligand_series" / series / "receptor.fasta"
    if not fa.is_file():
        return {}
    seq = "".join(ln.strip() for ln in fa.read_text().splitlines()
                  if not ln.startswith(">"))
    return {"seq": seq, "n": len(seq),
            "marks": [{"i": i, "t": txt, "k": kind}
                      for i, txt, kind in SEQ_MARKS.get(series, [])]}


def _submission_facts(series: str, n_files: int) -> dict:
    """Submission-level facts, with the archive measured rather than assumed.

    The filename is the one CASP expects (``L01LG129.tgz``), taken from the
    table above rather than rebuilt from parts -- assembling it as
    ``series + group`` is how this first shipped with the size blank, because
    ``L01129.tgz`` does not exist.
    """
    facts = dict(SUBMISSION[series], group=GROUP, author=AUTHOR, files=n_files)
    tgz = STAGE2 / facts["archive"]
    facts["bytes"] = tgz.stat().st_size if tgz.exists() else None
    return facts


def target_files(series_dir: Path, workdir: Path) -> list[tuple[str, Path]]:
    """One readable LG per target, whatever the on-disk packaging is.

    The submitted tarball is one file per MODEL (`<target>LG129_1` ...`_5`) --
    the ligand series is an Ensemble target category and the server rejects
    multi-MODEL members. The viewer wants the five alternates of a complex
    together, so the split members are stitched back into a temporary
    per-target file (header once, MODELs in suffix order). A directory that
    still holds one multi-MODEL file per target is read as-is.
    """
    groups: dict[str, list[Path]] = {}
    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        groups.setdefault(f.name.split("LG")[0], []).append(f)
    out: list[tuple[str, Path]] = []
    for target, fs in sorted(groups.items()):
        if len(fs) == 1:
            out.append((target, fs[0]))
            continue
        fs.sort(key=lambda x: int(x.name.rsplit("_", 1)[1]))
        head, bodies = None, []
        for f in fs:
            t = f.read_text()
            i = t.index("\nMODEL ") + 1
            if head is None:
                head = t[:i]
            bodies.append(t[i:])
        # Keep the member's own basename: downstream re-derives the target id
        # from the filename (``name.split("LG")[0]``), so a name like
        # ``<target>_merged.lg`` silently misses the evidence cache and the
        # template panel comes out empty.
        merged = workdir / fs[0].name
        merged.write_text(head + "".join(bodies))
        out.append((target, merged))
    return out


def build_payload(series: str, prev: Path | None, rescan: bool = False) -> dict:
    spec = load_stage2(series)
    merge_dir = EVIDENCE_CACHE / "_merged" / series
    merge_dir.mkdir(parents=True, exist_ok=True)
    files = target_files(STAGE2 / series, merge_dir)
    complexes = []
    for i, (target, f) in enumerate(files, 1):
        if target not in spec:
            raise SystemExit(f"{target}: not in ligands_stage2.csv")
        complexes.append(build_complex(f, spec[target], prev, rescan=rescan))
        print(f"  [{i}/{len(files)}] {target}", file=sys.stderr)
    return {
        "series": series,
        "scale": SCALE,
        "floors": {"receptor": RECEPTOR_FLOOR, "cofactor": COFACTOR_FLOOR},
        "anchor": COFACTOR_ANCHOR.get(series),
        "diversity": DIVERSITY_RMSD,
        "pocket_radius": POCKET_RADIUS,
        "submission": _submission_facts(series, len(files)),
        "sequence": _receptor_sequence(series),
        "complexes": complexes,
    }


TEMPLATE = REPO / "scripts" / "templates" / "ligand_series_stage2.html"


def render_page(payloads: dict[str, dict]) -> str:
    """Fold the payloads and the 3Dmol runtime into the authored template.

    3Dmol is inlined rather than pulled from a CDN because the page is meant to
    outlive the session that built it -- it gets published as an Artifact and
    opened again weeks later, possibly next to the submission deadline, and a
    viewer that renders a blank box because a script host moved is worse than
    half a megabyte.
    """
    html = TEMPLATE.read_text()
    blob = json.dumps(payloads, separators=(",", ":"))
    # keep the JSON from terminating its own <script> element
    blob = blob.replace("<", "\\u003c").replace("\u2028", "\\u2028")
    return (html
            .replace("__THREEDMOL__", ASSET.read_text())
            .replace("__PAYLOAD__", blob))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=["L01", "L02"],
                    choices=["L01", "L02"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="directory holding <dir>/<series> and the tarballs; "
                         "defaults to experiments/ligand_series/stage2")
    ap.add_argument("--prev", type=Path, default=None,
                    help="directory holding previous builds, as <prev>/<series>")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--rescan", action="store_true",
                    help="ignore the cached run evidence and read the run dirs again")
    ap.add_argument("--rescan-poses", action="store_true",
                    help="refresh only the pose sampling map of the cache")
    ap.add_argument("--rescan-templates", action="store_true",
                    help="refresh only the template block of the cache (skips the "
                         "pose scan, which is the slow half and never changes)")
    args = ap.parse_args()

    if args.build_dir is not None:
        # module-level constant: build_payload/_submission_facts read it directly
        global STAGE2
        STAGE2 = args.build_dir

    for flag, fn, label in ((args.rescan_poses, refresh_poses, "poses"),
                            (args.rescan_templates, refresh_templates, "templates")):
        if not flag:
            continue
        for s in args.series:
            for f in sorted((STAGE2 / s).iterdir()):
                tid = f.name.split("LG")[0]
                if f.name.rsplit("_", 1)[-1] not in ("1", ""):
                    continue          # split packaging: one pass per target
                ev = fn(tid, s)
                print(f"  {label} {tid}: "
                      + ("no cache entry" if ev is None else "ok"),
                      file=sys.stderr)

    payloads = {}
    for s in args.series:
        prev = None
        if args.prev is not None:
            cand = args.prev / s
            prev = cand if cand.is_dir() else None
            if prev is None:
                print(f"  (no previous build for {s} under {args.prev})", file=sys.stderr)
        print(f"{s}:", file=sys.stderr)
        payloads[s] = build_payload(s, prev, rescan=args.rescan)

    if args.json_only:
        args.out.write_text(json.dumps(payloads, separators=(",", ":")))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_page(payloads))
    print(f"{args.out}  {args.out.stat().st_size/1e6:.2f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
