#!/usr/bin/env python3
"""Superpose every RCSB donor of an ordered-solvent target and view them at once.

R2386's prediction is solvent transferred from experimental structures of the
same RNA, so the question that decides the submission is *what does the pooled
evidence actually look like* — which sites every crystal agrees on, which come
from a single entry, and which donors are re-refinements of one dataset voting
twice. That is a spatial question, and a table cannot answer it.

This builds one page holding all donors in a common frame:

  * the frame RNA as a cartoon, plus each donor's RNA as a P-atom trace,
    superposed with the same ``superpose()`` the submission builder uses — so
    what is on screen is the frame the solvent is actually predicted in;
  * every donor's Mg2+ / K+ / Na+ / water, transformed with its RNA, hoverable
    for species / source / B-factor / how many donors independently place a
    site there;
  * per-donor coverage of the target sequence, because the RCSB sequence search
    reports identity over the *aligned span* — a 26-nucleotide designed RNA
    matching at 100 % passes an identity filter that a genuine 390-nucleotide
    donor barely clears. Coverage, not identity, separates the two.

With ``--consensus`` the page also carries the merged sites that
``cluster_solvent_donors.py`` produced from the same donors. That layer answers
the question the raw pile cannot: after collapsing observations that no crystal
could show simultaneously, *how many distinct sites are there and how many
crystals stand behind each one*. Clicking a merged site selects exactly the
donors that support it, so a claim on screen can be traced back to the evidence
in one step.

Usage:
    uv run python scripts/make_solvent_donor_viewer.py \
      --target-id R2386 --sequence inputs/R2386.fasta \
      --templates data/solvent_templates/R2386/*.cif \
      --frame 3G78 \
      --consensus data/solvent_templates/R2386/consensus_sites.json \
      --out viz/standalone/R2386_donors.html
"""
from __future__ import annotations

import argparse
import html
import json
import string
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_solvent_ts_submission import (  # noqa: E402
    NUMBERING_OFFSET, SOLVENT_NAMES, check_offset, load_template, superpose,
)

ASSET = REPO_ROOT / "viz" / "assets" / "3Dmol-min.js"

#: One colour per solvent species. Water stays pale so the ions read first.
SPECIES_COLOR = {"MG": "#2ecc71", "K": "#c084fc", "NA": "#f39c12", "HOH": "#6fb1ff"}

#: The same four darkened for a white background — the dark-theme colours are
#: chosen to glow against #0f1115 and wash out completely on white.
SPECIES_COLOR_LIGHT = {"MG": "#0f8f45", "K": "#7c3aed", "NA": "#c2600a", "HOH": "#1f6fd0"}

#: Display radii, roughly ionic scale. Purely visual.
SPECIES_RADIUS = {"MG": 0.60, "K": 0.85, "NA": 0.65, "HOH": 0.35}

#: PDB element symbol per species (column 77-78).
SPECIES_ELEMENT = {"MG": "MG", "K": "K", "NA": "NA", "HOH": "O"}

#: Two sites this close, in different donors, are the same site seen twice.
SUPPORT_RADIUS = 1.5

#: A structure covering less of the target than this is not the same molecule at
#: full length — a designed RNA, an isolated domain, another organism's intron.
GOOD_COVERAGE = 85.0

#: Backbone RMSD below which two entries are the same RNA conformation. At 0.5 Å
#: the 41 usable R2386 entries collapse to 26 groups, the largest holding 12
#: re-refinements and metal soaks of one crystal.
CLUSTER_CUTOFF = 0.5

_CHAINS = string.ascii_uppercase + string.ascii_lowercase + string.digits

#: Support bins for the merged sites, as (lower bound, dark colour, light
#: colour, label). One crystal is a guess; ten agreeing crystals are not, and a
#: continuous ramp hides exactly that step. Binning also keeps the page fast:
#: each bin is one ``setStyle`` call instead of one per site.
SUPPORT_BINS = (
    (10, "#ff5c5c", "#c81e1e", "10+"),
    (5, "#ffa94d", "#c2600a", "5-9"),
    (3, "#ffe066", "#a17c00", "3-4"),
    (2, "#74c0fc", "#1f6fd0", "2"),
    (1, "#7f8797", "#8a919e", "1"),
)


def _read_sequence(spec: str) -> str:
    p = Path(spec)
    if p.is_file():
        return "".join(ln.strip() for ln in p.read_text().splitlines()
                       if not ln.startswith(">"))
    return spec.strip()


def _atom_line(record: str, serial: int, name: str, resname: str, chain: str,
               resseq: int, xyz, bfac: float, element: str) -> str:
    """One fixed-column PDB record.

    Columns matter here: name 13-16, altLoc 17, resName 18-20, chain 22,
    resSeq 23-26, coordinates 31-54, element 77-78. Getting altLoc's blank
    wrong shifts everything after it, and 3Dmol then reads the chain id out of
    the residue-name field — which is exactly how an earlier version of this
    script emitted traces that never drew.
    """
    an = name if len(name) >= 4 else f" {name:<3}"
    return ("%-6.6s%5d %-4.4s %3.3s %1.1s%4d    %8.3f%8.3f%8.3f  1.00%6.2f"
            "          %2.2s"
            % (record, serial % 100000, an, resname, chain, resseq,
               xyz[0], xyz[1], xyz[2], bfac, element))


#: Backbone atoms that give 3Dmol enough to draw a nucleic-acid ribbon rather
#: than a wire. Bases are omitted: they double the payload and the fold reads
#: from the backbone alone once there is more than one atom per residue.
_BACKBONE = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")


def _chain_payload(residues: dict, resnames: dict[int, str], R, tv) -> dict:
    """A structure's RNA backbone as numbers, not as PDB text.

    Emitting fixed-column PDB for 41 chains costs ~7.7 MB and the artifact has a
    hard size limit; the same coordinates as rounded arrays cost ~2 MB, and the
    page rebuilds the PDB record in JavaScript when a structure is actually
    drawn. ``miss`` marks residues lacking one of the backbone atoms so the
    client can skip them instead of writing zeros.
    """
    nums, letters, xyz = [], [], []
    for num in sorted(residues):
        row = []
        for name in _BACKBONE:
            v = residues[num].get(name)
            if v is None:
                row = []
                break
            w = np.asarray(v, float) @ R + tv
            row += [round(float(w[0]), 2), round(float(w[1]), 2), round(float(w[2]), 2)]
        if not row:
            continue
        nums.append(int(num))
        letters.append(resnames.get(num, "U"))
        xyz += row
    return {"nums": nums, "rn": "".join(letters), "xyz": xyz}


def _solvent_pdb(sites: list, chain: str) -> str:
    """Solvent sites as HETATM so 3Dmol can style and hover them as atoms."""
    return "\n".join(
        _atom_line("HETATM", i, SPECIES_ELEMENT.get(s[0], "O"), s[0], chain, i,
                   (s[1], s[2], s[3]), s[4], SPECIES_ELEMENT.get(s[0], "O"))
        for i, s in enumerate(sites, 1))


def _consensus_payload(path: Path, donors: list[dict]) -> tuple[dict, list[str]]:
    """Package ``cluster_solvent_donors.py``'s merged sites for the page.

    Supporting donors are stored as indices into ``donors`` rather than PDB
    codes, so clicking a site can drive the existing selection machinery
    directly. A donor named in the JSON but absent from the page's list (the two
    steps use different coverage windows) is reported rather than dropped
    silently — an unexplained gap between "3 crystals support this" and two
    highlighted rows is worse than a warning.
    """
    warnings: list[str] = []
    sites = json.loads(path.read_text())
    index = {d["pdb"]: i for i, d in enumerate(donors)}
    missing: set[str] = set()

    # A chain id no donor uses: hover and click identify a merged site by its
    # chain, and donors are handed chains off the same alphabet, so reusing one
    # would make an observation indistinguishable from the consensus.
    used = {_CHAINS[i % len(_CHAINS)] for i in range(len(donors))}
    chain = next((c for c in reversed(_CHAINS) if c not in used), "9")

    kinds, support, nobs, rna, idx, contested = [], [], [], [], [], []
    lines = []
    for i, s in enumerate(sites, 1):
        kind = s["kind"]
        kinds.append(kind)
        support.append(int(s.get("n_donors", 0)))
        nobs.append(int(s.get("n_obs", 0)))
        rna.append(round(float(s.get("rna_min", 0.0)), 2))
        contested.append(1 if len(s.get("votes", {})) > 1 else 0)
        here = []
        for pdb in s.get("donors", ()):
            if pdb in index:
                here.append(index[pdb])
            else:
                missing.add(pdb)
        idx.append(sorted(here))
        lines.append(_atom_line("HETATM", i, SPECIES_ELEMENT.get(kind, "O"), kind,
                                chain, i, [float(v) for v in s["xyz"]],
                                min(float(s.get("rna_min", 0.0)), 999.99),
                                SPECIES_ELEMENT.get(kind, "O")))
    if missing:
        warnings.append(
            f"consensus names {len(missing)} donor(s) the page does not list "
            f"({', '.join(sorted(missing))}) — their support is counted but not "
            f"selectable")
    payload = {
        "pdb": "\n".join(lines),
        "chain": chain,
        "kind": kinds,
        "support": support,
        "n_obs": nobs,
        "rna_min": rna,
        "donors": idx,
        "contested": contested,
        "max_support": max(support) if support else 0,
        "source": path.name,
    }
    return payload, warnings


def _frame_bfactors(path: Path, offset: int) -> dict[int, dict[str, float]]:
    """{target_resnum: {atom_name: B}} for the frame's RNA.

    ``load_template`` keeps coordinates only, but the B-factor is what makes the
    ribbon readable — it says which parts of the fold are ordered enough for the
    solvent around them to mean anything.
    """
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    best, best_n = None, 0
    for chain in st[0]:
        n = sum(1 for r in chain
                if (i := gemmi.find_tabulated_residue(r.name)) and i.is_nucleic_acid())
        if n > best_n:
            best, best_n = chain, n
    out: dict[int, dict[str, float]] = {}
    if best is None:
        return out
    for res in best:
        info = gemmi.find_tabulated_residue(res.name)
        if not (info and info.is_nucleic_acid()):
            continue
        out.setdefault(res.seqid.num + offset,
                       {a.name.strip(): float(a.b_iso) for a in res})
    return out


def _rna_clusters(donors: list[dict], cutoff: float) -> None:
    """Single-link cluster the structures by RNA conformation, in place.

    Most of these entries are the same fold in the same crystal form — several
    are literally re-refinements of one dataset. Pooling their solvent as if
    they were independent inflates every support count, and looking at 41
    superposed traces shows nothing. Clustering on backbone RMSD collapses that
    to a handful of distinct conformations with one representative each.

    Representative = best resolution in the cluster, since that is the entry
    whose solvent is modelled most reliably.
    """
    idx = [i for i, d in enumerate(donors) if d.get("pmap")]
    n = len(idx)
    if not n:
        return
    coords = [donors[i]["pmap"] for i in idx]

    def rmsd(a, b):
        shared = sorted(set(a) & set(b))
        if len(shared) < 20:
            return 9e9
        P = np.asarray([a[k] for k in shared], float)
        Q = np.asarray([b[k] for k in shared], float)
        pc, qc = P.mean(0), Q.mean(0)
        P, Q = P - pc, Q - qc
        U, _s, Vt = np.linalg.svd(P.T @ Q)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = rmsd(coords[i], coords[j])

    seen: set[int] = set()
    groups: list[list[int]] = []
    for i in range(n):
        if i in seen:
            continue
        stack, grp = [i], []
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            grp.append(a)
            stack += [b for b in range(n) if b not in seen and M[a, b] <= cutoff]
        groups.append(grp)

    groups.sort(key=len, reverse=True)
    for cid, grp in enumerate(groups, 1):
        members = [idx[g] for g in grp]
        rep = min(members, key=lambda m: (donors[m]["res"] or 99.0,
                                          -donors[m]["nsolv"]))
        for m in members:
            donors[m]["cluster"] = cid
            donors[m]["is_rep"] = (m == rep)
            donors[m]["cluster_n"] = len(members)


def _support(donors: list[dict]) -> None:
    """Annotate each site with how many *other* good donors place a site there.

    Support is the number of distinct donors within ``SUPPORT_RADIUS``, counted
    only over donors that actually are this molecule (coverage >= GOOD_COVERAGE)
    — a designed RNA's stray ion should not vouch for anything. This is the
    number the submission's ``min_support`` thresholds act on, so showing it per
    site makes the precision/recall models legible instead of abstract.
    """
    good = [d for d in donors if d["cov"] >= GOOD_COVERAGE and d["sites"]]
    if not good:
        return
    clouds = [(d["pdb"], np.asarray([[s[1], s[2], s[3]] for s in d["sites"]], float))
              for d in good]
    for d in donors:
        if not d["sites"]:
            continue
        P = np.asarray([[s[1], s[2], s[3]] for s in d["sites"]], float)
        n = np.zeros(len(P), dtype=int)
        for pdb, Q in clouds:
            if pdb == d["pdb"]:
                n += 1
                continue
            close = (np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
                     <= SUPPORT_RADIUS).any(axis=1)
            n += close
        for s, k in zip(d["sites"], n):
            s.append(int(k))


def build(target_id: str, sequence: str, cifs: list[Path], frame_id: str,
          offset: int, consensus: Path | None = None) -> tuple[dict, list[str]]:
    """Load, superpose and package every donor. Returns (payload, warnings)."""
    warnings: list[str] = []
    skipped: list[tuple[str, float, float]] = []
    tpls: dict[str, object] = {}
    for cif in cifs:
        try:
            t = load_template(cif, offset=offset)
        except Exception as exc:                                  # noqa: BLE001
            warnings.append(f"{cif.stem.upper()}: unreadable ({exc})")
            continue
        if not t.residues:
            warnings.append(f"{cif.stem.upper()}: no RNA chain")
            continue
        tpls[t.pdb_id] = t

    if frame_id.upper() not in tpls:
        raise SystemExit(f"frame {frame_id} not among the loaded donors")
    frame = tpls[frame_id.upper()]

    donors = []
    for pdb, t in sorted(tpls.items()):
        matched, total = check_offset(t, sequence)
        cov = 100.0 * matched / len(sequence)
        ident = (matched / total) if total else 0.0
        if pdb == frame.pdb_id:
            R, tv, rmsd, nsh = np.eye(3), np.zeros(3), 0.0, len(t.residues)
        else:
            try:
                R, tv, rmsd, nsh = superpose(t, frame)
            except Exception as exc:                              # noqa: BLE001
                warnings.append(f"{pdb}: not superposed ({exc})")
                continue
        trace = {n: tuple(np.asarray(a["P"], float) @ R + tv)
                 for n, a in t.residues.items() if "P" in a}
        resnames = dict(t.resnames)
        sites = []
        for kind, x, y, z, b in t.solvent:
            v = np.asarray([x, y, z], float) @ R + tv
            sites.append([kind, round(float(v[0]), 2), round(float(v[1]), 2),
                          round(float(v[2]), 2), round(float(b), 1)])
        counts = {k: sum(1 for s in sites if s[0] == k) for k in SOLVENT_NAMES}
        if cov < GOOD_COVERAGE:
            # Not this molecule at full length. It could never be selected, so
            # listing it only invited the question of why it was greyed out.
            skipped.append((pdb, round(cov, 1), t.resolution))
            continue
        donors.append({
            "pdb": pdb,
            "res": round(t.resolution, 2) if t.resolution else None,
            "cov": round(cov, 1),
            "ident": round(ident, 3),
            "nt": len(t.residues),
            "rmsd": round(float(rmsd), 2),
            "nshared": int(nsh),
            "counts": counts,
            "nsolv": len(sites),
            "sites": sites,
            "pmap": trace,
            "chain": _chain_payload(t.residues, resnames, R, tv),
        })

    donors.sort(key=lambda d: (-d["cov"], -d["nsolv"]))
    _rna_clusters(donors, CLUSTER_CUTOFF)
    _support(donors)
    # Cluster representatives get their real chain (backbone ribbon); the rest
    # keep the P trace. A non-representative is within CLUSTER_CUTOFF of its
    # representative by construction, so drawing its full backbone as well
    # would add megabytes to show the same curve twice.
    for d in donors:
        d.pop("pmap", None)
    for i, d in enumerate(donors):
        d["solvent_pdb"] = _solvent_pdb(d["sites"], _CHAINS[i % len(_CHAINS)])

    # Every atom of the frame, with real residue names: 3Dmol builds a nucleic
    # cartoon from the backbone *and* the bases, so a six-atom-per-residue
    # skeleton under a single fake residue name renders as bare wire. ~8k atoms
    # for one 389-nt chain is cheap; the donors stay P-only traces on purpose.
    bmap = _frame_bfactors(frame.path, offset)
    frame_pdb, bvals = [], []
    for num in sorted(frame.residues):
        resn = frame.resnames.get(num, "U")
        for name, xyz in frame.residues[num].items():
            elem = name.lstrip("0123456789")[:1] or "C"
            b = float(bmap.get(num, {}).get(name, 0.0))
            bvals.append(b)
            frame_pdb.append(_atom_line(
                "ATOM", len(frame_pdb) + 1, name, resn, "A", num, xyz, b, elem))
    barr = np.asarray(bvals, float) if bvals else np.zeros(1)
    payload = {
        "target": target_id,
        "frame": frame.pdb_id,
        "frame_pdb": "\n".join(frame_pdb),
        "frame_b": [round(float(np.percentile(barr, 5)), 1),
                    round(float(np.percentile(barr, 95)), 1)],
        "seq_len": len(sequence),
        "support_radius": SUPPORT_RADIUS,
        "good_coverage": GOOD_COVERAGE,
        "cluster_cutoff": CLUSTER_CUTOFF,
        "skipped": sorted(skipped, key=lambda r: -r[1]),
        "n_clusters": len({d["cluster"] for d in donors if d.get("cluster")}),
        "donors": donors,
    }
    if consensus is not None:
        payload["consensus"], cwarn = _consensus_payload(consensus, donors)
        warnings += cwarn
    return payload, warnings


PAGE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
         background:#0f1115; color:#e6e8ee; }}
  @media (prefers-color-scheme: light) {{ body {{ background:#f6f7f9; color:#1a1d23; }} }}
  :root[data-theme="light"] body {{ background:#f6f7f9; color:#1a1d23; }}
  :root[data-theme="dark"] body {{ background:#0f1115; color:#e6e8ee; }}
  #wrap {{ display:flex; height:100vh; width:100%; }}
  #panel {{ width:400px; min-width:280px; max-width:62vw; overflow:auto; padding:0 0 20px;
            border-right:1px solid #2a2f3a; }}
  #view {{ flex:1; position:relative; min-width:0; }}
  #panelShow {{ position:absolute; top:10px; left:10px; z-index:4; }}
  /* The header carries the two controls used constantly — species filter and
     background — so they stay reachable from anywhere in a long panel. */
  .ptop {{ position:sticky; top:0; z-index:3; background:#0f1115; padding:11px 13px 7px;
           border-bottom:1px solid #2a2f3a; }}
  .pbody {{ padding:0 13px; }}
  .titlebar {{ display:flex; align-items:flex-start; gap:8px; }}
  .titlebar h1 {{ flex:1; }}
  h1 {{ font-size:15px; margin:0 0 2px; }}
  .sub {{ color:#9aa3b2; font-size:11.5px; margin-bottom:10px; }}
  .sec {{ margin:13px 0 5px; font-weight:600; font-size:11px; letter-spacing:.06em;
          text-transform:uppercase; color:#9aa3b2; }}
  /* Sections collapse. Nine of them expanded at once is a 4000 px scroll in
     which the control you want is never on screen with the thing it changes. */
  details.acc {{ border-bottom:1px solid #20252f; }}
  details.acc > summary {{ cursor:pointer; list-style:none; padding:9px 2px 8px;
    font-weight:600; font-size:11px; letter-spacing:.06em; text-transform:uppercase;
    color:#9aa3b2; display:flex; align-items:center; gap:7px; }}
  details.acc > summary::-webkit-details-marker {{ display:none; }}
  details.acc > summary::before {{ content:"\\203A"; color:#5c6473; font-size:14px;
    line-height:1; transition:transform .15s ease; }}
  details.acc[open] > summary::before {{ transform:rotate(90deg); }}
  details.acc > summary:hover {{ color:#e6e8ee; }}
  details.acc > summary:focus-visible {{ outline:2px solid #6fb1ff; outline-offset:2px; }}
  .cnt {{ margin-left:auto; font-weight:500; letter-spacing:0; text-transform:none;
          font-size:10.5px; color:#6b7482; font-variant-numeric:tabular-nums; }}
  .accbody {{ padding:0 0 11px; }}
  @media (prefers-reduced-motion: reduce) {{
    details.acc > summary::before {{ transition:none; }}
  }}
  .btn {{ background:#1a1f2b; color:#e6e8ee; border:1px solid #2f3745; border-radius:6px;
          padding:3px 9px; margin:0 4px 4px 0; cursor:pointer; font-size:11px; }}
  .btn:hover {{ background:#232a38; }}
  .btn.on {{ border-color:#6fb1ff; color:#fff; background:#243044; }}
  .row {{ display:flex; align-items:center; gap:6px; padding:2px 0; font-size:12px;
          white-space:nowrap; cursor:pointer; }}
  .row:hover {{ background:#1a1f2b; }}
  .row.dim {{ opacity:.45; }}
  .row input {{ flex:none; }}
  /* 41 structures read as a table, not as a pile of chips: fixed columns and
     tabular figures so resolution, coverage and solvent count compare down the
     list at a glance. */
  .drow, .dhead {{ display:grid; grid-template-columns:15px 10px 1fr 42px 40px 42px;
                   align-items:center; gap:6px; padding:1px 3px; border-radius:5px;
                   font-size:12px; white-space:nowrap; }}
  .dhead {{ font-size:10px; letter-spacing:.05em; text-transform:uppercase;
            color:#6b7482; padding-top:6px; padding-bottom:2px; }}
  .drow:hover {{ background:#1a1f2b; }}
  .drow.dim {{ opacity:.45; }}
  .drow .n, .dhead .n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .drow .oneBtn {{ margin:0; padding:2px 7px; text-align:left; }}
  .cbadge {{ margin-left:5px; font-size:9.5px; color:#8b93a2; }}
  .pill {{ font-size:10px; padding:0 5px; border-radius:8px; background:#232a38; color:#9aa3b2; }}
  .warn {{ color:#ffcf6b; }}
  table.stat {{ border-collapse:collapse; font-size:11.5px; width:100%; }}
  table.stat td {{ padding:1px 5px 1px 0; }}
  table.stat td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%; flex:none; }}
  dl.gloss {{ margin:0; font-size:11.5px; }}
  dl.gloss dt {{ font-weight:600; margin-top:7px; }}
  dl.gloss dd {{ margin:1px 0 0; color:#9aa3b2; }}
  .intro {{ font-size:11.5px; color:#9aa3b2; background:#161b24; border:1px solid #262d3a;
            border-radius:7px; padding:7px 9px; margin:8px 0 0; }}
  .intro b {{ color:#e6e8ee; }}
  .bbar {{ display:inline-block; width:64px; height:8px; border-radius:4px; vertical-align:-1px;
           background:linear-gradient(90deg,#2c5fd8,#e8e8e8,#d84a2c); }}
  .cgrp {{ font-size:11.5px; padding:2px 4px; border-radius:6px;
           border:1px solid transparent; }}
  .cgrp.on {{ border-color:#6fb1ff; background:#182231; }}
  .divider {{ margin:8px 0 4px; padding:5px 7px; font-size:11px; line-height:1.4;
              color:#ffcf6b; background:#241d10; border:1px solid #4d3d16;
              border-radius:6px; white-space:normal; }}
  .cgrp .rep {{ font-weight:700; }}
  .cgrp .rest {{ color:#7f8797; }}
  .sl {{ display:flex; align-items:center; gap:8px; margin:5px 0 3px; font-size:11.5px; }}
  .sl input[type=range] {{ flex:1; accent-color:#6fb1ff; min-width:80px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:2px 9px; margin:5px 0 0;
             font-size:10.5px; color:#9aa3b2; }}
  .legend .k {{ display:inline-flex; align-items:center; gap:4px; }}
  #consInfo {{ margin:6px 0 0; padding:6px 8px; border-radius:7px; font-size:11.5px;
               background:#161b24; border:1px solid #262d3a; white-space:normal; }}
  #consInfo.empty {{ color:#7f8797; }}
  #consInfo b {{ color:#e6e8ee; }}
  .mono {{ font-variant-numeric:tabular-nums; }}
  @media (prefers-color-scheme: light) {{
    #panel {{ border-right-color:#d8dce4; }}
    .ptop {{ background:#f6f7f9; border-bottom-color:#d8dce4; }}
    details.acc {{ border-bottom-color:#e2e6ee; }}
    .drow:hover {{ background:#eef1f6; }}
    .cbadge {{ color:#6b7280; }}
    .btn {{ background:#fff; color:#1a1d23; border-color:#c9cfdb; }}
    .btn.on {{ background:#e3edff; border-color:#5a9bf6; }}
    .pill {{ background:#e9edf4; color:#5a6373; }}
    .row:hover {{ background:#eef1f6; }}
    dl.gloss dd {{ color:#5a6373; }}
    .intro {{ background:#eef2f8; border-color:#d3dae6; color:#4a5262; }}
    .intro b {{ color:#1a1d23; }}
    .cgrp.on {{ background:#e3edff; border-color:#5a9bf6; }}
    .divider {{ color:#7a5a10; background:#fdf5e0; border-color:#e2cf9a; }}
    .cgrp .rest {{ color:#7a8291; }}
    #consInfo {{ background:#eef2f8; border-color:#d3dae6; }}
    #consInfo b {{ color:#1a1d23; }}
    .legend {{ color:#5a6373; }}
  }}
</style>
<div id="wrap">
  <div id="panel">
    <div class="ptop">
      <div class="titlebar">
        <h1>{title}</h1>
        <button class="btn" id="panelHide" title="패널 접기">&#9664;</button>
      </div>
      <div class="sub" id="hdr" style="margin-bottom:7px"></div>
      <div id="spBtns"></div>
      <div style="margin-top:3px"><button class="btn" id="bgBtn">배경: 어두움</button></div>
    </div>
    <div class="pbody">
      <div class="intro" id="intro"></div>

      <details class="acc" id="consSec" open>
        <summary>합의 site<span class="cnt"><span id="nCons"></span>개</span></summary>
        <div class="accbody">
          <div class="sub" style="margin:0 0 6px">관측을 <b>물리적으로 공존할 수 없는 거리</b>에서
            병합한 결과다. 두 관측이 그 종류 쌍이 실제로 가질 수 있는 최단 거리보다 가까우면
            같은 자리로 본다 — 남은 것들은 한 모델에 동시에 써도 서로 부딪히지 않는다.</div>
          <div class="row"><input type="checkbox" id="showCons" checked><label for="showCons">합의 site 표시</label></div>
          <div class="sl"><label for="supMin">지지 구조 <span class="mono" id="supMinL">1</span>개 이상</label>
            <input type="range" id="supMin" min="1" value="1"></div>
          <div>
            <button class="btn" id="consOnly">합의만 보기</button>
            <button class="btn" id="consAll">관측 다시 켜기</button>
          </div>
          <div class="sub" style="margin:6px 0 2px">색 기준</div>
          <div id="consMode">
            <button class="btn cmBtn" data-m="support">지지 구조 수</button>
            <button class="btn cmBtn" data-m="species">용매 종류</button>
            <button class="btn cmBtn" data-m="rna">RNA 거리</button>
          </div>
          <div class="legend" id="consLeg"></div>
          <div id="consInfo" class="empty">화면에서 합의 site를 클릭하면 그 자리를 지지한
            실험 구조가 여기 나온다.</div>
        </div>
      </details>

      <details class="acc" open>
        <summary>실험 구조<span class="cnt"><span id="nDonor"></span>개</span></summary>
        <div class="accbody">
          <div>
            <button class="btn" id="dAll">전체</button>
            <button class="btn" id="dNone">해제</button>
            <button class="btn" id="dSolv">용매 많은 10개</button>
            <button class="btn" id="dFrame">기준 구조만</button>
          </div>
          <div class="dhead">
            <span></span><span></span><span>PDB</span><span class="n">해상도</span>
            <span class="n">덮음</span><span class="n">용매</span>
          </div>
          <div id="donors"></div>
          <div class="sub" style="margin:5px 0 0">PDB 코드를 누르면 그 구조 하나만,
            체크박스로는 여러 개를 겹쳐 본다. 행에 커서를 올리면 이온·물 내역과
            superposition RMSD가 나온다.</div>
        </div>
      </details>

      <details class="acc">
        <summary>RNA 형태 클러스터<span class="cnt"><span id="ccut"></span> &Aring; · 대표 <span id="nRepSum"></span></span></summary>
        <div class="accbody">
          <div class="sub" style="margin:0 0 5px">백본 RMSD가 이 값 이하면 같은 형태로 묶었다.
            대표는 클러스터에서 해상도가 가장 좋은 구조. <b>같은 클러스터의 구조를 독립
            증거로 세면 지지 수가 부풀려진다.</b></div>
          <div>
            <button class="btn" id="cRep">대표 전부 (<span id="nRep"></span>개)</button>
            <button class="btn" id="cAll">전체 (<span id="nClustered"></span>개)</button>
          </div>
          <div id="clusters"></div>
        </div>
      </details>

      <details class="acc">
        <summary>RNA 표시</summary>
        <div class="accbody">
          <div class="row"><input type="checkbox" id="showFrame" checked><label for="showFrame">기준 구조 리본 (<span id="frameName"></span>)</label></div>
          <div class="row"><input type="checkbox" id="colorB" checked><label for="colorB">리본을 B-factor로 색칠 <span class="pill" id="bLeg"></span></label></div>
          <div class="row"><input type="checkbox" id="colorByDonor"><label for="colorByDonor">용매를 종류 대신 구조별 색으로</label></div>
          <div class="sub" style="margin:5px 0 2px">선택된 구조의 RNA 사슬</div>
          <div id="rnaBtns">
            <button class="btn rnaBtn" data-m="auto">자동</button>
            <button class="btn rnaBtn" data-m="ribbon">리본 (염기 포함)</button>
            <button class="btn rnaBtn" data-m="tube">튜브</button>
            <button class="btn rnaBtn" data-m="line">가는 선</button>
            <button class="btn rnaBtn" data-m="off">끄기</button>
          </div>
          <div class="sub" style="margin:3px 0 0">자동 = 8개 이하면 리본, 그보다 많으면 튜브.
            튜브도 카툰이지만 잔기당 인산 하나만으로 그려서 41개를 다 겹쳐도 가볍다.</div>
          <div style="margin-top:5px"><button class="btn" id="allChains">RNA 전부 리본으로 겹쳐 보기</button></div>
        </div>
      </details>

      <details class="acc" open>
        <summary>지금 화면에 있는 것</summary>
        <div class="accbody"><table class="stat" id="totals"></table></div>
      </details>

      <details class="acc">
        <summary>각 수치의 뜻</summary>
        <div class="accbody"><dl class="gloss" id="gloss"></dl></div>
      </details>

      <details class="acc">
        <summary>이 목록을 어떻게 골랐나</summary>
        <div class="accbody"><div id="notes" class="sub" style="margin:0"></div></div>
      </details>
    </div>
  </div>
  <div id="view"><button class="btn" id="panelShow" hidden>&#9654; 패널</button></div>
</div>
<script>{asset}</script>
<script>
const DATA = {data};
const SPECIES_COLOR_DARK = {species_color};
const SPECIES_COLOR_LIGHT = {species_color_light};
let SPECIES_COLOR = SPECIES_COLOR_DARK;
const SPECIES_RADIUS = {species_radius};
const SP = ["MG", "K", "NA", "HOH"];
const SP_LABEL = {{ MG: "Mg2+", K: "K+", NA: "Na+", HOH: "물" }};
const $ = (id) => document.getElementById(id);
const viewer = $3Dmol.createViewer($("view"), {{ backgroundColor: "#0f1115" }});
function setBg(white) {{
  whiteBg = white;
  SPECIES_COLOR = white ? SPECIES_COLOR_LIGHT : SPECIES_COLOR_DARK;
  viewer.setBackgroundColor(white ? "#ffffff" : "#0f1115");
  // the swatches in the panel have to follow, or the legend lies
  document.querySelectorAll(".spBtn").forEach(b => {{
    const sw = b.querySelector(".swatch");
    if (sw) sw.style.background = SPECIES_COLOR[b.dataset.s];
  }});
  document.querySelectorAll("#totals .swatch").forEach((sw, i) => {{
    sw.style.background = SPECIES_COLOR[SP[i]];
  }});
  const b = $("bgBtn");
  if (b) b.textContent = white ? "배경: 흰색" : "배경: 어두움";
  render();
}}
let soloSpecies = null;          // null = show every species
let rnaMode = "ribbon";          // auto | ribbon | tube | line | off
let whiteBg = false;

// --- merged sites -------------------------------------------------------
// The consensus layer is the clustering result: observations that could not
// physically coexist collapsed into one site. It is optional, so everything
// below no-ops when the page was built without --consensus.
const CONS = DATA.consensus || null;
const SUPPORT_BINS = {support_bins}; // [lower bound, dark, light, label]
// Contact distance to the nearest RNA atom. Below ~2.6 A a water is inside a
// hydrogen bond that is already short; past 4 A it is bulk solvent that no
// assessor will score. Colouring by it shows which sites are actually bound.
// Labels go through innerHTML, so a bare "<" would be read as the start of a
// tag and swallow the rest of the legend.
const RNA_BINS = [[4.0, "#7f8797", "#8a919e", "4.0 &Aring;+ (벌크)"],
                  [3.2, "#74c0fc", "#1f6fd0", "3.2-4.0 &Aring;"],
                  [2.6, "#ffe066", "#a17c00", "2.6-3.2 &Aring;"],
                  [0.0, "#ff5c5c", "#c81e1e", "&lt; 2.6 &Aring; (밀착)"]];
let consMode = "support";        // support | species | rna
let consPick = null;             // index of the clicked site
let supMin = 1;

const binOf = (v, bins) => bins.find(b => v >= b[0]) || bins[bins.length - 1];
const binColor = (b) => whiteBg ? b[2] : b[1];
function consColor(i) {{
  if (consMode === "species") return SPECIES_COLOR[CONS.kind[i]];
  return consMode === "rna" ? binColor(binOf(CONS.rna_min[i], RNA_BINS))
                            : binColor(binOf(CONS.support[i], SUPPORT_BINS));
}}
function consRadius(i) {{
  // Merged sites sit slightly larger than the raw observations so the two
  // layers stay distinguishable when both are on.
  const base = SPECIES_RADIUS[CONS.kind[i]] * 1.15;
  if (consMode !== "support") return base;
  const s = CONS.support[i];
  return base * (s >= 10 ? 1.5 : s >= 5 ? 1.3 : s >= 3 ? 1.15 : s >= 2 ? 1.0 : 0.85);
}}

const donorColor = (i) => "hsl(" + ((i * 47) % 360) + ",70%,60%)";

// Chains ship as rounded coordinate arrays; rebuild the fixed-column PDB the
// first time a structure is drawn and keep it. Emitting the text server-side
// for all 41 chains would have blown the artifact size limit.
const BB = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"];
// NOTE: PAGE is a plain (non-raw) triple-quoted string, so a backslash-n
// written here becomes a real newline in the emitted JS and breaks the
// string literal it sits in. Build newlines with String.fromCharCode(10).
const _chainCache = {{}};
function pad(v, w) {{ return String(v).padStart(w); }}
function chainPdb(i) {{
  if (_chainCache[i]) return _chainCache[i];
  const c = DATA.donors[i].chain, out = [];
  let serial = 0;
  for (let r = 0; r < c.nums.length; r++) {{
    const resn = c.rn[r], num = c.nums[r];
    for (let a = 0; a < BB.length; a++) {{
      const o = (r * BB.length + a) * 3;
      const nm = BB[a], an = (nm.length >= 4 ? nm : " " + nm.padEnd(3));
      serial++;
      out.push("ATOM  " + pad(serial, 5) + " " + an + "   " + resn +
               " A" + pad(num, 4) + "    " +
               pad(c.xyz[o].toFixed(3), 8) + pad(c.xyz[o + 1].toFixed(3), 8) +
               pad(c.xyz[o + 2].toFixed(3), 8) + "  1.00  0.00          " +
               pad(nm[0], 2));
    }}
  }}
  return (_chainCache[i] = out.join(String.fromCharCode(10)));
}}

// The panel can be folded away: on a laptop it takes a third of the width, and
// comparing two structures is done in the 3D view, not in the list.
$("panelHide").onclick = () => {{
  $("panel").style.display = "none";
  $("panelShow").hidden = false;
  setTimeout(() => {{ viewer.resize(); viewer.render(); }}, 0);
}};
$("panelShow").onclick = () => {{
  $("panel").style.display = "";
  $("panelShow").hidden = true;
  setTimeout(() => {{ viewer.resize(); viewer.render(); }}, 0);
}};

$("frameName").textContent = DATA.frame;
$("ccut").textContent = DATA.cluster_cutoff.toFixed(2);
$("nRepSum").textContent = DATA.donors.filter(d => d.is_rep).length;
$("nRep").textContent = DATA.donors.filter(d => d.is_rep).length;
$("nDonor").textContent = DATA.donors.length;
$("nClustered").textContent = DATA.donors.filter(d => d.cluster).length;
$("bLeg").innerHTML = `낮음 <span class="bbar"></span> 높음 ` +
  `(${{DATA.frame_b[0]}}–${{DATA.frame_b[1]}})`;
$("hdr").innerHTML = "쓸 수 있는 실험 구조 " + DATA.donors.length +
  "개를 <b>" + DATA.frame + "</b> 좌표계에 중첩 · 타겟 " + DATA.seq_len + " nt";
$("intro").innerHTML =
  "이 타겟의 예측은 <b>같은 RNA의 실험 구조에서 용매를 옮겨오는 것</b>이다. " +
  "아래 목록의 구조 하나하나가 Mg<sup>2+</sup>·K<sup>+</sup>·Na<sup>+</sup>·물을 " +
  "자기 좌표계에 갖고 있고, 그것을 기준 구조 <b>" + DATA.frame + "</b>의 좌표계로 " +
  "옮겨 모은 것이 제출본이 된다. 그래서 보고 있는 것은 <b>실제 예측 좌표</b>다.";

$("spBtns").innerHTML = SP.map(s =>
  `<button class="btn spBtn" data-s="${{s}}">` +
  `<span class="swatch" style="background:${{SPECIES_COLOR[s]}}"></span> ` +
  `${{SP_LABEL[s]}} <span class="pill" id="sp_${{s}}">0</span></button>`).join("") +
  `<button class="btn" id="spAll">전체 종류</button>`;

$("donors").innerHTML = DATA.donors.map((d, i) => {{
  const c = d.counts;
  const dim = d.cov < DATA.good_coverage ? " dim" : "";
  // One aligned grid row per structure. Hanging five pills off each of 41 rows
  // meant nothing lined up and the list could not be scanned; the per-species
  // breakdown moves into the row title, where it is read once.
  return `<div class="drow${{dim}}" title="Mg ${{c.MG}} · K ${{c.K}} · Na ${{c.NA}} · ` +
    `물 ${{c.HOH}} — identity ${{d.ident}} (정렬 구간 기준) · ` +
    `모델링된 ${{d.nt}} nt · superposition RMSD ${{d.rmsd}} A (P 원자 ${{d.nshared}}개)">` +
    `<input type="checkbox" class="dChk" data-i="${{i}}">` +
    `<span class="swatch" style="background:${{donorColor(i)}}"></span>` +
    `<button class="btn oneBtn" data-i="${{i}}" title="이 구조만 보기">${{d.pdb}}` +
    (d.cluster ? `<span class="cbadge">C${{d.cluster}}${{d.is_rep ? "\u2605" : ""}}</span>` : "") +
    `</button>` +
    `<span class="n">${{d.res ? d.res.toFixed(2) : "\u2013"}}</span>` +
    `<span class="n">${{d.cov}}%</span>` +
    `<span class="n">${{d.nsolv}}</span></div>`;
}}).join("");

// cluster summary: representative first, the rest listed under it
const CLUSTERS = {{}};
DATA.donors.forEach((d, i) => {{
  if (!d.cluster) return;
  (CLUSTERS[d.cluster] = CLUSTERS[d.cluster] || []).push(i);
}});
$("clusters").innerHTML = Object.keys(CLUSTERS)
  .sort((a, b) => CLUSTERS[b].length - CLUSTERS[a].length)
  .map(cid => {{
    const mem = CLUSTERS[cid];
    const rep = mem.find(i => DATA.donors[i].is_rep);
    const rest = mem.filter(i => i !== rep).map(i => DATA.donors[i].pdb);
    const solv = mem.reduce((a, i) => a + DATA.donors[i].nsolv, 0);
    const r = DATA.donors[rep];
    return `<div class="cgrp" data-c="${{cid}}">` +
      `<button class="btn cRepBtn" data-c="${{cid}}" title="이 클러스터의 대표만 보기">` +
      `C${{cid}} · ${{r.pdb}}</button>` +
      `<span class="pill">${{r.res ? r.res.toFixed(2) + " A" : "-"}}</span> ` +
      (mem.length > 1
        ? `<button class="btn cMemBtn" data-c="${{cid}}" title="이 클러스터의 구조 전부 보기">` +
          `전체 ${{mem.length}}개</button>` : `<span class="pill">단독</span>`) +
      `<span class="pill">용매 ${{solv}}</span>` +
      (rest.length ? `<div class="rest" style="margin-left:8px">${{rest.join(" ")}}</div>` : "") +
      `</div>`;
  }}).join("");

const selected = () => [...document.querySelectorAll(".dChk")]
  .filter(c => c.checked).map(c => +c.dataset.i);
const speciesOn = (kind) => soloSpecies === null || soloSpecies === kind;

if (CONS) {{
  $("consSec").hidden = false;
  $("nCons").textContent = CONS.kind.length;
  const sl = $("supMin");
  sl.max = Math.max(CONS.max_support, 1);
  sl.oninput = () => {{
    supMin = +sl.value;
    $("supMinL").textContent = supMin;
    render();
  }};
  $("showCons").addEventListener("change", render);
  $("consOnly").onclick = () => {{
    $("showCons").checked = true;
    markCluster(null);
    setSel(() => false);          // the merged layer alone, nothing under it
  }};
  $("consAll").onclick = () => {{ markCluster(null); setSel(() => true); }};
  document.querySelectorAll(".cmBtn").forEach(b => {{
    b.onclick = () => {{ consMode = b.dataset.m; render(); }};
  }});
}}

function renderConsLegend() {{
  const shown = {{}};
  let n = 0;
  for (let i = 0; i < CONS.kind.length; i++) {{
    if (CONS.support[i] < supMin || !speciesOn(CONS.kind[i])) continue;
    n++;
    const key = consMode === "species" ? CONS.kind[i]
      : (consMode === "rna" ? binOf(CONS.rna_min[i], RNA_BINS)[3]
                            : binOf(CONS.support[i], SUPPORT_BINS)[3]);
    shown[key] = (shown[key] || 0) + 1;
  }}
  const order = consMode === "species" ? SP.filter(s => shown[s])
    : (consMode === "rna" ? RNA_BINS : SUPPORT_BINS).map(b => b[3]).filter(k => shown[k]);
  const colorOf = (key) => {{
    if (consMode === "species") return SPECIES_COLOR[key];
    const bins = consMode === "rna" ? RNA_BINS : SUPPORT_BINS;
    return binColor(bins.find(b => b[3] === key));
  }};
  const label = (key) => consMode === "species" ? SP_LABEL[key]
    : (consMode === "support" ? key + "개 구조" : key);
  $("consLeg").innerHTML = order.map(k =>
    `<span class="k"><span class="swatch" style="background:${{colorOf(k)}}"></span>` +
    `${{label(k)}} <span class="mono">${{shown[k]}}</span></span>`).join("") +
    `<span class="k">표시 <span class="mono">${{n}}</span> / ${{CONS.kind.length}}</span>`;
  document.querySelectorAll(".cmBtn").forEach(b =>
    b.classList.toggle("on", b.dataset.m === consMode));
}}

function pickCons(i) {{
  consPick = i;
  const pdbs = CONS.donors[i];
  const votes = CONS.contested[i]
    ? ` <span class="warn">종류가 갈리는 자리</span>` : "";
  $("consInfo").classList.remove("empty");
  $("consInfo").innerHTML =
    `<b>합의 site #${{i + 1}}</b> · ${{SP_LABEL[CONS.kind[i]] || CONS.kind[i]}}${{votes}}<br>` +
    `지지 구조 <b class="mono">${{CONS.support[i]}}</b>개 · 관측 ` +
    `<span class="mono">${{CONS.n_obs[i]}}</span>회 · RNA까지 ` +
    `<span class="mono">${{CONS.rna_min[i]}}</span> &Aring;<br>` +
    (pdbs.length
      ? pdbs.map(j => DATA.donors[j].pdb).join(" ") +
        `<br><button class="btn" id="consSel">이 구조들만 선택</button>`
      : "이 자리를 지지한 구조가 목록에 없다");
  const btn = $("consSel");
  if (btn) btn.onclick = () => {{
    markCluster(null);
    setSel((d, j) => pdbs.includes(j));
  }};
  render();
}}

function render() {{
  viewer.clear();
  const sel = selected(), byDonor = $("colorByDonor").checked;
  // A nucleic cartoon per structure is ~2.3k atoms; all 41 at once is 96k and
  // 3Dmol will not finish building them. Fall back to lines past a handful
  // unless the mode was chosen by hand.
  const rnaStyle = rnaMode === "auto" ? (sel.length <= 8 ? "ribbon" : "tube") : rnaMode;
  document.querySelectorAll(".rnaBtn").forEach(b =>
    b.classList.toggle("on", b.dataset.m === rnaMode));
  if ($("showFrame").checked && DATA.frame_pdb) {{
    const m = viewer.addModel(DATA.frame_pdb, "pdb");
    // 'oval' is 3Dmol's nucleic-acid ribbon; the base ladder makes the fold
    // readable, which a backbone tube alone does not. Colouring by B-factor
    // shows which parts of the fold are ordered enough for the solvent around
    // them to mean anything — a flat grey tube says nothing and reads as a
    // black stick against the site spheres.
    const cart = {{ style: "oval", thickness: 0.6, opacity: 1.0 }};
    if ($("colorB").checked) {{
      // rwb puts white in the middle, which vanishes on a white background;
      // roygb keeps every value visible there.
      cart.colorscheme = {{ prop: "b", gradient: whiteBg ? "roygb" : "rwb",
                           min: DATA.frame_b[1], max: DATA.frame_b[0] }};
    }} else {{
      cart.color = whiteBg ? "#5b6472" : "#8892a4";
    }}
    m.setStyle({{}}, {{ cartoon: cart }});
  }}
  const tot = {{ MG: 0, K: 0, NA: 0, HOH: 0 }};
  sel.forEach(i => {{
    const d = DATA.donors[i], col = donorColor(i);
    if (rnaStyle !== "off" && d.chain) {{
      const m = viewer.addModel(chainPdb(i), "pdb");
      if (rnaStyle === "ribbon") {{
        m.setStyle({{}}, {{ cartoon: {{ color: col, style: "oval",
                                     thickness: 0.45, opacity: 0.9 }} }});
      }} else if (rnaStyle === "tube") {{
        // 3Dmol builds cartoon geometry only from its backbone set
        // (P, O5', O3', C5', C2', N1, N3 for nucleic acids). Restricting the
        // style to P leaves one control point per residue instead of four, so
        // all 41 chains cost ~16k points rather than ~65k — still a ribbon.
        m.setStyle({{ atom: "P" }}, {{ cartoon: {{ color: col, style: "trace",
                                              thickness: 0.3, opacity: 0.85 }} }});
      }} else {{
        m.setStyle({{}}, {{ line: {{ color: col, opacity: 0.7 }} }});
      }}
    }}
    if (!d.solvent_pdb) return;
    const m = viewer.addModel(d.solvent_pdb, "pdb");
    m.setStyle({{}}, {{ }});                       // hide, then show per species
    SP.forEach(s => {{
      const n = d.counts[s] || 0;
      if (!n || !speciesOn(s)) return;
      tot[s] += n;
      m.setStyle({{ resn: s }}, {{ sphere: {{ radius: SPECIES_RADIUS[s],
                                          color: byDonor ? col : SPECIES_COLOR[s],
                                          opacity: s === "HOH" ? 0.8 : 0.95 }} }});
    }});
    m.__donor = i;
  }});
  let consShown = 0;
  if (CONS && $("showCons").checked) {{
    const m = viewer.addModel(CONS.pdb, "pdb");
    m.setStyle({{}}, {{ }});
    // One setStyle per colour/radius group, not per site: 829 individual calls
    // rebuild the geometry 829 times and the page stops responding.
    const groups = {{}};
    for (let i = 0; i < CONS.kind.length; i++) {{
      if (CONS.support[i] < supMin || !speciesOn(CONS.kind[i])) continue;
      consShown++;
      const key = consColor(i) + "|" + consRadius(i).toFixed(2);
      (groups[key] = groups[key] || []).push(i + 1);
    }}
    Object.keys(groups).forEach(key => {{
      const part = key.split("|");
      m.setStyle({{ resi: groups[key] }},
                 {{ sphere: {{ radius: +part[1], color: part[0], opacity: 0.95 }} }});
    }});
    if (consPick !== null && CONS.support[consPick] >= supMin) {{
      m.setStyle({{ resi: [consPick + 1] }},
                 {{ sphere: {{ radius: consRadius(consPick) * 2.0,
                            color: whiteBg ? "#1a1d23" : "#ffffff", opacity: 0.45 }} }});
    }}
    m.setClickable({{}}, true, (atom) => pickCons(atom.resi - 1));
  }}
  if (CONS) renderConsLegend();   // counts follow the species filter either way
  SP.forEach(s => {{ const e = $("sp_" + s); if (e) e.textContent = tot[s]; }});
  const sum = SP.reduce((a, s) => a + tot[s], 0);
  $("totals").innerHTML =
    SP.map(s => `<tr><td><span class="swatch" style="background:${{SPECIES_COLOR[s]}}"></span> ` +
                `${{SP_LABEL[s]}}</td><td class="n">${{tot[s]}}</td></tr>`).join("") +
    `<tr><td><b>표시된 관측</b></td><td class="n"><b>${{sum}}</b></td></tr>` +
    (CONS ? `<tr><td><b>표시된 합의 site</b></td><td class="n"><b>${{consShown}}</b></td></tr>` : "") +
    `<tr><td>구조</td><td class="n">${{sel.length}} / ${{DATA.donors.length}}</td></tr>`;

  // hover: identify a site without having to hunt for it in the table
  viewer.setHoverable({{}}, true,
    (atom, vw) => {{
      if (atom.label) return;
      let txt;
      if (CONS && atom.chain === CONS.chain) {{
        const i = atom.resi - 1;
        txt = `합의 site #${{i + 1}} · ${{SP_LABEL[CONS.kind[i]] || CONS.kind[i]}} · ` +
              `지지 ${{CONS.support[i]}}개 구조 · 관측 ${{CONS.n_obs[i]}}회 · ` +
              `RNA ${{CONS.rna_min[i]}} A` +
              (CONS.contested[i] ? " · 종류가 갈림" : "") + " — 눌러서 지지 구조 보기";
      }} else {{
        const site = ownerSite(atom);
        txt = site
          ? `${{SP_LABEL[site.kind] || site.kind}} · ${{site.pdb}} · B ${{site.b}} · ` +
            `구조 ${{site.support}}개가 이 자리에 site를 놓음`
          : `${{atom.resn}} · ${{atom.chain}}`;
      }}
      atom.label = vw.addLabel(txt, {{ position: atom, fontSize: 11,
                                      backgroundColor: whiteBg ? "#ffffff" : "#11151d",
                                      backgroundOpacity: 0.92,
                                      fontColor: whiteBg ? "#1a1d23" : "#e6e8ee" }});
    }},
    (atom, vw) => {{ if (atom.label) {{ vw.removeLabel(atom.label); delete atom.label; }} }});
  viewer.render();
}}

// map a hovered atom back to the site record it came from
function ownerSite(atom) {{
  for (const i of selected()) {{
    const d = DATA.donors[i];
    if (!d.solvent_pdb) continue;
    const s = d.sites[atom.serial - 1];
    if (s && s[0] === atom.resn &&
        Math.abs(s[1] - atom.x) < 0.02 && Math.abs(s[2] - atom.y) < 0.02) {{
      return {{ kind: s[0], pdb: d.pdb, b: s[4], support: s[5] === undefined ? "?" : s[5] }};
    }}
  }}
  return null;
}}

function setSel(pred) {{
  document.querySelectorAll(".dChk").forEach(c => {{
    c.checked = pred(DATA.donors[+c.dataset.i], +c.dataset.i);
  }});
  render();
}}
$("dAll").onclick = () => setSel(() => true);
$("dNone").onclick = () => setSel(() => false);
$("dFrame").onclick = () => setSel(d => d.pdb === DATA.frame);
$("dSolv").onclick = () => {{
  const rank = DATA.donors.map((d, i) => [d.nsolv, i]).sort((a, b) => b[0] - a[0])
                          .slice(0, 10).map(x => x[1]);
  setSel((d, i) => rank.includes(i));
}};
document.querySelectorAll(".spBtn").forEach(b => {{
  b.onclick = () => {{
    soloSpecies = (soloSpecies === b.dataset.s) ? null : b.dataset.s;
    document.querySelectorAll(".spBtn").forEach(x =>
      x.classList.toggle("on", x.dataset.s === soloSpecies));
    render();
  }};
}});
$("spAll").onclick = () => {{
  soloSpecies = null;
  document.querySelectorAll(".spBtn").forEach(x => x.classList.remove("on"));
  render();
}};
function markCluster(cid) {{
  document.querySelectorAll(".cgrp").forEach(g =>
    g.classList.toggle("on", cid !== null && g.dataset.c === String(cid)));
}}
$("cRep").onclick = () => {{ markCluster(null); setSel(d => !!d.is_rep); }};
$("cAll").onclick = () => {{ markCluster(null); setSel(d => !!d.cluster); }};
document.querySelectorAll(".cRepBtn").forEach(b => {{
  b.onclick = () => {{
    markCluster(b.dataset.c);
    setSel(d => d.is_rep && String(d.cluster) === b.dataset.c);
    viewer.zoomTo();
    viewer.render();
  }};
}});
document.querySelectorAll(".oneBtn").forEach(b => {{
  b.onclick = () => {{
    markCluster(null);
    setSel((d, i) => i === +b.dataset.i);
    viewer.zoomTo();
    viewer.render();
  }};
}});
document.querySelectorAll(".cMemBtn").forEach(b => {{
  b.onclick = () => {{
    markCluster(b.dataset.c);
    setSel(d => String(d.cluster) === b.dataset.c);
    viewer.zoomTo();
    viewer.render();
  }};
}});
document.querySelectorAll(".rnaBtn").forEach(b => {{
  b.onclick = () => {{ rnaMode = b.dataset.m; render(); }};
}});
$("bgBtn").onclick = () => setBg(!whiteBg);
$("allChains").onclick = () => {{
  rnaMode = "ribbon";
  markCluster(null);
  setSel(() => true);
  viewer.zoomTo();
  viewer.render();
}};
document.querySelectorAll(".dChk,#showFrame,#colorByDonor,#colorB")
  .forEach(el => el.addEventListener("change", render));

$("gloss").innerHTML = {gloss};
$("notes").innerHTML = {notes};

setSel(() => true);        // every usable structure, as ribbons
viewer.zoomTo();
viewer.render();
</script>
"""


def _check_js(page: str) -> None:
    """Syntax-check the emitted page script.

    ``PAGE`` is a plain triple-quoted string, so an escape written into the
    JavaScript is interpreted by Python first — a ``\\n`` meant for a JS string
    literal becomes a real newline and kills the whole script silently. The page
    still loads; the panel is simply never built. Catch it at build time.
    """
    import shutil
    import subprocess
    import tempfile
    node = shutil.which("node")
    if not node:
        print("  NOTE: node not found — page script not syntax-checked")
        return
    body = page[page.rindex("<script>") + len("<script>"):page.rindex("</script>")]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(body)
        tmp = fh.name
    r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    Path(tmp).unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit("emitted page script is not valid JavaScript:\n"
                         + (r.stderr or r.stdout))
    print("  page script: valid JavaScript")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--templates", nargs="+", type=Path, required=True)
    ap.add_argument("--frame", default="3G78")
    ap.add_argument("--offset", type=int, default=NUMBERING_OFFSET)
    ap.add_argument("--consensus", type=Path, default=None,
                    help="consensus_sites.json from cluster_solvent_donors.py; "
                         "adds the merged-site layer")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.consensus and not args.consensus.is_file():
        raise SystemExit(f"--consensus {args.consensus} does not exist")
    sequence = _read_sequence(args.sequence)
    cifs = [p for p in args.templates if p.is_file()]
    payload, warnings = build(args.target_id, sequence, cifs,
                              args.frame, args.offset, args.consensus)
    good = [d for d in payload["donors"] if d["cov"] >= GOOD_COVERAGE]
    n_rep = sum(1 for d in payload["donors"] if d.get("is_rep"))

    gloss = (
        f"<dt>cov (coverage)</dt><dd>{payload['seq_len']} nt 타겟 중 이 구조가 실제로 "
        "모델링한 비율. <b>진짜 실험 구조와 검색 잡음을 가르는 값</b> — 설계 RNA나 단독 "
        "도메인은 여기서 몇 %밖에 안 나온다. 85 % 미만은 숨기지 않고 흐리게 표시.</dd>"
        "<dt>identity</dt><dd>행에 커서를 올리면 나온다. <b>정렬된 구간에 대해서만</b> "
        "계산된 일치도라, 26 nt 조각이 완벽히 맞아도 1.00이 된다. 이 값만으로는 "
        "쓸 구조를 고를 수 없다.</dd>"
        f"<dt>superposition RMSD</dt><dd>제출 빌더와 같은 변환을 적용했을 때 그 구조의 "
        f"인산 백본이 {payload['frame']}에 얼마나 맞는지. 실제로 이 분자인 구조는 0.5~2.7 &Aring;, "
        "아닌 것은 수십 &Aring; 어긋난다.</dd>"
        "<dt>Mg / K / Na / W</dt><dd>그 구조가 기탁한 이온과 물의 개수. Tl<sup>+</sup>는 "
        "K<sup>+</sup>로 읽는다 — 탈륨은 칼륨 자리를 찾을 때 쓰는 표준 중원자 "
        "대체물이다.</dd>"
        f"<dt>support</dt><dd>용매 구슬에 커서를 올리면 나온다. 그 자리에서 "
        f"{payload['support_radius']} &Aring; 안에 site를 놓는 <b>서로 다른 실험 구조의 수</b>. "
        "이 분자인 구조만 센다. 제출본의 정밀도 모델이 임계값을 거는 바로 그 수치다 — "
        "한 번만 나온 자리는 추측이고, 열 개 결정에서 나온 자리는 아니다.</dd>"
        "<dt>B</dt><dd>기탁자가 준 B-factor. 낮을수록 단단히 정렬된 자리다.</dd>"
        "<dt>기준 구조 (frame)</dt><dd>모든 구조를 여기 좌표계로 중첩한다. 제출본의 "
        "용매도 이 좌표계에서 예측되므로, 화면에 보이는 것이 실제 예측 좌표다.</dd>"
        f"<dt>형태 클러스터 (C1, C2 …)</dt><dd>백본 RMSD가 "
        f"{CLUSTER_CUTOFF:.2f} &Aring; 이하인 구조끼리 묶은 것. 이 목록의 상당수는 "
        "같은 결정을 다시 정제했거나 금속만 바꿔 담근 것이라 사실상 같은 형태다. "
        "&#9733; 표시가 대표(클러스터 내 최고 해상도). <b>같은 클러스터의 구조들을 "
        "독립 증거로 세면 support가 부풀려진다.</b></dd>"
        "<dt>B-factor 색칠</dt><dd>기준 구조 리본에 적용. 파랑이 낮고(단단히 정렬) "
        "빨강이 높다(흐물거림). B가 높은 구간 주변의 용매 자리는 그만큼 덜 믿을 만하다.</dd>"
    )
    if "consensus" in payload:
        c = payload["consensus"]
        n_multi = sum(1 for v in c["support"] if v >= 2)
        gloss += (
            f"<dt>합의 site</dt><dd>위 구조들의 관측 "
            f"{sum(c['n_obs'])}개를 병합해 남은 <b>{len(c['kind'])}개</b> 자리. "
            "병합 기준은 고정 거리가 아니라 <b>그 종류 쌍이 실제로 가질 수 있는 최단 "
            "거리</b>다 — 물-물 2.4 &Aring;, Mg-O 1.95 &Aring; 같은 값. 그래서 살아남은 "
            "site들은 한 모델에 동시에 써도 서로 부딪히지 않는다.</dd>"
            f"<dt>지지 구조 수</dt><dd>그 자리를 놓은 <b>서로 다른 결정의 수</b>. "
            f"2개 이상이 {n_multi}개, 1개뿐인 자리가 {len(c['kind']) - n_multi}개다. "
            "<b>높다고 맞다는 뜻은 아니다</b> — 목록의 상당수가 같은 결정의 재정제라 "
            "서로 독립이 아니다. 위의 형태 클러스터가 그 중복을 보여준다.</dd>"
            "<dt>RNA 거리</dt><dd>합의 site에서 기준 구조의 가장 가까운 RNA 원자까지. "
            "4 &Aring;를 넘으면 결합한 자리가 아니라 벌크 용매에 가깝고, 채점 대상이 "
            "되기 어렵다.</dd>"
            "<dt>종류가 갈리는 자리</dt><dd>어떤 결정은 Mg로, 다른 결정은 물로 "
            "모델링한 자리. 자리가 둘인 게 아니라 <b>하나의 자리에 대한 해석이 "
            "갈리는</b> 것이다.</dd>"
        )
    dropped = payload["skipped"]
    dropped_txt = ", ".join(f"{pdb} {cov:.0f}%" for pdb, cov, _r in dropped[:8])
    note = (
        f"목록은 타겟의 {GOOD_COVERAGE:.0f} % 이상을 덮는 실험 구조 {len(good)}개다. "
        f"기본 선택은 <b>형태 클러스터 대표 {n_rep}개</b> — 서로 다른 RNA 형태 하나당 "
        "구조 하나다.<br><br>"
        f"서열 검색에 걸린 나머지 {len(dropped)}개는 <b>목록에서 뺐다</b>: 부분 구조, "
        "Domain 1 조각, 설계 RNA, 다른 균주의 인트론이라 이 분자가 아니고 용매를 "
        f"가져올 대상이 아니다 ({dropped_txt}{' 등' if len(dropped) > 8 else ''}). "
        "RCSB identity는 정렬된 구간에 대해서만 계산되므로 26 nt짜리 설계 RNA도 "
        "1.00으로 통과한다 — 그래서 coverage로 걸렀다."
    )
    if warnings:
        note += "<br><br><span class='warn'>" + "<br>".join(
            html.escape(w) for w in warnings) + "</span>"

    asset = ASSET.read_text() if ASSET.exists() else ""
    if not asset:
        print(f"WARNING: {ASSET} missing — page will not render offline")
    page = PAGE.format(
        title=f"{args.target_id} — ordered-solvent donors",
        asset=asset,
        data=json.dumps(payload, separators=(",", ":")),
        species_color=json.dumps(SPECIES_COLOR),
        species_color_light=json.dumps(SPECIES_COLOR_LIGHT),
        species_radius=json.dumps(SPECIES_RADIUS),
        support_bins=json.dumps([list(b) for b in SUPPORT_BINS]),
        gloss=json.dumps(gloss),
        notes=json.dumps(note),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    _check_js(page)

    tot = {k: sum(d["counts"][k] for d in good) for k in SOLVENT_NAMES}
    print(f"  donors loaded: {len(payload['donors'])}  "
          f"(coverage >= {GOOD_COVERAGE:.0f}%: {len(good)})")
    print("  solvent in the default selection: "
          + "  ".join(f"{k}={v}" for k, v in tot.items()))
    if "consensus" in payload:
        c = payload["consensus"]
        print(f"  consensus sites: {len(c['kind'])} from {sum(c['n_obs'])} "
              f"observations  (support >= 2: {sum(1 for v in c['support'] if v >= 2)}, "
              f"max {c['max_support']})  chain {c['chain']}")
    for w in warnings:
        print(f"  WARN {w}")
    print(f"built {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
